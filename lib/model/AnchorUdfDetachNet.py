import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.spatial import cKDTree as KDTree
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
import data_processing.implicit_waterproofing as iw
from .PointNet import PointNet
from .. import diff_operators
from ..geometry import index


def point_coord_xyz_func(feat_im, feat_pc, points):
    xy = points[:, :2, :]
    xyz_pc = points.transpose(1, 2)
    xyz_pc = xyz_pc[:, :, None, None, :]
    xyz_feat_pc = F.grid_sample(feat_pc, xyz_pc, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)  # out : (B,C,num_points)

    return torch.cat([index(feat_im, xy), xyz_feat_pc, points], 1)


class AnchorUdfDetachNet(BasePIFuNet):

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss(reduction='none'),
                 ):
        super(AnchorUdfDetachNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'AnchorUDFDetachNet'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilterAnchor(opt)

        self.surface_regressor = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            merge_layer=self.opt.merge_layer,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.ReLU())

        self.normalizer = DepthNormalizer(opt)

        L = 5
        self.freq = (2.0 ** torch.linspace(0, L-1, L).cuda()) * math.pi

        # Point net
        self.reso_grid = opt.reso_grid

        bb_min = -1.0
        bb_max = 1.0

        self.grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, self.reso_grid)
        self.kdtree = KDTree(self.grid_points)

        self.svr_net = PointNet(hidden_dim=opt.pn_hid_dim)

        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.hg_pc = None
        self.key_points = None
        self.proj_points = None
        self.point_local_feats = None
        self.fea_grid = None
        self.neighbors = None
        self.im_feat = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        with torch.no_grad():
            self.im_feat_list, self.tmpx, self.normx, self.hg_pc = self.image_filter(images)

        self.im_feat = self.im_feat_list[-1]

    def query(self, points, calibs, transforms=None, labels=None, key_points=None, neighbors=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        '''
        if labels is not None:
            self.labels = labels

        if key_points is not None:
            self.key_points = self.projection(key_points, calibs, transforms)

        if neighbors is not None:
            self.neighbors = self.projection(neighbors, calibs, transforms)

        self.proj_points = self.projection(points, calibs, transforms)

        xy = self.proj_points[:, :2, :]

        occupancies = self.hg_pc.new_zeros(self.hg_pc.size(0), len(self.grid_points))
        kp_pred = self.hg_pc.transpose(1, 2).detach().cpu().numpy()

        for b in range(self.hg_pc.size(0)):
            _, idx = self.kdtree.query(kp_pred[b])
            occupancies[b, idx] = 1

        voxel_kp_pred = occupancies.view(self.hg_pc.size(0), self.reso_grid, self.reso_grid, self.reso_grid)

        with torch.no_grad():
            self.fea_grid = self.svr_net(voxel_kp_pred.detach())

        vgrid = self.proj_points.transpose(1, 2)
        vgrid = vgrid[:, :, None, None, :]

        self.fea_grid = self.fea_grid.detach()
        self.fea_grid.requires_grad = True

        # acutally trilinear interpolation if mode = 'bilinear'
        xyz_fea_grid = F.grid_sample(self.fea_grid, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)  # out : (B,C,num_sample_inout)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        self.im_feat = self.im_feat.detach()
        self.im_feat.requires_grad = True

        point_local_feat_list = [self.index(self.im_feat, xy), xyz_fea_grid, self.proj_points]

        if self.opt.skip_hourglass:
            point_local_feat_list.append(tmpx_local_feature)

        point_local_feat = torch.cat(point_local_feat_list, 1)

        if self.opt.merge_layer != 0:
            pred, phi = self.surface_regressor(point_local_feat)
            self.phi = phi
        else:
            pred = self.surface_regressor(point_local_feat)

        self.intermediate_preds_list.append(pred)

        self.point_local_feats = point_local_feat
        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            loss_i = self.error_term(torch.clamp(preds.squeeze(1), max=self.opt.max_dist), torch.clamp(self.labels, max=self.opt.max_dist))  # out = (B,num_points)
            loss = loss_i.sum(-1).mean()  # loss_i summed over all #num_points -> out = (B,1) and mean over batch -> out = (1)

            error += loss

        error /= len(self.intermediate_preds_list)

        if self.opt.grad_constraint:
            self.proj_points.requires_grad = True
            gradient_dfeat = diff_operators.get_gradient(self.surface_regressor, self.point_local_feats)
            jacob_proj_points = diff_operators.get_batch_jacobian(point_coord_xyz_func, self.im_feat, self.fea_grid, self.proj_points, self.point_local_feats.shape[1])

            gradient = (gradient_dfeat.unsqueeze(2) * jacob_proj_points.detach()).sum(dim=1)

            grad_valid = (self.preds.squeeze(1) < self.opt.max_dist).float().detach()

            gt_direct = self.proj_points - self.neighbors
            direct_constraint = (1.0 - F.cosine_similarity(gradient, gt_direct, dim=1)) * grad_valid
            error_direct = direct_constraint.sum(-1).mean()

            return error, error_direct

        else:
            return error

    def forward(self, images, points, calibs, transforms=None, labels=None, key_points=None, neighbors=None):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels, key_points=key_points, neighbors=neighbors)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        if self.opt.grad_constraint:
            error, error_direct = self.get_error()
            return res, error, error_direct

        else:
            error = self.get_error()
            return res, error