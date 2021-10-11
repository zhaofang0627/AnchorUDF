import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
from .. import diff_operators


def point_netG_coord_xyz_func(netG, feat_im, feat_im_netG, feat_pc_netG, points):
    xy = points[:, :2, :]
    xyz_pc = points.transpose(1, 2)
    xyz_pc = xyz_pc[:, :, None, None, :]
    xyz_feat_pc = F.grid_sample(feat_pc_netG, xyz_pc, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1).squeeze(-1)  # out : (B,C,num_points)

    _, phi = netG.surface_regressor(torch.cat([index(feat_im_netG, xy), xyz_feat_pc, points], 1))

    return torch.cat([index(feat_im, xy), phi], 1)


class AnchorUdfMRDetachNet(BasePIFuNet):

    def __init__(self,
                 opt,
                 netG,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss(reduction='none'),
                 ):
        super(AnchorUdfMRDetachNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'AnchorUDFMRDetachNet'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilterHD(opt)

        self.surface_regressor = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_hd,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.ReLU())

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        init_net(self)

        self.netG = netG

    def filter_global(self, images):
        '''
        Filter the input images
        store all intermediate features.
        '''
        if self.opt.joint_train:
            self.netG.filter(images)
        else:
            with torch.no_grad():
                self.netG.filter(images)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        '''
        with torch.no_grad():
            self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        self.im_feat = self.im_feat_list[-1]

    def query(self, points, calibs, transforms=None, labels=None, key_points=None, neighbors=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        '''
        if labels is not None:
            self.labels = labels

        if neighbors is not None:
            self.neighbors = self.projection(neighbors, calibs, transforms)

        self.proj_points = self.projection(points, calibs, transforms)
        xy = self.proj_points[:, :2, :]

        self.netG.query(points=points, calibs=calibs, transforms=transforms, labels=labels, key_points=key_points)

        z_feat = self.netG.phi

        if not self.opt.joint_train:
            z_feat = z_feat.detach()

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        self.im_feat = self.im_feat.detach()
        self.im_feat.requires_grad = True

        point_local_feat_list = [self.index(self.im_feat, xy), z_feat]

        if self.opt.skip_hourglass:
            point_local_feat_list.append(tmpx_local_feature)

        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred = self.surface_regressor(point_local_feat)
        self.intermediate_preds_list.append(pred)

        self.point_local_feats = point_local_feat
        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            loss_i = self.error_term(torch.clamp(preds.squeeze(), max=self.opt.max_dist), torch.clamp(self.labels, max=self.opt.max_dist))  # out = (B,num_points)
            loss = loss_i.sum(-1).mean()  # loss_i summed over all #num_points -> out = (B,1) and mean over batch -> out = (1)

            error += loss

        error /= len(self.intermediate_preds_list)

        if self.opt.grad_constraint:
            self.proj_points.requires_grad = True
            gradient_dfeat = diff_operators.get_gradient(self.surface_regressor, self.point_local_feats)

            jacob_proj_points = diff_operators.get_batch_jacobian_netG(point_netG_coord_xyz_func, self.netG, self.im_feat, self.netG.im_feat, self.netG.fea_grid, self.proj_points, self.point_local_feats.shape[1])

            gradient = (gradient_dfeat.unsqueeze(2) * jacob_proj_points.detach()).sum(dim=1)

            grad_valid = (self.preds.squeeze(1) < self.opt.max_dist).float().detach()

            gt_direct = self.proj_points - self.neighbors
            direct_constraint = (1.0 - F.cosine_similarity(gradient, gt_direct, dim=1)) * grad_valid
            error_direct = direct_constraint.sum(-1).mean()

            return error, error_direct

        else:
            return error

    def forward(self, images, images_low, points, calibs, transforms=None, labels=None, key_points=None, neighbors=None):
        # Get image feature
        self.filter_global(images_low)

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
