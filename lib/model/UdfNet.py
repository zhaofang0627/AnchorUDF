import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net


class UdfNet(BasePIFuNet):

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss(reduction='none'),
                 ):
        super(UdfNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'UDFNet'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        self.surface_regressor = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            merge_layer=self.opt.merge_layer,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.ReLU())

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        z_feat = self.normalizer(z, calibs=calibs)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        intermediate_phi_list = []

        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            if self.opt.coord_xyz:
                point_local_feat_list = [self.index(im_feat, xy), xyz]
            else:
                point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            if self.opt.merge_layer != 0:
                pred, phi = self.surface_regressor(point_local_feat)
                intermediate_phi_list.append(phi)
            else:
                pred = self.surface_regressor(point_local_feat)

            self.intermediate_preds_list.append(pred)

        if self.opt.merge_layer != 0:
            self.phi = intermediate_phi_list[-1]

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
            loss_i = self.error_term(torch.clamp(preds.squeeze(), max=self.opt.max_dist), torch.clamp(self.labels, max=self.opt.max_dist))
            loss = loss_i.sum(-1).mean()

            error += loss

        error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images, points, calibs, transforms=None, labels=None):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        error = self.get_error()

        return res, error