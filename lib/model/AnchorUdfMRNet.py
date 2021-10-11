import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net


class AnchorUdfMRNet(BasePIFuNet):

    def __init__(self,
                 opt,
                 netG,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss(reduction='none'),
                 ):
        super(AnchorUdfMRNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'AnchorUDFMRNet'

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
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, transforms=None, labels=None, key_points=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        if self.opt.anchor:
            self.netG.query(points=points, calibs=calibs, transforms=transforms, labels=labels, key_points=key_points)
        else:
            self.netG.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        z_feat = self.netG.phi

        if not self.opt.joint_train:
            z_feat = z_feat.detach()

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            pred = self.surface_regressor(point_local_feat)
            self.intermediate_preds_list.append(pred)

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
            loss_i = self.error_term(torch.clamp(preds.squeeze(), max=self.opt.max_dist), torch.clamp(self.labels, max=self.opt.max_dist))  # out = (B,num_points)
            loss = loss_i.sum(-1).mean()  # loss_i summed over all #num_points -> out = (B,1) and mean over batch -> out = (1)

            error += loss

        error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images, images_low, points, calibs, transforms=None, labels=None, key_points=None):
        # Get image feature
        self.filter_global(images_low)

        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels, key_points=key_points)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        error = self.get_error()

        if self.opt.joint_train:
            if self.opt.anchor:
                error_netG, error_anchor_netG = self.netG.get_error()

                return res, error, error_netG, error_anchor_netG
            else:
                error_netG = self.netG.get_error()

                return res, error, error_netG

        else:
            return res, error
