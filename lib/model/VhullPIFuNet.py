import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet


class VhullPIFuNet(BasePIFuNet):
    '''
    Vhull Piximp network is a minimal network demonstrating how the template works
    also, it helps debugging the training/test schemes
    It does the following:
        1. Compute the masks of images and stores under self.im_feats
        2. Calculate calibration and indexing
        3. Return if the points fall into the intersection of all masks
    '''

    def __init__(self,
                 num_views,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(VhullPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)
        self.name = 'vhull'

        self.num_views = num_views

        self.im_feat = None

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        # If the image has alpha channel, use the alpha channel
        if images.shape[1] > 3:
            self.im_feat = images[:, 3:4, :, :]
        # Else, tell if it's not white
        else:
            self.im_feat = images[:, 0:1, :, :]

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        point_local_feat = self.index(self.im_feat, xy)
        local_shape = point_local_feat.shape
        point_feat = point_local_feat.view(
            local_shape[0] // self.num_views,
            local_shape[1] * self.num_views,
            -1)
        pred = torch.prod(point_feat, dim=1)

        self.preds = pred.unsqueeze(1)
