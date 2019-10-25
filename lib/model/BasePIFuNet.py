import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import index, orthogonal, perspective

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        """
        :param projection_mode:
        Either orthogonal or perspective.
        It will call the corresponding function for projection.
        :param error_term:
        nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        """
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None

    def forward(self, points, images, calibs, transforms=None):
        '''
        :param points: [B, 3, N] world space coordinates of points
        :param images: [B, C, H, W] input images
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :return: [B, Res, N] predictions for each point
        '''
        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        None

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
        None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds

    def get_error(self):
        '''
        Get the network loss from the last query
        :return: loss term
        '''
        return self.error_term(self.preds, self.labels)
