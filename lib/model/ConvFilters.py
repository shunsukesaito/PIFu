import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import torchvision.models.vgg as vgg


class MultiConv(nn.Module):
    def __init__(self, filter_channels):
        super(MultiConv, self).__init__()
        self.filters = []

        for l in range(0, len(filter_channels) - 1):
            self.filters.append(
                nn.Conv2d(filter_channels[l], filter_channels[l + 1], kernel_size=4, stride=2))
            self.add_module("conv%d" % l, self.filters[l])

    def forward(self, image):
        '''
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        '''
        y = image
        # y = F.relu(self.bn0(self.conv0(y)), True)
        feat_pyramid = [y]
        for i, f in enumerate(self.filters):
            y = f(y)
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            # y = F.max_pool2d(y, kernel_size=2, stride=2)
            feat_pyramid.append(y)
        return feat_pyramid


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


class ResNet(nn.Module):
    def __init__(self, model='resnet18'):
        super(ResNet, self).__init__()

        if model == 'resnet18':
            net = resnet.resnet18(pretrained=True)
        elif model == 'resnet34':
            net = resnet.resnet34(pretrained=True)
        elif model == 'resnet50':
            net = resnet.resnet50(pretrained=True)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv1 = net.conv1

        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, image):
        '''
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        '''

        y = image
        feat_pyramid = []
        y = self.layer0(y)
        feat_pyramid.append(y)
        y = self.layer1(self.pool(y))
        feat_pyramid.append(y)
        y = self.layer2(y)
        feat_pyramid.append(y)
        y = self.layer3(y)
        feat_pyramid.append(y)
        y = self.layer4(y)
        feat_pyramid.append(y)

        return feat_pyramid
