import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

import torchvision.models as models
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Decoder, self).__init__()

        self.conv = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1)  # keep ratio
        self.conv_trans = nn.ConvTranspose2d(mid_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        x = F.relu(self.conv_trans(x), inplace=True)
        return x


import torch.nn as nn
import torchvision.models


def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )


class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        # self.conv_last = nn.Conv2d(64, n_class, 1)
        self.conv_last = nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1, padding_mode='reflect')
        nn.init.constant_(self.conv_last.weight, 0.)
        nn.init.constant_(self.conv_last.bias, 0.)



    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class ResNetZoom(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetZoom, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_last = nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1, padding_mode='reflect')
        # self.tanh = nn.Tanh()
        # self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        # self.flatten = nn.Flatten()
        # self.fwhm_predicter = nn.Linear(49, 1)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        nn.init.constant_(self.conv_last.weight, 0.)
        nn.init.constant_(self.conv_last.bias, 0.)
        #
        # nn.init.normal_(self.conv_last, std=0.001)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)

        # x = self.relu(x)
        x = self.conv_last(x)
        # x = self.bn2(x)

        # x = self.tanh(x)

        return x


class FovSimModule(nn.Module):
    def __init__(self, cfg, in_channels=3, out_channels=4):
        # in_channels: num of channels corresponds to input image channels, e.g. 3
        # out_channels: num of channels corresponds to num of sclaes tested
        super(FovSimModule, self).__init__()
        BN_MOMENTUM = 0.1
        self.cfg = cfg
        self.fov_expand_1 = nn.Conv2d(in_channels=in_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False, stride=2)
        self.fov_expand_2 = nn.Conv2d(in_channels=8*out_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False, stride=2)
        # self.fov_squeeze_0 = nn.Conv2d(in_channels=16*out_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False, stride=1)
        self.fov_squeeze_1 = nn.Conv2d(in_channels=8*out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False, stride=2)
        # bn
        self.norm1 = nn.BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
        self.norm2 = nn.BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
        # self.norm0 = nn.BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
        self.norm3 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.act = nn.ReLU6(inplace=False)

        self.conv_last = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, x, reset_grad=True, train_mode=True):
        layer1 = self.act(self.norm1(self.fov_expand_1(x)))
        layer2 = self.act(self.norm2(self.fov_expand_2(layer1)))
        # layer2_5 = self.act(self.norm0(self.fov_squeeze_0(layer2)))
        layer3 = self.norm3(self.fov_squeeze_1(layer2))

        layer4 = self.conv_last(self.act(layer3))
        output = layer4
        return output


def saliency_network_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetZoom(BasicBlock, [2, 2, 2, 2], **kwargs).cuda()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        # pretrained_dict = torch.load('/root/autodl-tmp/too_warp_saliency.pth')
        # model.load_state_dict(pretrained_dict, strict=True)
        # for param in model.parameters():
        #     param.requires_grad = False

    # model = ResNetUNet()
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def fov_simple(cfg, pretrained=False, in_channels=3, out_channels=4):
    """
    simple three layer CNN as used in foveation work
    """
    model = FovSimModule(cfg, in_channels=in_channels, out_channels=out_channels)

    if pretrained:
        pretrained_path = './pretrained/foveater_cityscape_soft_e100.pth'
        model.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage), strict=False)

    return model
