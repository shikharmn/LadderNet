import torch
import torch.nn.functional as F
import torch.nn as nn

drop = 0.25


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes != planes:
            self.conv0 = conv3x3(inplanes, planes)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)

        out1 = self.conv1(out)
        # out1 = self.relu(out1)

        out2 = out1 + x

        return F.relu(out2)


class Discriminator(nn.Module):

    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                                kernel_size=3, stride=1, padding=1, bias=True)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(planes * (2 ** i), planes * (2 ** i)))

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_conv_list.append(
                nn.Conv2d(planes * 2 ** i, planes * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes * (2 ** layers), planes * (2 ** layers))

        # create module list for up branch
        self.penult_conv = nn.Conv2d(planes * (2 ** layers), planes * (2 ** (layers - 1)), kernel_size=3, stride=1,
                     padding=0, bias=True)
        self.final_fc = nn.Linear(planes * (2 ** (layers - 1)), 1)

    def forward(self, x):
        out = self.inconv(x)
        out = F.relu(out)
        down_out = []
        # down branch
        for i in range(0, self.layers):
            out = self.down_module_list[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        out = F.relu(out)
        out = self.penult_conv(out)
        out = F.relu(out).reshape((-1, self.planes * (2 ** (self.layers - 1))))
        out = self.final_fc(out)
        out = F.relu(out)

        return out