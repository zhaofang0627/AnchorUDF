import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super(PointNet, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 32 ->m.p. 16
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_0_1 = nn.Conv3d(32, hidden_dim, 3, padding=1, padding_mode='replicate')  # out: 16 ->m.p. 8
        self.conv_1 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_1_1 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='replicate')  # out: 8

        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(hidden_dim)
        self.conv1_1_bn = nn.BatchNorm3d(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        net = self.maxpool(net)  # out 16

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        net = self.maxpool(net)  # out 8

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        fea = self.conv1_1_bn(net)

        return fea
