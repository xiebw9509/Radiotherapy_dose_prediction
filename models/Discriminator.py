
import torch.nn as nn
import torch as t
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, kernel_size = 3, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size, stride, 1),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(True),
            nn.Conv3d(outchannel, outchannel, 3, 1, 1),
            nn.BatchNorm3d(outchannel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        out = F.relu(out, inplace=True)

        return out

class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(5, 8, 3, 1, 1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.MaxPool3d(4),
            nn.Conv3d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            ResidualBlock(16, 16),
            nn.MaxPool3d(4),
            nn.Conv3d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            nn.MaxPool3d(4),
            nn.Conv3d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            ResidualBlock(64, 64),
        )

        self.full_connects = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, ct, rois, dis_map, dose):
        x = t.cat([ct,rois, dis_map, dose], dim=1)
        x = self.net(x)

        x = self.full_connects(x.view(x.size(0), -1))
        return x

    def load(self, stat_dict):
        self.load_state_dict(stat_dict, strict=True)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    input = t.randn((2, 5, 96, 96, 96))

    input = input.cuda()
    net = Discriminator()
    net = net.cuda()

    output = net(input)
    print(output.size())