
import torch.nn as nn
import torch as t
import torch.nn.functional as F

from configs import opt

def UnetModule3D(inchannel, outchannel, kernal_size=3, stride=1, padding=1):
    conv = nn.Sequential(
        nn.Conv3d(inchannel, outchannel, kernal_size, stride, padding, bias=False),
        nn.BatchNorm3d(outchannel),
        nn.ReLU(True),
        nn.Conv3d(outchannel, outchannel, 3, 1, 1, bias=False),
        nn.BatchNorm3d(outchannel),
        nn.ReLU(True),
    )

    return conv

class up(nn.Module):
    def __init__(self, in_ch, low_ch, out_ch):
        super().__init__()
        # up
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2),
            nn.BatchNorm3d(in_ch // 2),
            nn.ReLU(inplace=True),
        )

        # conv
        self.conv = UnetModule3D(in_ch // 2 + low_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = t.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class ResidualUp(nn.Module):
    def __init__(self, in_ch, low_ch, out_ch, block_num):
        super().__init__()


        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2),
            nn.BatchNorm3d(in_ch//2),
            nn.ReLU(inplace=True),
        )

        blocks = []

        shortcut = nn.Sequential(
            nn.Conv3d(in_ch // 2 + low_ch, out_ch, 1),
            nn.BatchNorm3d(out_ch),
        )


        blocks.append(ResidualBlock(in_ch // 2 + low_ch, out_ch, shortcut=shortcut))

        for _ in range(1, block_num):
            blocks.append(ResidualBlock(out_ch, out_ch))

        self.conv = nn.Sequential(*blocks)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = t.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),
            nn.Conv3d(2, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

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

class UNet3D(nn.Module):
    def __init__(self, out_ch):
        super().__init__()

        self.pre = UnetModule3D(4, 8)

        self.down1 = self.make_residual_layer(8, 16, 2)
        self.down2 = self.make_residual_layer(16, 32,2)
        self.down3 = self.make_residual_layer(32, 64, 3)
        self.down4 = nn.Sequential(
            self.make_residual_layer(64, 128, 3),
            nn.Dropout3d(p=0.15)
        )

        self.up1 = ResidualUp(128, 64, 64, 3)
        self.up2 = ResidualUp(64, 32, 32, 2)
        self.up3 = ResidualUp(32, 16, 16, 2)
        self.up4 = up(16, 8, 8)

        self.outc = outconv(8, out_ch)

        if self.training:
            self.out_iso_doses = nn.Sequential(
                nn.Conv3d(8, opt.num_iso_doses + 1, 3, 1, 1, bias=False),
            )

    def forward(self, ct, rois, distance_map):
        x = t.cat([ct,rois, distance_map], dim=1)
        x0 = self.pre(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x0)

        out = self.outc(x8)

        if self.training:
            out_iso = self.out_iso_doses(x8)
            return out, out_iso
        else:
            return out

    def make_pre_layer(self, inchannel, outchannel):
        layers = UnetModule3D(inchannel, outchannel)

        return layers

    def make_plain_layer(self, inchannel, outchannel):
        layers = UnetModule3D(inchannel, outchannel, 4, 2, 1)

        return layers

    def make_residual_layer(self, inchannel, outchannel, block_num):
        layers = []

        shortcut = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, 1, 1, 0),
        )

        layers.append(ResidualBlock(inchannel, outchannel, 3, 1, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(
            nn.MaxPool3d(2, 2),
            *layers)

    def predict(self, ct, structure_masks, distance_map):
        pred = self(ct, structure_masks, distance_map)

        return pred

    def load(self, stat_dict):
        self.load_state_dict(stat_dict, strict=True)

if __name__ == '__main__':
    t.cuda.set_device(7)
    img = t.randn(1, 1, 32, 240, 240)
    import datetime

    net = UNet3D(1, 18)
    img = img.cuda()
    net = net.cuda()
    start = datetime.datetime.now()
    pred = net(img)

    print(pred.size())

    end = datetime.datetime.now()
    print(end - start)
