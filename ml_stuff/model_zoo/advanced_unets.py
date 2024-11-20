from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# code modified from https://github.com/LeeJunHyun/Image_Segmentation


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, kernel_size=3, stride=1, padding=1, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(
                ch_out,
                ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, t=2, num_rec_blocks=2):
        super().__init__()

        rec_blocks = [Recurrent_block(ch_out, kernel_size, stride, padding, t=t) for _ in range(num_rec_blocks)]
        self.RCNN = nn.Sequential(*rec_blocks)

        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# receptive field: 18px (in each direction)
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_features=32):
        super().__init__()

        ftrs = init_features

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=ftrs)
        self.Conv2 = conv_block(ch_in=ftrs, ch_out=ftrs * 2)
        self.Conv3 = conv_block(ch_in=ftrs * 2, ch_out=ftrs * 4)
        self.Conv4 = conv_block(ch_in=ftrs * 4, ch_out=ftrs * 8)
        self.Conv5 = conv_block(ch_in=ftrs * 8, ch_out=ftrs * 16)

        self.Up5 = up_conv(ch_in=ftrs * 16, ch_out=ftrs * 8)
        self.Att5 = Attention_block(F_g=ftrs * 8, F_l=ftrs * 8, F_int=ftrs * 4)
        self.Up_conv5 = conv_block(ch_in=ftrs * 16, ch_out=ftrs * 8)

        self.Up4 = up_conv(ch_in=ftrs * 8, ch_out=ftrs * 4)
        self.Att4 = Attention_block(F_g=ftrs * 4, F_l=ftrs * 4, F_int=ftrs * 2)
        self.Up_conv4 = conv_block(ch_in=ftrs * 8, ch_out=ftrs * 4)

        self.Up3 = up_conv(ch_in=ftrs * 4, ch_out=ftrs * 2)
        self.Att3 = Attention_block(F_g=ftrs * 2, F_l=ftrs * 2, F_int=ftrs)
        self.Up_conv3 = conv_block(ch_in=ftrs * 4, ch_out=ftrs * 2)

        self.Up2 = up_conv(ch_in=ftrs * 2, ch_out=ftrs)
        self.Att2 = Attention_block(F_g=ftrs, F_l=ftrs, F_int=ftrs // 2)
        self.Up_conv2 = conv_block(ch_in=ftrs * 2, ch_out=ftrs)

        self.Conv_1x1 = nn.Conv2d(ftrs, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# receptive field: 54px (in each direction)
class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_features=32, t=3):
        super().__init__()

        ftrs = init_features
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=ftrs, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=ftrs, ch_out=ftrs * 2, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=ftrs * 2, ch_out=ftrs * 4, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=ftrs * 4, ch_out=ftrs * 8, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=ftrs * 8, ch_out=ftrs * 16, t=t)

        self.Up5 = up_conv(ch_in=ftrs * 16, ch_out=ftrs * 8)
        self.Att5 = Attention_block(F_g=ftrs * 8, F_l=ftrs * 8, F_int=ftrs * 4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=ftrs * 16, ch_out=ftrs * 8, t=t)

        self.Up4 = up_conv(ch_in=ftrs * 8, ch_out=ftrs * 4)
        self.Att4 = Attention_block(F_g=ftrs * 4, F_l=ftrs * 4, F_int=ftrs * 2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=ftrs * 8, ch_out=ftrs * 4, t=t)

        self.Up3 = up_conv(ch_in=ftrs * 4, ch_out=ftrs * 2)
        self.Att3 = Attention_block(F_g=ftrs * 2, F_l=ftrs * 2, F_int=ftrs)
        self.Up_RRCNN3 = RRCNN_block(ch_in=ftrs * 4, ch_out=ftrs * 2, t=t)

        self.Up2 = up_conv(ch_in=ftrs * 2, ch_out=ftrs)
        self.Att2 = Attention_block(F_g=ftrs, F_l=ftrs, F_int=ftrs // 2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=ftrs * 2, ch_out=ftrs, t=t)

        self.Conv_1x1 = nn.Conv2d(ftrs, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        x2 = self.Maxpool(x1)

        x2 = self.RRCNN2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.RRCNN3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.RRCNN4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# Experimental section


class StackedR2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_features=16, t=2, num_stacks=2):
        super(StackedR2AttU_Net, self).__init__()
        self.stacks = nn.ModuleList()
        for i in range(num_stacks):
            self.stacks.append(
                R2AttU_Net(
                    img_ch=img_ch if i == 0 else init_features,
                    output_ch=init_features,
                    init_features=init_features,
                    t=t,
                )
            )

        # Final output layer to match the desired output channels
        self.final_conv = nn.Conv2d(init_features, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        for stack in self.stacks:
            x = stack(x)

        x = self.final_conv(x)
        return x


# receptive field: 216px (in each direction)
class R2AttU_Net_7x7_t4(nn.Module):  # recurrent residual attention u-net
    def __init__(self, img_ch=3, output_ch=1, init_features=32, t=4):
        super().__init__()

        ftrs = init_features
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=ftrs, kernel_size=7, stride=1, padding=3, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=ftrs, ch_out=ftrs * 2, kernel_size=7, stride=1, padding=3, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=ftrs * 2, ch_out=ftrs * 4, kernel_size=7, stride=1, padding=3, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=ftrs * 4, ch_out=ftrs * 8, kernel_size=7, stride=1, padding=3, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=ftrs * 8, ch_out=ftrs * 16, kernel_size=7, stride=1, padding=3, t=t)

        self.Up5 = up_conv(ch_in=ftrs * 16, ch_out=ftrs * 8)
        self.Att5 = Attention_block(F_g=ftrs * 8, F_l=ftrs * 8, F_int=ftrs * 4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=ftrs * 16, ch_out=ftrs * 8, kernel_size=7, stride=1, padding=3, t=t)

        self.Up4 = up_conv(ch_in=ftrs * 8, ch_out=ftrs * 4)
        self.Att4 = Attention_block(F_g=ftrs * 4, F_l=ftrs * 4, F_int=ftrs * 2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=ftrs * 8, ch_out=ftrs * 4, kernel_size=7, stride=1, padding=3, t=t)

        self.Up3 = up_conv(ch_in=ftrs * 4, ch_out=ftrs * 2)
        self.Att3 = Attention_block(F_g=ftrs * 2, F_l=ftrs * 2, F_int=ftrs)
        self.Up_RRCNN3 = RRCNN_block(ch_in=ftrs * 4, ch_out=ftrs * 2, kernel_size=7, stride=1, padding=3, t=t)

        self.Up2 = up_conv(ch_in=ftrs * 2, ch_out=ftrs)
        self.Att2 = Attention_block(F_g=ftrs, F_l=ftrs, F_int=ftrs // 2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=ftrs * 2, ch_out=ftrs, kernel_size=7, stride=1, padding=3, t=t)

        self.Conv_1x1 = nn.Conv2d(ftrs, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# receptive field: 432px (in each direction)
class R2AttU_Net_7x7_t4_4recblocks(nn.Module):  # recurrent residual attention u-net
    def __init__(self, img_ch=3, output_ch=1, init_features=32, t=4):
        super().__init__()

        ftrs = init_features
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(
            ch_in=img_ch,
            ch_out=ftrs,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )
        self.RRCNN2 = RRCNN_block(
            ch_in=ftrs,
            ch_out=ftrs * 2,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )
        self.RRCNN3 = RRCNN_block(
            ch_in=ftrs * 2,
            ch_out=ftrs * 4,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )
        self.RRCNN4 = RRCNN_block(
            ch_in=ftrs * 4,
            ch_out=ftrs * 8,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )
        self.RRCNN5 = RRCNN_block(
            ch_in=ftrs * 8,
            ch_out=ftrs * 16,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )

        self.Up5 = up_conv(ch_in=ftrs * 16, ch_out=ftrs * 8)
        self.Att5 = Attention_block(F_g=ftrs * 8, F_l=ftrs * 8, F_int=ftrs * 4)
        self.Up_RRCNN5 = RRCNN_block(
            ch_in=ftrs * 16,
            ch_out=ftrs * 8,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )

        self.Up4 = up_conv(ch_in=ftrs * 8, ch_out=ftrs * 4)
        self.Att4 = Attention_block(F_g=ftrs * 4, F_l=ftrs * 4, F_int=ftrs * 2)
        self.Up_RRCNN4 = RRCNN_block(
            ch_in=ftrs * 8,
            ch_out=ftrs * 4,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )

        self.Up3 = up_conv(ch_in=ftrs * 4, ch_out=ftrs * 2)
        self.Att3 = Attention_block(F_g=ftrs * 2, F_l=ftrs * 2, F_int=ftrs)
        self.Up_RRCNN3 = RRCNN_block(
            ch_in=ftrs * 4,
            ch_out=ftrs * 2,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )

        self.Up2 = up_conv(ch_in=ftrs * 2, ch_out=ftrs)
        self.Att2 = Attention_block(F_g=ftrs, F_l=ftrs, F_int=ftrs // 2)
        self.Up_RRCNN2 = RRCNN_block(
            ch_in=ftrs * 2,
            ch_out=ftrs,
            kernel_size=7,
            stride=1,
            padding=3,
            t=t,
            num_rec_blocks=4,
        )

        self.Conv_1x1 = nn.Conv2d(ftrs, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


#
# DILATED UNET
#


class Dilation_conv_block(nn.Module):  # drop-in replacement for the Recurrent_block
    def __init__(self, ch_out):
        super().__init__()
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))
        # Parallel dilated convolutions with different rates
        self.conv_d1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, dilation=1)
        self.conv_d3 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=3, dilation=3)
        self.conv_d5 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=5, dilation=5)
        self.conv_d7 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=7, dilation=7)
        self.conv_d9 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=9, dilation=9)

        # 1x1 convolution to merge the features
        self.conv1x1 = nn.Conv2d(ch_out * 5, ch_out, kernel_size=1, bias=False)

        # Skip connection
        self.conv_skip = nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)

    def forward(self, x):
        # Apply BN-ReLU
        x = self.bn_relu(x)

        # Apply parallel dilated convolutions
        out_d1 = self.conv_d1(x)
        out_d3 = self.conv_d3(x)
        out_d5 = self.conv_d5(x)
        out_d7 = self.conv_d7(x)
        out_d9 = self.conv_d9(x)

        # Concatenate features from different dilations
        out_concat = torch.cat((out_d1, out_d3, out_d5, out_d7, out_d9), 1)

        # Merge features with 1x1 convolution
        out = self.conv1x1(out_concat)

        # Add skip connection
        identity = self.conv_skip(x)
        out += identity

        # Apply final ReLU
        out = F.relu(out)

        return out


class RRCNN_dilation_block(nn.Module):
    def __init__(self, ch_in, ch_out, num_rec_blocks=2):
        super().__init__()

        rec_blocks = [Dilation_conv_block(ch_out) for _ in range(num_rec_blocks)]
        self.RCNN = nn.Sequential(*rec_blocks)

        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


# receptive field: ca. 450 (in each direction)
class R2AttU_Net_dilated(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_features=32):
        super().__init__()

        dilation = 1
        ftrs = init_features
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_dilation_block(ch_in=img_ch, ch_out=ftrs)
        self.RRCNN2 = RRCNN_dilation_block(ch_in=ftrs, ch_out=ftrs * 2)
        self.RRCNN3 = RRCNN_dilation_block(ch_in=ftrs * 2, ch_out=ftrs * 4)
        self.RRCNN4 = RRCNN_dilation_block(ch_in=ftrs * 4, ch_out=ftrs * 8)
        self.RRCNN5 = RRCNN_block(ch_in=ftrs * 8, ch_out=ftrs * 16, t=3)

        self.Up5 = up_conv(ch_in=ftrs * 16, ch_out=ftrs * 8)
        self.Att5 = Attention_block(F_g=ftrs * 8, F_l=ftrs * 8, F_int=ftrs * 4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=ftrs * 16, ch_out=ftrs * 8, t=3)

        self.Up4 = up_conv(ch_in=ftrs * 8, ch_out=ftrs * 4)
        self.Att4 = Attention_block(F_g=ftrs * 4, F_l=ftrs * 4, F_int=ftrs * 2)
        self.Up_RRCNN4 = RRCNN_dilation_block(ch_in=ftrs * 8, ch_out=ftrs * 4)

        self.Up3 = up_conv(ch_in=ftrs * 4, ch_out=ftrs * 2)
        self.Att3 = Attention_block(F_g=ftrs * 2, F_l=ftrs * 2, F_int=ftrs)
        self.Up_RRCNN3 = RRCNN_dilation_block(ch_in=ftrs * 4, ch_out=ftrs * 2)

        self.Up2 = up_conv(ch_in=ftrs * 2, ch_out=ftrs)
        self.Att2 = Attention_block(F_g=ftrs, F_l=ftrs, F_int=ftrs // 2)
        self.Up_RRCNN2 = RRCNN_dilation_block(ch_in=ftrs * 2, ch_out=ftrs)

        self.Conv_1x1 = nn.Conv2d(ftrs, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        x2 = self.Maxpool(x1)

        x2 = self.RRCNN2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.RRCNN3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.RRCNN4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
