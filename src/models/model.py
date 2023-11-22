# Created this unet using Medium.com tutorial @ https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(out_c)
        conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(out_c)
        relu = nn.ReLU()
        self.modules = nn.ModuleList([conv1, bn1, relu, conv2, bn2, relu])

    def forward(self, inputs):
        x = torch.clone(inputs)
        for _, m in enumerate(self.modules):
            x = m(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)  # reverse max pool
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class ClassicUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # Bottleneck
        self.b = ConvBlock(512, 1024)

        # Decoder 
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # Classifer
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, x = self.e1(inputs)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)

        x = self.b(x)

        x = self.d1(x, s4)
        x = self.d2(x, s3)
        x = self.d3(x, s2)
        x = self.d4(x, s1)

        outputs = self.outputs(x)
        return outputs
