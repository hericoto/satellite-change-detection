
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, image_a, image_b):


        a1 = self.enc1(image_a)
        a2 = self.enc2(self.pool(a1))
        a3 = self.enc3(self.pool(a2))
        a4 = self.enc4(self.pool(a3))


        b1 = self.enc1(image_b)
        b2 = self.enc2(self.pool(b1))
        b3 = self.enc3(self.pool(b2))
        b4 = self.enc4(self.pool(b3))


        f1 = torch.abs(a1 - b1)
        f2 = torch.abs(a2 - b2)
        f3 = torch.abs(a3 - b3)
        f4 = torch.abs(a4 - b4)

        x = self.bottleneck(self.pool(f4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, f4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, f3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, f2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, f1], dim=1))

        return self.final(x)
