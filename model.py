import torch
import torch.nn as nn
import numpy as np
import h5py

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = nn.functional.pad(x, (0, skip_input.shape[-1] - x.shape[-1], 0, skip_input.shape[-2] - x.shape[-2], 0, skip_input.shape[-3] - x.shape[-3]))
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        
        self.mid1 = UNetMid(256, 512, dropout=0.2)
        self.mid2 = UNetMid(512, 512, dropout=0.2)
        
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 3, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        m1 = self.mid1(d3)
        m2 = self.mid2(m1)
        
        u1 = self.up1(m2, d3)
        u2 = self.up2(u1, d2)
        uf = self.up3(u2, d1)
        
        return self.final(uf)

##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.final = nn.Conv3d(512, 1, 3, padding=1, bias=False)

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        intermediate = self.model(img_input)
        pad = nn.functional.pad(intermediate, pad=(1,0,1,0,1,0))
        return self.final(pad)


if __name__ == "__main__":
    # Initialize the Discriminator model
    discriminator = Discriminator(in_channels=2)

    batch_size = 4
    depth, height, width = 33, 33, 33
    img_A = torch.randn(batch_size, 2, depth, height, width)
    img_B = torch.randn(batch_size, 2, depth, height, width)
    output = discriminator(img_A, img_B)
    print(f'Input shape A: {img_A.shape}')
    print(f'Input shape B: {img_B.shape}')
    print(f'Output shape after discriminator: {output.shape}')

    # Example usage for GeneratorUNet
    in_channels = 2
    out_channels = 2  
    generator = GeneratorUNet(in_channels, out_channels)

    # dummy input tensor
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width)
    output_tensor = generator(input_tensor)
    print(f'Input shape: {input_tensor.shape}')
    print(f'Output shape: {output_tensor.shape}')


