"""
Train an autoencoder on synthetic lightfield images to predict missing images in the camera array
Architecture losely based on https://arxiv.org/pdf/1701.04949.pdf
:author: Fenja Kollasch
"""
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


#DENSE_DIM = (64, 30, 30)
DENSE_DIM = (64, 2, 30, 30) # Uncomment for 3D
DENSE_SIZE = np.prod(DENSE_DIM)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, threed=False, transpose=False):
        super(ResidualBlock, self).__init__()
        self.lr = nn.LeakyReLU()
        if threed:
            self.bn = nn.BatchNorm3d(in_channels)
            if transpose:
                self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
                self.conv2 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            else:
                self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
                self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        else:
            self.bn = nn.BatchNorm2d(in_channels)
            if transpose:
                self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
                self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
                self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        normed = self.bn(x)
        out = self.lr(self.conv1(normed)) + self.conv2(normed)
        #print(out.shape)
        return out


class Encoder(nn.Module):

    def __init__(self, channels, threed):
        super(Encoder, self).__init__()
        last_kernel = (3, 4, 4) if threed else 4
        self.grp1 = nn.Sequential(
            ResidualBlock(in_channels=channels, out_channels=16, kernel_size=3, stride=1, threed=threed),
            ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, threed=threed),
            ResidualBlock(in_channels=32, out_channels=64, kernel_size=last_kernel, stride=2, threed=threed),  # 64 x 61 x61
        )

    def forward(self, x):
        x1 = self.grp1(x)
        return x1


class Decoder(nn.Module):

    def __init__(self, channels, threed):
        super(Decoder, self).__init__()
        first_kernel = (3, 4, 4) if threed else 4
        self.grp1 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=32, kernel_size=first_kernel, stride=2, threed=threed, transpose=True),
            ResidualBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1, threed=threed, transpose=True),
            ResidualBlock(in_channels=16, out_channels=channels, kernel_size=3, stride=1, threed=threed, transpose=True)
        )

    def forward(self, x):
        x1 = self.grp1(x)
        return x1


class VAE(nn.Module):
    def __init__(self, dims=(9, 3, 512, 512), threed=False):
        super(VAE, self).__init__()
        self.dims = dims
        self.imgs = dims[0]
        self.channels = dims[1] if threed else dims[0]*dims[1]
        self.width = dims[2]
        self.height = dims[3]
        self.threed = threed

        self.encoder = Encoder(self.channels, threed)
        self.decoder = Decoder(self.channels, threed)
        if threed:
            self.mu_layer = nn.Conv3d(64, 64, (1, 3, 3), (1, 2, 2))  # 64 x 1 x 30 x 30
            self.var_layer = nn.Conv3d(64, 64, (1, 3, 3), (1, 2, 2))
            self.z_layer = nn.ConvTranspose3d(64, 64, (1, 3, 3), (1, 2, 2))
        else:
            self.mu_layer = nn.Conv2d(64, 64, 3, 2)  # 64 x 30 x 30
            self.var_layer = nn.Conv2d(64, 64, 3, 2)
            self.z_layer = nn.ConvTranspose2d(64, 64, 3, 2)

    def encode(self, x):
        if self.threed:
            x = x.view(-1, self.channels, self.imgs, self.width, self.height)
        else:
            x = x.view(-1, self.channels, self.width, self.height)
        h1 = self.encoder(x)
        return self.mu_layer(h1).view(-1, DENSE_SIZE), self.var_layer(h1).view(-1, DENSE_SIZE)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h3 = self.z_layer(z.view(-1, *DENSE_DIM))
        dec = self.decoder(h3)
        return dec.view(-1, *self.dims)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return recon + KLD




