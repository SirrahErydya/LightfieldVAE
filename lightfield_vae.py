"""
Train an autoencoder on synthetic lightfield images to predict missing images in the camera array
Architecture provided by https://www.researchgate.net/figure/Convolutional-variational-autoencoder-architecture-The-deep-learning-network-processes_fig1_329836538
:author: Fenja Kollasch
"""
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NUM_FILTERS = 32
DENSE_DIM = (32, 249, 249)
DENSE_SIZE = np.prod(DENSE_DIM)

class VAE(nn.Module):
    """
    Basic VAE from the pytorch examples
    """
    def __init__(self, latent_size=16, dims=(9, 3, 512, 512)):
        super(VAE, self).__init__()
        self.dims = dims

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=dims[0]*dims[1], out_channels=NUM_FILTERS, kernel_size=4, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, kernel_size=4, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, kernel_size=4, stride=1), nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, kernel_size=4, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS, kernel_size=4, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=NUM_FILTERS, out_channels=dims[0]*dims[1], kernel_size=4, stride=1), nn.Sigmoid()
        )
        self.dense_enc = nn.Linear(DENSE_SIZE, 256)
        self.dense_dec = nn.Linear(256, DENSE_SIZE)
        self.mu_layer = nn.Linear(256, latent_size)
        self.var_layer = nn.Linear(256, latent_size)
        self.z_layer = nn.Linear(latent_size, 256)

    def encode(self, x):
            conv = self.encoder(x)
            enc_input = conv.view(-1, DENSE_SIZE)
            h1 = self.dense_enc(enc_input)
            return self.mu_layer(h1), self.var_layer(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.dense_dec(F.relu(self.z_layer(z))).view(-1, *DENSE_DIM)
        return self.decoder(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


