"""
Train an autoencoder on synthetic lightfield images to predict missing images in the camera array
:author: Fenja Kollasch
"""
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """
    Basic VAE from the pytorch examples
    """
    def __init__(self, hidden_dims, latent_size, dims=(3, 512, 512)):
        super(VAE, self).__init__()

        num_hidden = len(hidden_dims)

        encoder_layers = [nn.Linear(np.prod(dims), hidden_dims[0]), nn.ReLU()]
        decoder_layers = [nn.Linear(latent_size, hidden_dims[num_hidden - 1]), nn.ReLU()]

        for i in range(1, num_hidden):
            encoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(hidden_dims[num_hidden-i], hidden_dims[num_hidden-(i+1)]))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.Linear(hidden_dims[0], np.prod(dims)))
        self.mu_layer = nn.Linear(hidden_dims[len(hidden_dims)-1], latent_size)
        self.var_layer = nn.Linear(hidden_dims[len(hidden_dims) - 1], latent_size)
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
            h1 = self.encoder(x)
            return self.mu_layer(h1), self.var_layer(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.decoder(z)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 512*512))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 512*512), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


