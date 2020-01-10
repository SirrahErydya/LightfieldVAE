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
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def show_scene(scene):
    # The scene was observed by a 9x9 camera array
    # Thus the loader contains 9 horizontal, 9 vertical, 9 increasing diagonal, and 9 decreasing diagonal images
    h_views, v_views, i_views, d_views, center, gt, mask, index = scene
    fig, axes = plt.subplots(9, 9, figsize=(18, 18))
    for y in range(9):
        for x in range(9):
            if x == 4 and y == 4:
                img = center
            elif x == 4:
                img = v_views[y]
            elif y == 4:
                img = h_views[x]
            elif x == y:
                img = d_views[x]
            elif x+y == 8:
                img = i_views[y]
            else:
                img = np.zeros((3, 512, 512))
            img = np.stack((img[0], img[1], img[2]), axis=-1)
            axes[y][x].imshow(img)
            axes[y][x].set_yticklabels([])
            axes[y][x].set_xticklabels([])
    plt.show()


