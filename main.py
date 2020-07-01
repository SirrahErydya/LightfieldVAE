# Training and testing routines similar to the PyTorch implementation
# but modified for lightfield arrays
import os
from dataloaders import hci4d, lfsequence
import torch
from torch import optim
import lightfield_vae as vae
from utils import show_view_sequence
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms

DATA_ROOT = os.path.join('data', 'SyntheticLightfieldData')
BATCH_SIZE = 6
use_cuda = torch.cuda.is_available()
print("Use cuda:", use_cuda)
kwargs = {'num_workers': 64, 'pin_memory': True} if use_cuda else {}

train_set = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'training'), transform=hci4d.DownSampling(4))
test_set = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'test'),transform=hci4d.DownSampling(4))
print("Training set length:", len(train_set))
print("Test set length:", len(test_set))
device = torch.device("cuda:1" if use_cuda else "cpu")
model = vae.VAE(dims=(9, 3, 128, 128), latent_size=2**11, bottleneck=2**10)
print("CPU model created")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Load horizontal lightfield data
# TODO: How to handle different directions
# Todo: Data Augmentation: RandomCrop, RedistColor, Contrast, Brightness, RandomRotate
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

n_train = len(train_loader.dataset)
n_test = len(test_loader.dataset)
print("Data samples for training:", n_train)
print("Data samples for testing:", n_test)


def train(epoch, log_interval=2):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data = data.view(-1, 27, 128, 128)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch, log_interval=3)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        h_views, v_views, i_views, d_views, center, gt, mask, index = test_set.get_scene(0)
        data_h = h_views[0].to(device)
        data_v = v_views[0].to(device)
        mu_h, var_h = model.encode(data_h.view(-1, 27, 128, 128))
        mu_v, var_v = model.encode(data_v.view(-1, 27, 128, 128))
        z_h, z_v = model.reparameterize(mu_h, var_h), model.reparameterize(mu_v, var_v)
        predicted = model.decode(z_h + z_v)
        ground_truth = d_views[0].to(device).view(-1, 27, 128, 128)
        test_loss += F.l1_loss(predicted, ground_truth)
        print(predicted.shape)
        print("Save predicition:")
        show_view_sequence(predicted.cpu().view(9, 3, 128, 128), save=True)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

