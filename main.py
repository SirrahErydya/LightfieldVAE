# Training and testing routines similar to the PyTorch implementation
# but modified for lightfield arrays
import os
from dataloaders import hci4d, lfsequence
import torch
from torch import optim
import lightfield_vae as vae
from utils import show_view_sequence, plot_loss
from torch.utils.data import DataLoader 
from torch.nn import functional as F
from torchvision import transforms
from matplotlib import pyplot as plt

DATA_ROOT = os.path.join('data', 'SyntheticLightfieldData')
BATCH_SIZE = 6
MODEL_NAME = "model_mse_2d"
EPOCHS = 200
use_cuda = torch.cuda.is_available()
print("Use cuda:", use_cuda)
kwargs = {'num_workers': 64, 'pin_memory': True} if use_cuda else {}
device = torch.device("cuda:0" if use_cuda else "cpu")
model = vae.VAE(dims=(9, 3, 128, 128))
print("CPU model created")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

total_loss = []


def train(train_loader, loader_num, log_interval=2):
    for epoch in range(1, EPOCHS+1):
        print("Training with Loader ", loader_num)
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
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
        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, avg_loss))
        total_loss.append(avg_loss)


if __name__ == '__main__':
    # Create folder to save results
    path = os.path.join(os.getcwd(), "{0}_{1}e".format(MODEL_NAME, EPOCHS*4))
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    ## Create multiple train sets to increase the training data
    train_set1 = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'training'),
                                       transform=hci4d.Crop((128, 128), (128, 128)))
    train_set2 = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'training'),
                                       transform=hci4d.Crop((128, 128), (128, 256)))
    train_set3 = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'training'),
                                       transform=hci4d.Crop((128, 128), (256, 128)))
    train_set4 = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'training'),
                                       transform=hci4d.Crop((128, 128), (256, 256)))
    test_set = lfsequence.LFSequence(os.path.join(DATA_ROOT, 'test'), transform=hci4d.CenterCrop(128))
    print("Training set length:", 4 * len(train_set1))
    print("Test set length:", len(test_set))

    # Dataloaders
    train_loader1 = DataLoader(train_set1, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train_loader2 = DataLoader(train_set2, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train_loader3 = DataLoader(train_set3, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train_loader4 = DataLoader(train_set4, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    n_train = 4 * len(train_loader1.dataset)
    n_test = len(test_loader.dataset)
    print("Data samples for training:", n_train)
    print("Data samples for testing:", n_test)

    # Model loading and training
    model_path = os.path.join(path, MODEL_NAME + ".mdl")
    try:
        print("Loading model ", MODEL_NAME)
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print("No such model. Training a new one!")
        train(train_loader1, 1)
        train(train_loader2, 2)
        train(train_loader3, 3)
        train(train_loader4, 4)
        plot_loss(total_loss, path)
        torch.save(model.state_dict(), model_path)

    model.eval()
    test_loss = 0
    results_path = os.path.join(path, "results")
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    with torch.no_grad():
        h_views, v_views, i_views, d_views, center, gt, mask, index = test_set.get_scene(0)
        data_h = torch.tensor(h_views, device=device).float()
        data_v = torch.tensor(v_views, device=device).float()
        data_d = torch.tensor(d_views, device=device).float()
        predicted_h, _, _ = model(data_h)
        predicted_h = predicted_h.view(9, 3, 128, 128)
        predicted_v, _, _ = model(data_v)
        predicted_v = predicted_v.view(9, 3, 128, 128)
        mu_h, _ = model.encode(data_v)
        mu_v, _ = model.encode(data_h)
        predicted_d = model.decode(mu_h + mu_v).view(9, 3, 128, 128)
        test_loss += F.l1_loss(predicted_d, data_d)
        print("Save predicition:")
        show_view_sequence(predicted_h.cpu(), "horizontal", savepath=results_path)
        show_view_sequence(predicted_v.cpu(), "vertical", savepath=results_path)
        show_view_sequence(predicted_d.cpu(), "diagonal", savepath=results_path)
        show_view_sequence(h_views, "h_truth", savepath=results_path)
        show_view_sequence(v_views, "v_truth", savepath=results_path)
        show_view_sequence(d_views, "d_truth", savepath=results_path)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

