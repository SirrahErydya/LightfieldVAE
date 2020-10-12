from dataloaders import hci4d, lfsequence
import torch
import lightfield_vae as vae
from utils import show_view_sequence

if __name__ == "__main__":
    print("A little setup. Please be patient.")
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 64, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = vae.VAE(dims=(9, 3, 128, 128), threed=False)
    print("CPU model created")
    model.to(device)

    test_set = lfsequence.LFSequence('test', transform=hci4d.CenterCrop(128))
    m = input("Please choose the model for this test run (MSE / KLD)").lower()
    if m == 'kld':
        model_name = 'model_KLD_2D.mdl'
    elif m == 'mse':
        model_name = 'model_MSE_2D.mdl'
    else:
        print("Your input confused me. Taking MSE model instead.")
        model_name = 'model_MSE_2D.mdl'

    model.load_state_dict(torch.load(model_name))
    print("Performing test with {0}...".format(model_name))
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
        show_view_sequence(predicted_d.cpu(), "test_prediction", savepath='test')
    print("All done! Please have a look into the test folder.")