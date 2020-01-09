"""
Train an autoencoder on synthetic lightfield images to predict missing images in the camera array
:author: Fenja Kollasch
"""
import os
from dataloaders import hci4d
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from vae import VAE, train, test


DATA_ROOT = os.path.join('data', 'SyntheticLightfieldData')

# declare dataloaders for train and test images
train_loader = hci4d.HCI4D(os.path.join(DATA_ROOT, 'training'))
test_loader = hci4d.HCI4D(os.path.join(DATA_ROOT, 'test'))


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


if __name__ == '__main__':
    epochs = 10
    model = VAE()
    train(epochs)
    test(epochs)
    show_scene(test_loader.load_scene(0))
