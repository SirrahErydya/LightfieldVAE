import matplotlib.pyplot as plt
from matplotlib import image as pltimg
import numpy as np
import skimage.io
from skimage.color import rgb2gray
import os
import csv

def show_scene(scene, show_dia=True):
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
            elif x == y and show_dia:
                img = d_views[x]
            elif x+y == 8 and show_dia:
                img = i_views[y]
            else:
                img = np.zeros((3, 512, 512))
            img = np.stack((img[0], img[1], img[2]), axis=-1)
            axes[y][x].imshow(img)
            axes[y][x].set_yticklabels([])
            axes[y][x].set_xticklabels([])
    plt.show()


def show_view_sequence(views, fn, savepath=None, cmap=None):
    length = views.shape[0]
    fig = plt.figure(figsize=(views.shape[2], views.shape[3]))
    for i in range(length):
        img = views[i]
        img = np.stack((img[0], img[1], img[2]), axis=-1)
        img = np.clip(img, -1, 1)
        if savepath:
            name = '{0}{1}.png'.format(fn, i)
            if cmap == "coolwarm":
                im = plt.imshow(rgb2gray(img), cmap=cmap)
                plt.colorbar(im)
                plt.savefig(os.path.join(savepath, name), cmap=cmap)
                plt.show()
            else:
                skimage.io.imsave(os.path.join(savepath, name), skimage.img_as_ubyte(img))


def save_stats(filename, model_name, loss_function, epochs, loss):
    with open(filename, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([model_name, loss_function, epochs, loss])


def plot_loss(loss, savepath):
    plt.plot(loss)
    plt.xlabel = "Epochs"
    plt.ylabel = "Average loss"
    plt.savefig(os.path.join(savepath, "loss.png"))


def show_ground_truth(scene):
    h_views, v_views, i_views, d_views, center, gt, mask, index = scene
    plt.imshow(gt)
    plt.show()
