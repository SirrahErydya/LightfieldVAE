import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os

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


def show_view_sequence(views, save=False):
    length = views.shape[0]
    fig, axes = plt.subplots(1, length, figsize=(20, 20*length))
    for i in range(length):
        if views.shape[1] == 3:
            img = np.stack((views[i][0], views[i][1], views[i][2]), axis=-1)
        else:
            img = views[i]
        axes[i].imshow(img)
        axes[i].set_yticklabels([])
        axes[i].set_xticklabels([])
        if save:
            skimage.io.imsave(os.path.join('results', 'pred{0}.png'.format(i)), skimage.img_as_ubyte(img))

    plt.show()


def show_ground_truth(scene):
    h_views, v_views, i_views, d_views, center, gt, mask, index = scene
    plt.imshow(gt)
    plt.show()
