"""
Data loader for hci4d data files
"""
from torch.utils.data import DataLoader, Dataset
import numpy as np


class LightfieldLoader(DataLoader):

    def __init__(self, dataset, direction=None, scene_idx=None, **kwargs):
        self.lightfield_dataset = dataset
        if direction is not None:
            prepared_dataset = self.make_dir_dataset(direction)
        elif scene_idx is not None:
            prepared_dataset = self.make_scene_dataset(scene_idx)
        else:
            prepared_dataset = dataset
        super().__init__(prepared_dataset, **kwargs)

    class LightfieldImages(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    def make_dir_dataset(self, direction):
        dir_data = []
        for idx in range(len(self.lightfield_dataset)):
            scene = self.lightfield_dataset.load_scene(idx)
            h_views, v_views, i_views, d_views, center, gt, mask, index = scene
            if direction == 'horizontal':
                views = h_views
            elif direction == 'vertical':
                views = v_views
            elif direction == 'increasing':
                views = i_views
            elif direction == 'decreasing':
                views = d_views
            else:
                raise ValueError("Invalid direction:", direction)
            for img in views:
                dir_data.append(img)

        return self.LightfieldImages(dir_data)

    def make_scene_dataset(self, scene_idx):
        scene = self.lightfield_dataset.load_scene(scene_idx)
        scene_data = np.concatenate([scene[0], scene[1]], axis=0)
        return self.LightfieldImages(scene_data)
