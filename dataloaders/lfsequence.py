from . import hci4d
import copy


class LFSequence(hci4d.HCI4D):
    """
    Dataset that holds the horizontal and vertical views of the synthetic HCI 4D Light Field Dataset
    Suitable for training an autoencoder generating the diagonal views
    """
    def __init__(self, root, nviews=(9, 9), transform=None, cache=False, length=0):
        super(LFSequence, self).__init__(root, nviews, transform, cache, length)

    def __len__(self):
        if self.length == 0:
            return len(self.scenes)*2

        return self.length

    def __getitem__(self, index):
        """
        Loads the next scene and returns the horizontal view if index < no of scenes, the vertical view else
        where the views are tensors of shape (w or h, 3, h_image, w_image),
        center is the center view and gt is the ground truth
        of shape (h_img, w_img) or zeroes if the dataset does not provide it.
        Index is just a scalar list index as numpy.ndarray.

        :param index: scene index in range(0, len(dataset))
        :type index: int
        """
        scene_idx = index % len(self.scenes)
        view_idx = index % len(self)

        assert scene_idx == view_idx or view_idx == scene_idx + len(self.scenes)
        h_views, v_views, i_views, d_views, _, _, _, _ =  self.get_scene(scene_idx)

        if scene_idx == view_idx:
            return h_views
        return v_views

    def get_scene(self, index):
        if self.cache:
            data = self.data[index]
        else:
            data = self.load_scene(index)

        if self.transform:
            data = copy.deepcopy(data)
            data = self.transform(data)
        return data

    def ground_truth(self, index):
        """
        Get the ground truth (aka the diagonal) to a scene
        :param index: scene index
        """
        h_views, v_views, i_views, d_views, _, _, _, _ =  self.get_scene(index)
        return d_views