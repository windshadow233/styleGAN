import random
import os
from PIL import Image
import numpy as np
import torch
import math


class ImageDataset(object):
    def __init__(self, data_root='data/flickr', transform=None, max_resolution=256):
        self.data_root = data_root
        self.images = []
        self.reload_images()
        self._length = len(self.images)
        self.transform = transform
        self.max_resolution = max_resolution
        self.R = int(math.log2(max_resolution))
        assert 2 ** self.R == max_resolution
        assert str(max_resolution) in os.listdir(self.data_root)

    def reload_images(self):
        self.images = os.listdir(f'{self.data_root}/4/data')
        random.shuffle(self.images)

    def __call__(self, batch_size, level):
        # assert level in range(2, self.R + 1)
        resolution = 2 ** level
        dir_path = os.path.join(self.data_root, str(resolution), 'data')
        if len(self.images) < batch_size:
            self.reload_images()
        img_files, self.images = self.images[:batch_size], self.images[batch_size:]
        if self.transform is not None:
            imgs = torch.stack([self.transform(Image.open(os.path.join(dir_path, img))) for img in img_files])
        else:
            imgs = np.stack([np.array(Image.open(os.path.join(dir_path, img))) for img in img_files]).transpose((0, 3, 1, 2))
        return imgs

    def __len__(self):
        return self._length
