import os
from os.path import *

from PIL import Image
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform

        self.samples = self.load_label(data_dir)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        image = self.load_image(filename)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def load_label(data_dir):
        labels = []
        images = []

        cls_names = [folder for folder in os.listdir(data_dir)]
        cls_names.sort()

        cls_to_idx = {cls_name: i for i, cls_name in enumerate(cls_names)}

        for root, dirs, filenames in os.walk(data_dir, False, followlinks=True):
            label = basename(relpath(root, data_dir) if (root != data_dir) else '')

            for filename in filenames:
                base, ext = splitext(filename)
                if ext.lower() in ('.png', '.jpg', '.jpeg'):
                    labels.append(label)
                    images.append(join(root, filename))

        return [(i, cls_to_idx[j]) for i, j in zip(images, labels) if j in cls_to_idx]


class TestDataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.samples = []
        filenames = os.listdir(f'{data_dir}/test')
        for filename in filenames:
            self.samples.append(f'{data_dir}/test/{filename}')

        cls_names = [folder for folder in os.listdir(f'{data_dir}/train')]
        cls_names.sort()
        self.idx_to_cls = {i: cls_name for i, cls_name in enumerate(cls_names)}

    def __getitem__(self, index):
        filename = self.samples[index]
        image = self.load_image(filename)
        image = self.transform(image)
        return image, basename(filename)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image
