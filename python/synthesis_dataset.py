""" Dataset class for ArchVizPro data

"""
import os
import json

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.utils import make_grid

from PIL import Image

import numpy as np

class SynthesisDataset(Dataset):
    """The dataset for the ArchVizPro data

    """
    def __init__(self, root_dir, scale=1.0, extension='.png'):

        # file locations
        self.root_dir = root_dir # file path of the image dataset
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.extension = extension
        self.file_list = os.listdir(self.root_dir) # gets all files in the directory
        self.img_list = [file for file in self.file_list if "img" in file] # only the images
        self.modalities = self.get_modalities() # all ground truths (rgb, semantic, outlines)
        self.dictionary = self.files_to_dict() # makes a dictionary out of image paths
        self.keys = list(self.dictionary.keys()) # all image sets (one image set contains multiple ground truths)
        self.meta_file = [file for file in self.file_list if "meta" in file]
        if len(self.meta_file) > 0:
            self.meta = self.load_json()

        # transformations
        self.toTensor = transforms.ToTensor() # from PIL image to Tensor
        self.toPilImage = transforms.ToPILImage() # Tensor to Pil Image

    def get_modalities(self):
        first_char = self.file_list[0][0]
        mods = [x for x in self.file_list if x.startswith(first_char) and x.endswith(self.extension)]
        return [x.replace('.', '_').split('_')[1] for x in mods]

    def files_to_dict(self):
        dictionary = {}
        for x in self.file_list:
            if x.endswith(self.extension):
                if x.split('_')[0] not in dictionary.keys():
                    dictionary[x.split('_')[0]] = {}
                dictionary[x.split('_')[0]][x.replace('.', '_').split('_')[1]] = x

        return dictionary

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        return pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images_dict = {}

        key = self.keys[idx]
        for mod in self.modalities:
            file = self.open_file(os.path.join(self.root_dir, self.dictionary[key][mod]))
            if self.scale != 1.0:
                file = self.preprocess(file, scale=self.scale, is_mask=mod!='img')
            images_dict[mod] = self.toTensor(file)

        return images_dict

    def open_file(self, path):
        if self.extension == '.png':
            return Image.open(path)
        elif self.extension == '.bin':
            return np.fromfile(path, dtype='float16').reshape(512,512,-1)

    def show_imgdir(self, images_dict):
        return self.toPilImage(make_grid([images_dict[key] for key in self.modalities]))

    def load_json(self):
        with open(os.path.join(
                self.root_dir, self.meta_file[0])) as f:
            return json.load(f)
