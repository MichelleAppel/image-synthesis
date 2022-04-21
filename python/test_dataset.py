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

torch.manual_seed(42)

class TestDataset(Dataset):
    """The dataset for the ArchVizPro data

    """
    def __init__(self, root_dir):

        # file locations
        self.root_dir = root_dir # file path of the image dataset
        files = os.listdir(root_dir)
        self.numbers = [file.split('_')[0] for file in files if 'predmask' in file]
        
    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        dict = {}

        i = self.numbers[idx]
        img_filename = os.path.join(self.root_dir, i + '_img.png')
        predmask_filename = os.path.join(self.root_dir, i + '_predmask.png')
        truemask_filename = os.path.join(self.root_dir, i + '_truemask.png')

        predmask = np.array(Image.open(predmask_filename))
        predmask_mono = predmask.transpose(2,0,1)[0:1]/255
        predmask_bin = np.concatenate((predmask_mono, 1-predmask_mono), 0)

        truemask = np.array(Image.open(truemask_filename))
        truemask_mono = (truemask.transpose(2,0,1)[0:1]/255)#.astype(int)

        dict['img'] = np.array(Image.open(img_filename))
        dict['predmask'] = predmask_mono
        dict['truemask'] = truemask_mono

        return dict
