""" Dataset class for ArchVizPro data

"""
import os
import json
import csv

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from models import networks

from PIL import Image

import numpy as np

from colorhash import ColorHash

import random

random.seed(42)
torch.manual_seed(42)

class SynthesisDataset(Dataset):
    """The dataset for the ArchVizPro data

    """
    def __init__(self, root_dir, scale=1.0, extension='.png', id_grouping=False, do_domain_transfer=True, net_G_path=None, random_crop=None):

        # file locations
        self.root_dir = root_dir # file path of the image dataset
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.random_crop=random_crop

        self.extension = extension
        self.file_list = os.listdir(self.root_dir) # gets all files in the directory
        self.img_list = [file for file in self.file_list if "img" in file] # only the images
        self.modalities = self.get_modalities() # all ground truths (rgb, semantic, outlines)
        self.dictionary = self.files_to_dict() # makes a dictionary out of image paths
        self.keys = list(self.dictionary.keys()) # all image sets (one image set contains multiple ground truths)
        self.meta_file = [file for file in self.file_list if "meta" in file]
        if len(self.meta_file) > 0:
            self.meta = self.load_json()

        self.n_classes = 2
        self.id_grouping = id_grouping
        if self.id_grouping:
            self.id_to_class = {}
            self.id_to_class_dict(os.path.join(self.root_dir, '../classes_53.csv'))
            self.add_class_to_meta()
            self.n_classes = len(self.id_to_class)

        self.do_domain_transfer = do_domain_transfer
        if self.do_domain_transfer:
            self.netG = networks.define_G(3, 3, 16, 'resnet_9blocks', 'instance', False)

            # device = torch.device('cuda:{}'.format(0))
            state_dict = torch.load(net_G_path)#, map_location=str(device))

            # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            #     self.__patch_instance_norm_state_dict(state_dict, self.netG_B, key.split('.'))

            self.netG.load_state_dict(state_dict)
            self.netG.eval()
                        
    
        # transformations
        self.toTensor = transforms.ToTensor() # from PIL image to Tensor
        self.toPilImage = transforms.ToPILImage() # Tensor to Pil Image
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if random_crop:
            self.crop = transforms.RandomCrop(random_crop)

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

        h, w = (self.random_crop, self.random_crop)
        crop_i = None
        crop_j = None

        for mod in self.modalities:
            file = self.open_file(os.path.join(self.root_dir, self.dictionary[key][mod]))

            if self.scale != 1.0:
                file = self.preprocess(file, scale=self.scale, is_mask=mod!='img')

            if self.random_crop:
                if crop_i == None:
                    crop_i = random.randint(0, max(0, file.size[1] - h))
                    crop_j = random.randint(0, max(0, file.size[0] - w))
                file = F.crop(file, crop_i, crop_j, h, w)
            
            images_dict[mod] = self.toTensor(file)

            if mod == 'img' and self.do_domain_transfer:
                images_dict['img'] = self.domain_transfer(self.normalize((images_dict['img'])))
            if mod == 'img':
                images_dict['img'] = self.normalize(images_dict['img'])

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

    def id_to_class_dict(self, csv_file='classes.csv'):
        with open(csv_file, newline='') as csvfile:
            classes = csv.reader(csvfile, delimiter=',')
            for i, c in enumerate(classes):
                self.id_to_class[i] = c

    def add_class_to_meta(self):
        for object in self.meta:
            objectname = object['objectname'].lower()
            parentsname = object['parentsname'].lower()
            for classid, classes in enumerate(self.id_to_class.values()):
                for classname in classes:
                    if classname in objectname or classname in parentsname:
                        object['classid'] = classid
                        object['classcategory'] = classes
                        break

    def image_to_class(self, batch):
        classimg = torch.clone(batch)

        for imgvalue in batch.unique():
            object = self.meta[imgvalue.item()]
            if 'classid' in object.keys():
                classimg[batch == imgvalue.item()] = object['classid']
            else:
                classimg[batch == imgvalue.item()] = 0
    
        return classimg

    def class_to_color(self, batch):
        colorimg = torch.Tensor(batch.shape[0], 3, batch.shape[2], batch.shape[3])

        for value in batch.unique():
            for i in range(colorimg.shape[1]):
                colorimg[:, i, :, :][batch[:, 0, :, :] == value] = torch.Tensor(ColorHash(value).rgb)[i]

        return colorimg

    def domain_transfer(self, batch):
        with torch.no_grad():
            output_batch = self.netG(batch.unsqueeze(0)).detach()
        return output_batch.squeeze(0)