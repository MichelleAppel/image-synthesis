""" Dataset class for ArchVizPro data

"""
import os
import json

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.utils import make_grid

from models import networks

from PIL import Image

import numpy as np

torch.manual_seed(42)

class TestDataset(Dataset):
    """The dataset for the ArchVizPro data

    """
    def __init__(self, root_dir, do_domain_transfer=False, net_G_path="checkpoints\fake-to-real_AVP3456\latest_net_G_A.pth"):

        # file locations
        self.root_dir = root_dir # file path of the image dataset
        files = os.listdir(root_dir)
        self.numbers = [file.split('_')[0] for file in files if 'predmask' in file]

        self.do_domain_transfer = do_domain_transfer
        if self.do_domain_transfer:
            self.netG_B = networks.define_G(3, 3, 16, 'resnet_9blocks', 'instance',
                                        False, 'normal', 0.02)
            load_path = net_G_path
            device = torch.device('cuda:{}'.format(0))
            state_dict = torch.load(load_path, map_location=str(device))
            self.netG_B.load_state_dict(state_dict)
            self.netG_B.eval()

        # transformations
        self.toTensor = transforms.ToTensor() # from PIL image to Tensor
        self.toPilImage = transforms.ToPILImage() # Tensor to Pil Image
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
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
        # predmask_bin = np.concatenate((predmask_mono, 1-predmask_mono), 0)

        truemask = np.array(Image.open(truemask_filename))
        truemask_mono = (truemask.transpose(2,0,1)[0:1]/255)#.astype(int)

        if self.do_domain_transfer:
            dict['img'] = self.domain_transfer(self.normalize(np.array(Image.open(img_filename))))
        else:
            dict['img'] = np.array(Image.open(img_filename))
        dict['predmask'] = predmask_mono
        dict['truemask'] = truemask_mono

        return dict

    def domain_transfer(self, batch):
        with torch.no_grad():
            output_batch = self.netG_B(batch.unsqueeze(0)).detach()
        return output_batch.squeeze(0)