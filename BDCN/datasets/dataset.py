import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
import random
from io import StringIO

import torchvision.transforms as transforms



def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if not cache.__contains__(path):
			with open(path, 'rb') as f:
				cache[path] = f.read()
		return Image.open(StringIO(cache[path]))
	return Image.open(path)


class Data(data.Dataset):
	def __init__(self, root, lst=None, yita=0.5,
		mean_bgr = np.array([104.00699, 116.66877, 122.67892]),
		crop_size=None, rgb=True, scale=None,
		do_domain_transfer = True,
		net_G_path=r'C:\Users\appel\Documents\Project\image-synthesis\checkpoints\fake2real_448\latest_net_G_A.pth'):
		self.mean_bgr = mean_bgr
		self.root = root

		if lst==None:
			self.lst = os.listdir(root)
		else:
			self.lst = lst
			
		self.yita = yita
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}

		self.files = [file for file in self.lst if "img" in file]

		self.do_domain_transfer = do_domain_transfer
		if self.do_domain_transfer:
			import networks
			self.netG_B = networks.define_G(3, 3, 16, 'resnet_9blocks', 'instance', False, 'normal', 0.02)

			load_path = net_G_path
			device = torch.device('cuda:{}'.format(0))
			state_dict = torch.load(load_path, map_location=str(device))
			self.netG_B.load_state_dict(state_dict)
			self.netG_B.eval()

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		# load img
		img_file = self.files[index]
		# img = load_image_with_cache(os.path.join(self.root, img_file), self.cache)
		img = Image.open(os.path.join(self.root, img_file))

		# load gt image
		gt_file = img_file.replace('img', 'outlines')
		# gt = load_image_with_cache(os.path.join(self.root, gt_file), self.cache)
		gt = Image.open(os.path.join(self.root, gt_file))

		if gt.mode == '1':
			gt  = gt.convert('L')

		img, gt = self.transform(img, gt)

		if self.do_domain_transfer:
			img = self.domain_transfer(img)

		return img, gt

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)
		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.
		gt[gt >= self.yita] = 1
		gt = torch.from_numpy(np.array([gt])).float()
		img = np.array(img, dtype=np.float32)
		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR
		img -= self.mean_bgr
		data = []
		if self.scale is not None:
			for scl in self.scale:
				img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
				data.append(torch.from_numpy(img_scale.transpose((2,0,1))).float())
			return data, gt
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img.copy()).float()
		if self.crop_size:
			_, h, w = gt.size()
			assert(self.crop_size < h and self.crop_size < w)
			i = random.randint(0, h - self.crop_size)
			j = random.randint(0, w - self.crop_size)
			img = img[:, i:i+self.crop_size, j:j+self.crop_size]
			gt = gt[:, i:i+self.crop_size, j:j+self.crop_size]
		return img, gt

	def domain_transfer(self, batch):
		with torch.no_grad():
			output_batch = self.netG_B(batch.unsqueeze(0)).detach()
		return output_batch.squeeze(0)

