from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from utils import *

class ImageDataset(Dataset):
    def __init__(self, path, augment=False, crop_size=400, bgr=False):
        self.root_dir = path
        self.inputs = []
        self.outputs = []

        self.augment = augment
        self.crop_size = crop_size
        self.bgr = bgr

        for i, filename in enumerate(os.listdir(path)):
            if filename.endswith(".jpg") or filename.endswith(".png") and 'img' in filename:
                self.inputs.append(filename)

        for i, filename in enumerate(os.listdir(path)):
            if filename.endswith(".jpg") or filename.endswith(".png") and 'outlines' in filename:
                self.outputs.append(filename)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = np.float32(Image.open(os.path.join(self.root_dir, self.inputs[idx])))
        if self.bgr:
            input = input[:,:,::-1]
        input -= np.array([104.00699, 116.66877, 122.67892])
        input = torch.from_numpy(input.transpose((2, 0, 1)).copy()).float()

        output = os.path.join(self.root_dir, self.outputs[idx])
        output = Image.open(output)

        output = transforms.ToTensor()(output).float()[0:1]

        if self.augment:
            if np.random.rand() > 0.5:
                input = torch.flip(input,[2])
                output = torch.flip(output,[2])

            x = np.random.randint(0, np.maximum(0, input.size()[1] - self.crop_size))
            y = np.random.randint(0, np.maximum(0, input.size()[2] - self.crop_size))

            input = input[:, x:x + self.crop_size, y:y + self.crop_size]
            output = output[:, x:x + self.crop_size, y:y + self.crop_size]

        return input, [output]
