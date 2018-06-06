import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
import torch.utils.data
import torchvision.transforms

import pdb


class AirSimDataSet(torch.utils.data.Dataset):
    """AirSim dataset."""

    def __init__(self, csv_filepath, transform=None):
        """
        Args:
            csv_file_path (string): Path to the csv file about data.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.airsim_dataframe = pd.read_csv(csv_filepath)
        self.transform = transform

    def __len__(self):
        return len(self.airsim_dataframe)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is target steering angle.
        """
        # Full path of image.
        img_path = self.airsim_dataframe.iloc[idx, 0]
        image = pil_loader(img_path) 
       
        # Target steering angle.
        label = self.airsim_dataframe.iloc[idx, -1]
        label = float(label)

        # Get state of this timestamp (size: 4)
        # 0: speed, 1: throttle, 2: brake, 3: gear
        car_state = self.airsim_dataframe.iloc[idx, 2:-1]
        car_state = car_state.tolist()
        if car_state[-1] == 'N':
            car_state[-1] = 0
        else:
            car_state[-1] = float(car_state[-1])
        car_state[0] *= 0.01
        car_state[-1] *= 0.1
        car_state = np.array(car_state, dtype=np.float64)


        sample = {'image':image, 'label':label, 'state':car_state}

        if self.transform is not None:
            sample = self.transform(sample)

        return (img_path, sample['image'], sample['state'], sample['label'])


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ROICrop(object):
    """Crop the ROI with square frame.

    Args:
        box (tuple):
            0 -> left
            1 -> up
            2 -> right
            4 -> bottom
    """
    def __init__(self, box):
        assert isinstance(box, tuple)
        self.box = box

    def __call__(self, sample):
        image = sample['image']
        sample['image'] = image.crop(self.box)

        return sample


class Resize(object):
    r"""Resize the image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
        (h, w), the output size will be matched to this. 
        interpolation (int, optional): Desired interpolation. Default is
            'PIL.Image.BILINEAR'
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        image = image.resize(size[::-1], interpolation)
        sample['image'] = image

        return sample


class BrightJitter(object):
    """Randomly change the brightness of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].

    """
    def __init__(self, brightness=0):
        self.brightness = brightness


    def __call__(self, sample):
        assert self.brightness > 0
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)

        image = sample['image']
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        sample['image'] = image

        return sample


class HorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, label, state = sample['image'], sample['label'], sample['state']

        if random.random() < self.prob:
            state[1] = -state[1]
            return {'image':image.transpose(Image.FLIP_LEFT_RIGHT),
                    'label':-label,
                    'state':state}
        else:
            return sample


class ToTensor(object):
    """Convert sample to Tensors."""
    
    def __call__(self, sample):
        pic, label, state = sample['image'], sample['label'], sample['state']

        # Support only PIL.Image.Image and 'RGB' mode
        assert isinstance(pic, Image.Image) and pic.mode == 'RGB'
        
        image = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        image = image.view(pic.size[1], pic.size[0], 3)
         
        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose(0, 1).transpose(0, 2).contiguous()
        image = image.float().div(255)

        label = torch.FloatTensor([label])
        state = torch.FloatTensor(state)

        return {'image':image, 'label':label, 'state':state}


def get_data_count(csv_file_path):
    """Get count of each class.
        
        Beacause label is float(continuous), 
        split into two groups by (0, not 0).

    Args:
        csv_file_path (string): path of data file

    Returns:
        tuple: count of (0, not 0)
    """
    dataframe = pd.read_csv(csv_file_path)
    nonzero_count = dataframe.astype(bool)['Label'].sum(axis=0)
    return dataframe.shape[0] - nonzero_count, nonzero_count


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    transform = torchvision.transforms.Compose([
        #ROICrop((0, 74, 256, 144)),
        #BrightJitter(0.7),
        ])
    airsim_dataset = AirSimDataSet(csv_filepath='cooked_data/train.csv', 
                                   transform=transform)
    for i in range(len(airsim_dataset)):
        img_path, image, state, label = airsim_dataset[i]
        pdb.set_trace()
        print(i, 'iamge:', image.size, 'label:', label)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(image)
        
        if i == 3:
            plt.show()
            break
