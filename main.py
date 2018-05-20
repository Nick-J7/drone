import argparse

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torchvision.transforms

from airsim_data import (AirSimDataSet, ROICrop, BrightJitter, HorizontalFlip,
                         ToTensor, get_data_count)

import pdb


def main(args):

    # Set parameter
    cudnn.benchmark = True

    train_csv = 'cooked_data/train.csv'
    val_csv = 'cooked_data/validation.csv'
    test_csv = 'cooked_data/test.csv'

    train_loader = get_loader(train_csv, usage='train', args=args)

    # TODO: Go to make model architecture.
    for i, (imgpath, input, target) in enumerate(train_loader):

        print(i, imgpath, input.shape, target)

        if i == 3:
            break


def get_loader(csv_filepath, usage, args):

    #TODO: Add Horizontal Flip
    box = (0, 74, 256, 144)
    if usage == 'train':
        transform = torchvision.transforms.Compose([
            ROICrop(box),
            BrightJitter(brightness=0.3),
            HorizontalFlip(prob=0.5),
            ToTensor(),
            ])
    else:
        transform = torchvision.transforms.Compose([
            ])


    dataset = AirSimDataSet(csv_filepath=csv_filepath,
                            transform=transform) 

    loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, shuffle=(usage == 'train'),
            num_workers=args.num_workers, pin_memory=True)

    return loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Drone driving training with Pytorch')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    main(args)

