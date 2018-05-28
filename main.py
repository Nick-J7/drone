import argparse
import logging

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torchvision.transforms
from torch.autograd import Variable

from airsim_data import (AirSimDataSet, ROICrop, BrightJitter, HorizontalFlip,
                         ToTensor, get_data_count)

import pdb


def main(args, logger):

    # Set parameter
    cudnn.benchmark = True

    train_csv = 'cooked_data/train.csv'
    val_csv = 'cooked_data/validation.csv'
    test_csv = 'cooked_data/test.csv'

    train_loader = get_loader(train_csv, usage='train', args=args)

    # TODO: Go to make model architecture.
    model = 

    for epoch in range(args.num_epoch):
        # TODO: scheduler


def get_loader(csv_filepath, usage, args):

    #box = (0, 74, 256, 144) # => (256, 70)
    box = (0, 8, 256, 136) # => (256, 128)
    if usage == 'train':
        transform = torchvision.transforms.Compose([
            ROICrop(box),
            BrightJitter(brightness=0.3),
            HorizontalFlip(prob=0.5),
            ToTensor(),
            ])
    else:
        transform = torchvision.transforms.Compose([
            ROICrop(box),
            ToTensor(),
            ])

    dataset = AirSimDataSet(csv_filepath=csv_filepath,
                            transform=transform) 

    pin_memroy = torch.cuda.is_available()
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, shuffle=(usage == 'train'),
            num_workers=args.num_workers, pin_memory=pin_memory)

    return dataloader


def train(dataloader, model, criterion, optimizer, scheduler, args):

    losses = RunningAverage()

    model.train()

    for i, (input, target) in enumerate(dataloader):

        if not torch.cuda.is_available():
            input = Variable(input)
            target = Variable(target)
        else:
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimzer.step()

        batch_size = input.size(0)
        losses.add(loss.item(), batch_size)

        if i % args.print_freq == 0:
            logger.debug("[{current:06d}/{total:06d}] Loss : {loss:2.5f}".format(
                current=batch_size*i,
                total=len(dataloader.dataset),
                loss=loss.item()))

    logger.debug("Finish training !\n"
            "Average Loss: {loss:2.5f}".format(
                loss=losses.mean))

    return losses.mean

def validate(dataloader, model, criterion, args):

    losses = RunningAverage()
    
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):

            if not torch.cuda.is_available():
                input = Variable(input)
                target = Variable(target)
            else:
                input = Variable(input).cuda()
                target = Variable(target).cuda(non_blocking=True)

            output = model(input)

            loss = criterion(output, target)

            batch_size = input.size(0)
            losses.add(loss.item(), batch_size)

            if i % args.print_freq == 0:
                logger.debug("[{current:06d}/{total:06d}] Loss : {loss:2.5f}".format(
                    current=batch_size*i,
                    total=len(dataloader.dataset),
                    loss=loss.item()))

    logger.debug("Finish validation !\n"
            "Average Loss: {loss:2.5f}".format(
                loss=losses.mean))

    return losses.mean


class RunningAverage(object):

    def __init__(self):
        self.mean = 0
        self.count = 0

    def add(self, value, count):
        self.mean = (self.mean * self.count + value * count) / (self.count + count)
        self.count += count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Drone driving training with Pytorch')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')

    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()

    logger = logging.getLogger("Main")
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    main(args, logger)


