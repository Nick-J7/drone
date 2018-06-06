import argparse
import logging
import shutil
import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torchvision.transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from airsim_data import (AirSimDataSet, ROICrop, BrightJitter, HorizontalFlip,
                         ToTensor, get_data_count)
from my_densenet import densenet_drone
from my_densenet import simple_net 

import pdb


def main(args, logger):

    board_writer = SummaryWriter() 

    train_csv = 'cooked_data/train.csv'
    val_csv = 'cooked_data/validation.csv'
    test_csv = 'cooked_data/test.csv'

    train_dataloader = get_loader(train_csv, usage='train', args=args)
    val_dataloader = get_loader(val_csv, usage='validate', args=args)
    #test_dataloader = get_loader(test_csv, usage='test', args=args)

    #model = densenet_drone()
    model = simple_net()

    model = model.cuda()

    criterion = nn.MSELoss()
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                    momentum=0.9, weight_decay=0.9)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=0.9)
    else:
        raise NameError("[!] {} is not prepared.".format(args.optim))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.debug("Loading checkpoint => {}".format(args.resume))
            checkpoint_state = torch.load(args.resume)
            start_epoch = checkpoint_state['epoch']
            best_loss = checkpoint_state['best_loss']
            model.load_state_dict(checkpoint_state['state_dict'])
            optimizer.load_state_dict(checkpoint_state['optimizer'])

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           factor=0.5,
                                                           patience=3, 
                                                           min_lr=0.00001,
                                                           verbose=True)

    
    best_loss = 100 
    for epoch in range(args.epochs):

        #scheduler.step()
#        logger.debug("[*] Epoch [{}/{}], Learning rate: {}"
#                     .format(epoch, args.epochs, scheduler.get_lr()))
        logger.debug("[*] Epoch [{}/{}]"
                     .format(epoch, args.epochs))
        
        train_loss = train(train_dataloader, model, criterion, optimizer, args, logger)
        val_loss = validate(val_dataloader, model, criterion, args, logger)
        scheduler.step(val_loss)
        

        #TODO: save parameters
        is_best = best_loss > val_loss
        best_loss = min(best_loss, val_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            }, is_best)

        board_writer.add_scalar('Train Loss', train_loss, epoch)
        board_writer.add_scalar('Val Loss', val_loss, epoch)

    logger.debug("[*] Finish Training !!!")


def get_loader(csv_filepath, usage, args):

    #box = (0, 74, 256, 144) # => (70, 256)
    box = (0, 8, 256, 136) # => (H 128, W 256)
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

    pin_memory = torch.cuda.is_available()
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, shuffle=(usage == 'train'),
            num_workers=args.num_workers, pin_memory=pin_memory)

    return dataloader


def train(dataloader, model, criterion, optimizer, args, logger):

    losses = RunningAverage()
    data_time = RunningAverage()
    model_time = RunningAverage()

    model.train()

    end = time.time()
    for i, (path, input, state, target) in enumerate(dataloader):
        cur_data_time = time.time() - end
        data_time.add(cur_data_time)

        if not torch.cuda.is_available():
            input = Variable(input)
            state = Variable(state)
            target = Variable(target)
        else:
            input = Variable(input).cuda()
            state = Variable(state).cuda()
            target = Variable(target).cuda(non_blocking=True)

        output = model(input, state)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_model_time = time.time() - end
        model_time.add(cur_model_time)
        end = time.time()

        batch_size = input.size(0)
        losses.add(loss.item(), batch_size)

        if i % args.print_freq == 0:
            logger.debug("[{current:06d}/{total:06d}] "
                    "Data time : {data_time:2.2f} | "
                    "Model time : {model_time:2.2f} | "
                    "Loss : {loss:2.5f}".format(
                current=batch_size*i,
                total=len(dataloader.dataset),
                data_time=cur_data_time,
                model_time=cur_model_time,
                loss=loss.item()))
#            logger.debug("output:, {}\n"
#                         "target:, {}".format(output[0].item(), target[0].item()))

    logger.debug("Finish training !\n"
                 "Average data time: {data_time:2.2f} | "
                 "Average model time: {model_time:2.2f} | "
                 "Average Loss: {loss:2.5f}\n".format(
                 data_time=data_time.mean,
                 model_time=model_time.mean,
                 loss=losses.mean))

    return losses.mean


def validate(dataloader, model, criterion, args, logger):

    losses = RunningAverage()
    data_time = RunningAverage()
    model_time = RunningAverage()
    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (path, input, state, target) in enumerate(dataloader):

            cur_data_time = time.time() - end
            data_time.add(cur_data_time)

            if not torch.cuda.is_available():
                input = Variable(input)
                state = Variable(state)
                target = Variable(target)
            else:
                input = Variable(input).cuda()
                state = Variable(state).cuda()
                target = Variable(target).cuda(non_blocking=True)

            output = model(input, state)

            loss = criterion(output, target)

            cur_model_time = time.time() - end
            model_time.add(cur_model_time)
            end = time.time()

            batch_size = input.size(0)
            losses.add(loss.item(), batch_size)

            if i % args.print_freq == 0:
                logger.debug("[{current:06d}/{total:06d}] "
                        "Data time : {data_time:2.2f} | "
                        "Model time : {model_time:2.2f} | "
                        "Loss : {loss:2.5f}".format(
                    current=batch_size*i,
                    total=len(dataloader.dataset),
                    data_time=cur_data_time,
                    model_time=cur_model_time,
                    loss=loss.item()))

    logger.debug("Finish validation !\n"
                 "Average data time: {data_time:2.2f} | "
                 "Average model time: {model_time:2.2f} | "
                 "Average Loss: {loss:2.5f}\n".format(
                 data_time=data_time.mean,
                 model_time=model_time.mean,
                 loss=losses.mean))

    return losses.mean


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class RunningAverage(object):

    def __init__(self):
        self.mean = 0
        self.count = 0

    def add(self, value, count=1):
        self.mean = (self.mean * self.count + value * count) / (self.count + count)
        self.count += count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Drone driving training with Pytorch')

    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')

#    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                        help='manual epoch number (useful on restarts)')
#    parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--optim', default='sgd', type=str, metavar='OPTIM', 
                        help='optimzer')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', 
                        help='initial learning rate')

    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--validate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    logger = logging.getLogger("Main")
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    main(args, logger)

