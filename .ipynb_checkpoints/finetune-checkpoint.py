from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam, lr_scheduler
from torchvision import transforms, datasets
from torchvision.models import resnet50

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import MyImageFolder
from networks.resnet import SupConResNet, LMCLResNet, LinearClassifier
from networks.layers import MarginCosineProduct, cosine_sim
from losses import SupConLoss

# from evaluate_lmcl import validate

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def set_loader(config):
    if config['dataset'] == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif config['dataset'] == 'cifar100' or config['dataset'] == 'path':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif config.get('mean') is not None and config.get('std') is not None:
        mean = eval(config['mean'])
        std = eval(config['std'])
    else:
        raise ValueError('dataset not supported: {}'.format(config['dataset']))
        
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=config['image_size'], scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.15, 0.1, 0.1)
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(31, 2)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(size=(config['image_size'], config['image_size'])), # support batching
        transforms.ToTensor(),
        normalize,
    ])
    
#     if config['model'] == 'supcon':
#         train_transform = TwoCropTransform(train_transform)
    
    if config['dataset'] == 'cifar10':
        train_dataset = datasets.CIFAR10(root=config['data_folder'],
                                         transform=train_transform,
                                         download=True)
    elif config['dataset'] == 'cifar100':
        train_dataset = datasets.CIFAR100(root=config['data_folder'],
                                          transform=train_transform,
                                          download=True)
    elif config['dataset'] == 'path':
        train_dataset = MyImageFolder(root=config['data_folder']+"/train",
                                            transform=train_transform)
        print("[Dataset] Load training data from ", config['data_folder']+"/train")
        print(train_dataset.class_to_idx)
        val_dataset = MyImageFolder(root=config['data_folder']+"/val",
                                            transform=val_transform)
        print("[Dataset] Load validation data from ", config['data_folder']+"/val")
        print(val_dataset.class_to_idx)
    
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=False,
    num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader




def set_model(config):
    if config['model'] == 'lmcl':
        model = LMCLResNet(**config['model_args'])
        loss = nn.CrossEntropyLoss()
    elif config['model'] == 'supcon':
        model = SupConResNet(**config['model_args'])
        loss = torch.nn.CrossEntropyLoss()
        classifier = LinearClassifier(**config['classifier_args'])
        
    if config.get('load'):
        print('Loading pretrained model from: ', config['load'])
        model.load_state_dict(torch.load(config['load'])['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()
        classifier = classifier.cuda()
    return model, classifier, loss


def set_optimizer(config, params):
    if config['optimizer'] == 'sgd':
        optimizer = SGD(params, **config['optimizer_args'])
    elif config['optimizer'] == 'adam':
        optimizer = Adam(params) if config.get('optimizer_args') is None else Adam(params, **config['optimizer_args'])
    
    if config.get('scheduler') is None:
        scheduler = None
        print("[Scheduler] do not use lr shceduler")
    elif config['scheduler'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=300) if config.get('scheduler_args') is None else lr_scheduler.StepLR(optimizer, **config['scheduler_args'])
    elif config['scheduler'] == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99) if config.get('scheduler_args') is None else lr_scheduler.ExponentialLR(optimizer, **config['scheduler_args'])
    elif config['scheduler'] == 'cos':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=5) if config.get('scheduler_args') is None else lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config['scheduler_args'])
    elif config['scheduler'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max') if config.get('scheduler_args') is None else lr_scheduler.ReduceLROnPlateau(optimizer, **config['scheduler_args'])

    return optimizer, scheduler



def train(train_loader, model, criterion, classifier, optimizer, scheduler, config):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
#         if config['model'] == 'supcon':
#             images = torch.cat([images[0], images[1]], dim=0) # torch.Size([2*B, 3, 32, 32])

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        while features.dim() > 2:
            features = torch.squeeze(features, dim=2)
        output = classifier(features.detach())
        loss = criterion(output, labels)
                
        
        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # scheduler
        if scheduler is not None:
            if config['scheduler'] != 'plateau':
                scheduler.step()
            else:
                scheduler.step(acc1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, config, classifier=None):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model.encoder(images)
            while features.dim() > 2:
                features = torch.squeeze(features, dim=2)
            output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.avg, top1.avg

def set_save(config):
    # set the path according to the environment
    if not os.path.isdir('save'):
        os.makedirs('save')
    save_model_path = 'save/{}_models'.format(config['model'])
    save_logger_path = 'save/{}_tensorboard'.format(config['model'])

    model_name = 'model_{}_load_pt_encoder_{}_optimizer_{}_bs_{}'.format(config['model'], config['model_args']['load_pt_encoder'], config['optimizer'], config['batch_size'])
    if config.get('scheduler') is not None:
        model_name += '_scheduler_{}'.format(config['scheduler'])

    save_logger_path = os.path.join(save_logger_path, model_name)
    if not os.path.isdir(save_logger_path):
        os.makedirs(save_logger_path)

    save_model_path = os.path.join(save_model_path, model_name)
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
        
    return save_model_path, save_logger_path

def main(config):

    best_acc = 0
    best_epoch = 0
    
    ### Dataset and DataLoader ###
    train_loader, val_loader = set_loader(config)

    ### Model and Loss ###
    model, classifier, criterion = set_model(config)

    ### Optimizer ###
    optimizer, scheduler = set_optimizer(config, classifier.parameters())

    ### Save ###
    save_model_path, save_logger_path = set_save(config)
    
    logger = tb_logger.Logger(logdir=save_logger_path, flush_secs=2)

    # training routine
    for epoch in range(1, config['epochs'] + 1):
#         adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, classifier, optimizer, scheduler, config)
        time2 = time.time()
        # eval
        val_loss, val_acc = validate(val_loader, model, criterion, config, classifier)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        
        if epoch % config['print_freq'] == 0:
            print('epoch {}, train time {:.2f}, train_loss {:.2f}, train_acc {:.2f}; val_loss {:.2f}, val_acc {:.2f}'.format(epoch, time2 - time1, train_loss, train_acc, val_loss, val_acc))

        
        # tensorboard logger
        logger.log_value('train loss', train_loss, epoch)
#         logger.log_value('train acc', train_acc, epoch)
#         logger.log_value('val loss', val_loss, epoch)
#         logger.log_value('val acc', val_acc, epoch)
#         logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        if epoch % config['save_freq'] == 0:
            save_file = os.path.join(
                save_model_path, 'finetune_ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, config, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        save_model_path, 'last.pth')
    save_model(model, optimizer, config, config['epochs'], save_file)
    
    print('Finetuning model: {}, best validation accuracy: {:.2f}, epoch: {}'.format(config['load'], best_acc, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    main(config)
