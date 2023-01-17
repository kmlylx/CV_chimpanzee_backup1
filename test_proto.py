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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam, lr_scheduler
from torchvision import transforms, datasets
from torchvision.models import resnet50

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import MyImageFolder
from networks.resnet_big import SupConResNet, LMCLResNet, LinearClassifier
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




def test(model, train_loader, val_loader, mode='eu'):
    model.eval()
    protos = model.return_proto(train_loader).cuda() # (c, d)
    features = [] # (m, d)
    labels = [] # (m, )
    preds = [] # (m, )
    for idx, (imgs, lbs) in enumerate(val_loader):
        imgs = imgs.float().cuda()
        lbs = lbs.cuda()
        feats = model.encoder(imgs)
        while feats.dim() > 2:
            feats = torch.squeeze(feats, dim=2)
#         feats = model.head(feats)
        features.append(feats)
        labels.append(lbs)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).long()
    
    if mode == 'eu':
        # Euclidean distance
        features = torch.unsqueeze(features, dim=1)
        distances = torch.norm(features-protos, dim=2) # (m, c)
        preds = torch.argmin(distances, dim=1).long()
    elif mode == 'weight':
        # Weights
        outputs = features @ protos.t() # (m, c)
        p = F.softmax(outputs, dim=1)
        preds = torch.argmax(p, dim=1).long()
    elif mode == 'cos':
        # Cosine distance
        features = F.normalize(features, dim=1)
        protos = F.normalize(protos, dim=1)
        outputs = features @ protos.t() # (m, c)
        p = F.softmax(outputs, dim=1)
        preds = torch.argmax(p, dim=1).long()
   
    acc = 100 * (preds == labels).sum() / len(preds)
    return acc
    

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
    
    ### Get Prototypical Features ###
    train_set_acc = test(model, train_loader, train_loader, mode='cos')
    val_set_acc = test(model, train_loader, val_loader, mode='cos')
    print('Testing model with proto head and inner product: {}, train set accuracy: {:.2f}, val set accuracy: {:.2f}'.format(config['load'], train_set_acc, val_set_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    main(config)
