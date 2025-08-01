import argparse
import time
import os
import torch
import numpy as np
from numpy import random
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
#import torchvision.models as models
from datetime import timedelta
from net_models import *
import sys
sys.path.append("ViT-pytorch")
sys.path.append("ViT-pytorch/models")
from models.modeling import VisionTransformer, CONFIGS
from transformers import AutoImageProcessor, AutoModelForImageClassification ,ConvNextForImageClassification

def get_expr_name(ldb=True, model='resnet', optimizer='sgd', lr=0.1,momentum=0, droprate=0, dataset='cifar10'):
    name = {
        'sgd': 'lr{}-momentum{}--droprate{}'.format(lr, momentum, droprate)
    }[optimizer]
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    #time of we run the script
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    
    if ldb==True:
        method="LayerDropBack"
    else:
        method="Standard"

    return '{}-{}-{}-{}-{}-{}'.format(TIME_NOW,method, dataset, model, optimizer, name)


def build_model(args,device):
    print('==> Building model..')

    no_of_class = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'svhn': 10,
        'imagenet': 1000
    }[args.dataset]
    config = CONFIGS[args.model]
    pretrain_weights = {
    'ViT-B_16' : 'pretrain/ViT-B_16.npz',
    'ViT-L_16' : 'pretrain/ViT-L_16.npz',
    'ViT-H_16' : 'pretrain/ViT-H_16.npz'
        }

    if args.model in pretrain_weights.keys():
        net = VisionTransformer(config, (args.image_resolution,args.image_resolution), zero_head=True, num_classes=no_of_class)
        net.load_from(np.load(pretrain_weights[args.model]))
    if args.model == "SwinTransformer":
        net = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        net.classifier = nn.Linear(net.classifier.in_features,no_of_class)
    net = net.to(device)
    return net

def build_dataset(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.dataset == 'cifar10':
            dataset = torchvision.datasets.CIFAR10
            TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
            TRAIN_STD = (0.2023, 0.1994, 0.2010)
        else:
            dataset = torchvision.datasets.CIFAR100
            TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.Resize((args.image_resolution,args.image_resolution)),
            transforms.RandomCrop(args.image_resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((args.image_resolution,args.image_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])

        trainset                    = dataset(root='./data', train=True, download=True, transform=transform_train)
        trainloader                 = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=32)
        trainloader_double_batch    = torch.utils.data.DataLoader(trainset, batch_size=args.batch*2, shuffle=True, num_workers=32)

        testset = dataset(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch, shuffle=False, num_workers=32)

    if args.dataset=='svhn':
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=1)

        testset = torchvision.datasets.SVHN(
            root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch, shuffle=False, num_workers=1)
        
    elif args.dataset == 'imagenet':
        path = args.imagnet_path
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
                                                                    transforms.Resize((args.image_resolution,args.image_resolution)),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(15),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=32, pin_memory=True,shuffle=True)
        trainloader_double_batch = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch*2, num_workers=32, pin_memory=True,shuffle=True)

        val_dateset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize((args.image_resolution,args.image_resolution)),
                                                        transforms.ToTensor(),
                                                        normalize]))
        val_loader = torch.utils.data.DataLoader(val_dateset,
                                                    batch_size=args.batch, shuffle=True,
                                                    num_workers=32, pin_memory=False)
        
        return train_loader , val_loader,trainloader_double_batch

    return trainloader, testloader,trainloader_double_batch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 

def set_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

def set_grad(p,Val):
    p.requires_grad=Val

def train(epoch, model, device, train_loader_single_batch, optimizer, criterion, args, params,train_loader_double_batch):
    model.train()
    train_loss      = 0
    train_total     = 0
    train_correct   = 0
    
    start_time      = time.process_time()

    train_loader=train_loader_single_batch
    if params is not None:
        if args.ldb and epoch % args.skip == 0 and epoch>0:
            print("setting droplayer")
            original_lr = get_lr(optimizer)
            set_lr(optimizer,original_lr/(1-args.droprate+0.000001))                   
            train_loader=train_loader_double_batch
        else:  
            [set_grad(p,True) for p in params]    

    for _, (data, target) in enumerate(train_loader, start=0):  
        data = data.to(device)
        target = target.to(device)        
        optimizer.zero_grad()  
        
        # drop layers
        if params is not None and args.ldb and epoch % args.skip == 0 and epoch>0:
            [set_grad(p,False) if np.random.uniform() < args.droprate else set_grad(p,True) for p in params]
        output = model(data) 
        if type(output) is tuple:
            output = output[0]
        else:
            output = output.logits
        loss = criterion(output, target)
        train_loss += loss.item()
        _, predictions = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += int(sum(predictions == target))
        loss.backward()
        optimizer.step()


    end_time=time.process_time() - start_time
    acc = round((train_correct / train_total) * 100, 2)
    
    if params is not None:
        if args.ldb and epoch % args.skip == 0 and epoch>0:
           set_lr(optimizer,original_lr)
        print('LDB:   Epoch [{}], Train Loss: {}, Train Accuracy: {}, Epoch Time {}'.format(epoch, train_loss / train_total, acc,end_time), end='') 
    return end_time


def test(epoch,model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) 
            if type(output) is tuple: #for ViT
                output = output[0]
            else:
                output = output.logits # for hugging face models
            test_loss += criterion(output, target).item()
            scores, predictions = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += int(sum(predictions == target))
    acc = round((test_correct / test_total) * 100, 2)
    print(' Test_loss: {}, Test_accuracy: {}'.format(test_loss / test_total, acc))
    return acc
    
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='ResNet18', type=str, help='model')
    parser.add_argument('--dataset', type=str, default="cifar10", help='dataset', choices=['cifar10', 'cifar100', 'mnist', 'svhn','imagenet'])
    parser.add_argument('--imagnet_path', default='data/' ,type=str, help='Path to imagenet if used') 
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default="linear", type=str, help='scheduler', choices=['linear', 'cosine'] )
    parser.add_argument('--lr_gamma', default=0.2, type=float, help='learning rate gamma')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--droprate', default=0.5, type=float, help='layer drop percentage')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--skip', default=2, type=int, help='skip index')
    parser.add_argument('--epochs', default=202, type=int, help='number of epochs')
    parser.add_argument('--optim', type=str, default="sgd", help='optimizer', choices=['sgd', 'dlrd', 'lrd', 'adam'])
    parser.add_argument('--seed', default=2, type=int, help='seed')
    parser.add_argument("--no_ldb", action="store_true",help="do not use layerdropback") 
    parser.add_argument('--milestones', type=str,default="60,120,160,180",help="milestones for lr decay")
    parser.add_argument('--image_resolution', default=224, type=int, help='input image resolution to be resized to') 

    return parser
      

def main():    
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    
    parser          = get_parser()
    args            = parser.parse_args()
    args.milestones = [int(item) for item in args.milestones.split(',')]
    args.ldb        = not args.no_ldb # more easy, ldb positive as default
    args.expr_name  = get_expr_name(args.ldb,model=args.model, optimizer=args.optim, lr=args.lr,
                                     momentum=args.momentum,droprate = args.droprate, dataset=args.dataset)
    
         
    trainloader_single_batch, testloader,trainloader_double_batch = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: " + str(device))
    print("Expr_name: " + args.expr_name)
    print("Args: ",args)


    layerdropback_net           = build_model(args, device)
    if sys.platform.lower() == 'linux':
        print("using two gpus")
        layerdropback_net           = nn.DataParallel(layerdropback_net)

    layerdropback_criterion     = nn.CrossEntropyLoss()
    layerdropback_optimizer     = optim.SGD(layerdropback_net.parameters(), args.lr, momentum=args.momentum,weight_decay=1e-4,  nesterov=True) #weight_decay=1e-4, nesterov=True) #weight_decay=5e-4) #
    if args.lr_scheduler == 'cosine':
        layerdropback_lr_scheduler  = optim.lr_scheduler.CosineAnnealingLR(layerdropback_optimizer, T_max=200)
    if args.lr_scheduler == 'linear':
        milestones             = [round(args.epochs * 0.3), round(args.epochs * 0.6), round(args.epochs * 0.85)]
        layerdropback_lr_scheduler  = optim.lr_scheduler.MultiStepLR(layerdropback_optimizer, milestones=milestones, gamma=args.lr_gamma)
    
    
    layerdropback_total_time = 0  
    layerdropback_max_acc    = 0
    

    if args.ldb==True:
        param_list = []
        for name, param in layerdropback_net.named_parameters():
            param_list.append(param)
        # do not drop the last two layers    
        param_list=param_list[:-2]
    else:
        param_list=None
            
    for epoch in range(args.epochs):
        ### training
        end_time=train(epoch, layerdropback_net, device, trainloader_single_batch, layerdropback_optimizer, layerdropback_criterion, args,param_list,trainloader_double_batch)
        acc=test(epoch, layerdropback_net, device, testloader, layerdropback_criterion)
        if acc>layerdropback_max_acc:
            layerdropback_max_acc=acc        
        layerdropback_total_time += end_time
        layerdropback_lr_scheduler.step()
       
               
        print('LDB Time: {}, LDB Max Acc {}'.format(layerdropback_total_time,layerdropback_max_acc))
        
    print('Finished Training')

if __name__ == '__main__':
    main()
    


