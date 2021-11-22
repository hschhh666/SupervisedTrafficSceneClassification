"""
Train supervised classification task with AlexNet

This code refers to CMC:https://github.com/HobbitLong/CMC/#contrastive-multiview-coding

Author: Shaochi Hu
"""
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import numpy as np

from torchvision import transforms, datasets
import torchvision

import tensorboard_logger as tb_logger
from torchvision.transforms.transforms import RandomVerticalFlip

from myModel import myResnet50

from util import accuracy, adjust_learning_rate, AverageMeter, classify,print_running_time, Logger
from dataset import ImageFolderInstance



def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print every print_freq batchs')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save model checkpoint every save_freq epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

    parser.add_argument('--class_num', type=int, default=4)

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to training data') # 训练数据文件夹，即锚点/正负样本文件夹
    parser.add_argument('--test_data_folder', type=str, default=None, help='path to testing data') # 测试数据文件夹，即所有视频帧的文件夹
    parser.add_argument('--validation_frequency',type=int, default=1) # 训练过程中验证分类精度的频率，比如每个epoch都在验证集上测试并输出分类精度，或者每十个epoch在验证集上测试等
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')
    parser.add_argument('--log_txt_path', type=str, default=None, help='path to log file')
    parser.add_argument('--result_path', type=str, default=None, help='path to sample dis and img case study') # 训练结束后，图像间距离的case study保存在这个路径下

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.8, help='low area in crop')

    parser.add_argument('--comment_info', type=str, default='', help='Comment message, donot influence program')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    
    opt.model_name = '{}_lr_{}_decay_{}_bsz_{}_{}'.format(curTime, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.comment_info)

    if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None) or (opt.log_txt_path is None) or (opt.result_path is None) or (opt.test_data_folder is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path | log_txt_path | result_path | test_data_folder')

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.log_txt_path):
        os.makedirs(opt.log_txt_path)

    opt.result_path = os.path.join(opt.result_path, opt.model_name)
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)
    
    log_file_name = os.path.join(opt.log_txt_path, 'log_'+opt.model_name+'.txt') 
    sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中

    if opt.comment_info != '':
        print('comment message : ', opt.comment_info)

    print('start program at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    print('Dataset :', opt.data_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    data_folder = os.path.join(args.data_folder, 'train')

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)

    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return train_loader

def get_val_loader(args):
    data_folder = os.path.join(args.data_folder,'val')
    if not os.path.exists(data_folder):
        print('No validation data. It\'s ok, but there are no validation classiying results.')
        return None
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_dataset = ImageFolderInstance(data_folder, transform=transform)
    n_data = len(val_dataset)
    print('number of validation samples: {}'.format(n_data))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return val_loader

def set_model(args):

    model = myResnet50(args.class_num)

    if args.resume:
        if torch.cuda.is_available():
            ckpt = torch.load(args.resume)
        else:
            ckpt = torch.load(args.resume,map_location=torch.device('cpu'))
        print("==> loaded pre-trained checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
        model.load_state_dict(ckpt['model'])
        print('==> done')

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

def train_e2e(epoch,train_loader, model, criterion, optimizer, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    end = time.time()
    for idx,(img, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)


        bsz = img.size(0)
        if torch.cuda.is_available():
            index = index.cuda()
            target = target.cuda()
            img = img.cuda()

        # ===================forward=====================
        out = model(img)
        loss = criterion(out, target)
        acc = accuracy(out, target)[0]

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        acces.update(acc.item(), bsz)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc = acces))
            sys.stdout.flush()

    return losses.avg


def main():
    # parse the args
    args = parse_option()
    args.start_epoch = 1

    # set the loader
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)

    # set the model
    model, criterion = set_model(args)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # train by epoch
    print('start training at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    min_loss = np.inf
    max_val_acc = 0
    best_model_path = args.resume
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)

        loss = train_e2e(epoch, train_loader, model, criterion, optimizer, args)

        #<----------------------------输出在验证集上的分类精度---------------------------->#
        if val_loader != None and epoch % args.validation_frequency == 0:
            model.eval()
            val_acces = AverageMeter()
            # 数据加载完毕
            for idx,(img, target, index) in enumerate(val_loader):
                bsz = img.size(0)
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                out = model(img)
                acc = accuracy(out, target)[0]
                val_acces.update(acc.item(),bsz)

            best_acc_tmp = val_acces.avg if max_val_acc < val_acces.avg else max_val_acc
            print('Epoch %d Validation accuracy %.3f%%, best validation accuracy %.3f%%'%(epoch, val_acces.avg, best_acc_tmp))

            #<----------------------------保存最好的模型---------------------------->#
            if max_val_acc < val_acces.avg:
                if max_val_acc != 0:
                    os.remove(best_model_path)
                max_val_acc = val_acces.avg
                best_model_path = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}_Best.pth'.format(epoch=epoch))
                print('==> Saving best model...')
                state = {
                    'opt': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                # if args.amp:
                #     state['amp'] = amp.state_dict()
                torch.save(state, best_model_path)
                # help release GPU memory
                del state

        print_running_time(start_time) # 输出截至当前运行的时长

        #<----------------------------按照一定频率保存模型---------------------------->#
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # if args.amp:
            #     state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state


    print("==================== Training finished. Start testing ====================")
    print('==> loading best model')
    # print('min loss = %.3f'%min_loss)
    print('max val acc = %.3f'%max_val_acc)
    if torch.cuda.is_available():
        ckpt = torch.load(best_model_path)
    else:
        ckpt = torch.load(best_model_path,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(best_model_path, ckpt['epoch']))
    print('==> done')
    model.eval()
    
    print('Classifying...')
    classify(model, args)

    print('Done.\n\n')
    print('Program exit normally.')

if __name__ == '__main__':
    main()


