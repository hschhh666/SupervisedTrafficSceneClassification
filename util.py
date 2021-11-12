from __future__ import print_function

from numpy.core.defchararray import mod
from myModel import myResnet50

from sklearn.metrics import classification_report

import torch
import numpy as np
import time
import sys
import os
from dataset import ImageFolderInstance
from torchvision import transforms
from torch.nn import functional as F

def print_running_time(start_time):
    print()
    print('='*20,end = ' ')
    print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
    using_time = time.time()-start_time
    hours = int(using_time/3600)
    using_time -= hours*3600
    minutes = int(using_time/60)
    using_time -= minutes*60
    print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)),end = ' ')
    print('='*20)
    print()


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger(object): # 旨在把程序中所有print出来的内容都保存到文件中
    def __init__(self, filename="Default.log"):
        path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(path,filename)
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass



def classify(model, args):

    # 加载数据
    data_folder = args.test_data_folder
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = ImageFolderInstance(data_folder, transform=transform)
    n_data = len(dataset)
    print('number of samples: {}'.format(n_data))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    labels = []
    outs = []
    targets = []
    # 数据加载完毕
    for idx,(img, target, index) in enumerate(data_loader):
        if torch.cuda.is_available():
            img = img.cuda()
            model = model.cuda()
        with torch.no_grad(): # 不计算梯度，节省显存
            out = model(img)
        _, pred = out.topk(1, 1, True, True)
        pred = pred.t()
        if torch.cuda.is_available():
            pred = pred.cpu()
            out = out.detach().cpu()
            out = F.softmax(out, dim=1)
        pred = list(pred.numpy()[0])
        out = list(out.numpy())
        targets += list(target.numpy())
        labels += pred
        outs += out
        torch.cuda.empty_cache()
    targets = np.array(targets)
    labels = np.array(labels)
    outs = np.array(outs)
    print('<================Val classification report================>')
    print(classification_report(targets, labels, target_names=dataset.classes, digits=3))
    print('<================Val classification report================>')
    np.save(os.path.join(args.result_path, 'gt.npy'),targets)
    np.save(os.path.join(args.result_path, 'lables.npy'),labels)
    np.save(os.path.join(args.result_path, 'prob.npy'),outs)


if __name__ == '__main__':
    data = torch.randn((5,3))
    target = torch.tensor([0,0,0,0,0], dtype=torch.int64)
    print(data)
    print(target)
    print(accuracy(data, target))
    model = myResnet50()
    classify(model)

