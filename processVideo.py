import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from util import print_time
from myModel import myResnet50
from PIL import Image


@print_time
def process_video(model, args):
    if args.video_path is None: return 
    video_path = args.video_path
    mp4_file = []
    
    for _,_,file in os.walk(video_path):
        for i in file:
            if i.endswith('mp4'):
                mp4_file.append(os.path.join(video_path,i))
        break

    for v in mp4_file:
        args.video = v
        args.resultPath = args.pretrained.replace('modelPath','resultPath')
        args.resultPath = '/'.join(args.resultPath.split('/')[:-1])
        args.tarNpyPath = os.path.join(args.resultPath, 'video_feat')
        if not os.path.exists(args.tarNpyPath):
            os.makedirs(args.tarNpyPath)

        args.tarNpyPath =os.path.join(args.tarNpyPath, args.video.split('/')[-1][:-4])
        # ===============读取视频及信息===============
        cap = cv2.VideoCapture(args.video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ===============配置特征计算模型===============
        model.cuda()
        model.eval()

        # ===============特征计算的transforms===============  
        featCal_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ])

        # ==============================开始处理视频==============================  
        pred_labels = []
        print(args.video)
        pbar = tqdm(range(frame_num))
        for i in pbar:
            fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, img = cap.read()
            if not ret:
                print('Read error at frame %d in %s'%(fno, args.video))
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img = cv2.resize(img,(400,224))
            
            img = Image.fromarray(img).convert('RGB')
            
            img = featCal_transforms(img)
            img = img.unsqueeze(0)
            img = img.cuda()
            with torch.no_grad():
                out = model(img)
            _, pred = out.topk(1, 1, True, True)
            pred = pred.t()
            if torch.cuda.is_available():
                pred = pred.cpu()
            pred = pred.numpy()[0][0]
            pred_labels.append(pred)

        pred_labels = np.array(pred_labels, dtype=int)
        np.save(args.tarNpyPath+'_predLabels.npy', pred_labels)
        print('Save feature npy file to %s. Done.'%(args.tarNpyPath+'_predLabels.npy'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.class_num = 6
    args.video_path = '/home/hsc/Research/TrafficSceneClassification/data/video/tmp'
    args.pretrained = '/home/hsc/Research/TrafficSceneClassification/runningSavePath/modelPath/20220420_21_52_26_lr_0.03_decay_0.0001_bsz_128_BDD100K/ckpt_epoch_6_Best.pth'

    ckpt = torch.load(args.pretrained)
    model = myResnet50(args.class_num)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    process_video(model, args)