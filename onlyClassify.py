import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from util import classify
from processFeature import set_model


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--num_class', default=4, type=int,help='Number of class.')
parser.add_argument('--batch-size', default=64, type=int,metavar='N')
parser.add_argument('--num_workers', default=6, type=int, metavar='N',help='number of data loading workers (default: 6)')

args = parser.parse_args()
args.pretrained = '/home/hsc/Research/TrafficSceneClassification/runningSavePath/modelPath/20220327_21_30_08_lr_0.03_decay_0.0001_bsz_128_/ckpt_epoch_7_Best.pth'
args.test_data_folder = '/home/hsc/Research/TrafficSceneClassification/data/fineGrain/tmp'
args.result_path = '/home/hsc/Research/TrafficSceneClassification/runningSavePath/resultPath/tmp'

if  not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

model = set_model(args)
classify(model,args)

gt = np.load(os.path.join(args.result_path, 'gt.npy'))
pred = np.load(os.path.join(args.result_path, 'lables.npy'))

cm = confusion_matrix(gt, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Highway','Local','Ramp','Urban'])
disp.plot()
plt.savefig(os.path.join(args.result_path, 'confusionMatrix.png'))