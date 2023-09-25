from __future__ import print_function
import argparse

from models import resnetv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import models.kernel_densenet as dn
import util.svhn_loader as svhn
import numpy as np
import time
from util.metrics import compute_traditional_ood

from util.score import get_score
def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95
def cal_metric(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

#%%
import argparse
import os
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--out-dataset', default="SVHN", type=str, help='out-distribution dataset')
parser.add_argument('--name', default="densenet", type=str,
                    help='neural network name and training set')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--p', default=None, type=int, help='sparsity level')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')

parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')

parser.add_argument('--method', default='energy', type=str, help='odin mahalanobis')
parser.add_argument('--cal-metric', help='calculatse metric directly', action='store_true')
parser.add_argument('--clip_threshold', default=1e5, type=float, help='odin mahalanobis')

parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon')
parser.add_argument('--iters', default=40, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')

parser.add_argument('--severity-level', default=5, type=int, help='severity level')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    help='mini-batch size')

parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

parser.add_argument('--case', default='each', type=str,
                    help='LA(lapalce) or No(normal) ')
parser.add_argument('--trainset', default=False, type=bool,
                    help='LA(lapalce) or No(normal) ')



parser.add_argument('--lamb', default=1, type=float,
                    help='')
parser.add_argument('--q1', default=0, type=float,
                    help='')
parser.add_argument('--q2', default=1, type=float,
                    help='')

parser.add_argument('--a', default=0, type=float,
                    help='')
parser.add_argument('--m', default=1, type=float,
                    help='')


parser.add_argument('--featmin', default=-1e6, type=float,
                    help='')

parser.add_argument('--featmax', default=1e6, type=float,
                    help='')

parser.add_argument('--bin', default=6, type=int,
                    help='')
parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
if args.in_dataset == 'CIFAR-10' or args.in_dataset == 'CIFAR-100':
    args.out_datasets = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
    weight = torch.Tensor(np.load(args.in_dataset+args.model_arch+'weight.npy'))
    bias = torch.Tensor(np.load(args.in_dataset+args.model_arch+'bias.npy'))
    in_feat = torch.Tensor(np.load('bin'+args.in_dataset+args.model_arch+'feat.npy'))
    
if args.in_dataset == 'imagenet':
    args.out_datasets = ['dtd', 'sun', 'inat', 'places']
    weight = torch.Tensor(np.load(args.in_dataset+args.model_arch+'weight.npy'))
    bias = torch.Tensor(np.load(args.in_dataset+args.model_arch+'bias.npy'))
    in_feat = torch.Tensor(np.load('bin'+args.in_dataset+args.model_arch+'feat.npy'))
print(in_feat.shape)



#in_feats = torch.Tensor(np.load(args.in_dataset+args.model_arch+'feat.npy'))
# for out_dataset in args.out_datasets:
AUROC = []
FPR = []
AUIN = []
if args.case == 'all':
    out_feats = None
    for out_dataset in args.out_datasets:
        if out_feats  ==None:
            out_feats = torch.Tensor(np.load('bin'+out_dataset+args.model_arch+'feat.npy'))
        else:
            out_feats = torch.cat((out_feats,torch.Tensor(np.load('bin'+out_dataset+args.model_arch+'feat.npy'))))
if args.trainset :
    in_feats =  torch.Tensor(np.load('feat.npy'))
    print(in_feats.shape)
for out_dataset in args.out_datasets:
    out_feat = torch.Tensor(np.load('bin'+out_dataset+args.model_arch+'feat.npy'))
    #compute p_in and p_ood
    # p_in = []
    # p_ood = []
    # xiuz = []
    x = []
    for i in range(in_feat.shape[1]):
        if args.case == 'each':
            maxi = float(max(in_feat[:,i].max(),out_feat[:,i].max()))
        else:
            maxi = float(max(in_feat[:,i].max(),out_feats[:,i].max()))
            
        if args.trainset:
            if args.case == 'each':
                maxi = float(max(in_feats[:,i].max(),out_feat[:,i].max()))
            else:
                maxi = float(max(in_feats[:,i].max(),out_feats[:,i].max()))            
        # a,_=torch.histogram(in_feat[:,i], range=(0,maxi),bins=args.bin)
        # a = a/len(in_feat)
        #maxi = float(max(in_feat[:,i].max(),out_feat[:,i].max()))
        # print(in_feat[:,i].max(),out_feat[:,i].max())
        if args.trainset:
            a,_=torch.histogram(in_feats[:,i], range=(0,maxi),bins=args.bin)
            a = a/len(in_feats)
        else:
            a,_=torch.histogram(in_feat[:,i], range=(0,maxi),bins=args.bin)
            a = a/len(in_feat)
            
        if args.case == 'each':
            b,_ = torch.histogram(out_feat[:,i], range=(0,maxi),bins=args.bin)
            b = b/len(out_feat)
        if args.case == 'all':
            b,_ = torch.histogram(out_feats[:,i], range=(0,maxi),bins=args.bin)
            b = b/len(out_feats)
        
        # print(torch.sum(a))
        # print(torch.sum(b))
        modify = 1-((b+1e-8)/(a+1e-8))
        #modify = a - b

        in_feat[:,i] = in_feat[:,i]+args.lamb*modify[(args.bin*in_feat[:,i]/maxi).int().clip(max=args.bin-1).tolist()]
        out_feat[:,i] = out_feat[:,i]+args.lamb*modify[(args.bin*out_feat[:,i]/maxi).int().clip(max=args.bin-1).tolist()]
        # assert 1==2
        # #%%
        # for j in range(len(in_feat)):
        #     in_feat[j,i] = in_feat[j,i]+args.lamb*xiuz[min(int(args.bin*in_feat[j,i]/maxi),args.bin-1)]
        # for j in range(len(out_feat)):
        #     out_feat[j,i] = out_feat[j,i]+args.lamb*xiuz[min(int(args.bin*out_feat[j,i]/maxi),args.bin-1)]            
        
    in_feat = in_feat.clip(min=0)
    out_feat = out_feat.clip(min=0)
    #%%
    in_energy_score = torch.logsumexp(torch.matmul(in_feat,weight.T)+bias,dim=1)
    ood_energy_score = torch.logsumexp(torch.matmul(out_feat,weight.T)+bias,dim=1)
#%%
    # print(torch.topk(ood_energy_score,k=5))
    in_energy_score = in_energy_score.numpy()
    ood_energy_score = ood_energy_score.numpy()
    
    result = cal_metric(in_energy_score,ood_energy_score,None)
    print(out_dataset,result)
    AUROC.append(result['AUROC'])
    FPR.append(result['FPR'])
    AUIN.append(result['AUIN'])
print('AUROC',sum(AUROC)/len(AUROC))
print('FPR:',sum(FPR)/len(FPR))
print('AUIN',sum(AUIN)/len(AUROC))













