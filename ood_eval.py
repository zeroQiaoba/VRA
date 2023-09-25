from __future__ import print_function
import argparse
import os
from models import resnetv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import models.densenet as dn
import util.svhn_loader as svhn
import numpy as np
import time
import tqdm

from util.metrics import compute_traditional_ood
from util.score import get_score
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
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

parser.add_argument('--case', default='clip2', type=str,
                    help='train or clip2 ')
parser.add_argument('--lamb', default=0, type=float,
                    help='for VRA+')
parser.add_argument('--q1', default=0, type=float,
                    help='q1 quantile')
parser.add_argument('--q2', default=1, type=float,
                    help='q2 quantile')
parser.add_argument('--a', default=0, type=float,
                    help='for VRA++')
parser.add_argument('--m', default=1, type=float,
                    help='for VRA++')
parser.add_argument('--feat', default=False, type=bool,
                    help='store the features of trainingset')

parser.add_argument('--featmin', default=-1e6, type=float,
                    help='less than featmin to 0')
parser.add_argument('--featmax', default=1e6, type=float,
                    help='larger than featmax to featmax')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def eval_ood_detector(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    batch_size = args.batch_size
    method = args.method
    method_args = args.method_args
    name = args.name
    epochs = args.epochs

    in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'nat'+str(args.model_arch))
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)


    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test_largescale = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if     'Bit' in args.model_arch :
        transform_test_largescale = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if 'efficientnet' == args.model_arch:
        transform_test_largescale=transforms.Compose([
                transforms.Resize(380),
                transforms.CenterCrop(380),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=(args.case =='train'), download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 10

    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=(args.case =='train'), download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 100
    elif in_dataset == "imagenet":
        testloaderIn = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(root='./dataset/ILSVRC-2012', transform=transform_test_largescale),
            batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 1000

    method_args['num_classes'] = num_classes


    info = None
    featmin=args.featmin
    featmax=args.featmax
    if featmin == -1e6 and featmax ==  1e6:
        if  args.case == 'clip2':
            feat = torch.Tensor(np.load(args.in_dataset+args.model_arch+'feat.npy')).cuda()
            if args.q1>0:
                featmin = torch.quantile(feat,q=args.q1,dim=0)
            if args.q2<1:
                featmax = torch.quantile(feat,q=args.q2,dim=0)
    print(args.model_arch)
    print(args.method)
    if args.model_arch == 'densenet':
        info = np.load(f"cache/{args.in_dataset}_{args.model_arch}_feat_stat.npy")
        model = dn.DenseNet3(args.layers, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None, p=args.p, data_type=args.case,info=info,lamb = args.lamb,featmin=featmin,featmax=featmax)
        checkpoint = torch.load(
            "./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=in_dataset, name=name,
                                                                                   epochs=epochs))
        model.load_state_dict(checkpoint['state_dict'])
    elif args.model_arch == 'resnet50':
        info = np.load(f"cache/{args.in_dataset}_{args.model_arch}_feat_stat.npy")
        num_classes = 1000
        from models.resnet import resnet50
        model = resnet50(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)
    elif args.model_arch == 'resnet18':
        num_classes = 1000
        from models.resnet import resnet18
        model = resnet18(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)
    elif args.model_arch == 'resnet34':
        num_classes = 1000
        from models.resnet import resnet34
        model = resnet34(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)
    elif args.model_arch == 'resnet101':
        num_classes = 1000
        from models.resnet import resnet101
        model = resnet101(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)
    elif args.model_arch == 'resnet152':
        num_classes = 1000
        from models.resnet import resnet152
        model = resnet152(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)    
    elif args.model_arch == 'Mos_Bit':
        model = resnetv2.KNOWN_MODELS['BiT-S-R101x1'](head_size=1000,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)
        state_dict = torch.load('./checkpoints/pretrain/BiT-S-R101x1-flat-finetune.pth.tar')
        model.load_state_dict_custom(state_dict['model'])
    elif args.model_arch == 'vgg16':   
        from models.vgg16 import vgg16
        model = vgg16(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)   
    elif args.model_arch == 'vgg16bn':   
        from models.vgg16 import vgg16_bn as vgg16
        model = vgg16(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)               
    elif args.model_arch == 'efficientnet':
        from  models.efficientnet import efficientnet_v2_s as eff
        model = eff(num_classes=num_classes, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)   
    elif args.model_arch == 'mobilenetv3':
        info = None
        from models.mobilenetv3 import mobilenet_v3_large as MobileNetV3
        model =MobileNetV3(pretrained=True,p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)
    elif args.model_arch == 'Regnet':
        info = None
        from models.regnet import regnet_x_8gf as Regnet
        model = Regnet(pretrained=True,p=args.p, info=info, clip_threshold=args.clip_threshold,data_type=args.case,lamb = args.lamb,featmin=featmin,featmax=featmax)               

    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)
    model.eval()
    model.cuda()
    if not mode_args['out_dist_only']:
        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')
    ########################################In-distribution###########################################
        print("Processing in-distribution images")
        label_matrix = []
        feat_matrix = []
        logits_matrix = []
        N = len(testloaderIn.dataset)
        print(N)
        count = 0
        for j, data in tqdm.tqdm(enumerate(testloaderIn)):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]
            if args.case =='train':
                if args.feat == True:
                    feat = model(images,feat=True)
                    feat_matrix+=feat.cpu().detach().numpy().tolist()
            else:
                inputs = images
                scores = get_score(inputs, model, method, method_args)
                for score in scores:
                    f1.write("{}\n".format(score))
    
                outputs = F.softmax(model(inputs)[:, :num_classes], dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)
    
                for k in range(preds.shape[0]):
                    g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))
    
                count += curr_batch_size
                t0 = time.time()
        f1.close()
        g1.close()
    if args.case =='train':
        if args.feat == True:
            print(np.array(feat_matrix).shape)
            np.save(args.in_dataset+args.model_arch+'feat.npy',np.array(feat_matrix))
        print ('finish extraction features, and then return')
        assert 1 == 2
    if mode_args['in_dist_only']:
        return
    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('./dataset/ood_datasets/svhn', split='test', transform=transform_test, download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=2)
        elif out_dataset == 'dtd':
            transform = transform_test_largescale if in_dataset in {'imagenet'} else transform_test
            testsetout = torchvision.datasets.ImageFolder(root="./dataset/ood_datasets/dtd/images", transform=transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="./dataset/ood_datasets/places365/", transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'CIFAR-100':
            testsetout = torchvision.datasets.CIFAR100(root='./dataset/CIFAR100', train=False, download=True, transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'inat':
            testloaderOut = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./dataset/ood_datasets/iNaturalist", transform=transform_test_largescale), batch_size=batch_size, shuffle=False, num_workers=2)
        elif out_dataset == 'places':
            testloaderOut = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./dataset/ood_datasets/Places", transform=transform_test_largescale), batch_size=batch_size, shuffle=False, num_workers=2)
        elif out_dataset == 'sun':
            testloaderOut = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./dataset/ood_datasets/SUN", transform=transform_test_largescale), batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            testsetout = torchvision.datasets.ImageFolder("./dataset/ood_datasets/{}".format(out_dataset), transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=2)

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        #print(N)
        count = 0
        for j, data in tqdm.tqdm(enumerate(testloaderOut)):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            t0 = time.time()
        f2.close()
    return in_save_dir

if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()
    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only
    if args.in_dataset == 'imagenet':
        args.out_datasets = ['dtd', 'sun', 'inat', 'places']
    else:
        args.out_datasets = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
    if args.method == 'odin':
        if args.in_dataset == 'imagenet':
            args.method_args['magnitude']=0
        else:
            args.method_args['magnitude']=0.004
    if args.method == 'energy':
        args.method_args['temperature'] = 1000.0
        in_save_dir=eval_ood_detector(args, mode_args)
    elif args.method =='vnorm':
        args.method_args['a'] = args.a
        args.method_args['m'] = args.m
        in_save_dir=eval_ood_detector(args, mode_args)
    else:
        args.method_args['temperature'] = 1000.0
        in_save_dir = eval_ood_detector(args, mode_args)
    compute_traditional_ood(args.base_dir, args.in_dataset, args.out_datasets, args.method, args.name,in_save_dir)
