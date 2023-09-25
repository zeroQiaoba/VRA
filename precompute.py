from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import models.densenet as dn

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', '-d', default='imagenet', type=str, help='dataset')

args = parser.parse_args()
if args.dataset == 'CIFAR-100':
    num_classes = 100
    model = dn.DenseNet3(100, num_classes, normalizer=None, p=None)
    checkpoint = torch.load("./checkpoints/CIFAR-100/densenet/checkpoint_100.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    featdim = 342
elif args.dataset == 'CIFAR-10':
    num_classes = 10
    model = dn.DenseNet3(100, num_classes, normalizer=None, p=None)
    checkpoint = torch.load("./checkpoints/CIFAR-10/densenet/checkpoint_100.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    featdim = 342
elif args.dataset == 'imagenet':
    num_classes = 1000
    from models.resnet import resnet50
    model = resnet50(num_classes=num_classes, pretrained=True)
    featdim = 2048

net = model

# checkpoint = torch.load("checkpoints/CIFAR-10/resnet18_t5_SOFL/checkpoint_200.pth.tar")
# net.load_state_dict(checkpoint['state_dict'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 64
test_batch_size = 64

net = net.to(device)
if args.dataset in {'CIFAR-10', 'CIFAR-100'}:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = {
        'CIFAR-10': torchvision.datasets.CIFAR10,
        'CIFAR-100': torchvision.datasets.CIFAR100,
    }
    trainset = dataset[args.dataset](root='./data', train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    id_train_size = 50000

    cache_name = f"cache/{args.dataset}_train_densenet_in.npy"
    if not os.path.exists(cache_name):
        feat_log = np.zeros((id_train_size, featdim))
        score_log = np.zeros((id_train_size, num_classes))
        label_log = np.zeros(id_train_size)

        net.eval()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(trainset))

            outputs = net.features(inputs)
            out = F.adaptive_avg_pool2d(outputs, 1)
            out = out.view(out.size(0), -1)
            score = net.fc(out)
            # score = net(inputs)
            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 10 == 0:
                print(batch_idx)
        np.save(cache_name, (feat_log.T, score_log.T, label_log))
    else:
        feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log, score_log = feat_log.T, score_log.T

    np.save(f"cache/{args.dataset}_densenet_feat_stat.npy", feat_log.mean(0))
    print("done")
else:
    transform_test_largescale = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join('datasets/ILSVRC-2012', 'train'), transform_test_largescale),
        batch_size=test_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join('datasets/ILSVRC-2012', 'val'), transform_test_largescale),
        batch_size=test_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    id_train_size = 1281167

    feat_log = np.zeros((id_train_size, featdim))
    score_log = np.zeros((id_train_size, num_classes))
    label_log = np.zeros(id_train_size)

    net.eval()
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        start_ind = batch_idx * batch_size
        end_ind = min((batch_idx + 1) * batch_size, len(trainloader.dataset))

        outputs = net.features(inputs)
        out = F.adaptive_avg_pool2d(outputs, 1)
        out = out.view(out.size(0), -1)
        score = net.fc(out)
        # score = net(inputs)
        feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
        label_log[start_ind:end_ind] = targets.data.cpu().numpy()
        score_log[start_ind:end_ind] = score.data.cpu().numpy()
        if batch_idx % 10 == 0:
            print(f"{batch_idx}/{len(trainloader)}")


    np.save(f"cache/{args.dataset}_resnet50_feat_stat.npy", feat_log.mean(0))
    print("done")
