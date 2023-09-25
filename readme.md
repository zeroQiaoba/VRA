## VRA: Variational Rectifed Activation for Out-of-distribution Detection  


Correspondence to: 

  - Mingyu Xu  (xumingyu2021@ia.ac.cn)
  - Zheng Lian (lianzheng2016@ia.ac.cn)

## Paper
[**VRA: Variational Rectifed Activation for Out-of-distribution Detection**](https://arxiv.org/pdf/2302.11716.pdf)<br>
Mingyu Xu, Zheng Lian, Bin Liu, Jianhua Tao<br>

Please cite our paper if you find our work useful for your research:

```tex
@article{xu2023vra,
  title={VRA: Variational Rectifed Activation for Out-of-distribution Detection},
  author={Xu, Mingyu and Lian, Zheng and Liu, Bin and Tao, Jianhua},
  journal={NeurIPS},
  year={2023}
}
```



## Usage



### Run VRA on CIFAR

```
1.Download datasets
CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -> ./dataset/CIFAR10
CIFAR100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz  -> ./dataset/CIFAR100
SVHN: http://ufldl.stanford.edu/housenumbers/test_32x32.mat -> ./dataset/ood_datasets/svhn, Then run `python select_svhn_data.py` to generate test subset.
LSUN-C: https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz -> ./dataset/ood_datasets/LSUN
LSUN-R: https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz ./dataset/ood_datasets/LSUN_resize
iSUN: https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz -> ./dataset/ood_datasets/iSUN
Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -> ./dataset/ood_datasets/dtd
Places365: http://data.csail.mit.edu/places/places365/test_256.tar -> ./dataset/ood_datasets/places365

2.Extract training data features to facilitate subsequent calculation of quantile:
python ood_eval.py --in-dataset CIFAR-10  --case train --feat True
python ood_eval.py --in-dataset CIFAR-100 --case train --feat True

3.Apple VRA, VRA+, VRA++ to CIFAR-10:
VRA:   python ood_eval.py --in-dataset CIFAR-10 --case clip2 --q1 0.6 --q2 0.9
VRA+:  python ood_eval.py --in-dataset CIFAR-10 --case clip2 --q1 0.6 --q2 0.95 --lamb 0.6
VRA++: python ood_eval.py --in-dataset CIFAR-10 --case clip2 --method vnorm --a 0.01 --m 4 --q1 0.5 --q2 0.85

4.Apple VRA, VRA+, VRA++ to CIFAR-100:
VRA:   python ood_eval.py --in-dataset CIFAR-100 --case clip2 --q1 0.6 --q2 0.95 --method odin
VRA+:  python ood_eval.py --in-dataset CIFAR-100 --case clip2 --q1 0.6 --q2 0.95 --lamb 0.6 --method odin
VRA++: python ood_eval.py --in-dataset CIFAR-100 --case clip2 --method vnorm --a 0.01 --m 4 --q1 0.5 --q2 0.85
```



### Run VRA on ImageNet

```
1.Download datasets
ImageNet: http://www.image-net.org/challenges/LSVRC/2012/index (after login, using https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) -> ./dataset/ILSVRC-2012/val
iNaturalist: http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz -> ./dataset/ood_datasets/iNaturalist
SUN: http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz -> ./dataset/ood_datasets/SUN
Places: http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz -> ./dataset/ood_datasets/Places
Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -> ./dataset/ood_datasets/dtd

2.Apply VRA, VRA+, VRA++ to ImageNet (resnet-50)
VRA:   python ood_eval.py --in-dataset imagenet --model-arch resnet50 --case clip2 --featmin 0.5 --featmax 1 
VRA+:  python ood_eval.py --in-dataset imagenet --model-arch resnet50 --case clip2 --featmin 0.6 --featmax 0.7 --lamb 0.3
VRA++: python ood_eval.py --in-dataset imagenet --model-arch resnet50 --case clip2 --method vnorm --a 0.001 --m 1.5 --featmin 0.3 --featmax 0.8

3.Apply VRA, VRA+, VRA++ to ImageNet (resnetv2-101)
download pretrained model from http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-flat-finetune.pth.tar -> ./checkpoints/pretrain/BiT-S-R101x1-flat-finetune.pth.tar
VRA:   python ood_eval.py --in-dataset imagenet --model-arch Mos_Bit --case clip2 --featmin 1   --featmax 2
VRA+:  python ood_eval.py --in-dataset imagenet --model-arch Mos_Bit --case clip2 --featmin 1   --featmax 2 --lamb 0.6
VRA++: python ood_eval.py --in-dataset imagenet --model-arch Mos_Bit --case clip2 --featmin 1   --method vnorm  --a 0.005 --m 1.5

4.Apply to other backbones
python ood_eval.py --in-dataset imagenet --model-arch resnet18 --case clip2 --featmin 0.5 --featmax 1 
python ood_eval.py --in-dataset imagenet --model-arch resnet34 --case clip2 --featmin 0.5 --featmax 1 
python ood_eval.py --in-dataset imagenet --model-arch resnet101 --case clip2 --featmin 0.5 --featmax 1 
python ood_eval.py --in-dataset imagenet --model-arch resnet152 --case clip2 --featmin 0.5 --featmax 1 
python ood_eval.py --in-dataset iamgenet --model-arch Regnet --case clip2 --featmin 0.5 --featmax 1 
python ood_eval.py --in-dataset iamgenet --model-arch Mos_Bit --case clip2 --featmin 1 --featmax 2 
python ood_eval.py --in-dataset iamgenet --model-arch efficientnet --case clip2 --featmin -0.2 --featmax 0.5 
python ood_eval.py --in-dataset iamgenet --model-arch mobilenetv3 --case clip2 --featmin 0 --featmax 4 
python ood_eval.py --in-dataset iamgenet --model-arch vgg16 --case clip2 --featmin 4 
python ood_eval.py --in-dataset iamgenet --model-arch vgg16bn --case clip2 --featmin 1
```
