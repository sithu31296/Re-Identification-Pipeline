# Re-Identification Pipeline

* [Introduction](##Introduction)
* [Features](##Features)
* [Datasets](##Datasets)
* [Configuration](##Configuration)
* [Training](##Training)
* [Evaluation](##Evaluation)


## Introduction

Person Re-Identification can be seen as an image retrieval problem. Given one query image in one camera, the objective is to find the images of the same person in other cameras. 

This project is currently in beta release.


## Features

Datasets:
* [Market-1501](http://www.liangzheng.org/Project/project_reid.html)
* [DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation) (Coming Soon)
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) (Coming Soon)
* [MSMT17](http://www.pkuvmc.com/publications/msmt17.html) (Coming Soon)
* [VeRi](https://vehiclereid.github.io/VeRi/) (Coming Soon)

Models:
* Baseline
* [PCB](https://arxiv.org/abs/1711.09349) 

Losses:
* [Triplet Loss]()
* [Circle Loss](https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py)
* [CrossEntropyLabelSmooth Loss]()


Augmentations:
* [Random Erasing](https://arxiv.org/pdf/1708.04896)
* [Random GrayScale](https://arxiv.org/abs/2101.08533)
* [AutoAugment](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py) (Coming Soon)
* [AugMix](https://github.com/google-research/augmix) (Coming Soon)

Features coming soon:
* [Native DDP](https://pytorch.org/docs/stable/notes/ddp.html)
* [Native AMP](https://pytorch.org/docs/stable/notes/amp_examples.html)
* [TorchScript Support]()


## Datasets

### Market1501

The Market1501 dataset contains 1501 different individuals (people) collected from 6 cameras in Tsinghua University.

Train Set: 751 different people, 12936 images
Test Set: 750 different people, 3368 query images, 19732 testing images

Dataset Structure:

```
|__ Market-1501-v15.09.15/
    |__ bounding_box_test/      /* 19732 images for testing
    |__ bounding_box_train/     /* 12936 images for training
    |__ gt_bbox/                /* 25259 images (not use)
    |__ gt_query/               /* 3368 queries (not use)
    |__ query/                  /* 3368 query images (id is searched in "bounding_box_test/" folder)
```

Each image is named like `0000_c1s1_000151_01.jpg`. For Market1501, the image name contains the identity label and camera id. Check the naming rule at [here](http://www.liangzheng.org/Project/project_reid.html).

## Configuration

Create a configuration file in `configs/` folder. Sample configuration can be found [here](configs/defaults.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training and evaluation scripts.

## Training

```bash
$ python train.py --cfg configs/CONFIG_FILE_NAME.yaml
```

## Evaluation

```bash
$ python val.py --cfg configs/CONFIG_FILE_NAME.yaml
```