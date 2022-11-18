# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
create train or eval dataset.
"""
import numpy as np
import multiprocessing
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.py_transforms as P2
from mindspore.communication.management import init, get_rank, get_group_size
from PIL import ImageFilter
import random


def create_dataset_moco(dataset_path, aug_plus, repeat_num=1, batch_size=32, target="Ascend", workers=32, distribute=False):
    """
    create a train imagenet2012 dataset for Unsupervised Training

    Args:
        dataset_path(string): the path of dataset.
        aug_plus(bool): whether dataset is used for MoCo v2's aug or MoCo v1's aug.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    #if distribute: batch_size=32； 1p: batch_size=256
    #distribute待验证sampler=train_sampler: train_dataset, train_datasetv2, train_datasetv1
    
    device_num, rank_id = _get_rank_info(distribute)


    if distribute:
        train_sampler = ds.DistributedSampler(10, shard_id=rank_id) #？(10, shard_id=rank_id)不确定
    else:
        train_sampler = None

    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    # define map operations
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        trans = [
            #C.RandomCropDecodeResize(image_size, scale=(0.2, 1.)),
            #C2.RandomApply([
            #    C.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)  # not strengthened #?RandomColorAdjust(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0))
            #], prob=0.8),
            #P.RandomGrayscale(prob=0.2),
            #C2.RandomApply([GaussianBlur([.1, 2.])], prob=0.5),
            #C.RandomHorizontalFlip(),
            #C.Normalize(mean=mean, std=std),
            #C.HWC2CHW()
            P.Decode(),
            P.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            P2.RandomApply([
                P.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)  # not strengthened #?RandomColorAdjust(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0))
            ], prob=0.8),
            P.RandomGrayscale(prob=0.2),
            P2.RandomApply([GaussianBlur([.1, 2.])], prob=0.5),
            P.RandomHorizontalFlip(),
            P.ToTensor(),
            P.Normalize(mean=mean, std=std)
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        trans = [
            #C.RandomCropDecodeResize(image_size, scale=(0.2, 1.)),
            #P.RandomGrayscale(prob=0.2),
            #C.RandomColorAdjust(0.4, 0.4, 0.4, 0.4), #?RandomColorAdjust(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0))
            #C.RandomHorizontalFlip(),
            #C.Normalize(mean=mean, std=std),
            #C.HWC2CHW()
            P.Decode(),
            P.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            P.RandomGrayscale(prob=0.2),
            P.RandomColorAdjust(0.4, 0.4, 0.4, 0.4), #?RandomColorAdjust(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0))
            P.RandomHorizontalFlip(),
            P.ToTensor(),
            P.Normalize(mean=mean, std=std)
        ]

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=None, sampler=train_sampler)
    type_cast_op = C2.TypeCast(ms.int32)

    if device_num == 1:
        trans_work_num = 24
    else:
        trans_work_num = 12


    data_set = data_set.map(operations=TwoCropsTransform(P2.Compose(trans)), input_columns="image", output_columns=["im_q", "im_k"],
                            column_order=["im_q", "im_k", "label"], num_parallel_workers=get_num_parallel_workers(trans_work_num))
    data_set = data_set.map(operations=type_cast_op, input_columns="label", 
                            num_parallel_workers=get_num_parallel_workers(12))




    # apply batch operations
    data_set = data_set.batch(1, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset_lincls(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", workers=32, distribute=False):
    """
    create a train or eval imagenet2012 dataset for Linear Classification

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    device_num, rank_id = _get_rank_info(distribute)


    if distribute:
        train_sampler = ds.DistributedSampler(10, shard_id=rank_id) #？(10, shard_id=rank_id)不确定,待8卡检测
    else:
        train_sampler = None

    image_size = 224

    #mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    #std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # define map operations
    if do_train:
        trans = [
            #C.RandomCropDecodeResize(image_size),
            #C.RandomHorizontalFlip(),
            #C.Normalize(mean=mean, std=std),
            #C.HWC2CHW()
            P.Decode(),
            P.RandomResizedCrop(image_size),
            P.RandomHorizontalFlip(),
            P.ToTensor(),
            P.Normalize(mean=mean, std=std)
        ]
        if device_num == 1:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=None, sampler=train_sampler)
    else:
        trans = [
            #C.Decode(),
            #C.Resize(256),
            #C.CenterCrop(image_size),
            #C.Normalize(mean=mean, std=std),
            #C.HWC2CHW()
            P.Decode(),
            P.Resize(256),
            P.CenterCrop(image_size),
            P.ToTensor(),
            P.Normalize(mean=mean, std=std)
        ]
        if device_num == 1:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=False)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=workers, shuffle=False)

    type_cast_op = C2.TypeCast(ms.int32)
    if device_num == 1:
        trans_work_num = 24
    else:
        trans_work_num = 12


    data_set = data_set.map(operations=P2.Compose(trans), input_columns="image",
                            num_parallel_workers=get_num_parallel_workers(trans_work_num))
    #data_set = data_set.map(operations=C2.Compose(trans), input_columns="image",
    #                        num_parallel_workers=get_num_parallel_workers(trans_work_num))
    data_set = data_set.map(operations=type_cast_op, input_columns="label",
                            num_parallel_workers=get_num_parallel_workers(12))

    # apply batch operations
    data_set = data_set.batch(batch_size)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info(distribute):
    """
    get rank size and rank id
    """
    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        q = np.array(q)
        k = np.array(k)
        q = np.squeeze(q)
        k = np.squeeze(k)
        return q, k


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
