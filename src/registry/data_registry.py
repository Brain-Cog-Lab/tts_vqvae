from src.data.load_dataset import load_from_imgfolder_ucond, load_from_imgfolder_ucond_key, load_lsun_church_key
from src.data.load_dataset import load_from_imgfolder_ucond_unnorm, load_cifar10_unnorm
from src.data.load_dataset import load_celeba, load_cifar10, load_celeba_unnorm, load_lsun_church_unnorm
from src.data.load_dataset import load_nmnist, load_dvsc10_data
from torch.utils.data.dataset import random_split
from torch.utils.data import ConcatDataset
from src.data.custom_dataset import AddKey
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import torch
import os


def celeba_64_key(data_root, split='full'):
    dataset = load_celeba(data_root=data_root, image_size=64)
    dataset = AddKey(origin_dataset=dataset)

    if split == 'full':
        return dataset

    train_size = int(0.85 * len(dataset))
    valsize = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, valsize])
    if split == 'train':
        return trainset
    elif split == 'val':
        return valset


def lsun_bedroom_256_20(data_root, split='full'):
    """
    load 20% of lsun bedroom
    """
    dataset = load_from_imgfolder_ucond(data_path=data_root,
                                        image_size=256,
                                        recursive=True)
    if split == 'full':
        return dataset

    train_size = int(0.85 * len(dataset))
    valsize = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, valsize])
    if split == 'train':
        return trainset
    elif split == 'val':
        return valset


def lsun_bedroom_256_20_key(data_root, split='full'):
    """
    load 20% of lsun bedroom
    """
    dataset = load_from_imgfolder_ucond_key(data_path=data_root,
                                            image_size=256,
                                            recursive=True)
    if split == 'full':
        return dataset

    train_size = int(0.85 * len(dataset))
    valsize = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, valsize])
    if split == 'train':
        return trainset
    elif split == 'val':
        return valset


def nmnist_64_key(data_root, step=4, split='full', pad=True):
    train_dataset, test_dataset = load_nmnist(step=step,
                                              size=64,
                                              data_root=data_root,
                                              pad=pad)
    if split == 'full':
        return AddKey(ConcatDataset([train_dataset, test_dataset]))
    elif split == 'train':
        return AddKey(train_dataset)
    elif split == 'test':
        return AddKey(test_dataset)
