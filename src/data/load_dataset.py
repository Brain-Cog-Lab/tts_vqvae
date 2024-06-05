import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .custom_dataset import CustomImageDatasetUcond, CustomImageDatasetKey, AddKey
from torchvision import datasets
import tonic
from tonic import DiskCachedDataset
import torch
import os
from einops import repeat
import torch.nn.functional as F


def dvs_channel_check_expend(x):
    """
    检查是否存在DVS数据缺失, N-Car中有的数据会缺少一个通道
    :param x: 输入的tensor
    :return: 补全之后的数据
    """
    if x.shape[1] == 1:
        return repeat(x, 'b c w h -> b (r c) w h', r=2)
    else:
        return x


def padding_dvs(x):
    # x.shape = (t,2,h,w)
    pad = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.concat([x, pad], dim=1)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                     resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(
        round(x * scale) for x in pil_image.size),
                                 resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size,
                               crop_x:crop_x + image_size])


def load_from_imgfolder(data_path, image_size):
    """
    load image from a folder with subfolders that corresponds to different classes
    """
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True)
    ])
    dataset = ImageFolder(data_path, transform=transform)
    return dataset


def load_from_imgfolder_ucond(data_path, image_size, recursive=False):
    """
    data_path has no subfolders, unconditional
    """
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True)
    ])

    datasets = CustomImageDatasetUcond(root_dir=data_path,
                                       transform=transform,
                                       recursive=recursive)
    return datasets


def load_from_imgfolder_ucond_unnorm(data_path, image_size, recursive=False):
    """
    data_path has no subfolders, unconditional
    return in (0,1), unnorm
    """
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor()
    ])

    datasets = CustomImageDatasetUcond(root_dir=data_path,
                                       transform=transform,
                                       recursive=recursive)
    return datasets


def load_from_imgfolder_ucond_key(data_path, image_size, recursive=False):
    """
    data_path has no subfolders, unconditional
    """
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True)
    ])

    datasets = CustomImageDatasetKey(root_dir=data_path,
                                     transform=transform,
                                     recursive=recursive)
    return datasets


def load_lsun_church_key(data_root, image_size=256):
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True)
    ])
    dataset = datasets.LSUN(root=data_root,
                            classes=['church_outdoor_train'],
                            transform=transform)
    dataset = AddKey(origin_dataset=dataset)

    return dataset


def load_lsun_church_unnorm(data_root, image_size=256):
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.LSUN(root=data_root,
                            classes=['church_outdoor_train'],
                            transform=transform)
    return dataset


def load_celeba(data_root, image_size=64):
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True)
    ])

    train_dataset = datasets.CelebA(root=data_root,
                                    split='train',
                                    transform=transform,
                                    download=True)
    '''eval_dataset = datasets.CelebA(root=data_root,
                                   split='valid',
                                   transform=transform,
                                   download=True)'''
    return train_dataset


def load_celeba_unnorm(data_root, image_size=64):
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CelebA(root=data_root,
                                    split='train',
                                    transform=transform,
                                    download=True)
    '''eval_dataset = datasets.CelebA(root=data_root,
                                   split='valid',
                                   transform=transform,
                                   download=True)'''
    return train_dataset


def load_cifar10(data_root):
    """
    using ToTensor, [0,1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))  # Normalize images
    ])
    train_dataset = datasets.CIFAR10(root=data_root,
                                     train=True,
                                     transform=transform,
                                     download=True)
    '''test_dataset = datasets.CIFAR10(root=data_root,
                                    train=False,
                                    transform=transform,
                                    download=True)'''
    return train_dataset


def load_cifar10_unnorm(data_root):
    """
    using ToTensor, [0,1]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])
    train_dataset = datasets.CIFAR10(root=data_root,
                                     train=True,
                                     transform=transform,
                                     download=True)
    '''test_dataset = datasets.CIFAR10(root=data_root,
                                    train=False,
                                    transform=transform,
                                    download=True)'''
    return train_dataset


def load_nmnist(step, size, data_root='./dataset', pad=True):

    sensor_size = tonic.datasets.NMNIST.sensor_size
    filter_time = 10000

    train_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=filter_time),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=filter_time),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])

    train_dataset = tonic.datasets.NMNIST(os.path.join(data_root,
                                                       'DVS/NMNIST'),
                                          transform=train_transform,
                                          train=True)
    test_dataset = tonic.datasets.NMNIST(os.path.join(data_root, 'DVS/NMNIST'),
                                         transform=test_transform,
                                         train=False)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(
            x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
        lambda x: 2 * x / (torch.max(x) - torch.min(x)) - 1,  # to (-1,1)
        # transforms.RandomCrop(size, padding=size // 12),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15)
    ])

    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(
            x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
        lambda x: 2 * x / (torch.max(x) - torch.min(x)) - 1,  # to (-1,1)
    ])

    if pad:
        train_transform.transforms.append(transforms.Lambda(padding_dvs))
        test_transform.transforms.append(transforms.Lambda(padding_dvs))

    train_dataset = DiskCachedDataset(
        train_dataset,
        cache_path=os.path.join(
            data_root, 'DVS/NMNIST/train_cache_{}_denoise'.format(step)),
        transform=train_transform,
        num_copies=3)
    test_dataset = DiskCachedDataset(
        test_dataset,
        cache_path=os.path.join(
            data_root, 'DVS/NMNIST/test_cache_{}_denoise'.format(step)),
        transform=test_transform,
        num_copies=3)

    return train_dataset, test_dataset


def load_dvsc10_data(step, root, size):
    """
    load dvs cifar-10, return torch dataset
    """
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(
        root, 'DVS/DVS_Cifar10'),
                                              transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(
        root, 'DVS/DVS_Cifar10'),
                                             transform=test_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(
            x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: 2 * x / (torch.max(x) - torch.min(x)) - 1,  # to (-1,1)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(
            x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: 2 * x / (torch.max(x) - torch.min(x)) - 1,  # to (-1,1)
    ])

    train_dataset = DiskCachedDataset(
        train_dataset,
        cache_path=os.path.join(root,
                                'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
        transform=train_transform)
    test_dataset = DiskCachedDataset(
        test_dataset,
        cache_path=os.path.join(root,
                                'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
        transform=test_transform)

    return train_dataset, test_dataset
