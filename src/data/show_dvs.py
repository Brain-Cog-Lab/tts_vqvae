import os
import sys

sys.path.append(os.getcwd())

import tonic
from tonic import DiskCachedDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from einops import repeat
from spikingjelly.datasets import play_frame
from src.utils.dvs import save_frame_dvs
from braincog.datasets.datasets import get_dvsc10_data, DATA_DIR

DATA_DIR = "../dataset"


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


def load_nmnist(batch_size, step, size):

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

    train_dataset = tonic.datasets.NMNIST(os.path.join(DATA_DIR, 'DVS/NMNIST'),
                                          transform=train_transform,
                                          train=True)
    test_dataset = tonic.datasets.NMNIST(os.path.join(DATA_DIR, 'DVS/NMNIST'),
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

    train_dataset = DiskCachedDataset(
        train_dataset,
        cache_path=os.path.join(
            DATA_DIR, 'DVS/NMNIST/train_cache_{}_denoise'.format(step)),
        transform=train_transform,
        num_copies=3)
    test_dataset = DiskCachedDataset(
        test_dataset,
        cache_path=os.path.join(
            DATA_DIR, 'DVS/NMNIST/test_cache_{}_denoise'.format(step)),
        transform=test_transform,
        num_copies=3)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=2,
        shuffle=False,
    )

    return train_loader, test_loader


def load_dvsc10_data(batch_size, step, root=DATA_DIR, size=48):
    """
    获取DVS CIFAR10数据
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
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

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               drop_last=True,
                                               num_workers=8,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              drop_last=False,
                                              num_workers=2,
                                              shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # train_loader, test_loader = load_nmnist(batch_size=1, step=4, size=64)
    train_loader, test_loader = load_dvsc10_data(batch_size=16,
                                                 step=4,
                                                 root='../dataset',
                                                 size=64)
    X, _ = next(iter(train_loader))  # (b,t,2,h,w)
    print(torch.min(X), torch.max(X))
    X = X[0]
    X = (X + 1.0) * 0.5
    save_frame_dvs(X, save_gif_to='show_dvs.gif')
