import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDatasetUcond(Dataset):

    def __init__(self, root_dir, transform=None, recursive=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.

        unconditional: all labels are 0 !
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        if not recursive:
            self.image_files = [
                f for f in os.listdir(root_dir)
                if os.path.isfile(os.path.join(root_dir, f))
            ]
            self.image_files = [
                os.path.join(self.root_dir, f) for f in self.image_files
            ]
        else:
            self.image_files = []
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, 0


class CustomImageDatasetKey(CustomImageDatasetUcond):

    def __init__(self, root_dir, transform=None, recursive=None):
        """
        return dict, with key "image"
        """
        super().__init__(root_dir=root_dir,
                         transform=transform,
                         recursive=recursive)

    def __getitem__(self, idx):
        example = dict()
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        example['image'] = image

        return example


class AddKey(Dataset):

    def __init__(self, origin_dataset) -> None:
        super().__init__()
        self.origin_dataset = origin_dataset

    def __len__(self):
        return len(self.origin_dataset)

    def __getitem__(self, index):
        ex = dict()
        ex['image'] = self.origin_dataset[index][0]
        return ex


class CustomImageDatasetUcond2(Dataset):
    """
    not return list, return tensor without label
    """

    def __init__(self, root_dir, transform=None, recursive=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.

        unconditional: all labels are 0 !
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        if not recursive:
            self.image_files = [
                f for f in os.listdir(root_dir)
                if os.path.isfile(os.path.join(root_dir, f))
            ]
            self.image_files = [
                os.path.join(self.root_dir, f) for f in self.image_files
            ]
        else:
            self.image_files = []
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
