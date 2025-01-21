import math
import os
import cv2
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
import torch.utils.data as data
from pathlib import Path


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["", "data/data_splits/split_zhou_EuroSAT.json"]
}

def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, mode, n_shot, transform)
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

class CustomCifarDataset(data.Dataset):
    def __init__(self, samples, transform=None):
        super(CustomCifarDataset, self).__init__()

        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        img, label, domain = self.samples[index]
        if self.transform is not None:
            img2 = Image.fromarray(np.uint8(img * 255.)).convert('RGB')
            img2 = self.transform(img2)
        else:
            img = torch.tensor(img.transpose((2, 0, 1)))

        return img2, torch.tensor(label)#, domain

    def __len__(self):
        return len(self.samples)

def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    transform=None,):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'CIFAR-10-C':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'CIFAR-100-C':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)

def load_cifar10c(
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './dataset',
        shuffle: bool = False,
        corruptions: Sequence[str] = None,
        prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(n_examples,
                                  severity, data_dir, corruptions, shuffle)

def load_cifar100c(
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './dataset',
        shuffle: bool = False,
        corruptions: Sequence[str] = None,
        prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(n_examples,
                                  severity, data_dir, corruptions, shuffle)

def load_corruptions_cifar(
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './data',
        corruptions: Sequence[str] = None,
        shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir

    # Download labels
    labels_path = data_root_dir / 'labels.npy'
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test
