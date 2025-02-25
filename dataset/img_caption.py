# Code adapted from https://github.com/facebookresearch/SLIP
from collections import defaultdict
import json
import os
import numpy as np
from PIL import Image, ImageFile
from typing import Optional
import torch
import re
from torchvision import transforms
from torchvision import datasets as t_datasets
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_lightning import LightningDataModule
from dataset.tokenizer import SimpleTokenizer
from utils import GaussianBlur


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCaptionDataModule(LightningDataModule):
    """Data module for vision-language dataset of image-captions including:
        - RedCaps (12M images)
        - COCO (300K images)
        - CC3M
        - CC12M
    """
    def __init__(self, dataset: str,
                 model: str,
                 tokenizer = None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 augment: Optional[str] = None):
        """
        :param dataset: {'redcaps', 'coco', 'cc12m', 'cc3m'}
            Large-scale image-captions dataset (usually for pre-training).
        :param model: {'SimCLR', 'CLIP', 'SLIP', 'VirTex', 'CoMM'}
            The model defines the augmentations to apply to the data.
        :param tokenizer:
        :param batch_size: Batch size to pass to Dataloaders
        :param num_workers: Number of workers to pass to Dataloaders
        :param augment: Specify the augmentations strength to apply,
        e.g. "crop-0.1" from cropping from 0.1 to 1 or "crop-to-0.8" for cropping from 0 to 0.8
        """
        super().__init__()

        self.dataset = dataset
        self.model = model
        catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "catalog.json")
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])
        scale = (0.08, 1.0) # default value
        if augment is not None:
            if re.match("(crop-(\d*\.?\d+)|crop-to-(\d*\.?\d+))", augment):
                if re.match("crop-to-(\d*\.?\d+)", augment):
                    scale = (0.01, float(re.match("crop-to-(\d*\.?\d+)", augment)[1]))
                else:
                    scale = (float(re.match("crop-(\d*\.?\d+)", augment)[1]), 1.0)
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # Todo
        self.text_augment = lambda x: x

    def setup(self, stage: str):
        self.val_dataset = None
        root, metadata = self.catalog[self.dataset]["path"], self.catalog[self.dataset]["metadata"]
        
        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer()

        if self.model.startswith('SimCLR'):
            self.train_dataset = ImageCaptionDatasetSSL(self.dataset, root, metadata, self.augment, "train")
            self.val_dataset = ImageCaptionDatasetSSL(self.dataset, root, metadata, self.augment, "val")
        elif self.model.startswith('CLIP') or self.model == "VirTex":
            self.train_dataset = ImageCaptionDatasetCLIP(self.dataset, root, metadata, "train",
                                                         self.img_transform, self.tokenizer)
            self.val_dataset = ImageCaptionDatasetCLIP(self.dataset, root, metadata,"val",
                                                         self.img_transform, self.tokenizer)
        elif self.model.startswith('SLIP'):
            self.train_dataset = ImageCaptionDatasetSLIP(self.dataset, root, metadata,
                                                         self.img_transform, self.augment, "train",
                                                         self.tokenizer)
            self.val_dataset = ImageCaptionDatasetSLIP(self.dataset, root, metadata,
                                                       self.img_transform, self.augment, "val",
                                                       self.tokenizer)
        elif self.model == "CoMM":
            self.train_dataset = ImageCaptionDatasetMMSSL(self.dataset, root, metadata,
                                                          self.img_transform, self.augment,
                                                          self.text_augment,"train",self.tokenizer)
            self.val_dataset = ImageCaptionDatasetMMSSL(self.dataset, root, metadata,
                                                          self.img_transform, self.augment,
                                                          self.text_augment, "val",self.tokenizer)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, drop_last=False)
        return None

    def test_dataloader(self):
        return self.val_dataloader()


class DownstreamVisionDataModule(LightningDataModule):

    def __init__(self, dataset: str,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.dataset = dataset
        catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "catalog.json")
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = self.train_transform

        entry = self.catalog[self.dataset]
        root = entry['path']
        datasets = {0: None, 1: None}

        for is_train in [True, False]:
            transform = self.train_transform if is_train else self.test_transform
            if entry['type'] == 'imagefolder':
                datasets[is_train] = t_datasets.ImageFolder(
                    os.path.join(root, entry['train'] if is_train else entry['test']),
                    transform=transform)
            elif entry['type'] == 'special':
                if self.dataset == 'cifar10':
                    datasets[is_train] = t_datasets.CIFAR10(root, train=is_train,
                                                            transform=transform, download=True)
                elif self.dataset == 'cifar100':
                    datasets[is_train] = t_datasets.CIFAR100(root, train=is_train,
                                                             transform=transform, download=True)
                elif self.dataset == 'stl10':
                    datasets[is_train] = t_datasets.STL10(root, split='train' if is_train else 'test',
                                                          transform=transform, download=True)
                elif self.dataset == 'mnist':
                    datasets[is_train] = t_datasets.MNIST(root, train=is_train,
                                                          transform=transform, download=True)
                elif self.dataset == "cars":
                    datasets[is_train] = t_datasets.StanfordCars(root, split="train" if is_train else "test",
                                                                 transform=transform, download=True)
                elif self.dataset == "food101":
                    datasets[is_train] = t_datasets.Food101(root, split="train" if is_train else "test",
                                                            transform=transform, download=True)
                elif self.dataset == "caltech101":
                    dataset = t_datasets.Caltech101(root, transform=transform, download=True)
                    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=48).
                                               split(np.ones((len(dataset),)), dataset.y))
                    datasets[is_train] = torch.utils.data.Subset(dataset, train_idx if is_train else test_idx)
                elif self.dataset == "sun397":
                    dataset = t_datasets.SUN397(root, transform=transform, download=True)
                    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=48).
                                               split(np.ones((len(dataset),)), dataset._labels))
                    datasets[is_train] = torch.utils.data.Subset(dataset, train_idx if is_train else test_idx)
                elif self.dataset == "aircraft":
                    datasets[is_train] = t_datasets.FGVCAircraft(root, split="trainval" if is_train else "test",
                                                                 transform=transform, download=True)
                elif self.dataset == "dtd":
                    datasets[is_train] = t_datasets.DTD(root, split="train" if is_train else "test",
                                                        transform=transform, download=True)
                elif self.dataset == "pets":
                    datasets[is_train] = t_datasets.OxfordIIITPet(root, split="trainval" if is_train else "test",
                                                                  transform=transform, download=True)
                elif self.dataset == "flowers":
                    datasets[is_train] = t_datasets.Flowers102(root, split="train" if is_train else "test",
                                                               transform=transform, download=True)
                elif self.dataset == "eurosat":
                    dataset = t_datasets.EuroSAT(root, transform=transform, download=True)
                    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=48).
                                               split(np.ones((len(dataset),)), dataset.targets))
                    datasets[is_train] = torch.utils.data.Subset(dataset, train_idx if is_train else test_idx)
                elif self.dataset == "patch_camelyon":
                    datasets[is_train] = t_datasets.PCAM(root, split="train" if is_train else "test",
                                                         transform=transform, download=True)
            else:
                raise Exception('Unknown dataset type: %s' % entry["type"])
        self.train_dataset = datasets[1]
        self.test_dataset = datasets[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, split: str = "train"):
        self.dataset = dataset
        self.root = root
        self.split = split
        if self.dataset == 'coco':
            samples = defaultdict(list)
            metadata = os.path.join(metadata, f"captions_{self.split}2017.json")
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            if self.split in ["val", "test"]:
                raise NotImplementedError()
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == 'redcaps':
            self.samples = []
            for annot in sorted(os.listdir(metadata)):
                with open(os.path.join(metadata, annot)) as f:
                    annotations = json.load(f)["annotations"]
                annotations = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations
                               if os.path.isfile(os.path.join(self.root, ann['subreddit'], f"{ann['image_id']}.jpg"))]
                self.samples.extend(annotations)

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_raw_item(self, i):
        if self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, f'{self.split}2017', '{:012d}.jpg'.format(index))
            img = self.pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc3m':
            ann = self.samples[i]
            filename, captions = ann['image_id'], ann['captions']
            path = os.path.join(self.root, str(filename))
            img = self.pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc12m':
            ann = self.samples[i]
            filename, captions = ann['image_name'], ann['captions']
            path = os.path.join(self.root, filename)
            img = self.pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = self.pil_loader(path)

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, split: str = "train", transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata, split=split)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return img, caption


class ImageCaptionDatasetSLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, augment, split: str = "train", tokenizer=None):
        super().__init__(dataset, root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2


class ImageCaptionDatasetSSL(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, augment, split: str = "train"):
        super().__init__(dataset, root, metadata, split=split)

        self.augment = augment

    def __getitem__(self, i):
        img, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class ImageCaptionDatasetMMSSL(ImageCaptionDatasetBase):
    """Apply augmentations jointly to both image and text modalities."""
    def __init__(self, dataset, root, metadata, transform,
                 augment, text_augment, split: str = "train", tokenizer=None):
        super().__init__(dataset, root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.text_augment = text_augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)
        cap1 = self.text_augment(caption)
        cap2 = self.text_augment(caption)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return [aug1, cap1], [aug2, cap2]



if __name__ == "__main__":
    ds = ImageCaptionDatasetCLIP("redcaps", "/data/redcaps/images", "/data/redcaps/annotations")
    print(ds[0])