import json
import os
import numpy as np
from PIL import Image
import torch
from typing import Optional
import warnings
import re
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
# Local import
from dataset.img_caption import ImageCaptionDataModule
from utils import GaussianBlur


# Disable decompression bombs warning for large images
Image.MAX_IMAGE_PIXELS = None
# Silence repeated user warnings from scikit-learn multilabel binarizer for unknown classes.
warnings.filterwarnings("ignore", category=UserWarning)


class MMIMDBDataModule(ImageCaptionDataModule):
    """
    Data module for MM-IMBD vision-language dataset [1] including
    movies description (text) + poster (image).
    The downstream task is to predict the movie genre.

    [1] Gated Multimodal Units for Information Fusion, John Arevalo et al., ICLR-Workshop 2017
    """

    def __init__(self, model: str,
                 tokenizer=None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 img_augment: Optional[str] = None
                 ):

        """
        :param model: {'Sup', 'SimCLR', 'CLIP', 'SLIP', 'BLIP2, 'CoMM'}
            The model defines the augmentations to apply to the data.
        :param tokenizer: Which tokenizer use for encoding text with integers
        :param batch_size: Batch size to pass to Dataloaders
        :param num_workers: Number of workers to pass to Dataloaders
        :param img_augment: What specific image augmentation to perform for SSL
        """

        super().__init__("mmimdb", model, tokenizer, batch_size, num_workers)

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if img_augment is not None:
            if re.match("crop-(\d*\.?\d+)", img_augment):
                s = float(re.match("crop-(\d*\.?\d+)", img_augment)[1])
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                self.augment = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(s, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                raise ValueError(f"Unknown image augmentation: {img_augment}")
        self.setup("test")

    def setup(self, stage: str):
        self.val_dataset = None
        root, metadata = self.catalog[self.dataset]["path"], self.catalog[self.dataset]["metadata"]

        if self.model == 'Sup':
            self.train_dataset = MMIMDBDatasetSup(root, metadata, "train", self.test_transform, self.tokenizer)
            self.val_dataset = MMIMDBDatasetSup(root, metadata, "dev", self.test_transform, self.tokenizer)
            self.test_dataset = MMIMDBDatasetSup(root, metadata, "test", self.test_transform, self.tokenizer)
        elif self.model == 'SupervisedClassifier':
            self.train_dataset = MMIMDBDatasetSup(root, metadata, "train", self.img_transform, self.tokenizer)
            self.val_dataset = MMIMDBDatasetSup(root, metadata, "dev", self.test_transform, self.tokenizer)
            self.test_dataset = MMIMDBDatasetSup(root, metadata, "test", self.test_transform, self.tokenizer)
        elif self.model == 'SimCLR':
            self.train_dataset = MMIMDBDatasetSSL(root, metadata, self.augment, "train")
            self.val_dataset = MMIMDBDatasetSSL(root, metadata, self.augment, "dev")
            self.test_dataset = MMIMDBDatasetSSL(root, metadata, self.augment, "test")
        elif self.model in ['CLIP', 'BLIP2']:
            self.train_dataset = MMIMDBDatasetCLIP(root, metadata, "train", self.img_transform, self.tokenizer)
            self.val_dataset = MMIMDBDatasetCLIP(root, metadata, "dev", self.img_transform, self.tokenizer)
            self.test_dataset = MMIMDBDatasetCLIP(root, metadata, "test", self.img_transform, self.tokenizer)
        elif self.model == 'SLIP':
            self.train_dataset = MMIMDBDatasetSLIP(root, metadata, self.img_transform,
                                                   self.augment, "train", self.tokenizer)
            self.val_dataset = MMIMDBDatasetSLIP(root, metadata, self.img_transform,
                                                 self.augment, "dev", self.tokenizer)
            self.test_dataset = MMIMDBDatasetSLIP(root, metadata, self.img_transform,
                                                 self.augment, "test", self.tokenizer)
        elif self.model == "CoMM":
            self.train_dataset = MMIMDBDatasetMMSSL(root, metadata, self.img_transform,
                                                    self.augment, self.text_augment, "train", self.tokenizer)
            self.val_dataset = MMIMDBDatasetMMSSL(root, metadata, self.img_transform, self.augment,
                                                  self.text_augment, "dev", self.tokenizer)
            self.test_dataset = MMIMDBDatasetMMSSL(root, metadata, self.img_transform, self.augment,
                                                  self.text_augment, "test", self.tokenizer)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, drop_last=False)

genres_ = [
    "drama", "comedy", "romance", "thriller", "crime", "action", "adventure",
    "horror", "documentary", "mystery", "sci-fi", "music", "fantasy", "family",
    "biography", "war", "history", "animation", "musical", "western", "sport",
    "short", "film-noir"
]

class MMIMDBDatasetBase(torch.utils.data.Dataset):
    def __init__(self, root: str, metadata: str, split: str = "train"):
        """
        :param root: /path/to/mmimdb
        :param metadata: /path/to/mmimdb/split/ where `split.json` is located
        :param split: "train", "dev" (i.e. validation) or "test"
        """
        self.root = root
        self.split = split
        self.samples = []
        metadata = os.path.join(metadata, "split.json")
        with open(metadata) as f:
            ids = json.load(f)[self.split]
        for img_id in ids:
            sample_path = os.path.join(self.root, 'dataset', f'{img_id}.json')
            with open(sample_path) as f:
                meta = json.load(f)
                plot = meta['plot']
                genres = meta['genres']
            self.samples.append((img_id, plot, genres))

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_raw_item(self, i):
        index, captions, genres = self.samples[i]
        path = os.path.join(self.root, 'dataset', '{}.jpeg'.format(index))
        img = self.pil_loader(path)
        caption = np.random.choice(captions)

        return img, caption, genres

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class MMIMDBDatasetSup(MMIMDBDatasetBase):
    def __init__(self, root, metadata, split: str = "train", transform=None, tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.tokenizer = tokenizer
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([genres_])

    def __getitem__(self, i):
        img, caption, genres = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # one-hot encoding of genres
        genres = [genre.lower() for genre in genres]
        genres = self.mlb.transform([genres])[0]

        return (img, caption), genres


class MMIMDBDatasetCLIP(MMIMDBDatasetBase):
    def __init__(self, root, metadata, split: str = "train", transform=None, tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption, _ = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return img, caption


class MMIMDBDatasetSLIP(MMIMDBDatasetBase):
    def __init__(self, root, metadata, transform, augment, split: str = "train", tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption, _ = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2


class MMIMDBDatasetSSL(MMIMDBDatasetBase):
    def __init__(self, root, metadata, augment, split: str = "train"):
        super().__init__(root, metadata, split=split)

        self.augment = augment

    def __getitem__(self, i):
        img, _, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class MMIMDBDatasetMMSSL(MMIMDBDatasetBase):
    """Apply augmentations jointly to both image and text modalities."""

    def __init__(self, root, metadata, transform,
                 augment, text_augment, split: str = "train", tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.text_augment = text_augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        cap1 = self.text_augment(caption)
        cap2 = self.text_augment(caption)

        return [aug1, cap1], [aug2, cap2]

