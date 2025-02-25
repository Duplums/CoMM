import json
import os
from PIL import Image
from typing import Optional
import torch
import warnings
from torchvision import transforms
# Local import
from dataset.img_caption import ImageCaptionDataModule


# Disable decompression bombs warning for large images
Image.MAX_IMAGE_PIXELS = None
# Silence repeated user warnings from scikit-learn multilabel binarizer for unknown classes.
warnings.filterwarnings("ignore", category=UserWarning)


class HatefulMemesDataModule(ImageCaptionDataModule):
    """
    Data module for Hateful Memes vision-language dataset [1] including
    memes (images) + captions describing hateful intentions (text).
    The downstream task is to predict whether the meme promotes hateful intentions or not.

    [1] The hateful memes challenge: Detecting hate speech in multimodal memes. Douwe Kiela, et al., NeurIPS 2020.
    """

    def __init__(self, model: str,
                 tokenizer=None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 augment: Optional[str] = None
                 ):

        """
        :param model: {'Sup', 'SimCLR', 'CLIP', 'SLIP', 'BLIP2, 'CoMM'}
            The model defines the augmentations to apply to the data.
        :param tokenizer: Which tokenizer use for encoding text with integers
        :param batch_size: Batch size to pass to Dataloaders
        :param num_workers: Number of workers to pass to Dataloaders
        :param augment: Specify the augmentations strength to apply,
        e.g. "crop-0.1" from cropping from 0.1 to 1 or "crop-to-0.8" for cropping from 0 to 0.8
        """

        super().__init__("hateful_memes", model, tokenizer, batch_size, num_workers, augment)

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.setup("test")

    def setup(self, stage: str):
        self.val_dataset = None
        root, metadata = self.catalog[self.dataset]["path"], self.catalog[self.dataset]["metadata"]

        if self.model == 'Sup':
            self.train_dataset = HatefulMemesDatasetSup(root, metadata, "train", self.test_transform, self.tokenizer)
            self.val_dataset = HatefulMemesDatasetSup(root, metadata, "dev", self.test_transform, self.tokenizer)
            self.test_dataset = HatefulMemesDatasetSup(root, metadata, "dev", self.test_transform, self.tokenizer)
        elif self.model == 'SupervisedClassifier':
            self.train_dataset = HatefulMemesDatasetSupCon(root, metadata, self.augment, self.text_augment,
                                                           "train", self.tokenizer)
            self.val_dataset = HatefulMemesDatasetSupCon(root, metadata, self.augment, self.text_augment,
                                                         "dev", self.tokenizer)
            self.test_dataset = HatefulMemesDatasetSupCon(root, metadata, self.augment, self.text_augment,
                                                          "dev", self.tokenizer)
        elif self.model == 'SimCLR':
            self.train_dataset = HatefulMemesDatasetSSL(root, metadata, self.augment, "train")
            self.val_dataset = HatefulMemesDatasetSSL(root, metadata, self.augment, "dev")
            self.test_dataset = HatefulMemesDatasetSSL(root, metadata, self.augment, "dev")
        elif self.model in ['CLIP', 'BLIP2']:
            self.train_dataset = HatefulMemesDatasetCLIP(root, metadata, "train", self.img_transform, self.tokenizer)
            self.val_dataset = HatefulMemesDatasetCLIP(root, metadata, "dev", self.img_transform, self.tokenizer)
            self.test_dataset = HatefulMemesDatasetCLIP(root, metadata, "dev", self.img_transform, self.tokenizer)
        elif self.model == 'SLIP':
            self.train_dataset = HatefulMemesDatasetSLIP(root, metadata, self.img_transform,
                                                   self.augment, "train", self.tokenizer)
            self.val_dataset = HatefulMemesDatasetSLIP(root, metadata, self.img_transform,
                                                 self.augment, "dev", self.tokenizer)
            self.test_dataset = HatefulMemesDatasetSLIP(root, metadata, self.img_transform,
                                                 self.augment, "dev", self.tokenizer)
        elif self.model == "CoMM":
            self.train_dataset = HatefulMemesDatasetMMSSL(root, metadata, self.img_transform,
                                                    self.augment, self.text_augment, "train", self.tokenizer)
            self.val_dataset = HatefulMemesDatasetMMSSL(root, metadata, self.img_transform, self.augment,
                                                  self.text_augment, "dev", self.tokenizer)
            self.test_dataset = HatefulMemesDatasetMMSSL(root, metadata, self.img_transform, self.augment,
                                                  self.text_augment, "dev", self.tokenizer)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, drop_last=False)


class HatefulMemesDatasetBase(torch.utils.data.Dataset):
    def __init__(self, root: str, metadata: str, split: str = "train"):
        """
        :param root: /path/to/HatefulMemes
        :param metadata: /path/to/HatefulMemes/split/ where `split.json` is located
        :param split: "train", "dev" (i.e. validation) or "test"
        """
        self.root = root
        self.split = split
        self.samples = []
        metadata = os.path.join(metadata, f"{split}.jsonl")
        with open(metadata, 'r') as json_file:
            infos = list(json_file)
        for info in infos:
            info = json.loads(info)
            self.samples.append((info["img"], info["text"], info["label"]))

    @staticmethod
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_raw_item(self, i):
        img_path, text, is_hateful = self.samples[i]
        path = os.path.join(self.root, img_path)
        img = self.pil_loader(path)
        return img, text, is_hateful

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class HatefulMemesDatasetSup(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, split: str = "train", transform=None, tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, is_hateful = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize text
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return (img, text), is_hateful


class HatefulMemesDatasetSupCon(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, augment, text_augment, split: str = "train", tokenizer=None):
        super().__init__(root, metadata, split=split)
        self.augment = augment
        self.text_augment = text_augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, is_hateful = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        text1 = self.text_augment(text)
        text2 = self.text_augment(text)

        return [aug1, text1], [aug2, text2], is_hateful


class HatefulMemesDatasetCLIP(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, split: str = "train", transform=None, tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, _ = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize text
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return img, text


class HatefulMemesDatasetSLIP(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, transform, augment, split: str = "train", tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, _ = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize text
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        return image, text, aug1, aug2


class HatefulMemesDatasetSSL(HatefulMemesDatasetBase):
    def __init__(self, root, metadata, augment, split: str = "train"):
        super().__init__(root, metadata, split=split)

        self.augment = augment

    def __getitem__(self, i):
        img, _, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class HatefulMemesDatasetMMSSL(HatefulMemesDatasetBase):
    """Apply augmentations jointly to both image and text modalities."""

    def __init__(self, root, metadata, transform,
                 augment, text_augment, split: str = "train", tokenizer=None):
        super().__init__(root, metadata, split=split)

        self.transform = transform
        self.augment = augment
        self.text_augment = text_augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, text, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            text = self.tokenizer(text)

        text1 = self.text_augment(text)
        text2 = self.text_augment(text)

        return [aug1, text1], [aug2, text2]

