import numpy as np
import scipy.ndimage
from PIL import Image
import torch
import os
import re
import json
from typing import Callable, Optional, Type, Tuple
from pytorch_lightning import LightningDataModule
from torchvision import transforms
# Local import
from utils import make_dirs, GaussianBlur


class TrifeaturesDataModule(LightningDataModule):
    """Data module for Trifeatures/BimodalTrifeatures dataset"""

    def __init__(self, model: str,
                 dataset: str = "bimodal",
                 batch_size: int = 32,
                 num_workers: int = 0,
                 augment: Optional[Tuple[str]] = None,
                 **kwargs):
        """
        :param model: {'Sup', 'CLIP', 'CrossSelf', 'CoMM'}
            The model defines the augmentations to apply:
                - Sup: no augmentation, returns the image + label(s)
                - CLIP: no augmentation, returns pairs of modalities
                - CrossSelf: pairs of augmented images + original
                - CoMM: augmented pairs of modalities
        :param dataset: either "bimodal" or "unimodal"
        :param batch_size: Batch size to pass to Dataloaders
        :param num_workers: Number of workers to pass to Dataloaders
        :param kwargs: keywords args given to Trifeatures/BimodalTrifeatures dataset
        """
        super().__init__()

        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "catalog.json")
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        root = self.catalog["trifeatures"]["path"]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        if augment is not None:
            _augment_parsed = []
            for aug in augment:
                if aug == "all":
                    _augment_parsed.append(self.augment)
                elif aug is None:
                    _augment_parsed.append(self.img_transform)
                elif aug == "all_wo_crop":
                    _augment_parsed.append(
                        transforms.Compose([
                            transforms.RandomApply([
                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                            ], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
                elif re.match(r"(crop-(\d*\.?\d+)|crop-to-(\d*\.?\d+))", aug):
                    if re.match(r"crop-to-(\d*\.?\d+)", aug):
                        scale = (0.01, float(re.match(r"crop-to-(\d*\.?\d+)", aug)[1]))
                    else:
                        scale = (float(re.match(r"crop-(\d*\.?\d+)", aug)[1]), 1.0)
                    _augment_parsed.append(
                        transforms.Compose([
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
                    )
                else:
                    raise ValueError(f"Unknown augmentation: {aug}")
            self.augment = _augment_parsed

        if self.model == "Sup":
            dset = Trifeatures if dataset == "unimodal" else BimodalTrifeatures
            self.train_dataset = dset(root, split="train", transform=self.img_transform, **kwargs)
            self.val_dataset = dset(root, split="test", transform=self.img_transform, **kwargs)
        elif self.model.startswith('SimCLR'):
            assert dataset == "unimodal"
            dset = TrifeaturesSSL
            self.train_dataset = dset(root, split="train", augment=self.augment, **kwargs)
            self.val_dataset = dset(root, split="test", augment=self.augment, **kwargs)
        elif self.model.startswith('CLIP'):
            dset = TrifeaturesCLIP if dataset == "unimodal" else BimodalTrifeaturesCLIP
            self.train_dataset = dset(root, split="train", transform=self.img_transform, **kwargs)
            self.val_dataset = dset(root, split="test", transform=self.img_transform, **kwargs)
        elif self.model.startswith('CrossSelf'):
            assert dataset == "bimodal"
            dset = BimodalTrifeaturesCrossSelf
            self.train_dataset = dset(root, split="train", transform=self.img_transform,
                                      augment=self.augment, **kwargs)
            self.val_dataset = dset(root, split="test", transform=self.img_transform,
                                    augment=self.augment, **kwargs)
        elif self.model == "CoMM":
            dset = TrifeaturesMMSSL if dataset == "unimodal" else BimodalTrifeaturesMMSSL
            if dataset == "unimodal":
                kwargs.update(text_augment=(lambda x: x))
            self.train_dataset = dset(root, split="train", augment=self.augment, **kwargs)
            self.val_dataset = dset(root, split="test", augment=self.augment, **kwargs)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def test_dataloader(self):  # val == test here
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)


class Trifeatures(torch.utils.data.Dataset):
    """ Trifeatures dataset [1] of images from 10 shapes, 10 colors and 10 textures.
    Each image is rendered several times for various positions and angles in the shape + texture.

    [1] What shapes feature representations? Exploring dataset, architectures, and training, Hermann & Lampinen, NeurIPS 2020
    """

    BASE_SHAPES = ["triangle", "square",
                   "plus", "circle", "tee",
                   "rhombus", "pentagon",
                   "star", "fivesquare", "trapezoid"]
    BASE_COLORS = {
        "red": (1., 0., 0.),
        "green": (0., 1., 0.),
        "blue": (0., 0., 1.),
        "yellow": (1., 1, 0.),
        "pink": (1., 0.4, 1.),
        "cyan": (0., 1., 1.),
        "purple": (0.3, 0., 0.5),
        "ocean": (0.1, 0.4, 0.5),
        "orange": (1., 0.6, 0.),
        "white": (1., 1., 1.),
    }
    BG_COLOR = np.array((0.5, 0.5, 0.5), dtype=np.float32)
    BASE_TEXTURES = ["solid", "stripes", "grid", "hexgrid", "dots", "noise",
                     "triangles", "zigzags", "rain", "pluses"]
    BASE_SIZE = 128
    RENDER_SIZE = 224
    RANDOM_ANGLE_RANGE = 45
    TEXTURE_SCALE = 10
    BASE_COLORS = {n: np.array(c, dtype=np.float32) for n, c in BASE_COLORS.items()}

    def __init__(self,
                 root: str,
                 split: str = "train",
                 targets: str = "shape+color+texture",
                 targets_format: Type = str,
                 num_per_combination: int = 3,
                 transform: Optional[Callable] = None,
                 seed: int = 42):
        """
        :param root: Path to directory where train/<images.png> and test/<images.png> are stored.
        :param num_per_combination: Number of views for each triplet (color, shape, texture).
            Number of total images == `num_per_combination` x 1000
        :param split: "train" or "test"
        :param targets: Which targets to return ("shape", "color", "texture" or "shape+color+texture")
        :param targets_format: Format of the targets (str for sentences, int for integer encoding).
        :param transform: Transformations applied to images
        :param seed: random seed for reproducibility
        """
        assert targets in {"shape", "color", "texture", "shape+color+texture"}, f"Incorrect targets: {targets}"
        assert targets_format in {str, int}, f"Incorrect targets format: {targets_format}"
        assert split in {"train", "test"}, f"Incorrect split: {split}"

        self.root = root
        self.transform = transform
        self.split = split
        self.num_per_combination = num_per_combination
        self.targets = targets
        self.targets_format = targets_format
        self.rng = np.random.default_rng(seed)
        self.split_ratio = 0.8  # ratio between number of training/test images
        self._base_templates = {(s, Trifeatures.BASE_SIZE): Trifeatures._render_plain_shape(s)
                                for s in self.BASE_SHAPES}

        # If images are absent or missing, generate them
        if not self._check_integrity():
            self.generate_data()

        self.root = os.path.join(self.root, self.split)
        self.name_images = [file for file in sorted(os.listdir(self.root)) if file.endswith('.png')]
        self.color_to_index = {label: idx for idx, label in enumerate(Trifeatures.BASE_COLORS)}
        self.shape_to_index = {label: idx for idx, label in enumerate(Trifeatures.BASE_SHAPES)}
        self.texture_to_index = {label: idx for idx, label in enumerate(Trifeatures.BASE_TEXTURES)}

    def __len__(self):
        return len(self.name_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.name_images[idx])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        target = self.target_transform(self.name_images[idx])
        return image, target

    def target_transform(self, filename):
        """From filename, generate a sentence or integers encoding the shape, color or texture."""
        labels = re.split('_', filename.split('.')[0])  # {shape}_{texture}_{color}_{id}.png
        if self.targets == "color":
            if self.targets_format == str:
                return f"This is a picture of an object in {filename.split('_')[2]}"
            else:
                return self.color_to_index[labels[2]]
        elif self.targets == "shape":
            if self.targets_format == str:
                return f"This is a picture of a {filename.split('_')[0]}"
            else:
                return self.shape_to_index[labels[0]]
        elif self.targets == "texture":
            if self.targets_format == str:
                return f"This is a picture of an object made of {filename.split('_')[1]}"
            else:
                return self.texture_to_index[labels[1]]
        elif self.targets == "shape+color+texture":
            if self.targets_format == str:
                return (f"This is a picture of a {filename.split('_')[0]} in {filename.split('_')[2]} "
                        f"made of {filename.split('_')[1]}")
            else:
                return self.shape_to_index[labels[0]], self.color_to_index[labels[2]], self.texture_to_index[labels[1]]

        raise ValueError(f"Unknown targets: {self.targets}")

    def _check_integrity(self):
        for split in ["train", "test"]:
            nb_images = 0
            pattern = r"(\w+)_(\w+)_(\w+)(_[0-9]+)?\.png"
            if not os.path.exists(os.path.join(self.root, split)):
                return False
            for f in os.listdir(os.path.join(self.root, split)):
                if os.path.isfile(os.path.join(self.root, split, f)):
                    match = re.match(pattern, f)
                    if match:
                        nb_images += 1

            total_elements = len(self.BASE_SHAPES) * len(self.BASE_COLORS) * len(self.BASE_TEXTURES) # == 1000
            training_images = int(self.split_ratio * self.num_per_combination * total_elements)
            test_images = total_elements - int(self.split_ratio * total_elements)
            if (nb_images != training_images and split == "train") or (nb_images != test_images and split == "test"):
                return False
            print(f"{nb_images} images in {split} found.")
        return True

    def generate_data(self):
        """Saves stimuli where all features vary orthogonally. Split it into train/test set"""
        # create an array of size BASE_SHAPESxBASE_COLORSxBASE_TEXTURES
        print("Images not found. Generating dataset...")
        split = self._split_array()
        train_directory = os.path.join(self.root, "train")
        test_directory = os.path.join(self.root, "test")
        make_dirs([train_directory])
        make_dirs([test_directory])
        idx = 0
        for s in self.BASE_SHAPES:
            for t in self.BASE_TEXTURES:
                for c in self.BASE_COLORS.keys():
                    print(idx + 1, s, t, c)
                    if split[idx] == 1:
                        directory = test_directory
                        image_array = self._render_stimulus(s, c, t)
                        image = Image.fromarray((image_array * 255.).astype(np.uint8), mode='RGB')
                        image.save(os.path.join(directory, "%s_%s_%s.png" % (s, t, c)))
                    else:
                        directory = train_directory
                        for i in range(self.num_per_combination):
                            image_array = self._render_stimulus(s, c, t)
                            image = Image.fromarray((image_array * 255.).astype(np.uint8), mode='RGB')
                            image.save(os.path.join(directory, "%s_%s_%s_%i.png" % (s, t, c, i)))
                    idx += 1
        print("Dataset generated !")

    @staticmethod
    def _render_plain_shape(name):
        """Shape without color dimension."""
        size = Trifeatures.BASE_SIZE
        size = int(size)
        shape = np.zeros([size, size], np.float32)
        if name == "square":
            shape[:, :] = 1.
        elif name == "circle":
            for i in range(size):
                for j in range(size):
                    if np.square(i + 0.5 - size // 2) + np.square(j + 0.5 - size // 2) < np.square(size // 2):
                        shape[i, j] = 1.
        elif name == "triangle":
            for i in range(size):
                for j in range(size):
                    if np.abs(j - size // 2) - np.abs(i // 2) < 1:
                        shape[i, j] = 1.
        elif name == "plus":
            shape[:, size // 2 - size // 6: size // 2 + size // 6 + 1] = 1.
            shape[size // 2 - size // 6: size // 2 + size // 6 + 1, :] = 1.
        elif name == "tee":
            shape[:, size // 2 - size // 6: size // 2 + size // 6 + 1] = 1.
            shape[:size // 3, :] = 1.
        elif name == "rhombus":
            for i in range(size):
                for j in range(size):
                    if 0 < j - size // 2 + i // 2 < size // 2:
                        shape[i, j] = 1.
        elif name == "pentagon":
            midline = int(size * 0.4)
            for i in range(midline):
                for j in range(size):
                    if np.abs(j - size // 2) - np.abs(i * 1.25) < 1:
                        shape[i, j] = 1.
            for i in range(midline, size):
                x_off = (i - midline) // 3.1
                for j in range(size):
                    if x_off < j < size - x_off:
                        shape[i, j] = 1.
        elif name == "star":
            line = int(size * 0.4)
            line2 = line + int(0.2 * size)
            line3 = line + int(0.15 * size)
            for i in range(line):
                for j in range(size):
                    if np.abs(j - size // 2) - np.abs(i // 4) < 1:
                        shape[i, j] = 1.
            for i in range(line, line2):
                x_off = (i - line) * 2.4
                for j in range(size):
                    if x_off < j < size - x_off:
                        shape[i, j] = 1.
            for i in range(line3, size):
                x_off_1 = (size * 0.33) - 0.43 * (i - line3)
                x_off_2 = (size * 0.62) - 1.05 * (i - line3)
                for j in range(size):
                    if x_off_1 < j < x_off_2 or x_off_1 < size - j < x_off_2:
                        shape[i, j] = 1.
        elif name == "fivesquare":
            shape[:, :] = 1.
            shape[:, size // 3: 2 * size // 3] = 0.
            shape[size // 3: 2 * size // 3, :] = 0.
            shape[size // 3: 2 * size // 3, size // 3: 2 * size // 3] = 1.
        elif name == "trapezoid":
            for i in range(size):
                x_off = i // 3.1
                for j in range(size):
                    if x_off < j < size - x_off:
                        shape[i, j] = 1.

        return shape

    def get_texture(self, size, texture_name):
        scale = Trifeatures.TEXTURE_SCALE
        lwidth = scale // 3
        small_lwidth = scale // 5
        texture_offset_x = self.rng.integers(0, scale)
        texture_offset_y = self.rng.integers(0, scale)
        texture = np.zeros([size, size], dtype=np.float32)
        if texture_name == "solid":
            return np.ones_like(texture)
        elif texture_name == "stripes":
            for i in range(size):
                if (i + texture_offset_y) % scale < lwidth:
                    texture[i, :] = 1.
        elif texture_name == "grid":
            for i in range(size):
                if (i + texture_offset_y) % scale < lwidth:
                    texture[i, :] = 1.
            for j in range(size):
                if (j + texture_offset_x) % scale < lwidth:
                    texture[:, j] = 1.
        elif texture_name == "hexgrid":
            for i in range(size):
                for j in range(size):
                    y = (i + texture_offset_y)
                    x = (j + texture_offset_x)
                    # if y < lwidth or (x + int(1.73 * y)) % scale < lwidth:
                    if (x + int(1.73 * y)) % scale < small_lwidth or (
                            x - int(1.73 * y)) % scale < small_lwidth or y % scale < small_lwidth:
                        texture[i, j] = 1.
        elif texture_name == "dots":
            rad_squared = (3 * scale // 7) ** 2
            for i in range(size):
                for j in range(size):
                    y = ((i + texture_offset_y) % scale) - scale // 2
                    x = ((j + texture_offset_x) % scale) - scale // 2
                    # if y < lwidth or (x + int(1.73 * y)) % scale < lwidth:
                    if (x ** 2) + (y ** 2) < rad_squared:
                        texture[i, j] = 1.
        elif texture_name == "noise":
            texture = self.rng.binomial(1, 0.5, texture.shape)
        elif texture_name == "triangles":
            for i in range(size):
                for j in range(size):
                    y = (i + texture_offset_y) % scale
                    x = (j + texture_offset_x) % scale
                    # if y < lwidth or (x + int(1.73 * y)) % scale < lwidth:
                    if y // 2 - np.abs(x - scale // 2) > 0:
                        texture[i, j] = 1.
        elif texture_name == "zigzags":
            scale_off = scale - scale // 2
            for i in range(size):
                slopesign = ((i + texture_offset_y) // scale) % 2
                slopesign2 = ((i + texture_offset_y) // (2 * scale)) % 2
                for j in range(size):
                    y = (i + texture_offset_y) % scale
                    x = (j + texture_offset_x) % scale
                    if slopesign:
                        x = scale - x - 1
                    off = y // 2
                    if off < x < scale_off + off:
                        texture[i, j] = 1.
        elif texture_name == "rain":
            rainheight = scale - scale // 3
            rainwidth = 1
            rainprob = 0.05
            this_offset_x = self.rng.integers(0, scale)
            for i in range(size):
                for j in range(size):
                    if self.rng.binomial(1, rainprob):
                        texture[i: i + rainheight, j:j + rainwidth] = 1.
        elif texture_name == "pluses":
            pl_half_width = 1.5
            for i in range(size):
                slopesign = ((i + texture_offset_y) // scale) % 2
                for j in range(size):
                    y = (i + texture_offset_y) % scale
                    x = (j + texture_offset_x) % scale
                    if slopesign:
                        if (np.abs(x) < pl_half_width) or (scale - x < pl_half_width) or (
                                (np.abs(y - scale // 2) < pl_half_width) and
                                np.abs(x - scale // 2) > pl_half_width):
                            texture[i, j] = 1.
                    else:
                        if (np.abs(x - scale // 2) < pl_half_width) or (np.abs(y - scale // 2) < pl_half_width):
                            texture[i, j] = 1.

        return texture

    def _render_uncolored_shape(self, name):
        "Shape without color dimension, at random rotation and position."
        template = self._base_templates[(name, Trifeatures.BASE_SIZE)]
        angle = self.rng.integers(-Trifeatures.RANDOM_ANGLE_RANGE, Trifeatures.RANDOM_ANGLE_RANGE)
        shape = scipy.ndimage.rotate(template, angle, order=1)
        new_size = shape.shape
        image = np.zeros([Trifeatures.RENDER_SIZE, Trifeatures.RENDER_SIZE], np.float32)
        offset_x = self.rng.integers(0, Trifeatures.RENDER_SIZE - new_size[0])
        offset_y = self.rng.integers(0, Trifeatures.RENDER_SIZE - new_size[1])
        image[offset_x:offset_x + new_size[0],
        offset_y:offset_y + new_size[1]] = shape
        return image

    def _render_stimulus(self, shape, color, texture):
        image = self._render_uncolored_shape(shape)
        t_size = 2 * Trifeatures.RENDER_SIZE
        texture = self.get_texture(t_size, texture)
        angle = self.rng.integers(-Trifeatures.RANDOM_ANGLE_RANGE, Trifeatures.RANDOM_ANGLE_RANGE)
        texture = scipy.ndimage.rotate(texture, angle, order=0, reshape=False)
        texture = texture[Trifeatures.RENDER_SIZE // 2:-Trifeatures.RENDER_SIZE // 2,
                  Trifeatures.RENDER_SIZE // 2:-Trifeatures.RENDER_SIZE // 2]
        image = np.multiply(image, texture)
        color_image = image[:, :, None] * Trifeatures.BASE_COLORS[color][None, None, :]
        color_image += (1 - image)[:, :, None] * Trifeatures.BG_COLOR[None, None, :]

        return color_image

    def _split_array(self):
        total_elements = len(self.BASE_SHAPES) * len(self.BASE_COLORS) * len(self.BASE_TEXTURES)
        # Calculate the number of ones
        num_zeros = int(total_elements * self.split_ratio)
        # Calculate the number of zeros
        num_ones = total_elements - num_zeros
        # Create an array with the required number of ones and zeros
        array = np.array([1] * num_ones + [0] * num_zeros)
        # Shuffle the array to randomize the 0s and 1s
        self.rng.shuffle(array)
        return array

    def __repr__(self):
        return f"{self.__class__.__name__}(split={self.split}, num_per_combination={self.num_per_combination})"

class TrifeaturesCLIP(Trifeatures):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        assert self.targets_format == str, "Target must be caption for CLIP"
        img, caption = super().__getitem__(i)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return img, caption


class TrifeaturesSLIP(Trifeatures):
    def __init__(self, *args, augment, tokenizer=None, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.img_transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        assert self.targets_format == str, "Target must be caption for SLIP"
        img, caption = super().__getitem__(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return img, caption, aug1, aug2


class TrifeaturesSSL(Trifeatures):
    def __init__(self, *args, augment, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment

    def __getitem__(self, i):
        img, _ = super().__getitem__(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class TrifeaturesMMSSL(Trifeatures):
    """Apply augmentations jointly to both image and text modalities."""
    def __init__(self, *args, augment, text_augment, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.text_augment = text_augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        assert self.targets_format == str, "Target must be caption for MMSSL"
        img, caption = super().__getitem__(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        cap1 = self.text_augment(caption)
        cap2 = self.text_augment(caption)

        return [aug1, cap1], [aug2, cap2]


class BimodalTrifeatures(Trifeatures):
    """
        Datasets of pairs (image, image) from trifeatures (viewed as modalities) such that
        all pairs share only one attribute, the other two being independent.
        The size of the dataset is custom and the pairs are uniformly sampled among all possible combinations.
        The tasks (labels) can be either:
            - "share": returns the share attribute between images
            - "unique1" or "unique2": returns the unique attribute for each modality (1 or 2)
            - "synergy": returns 1 iff synergy attribute is present in both modalities, 0 otherwise
    """

    def __init__(self, root: str,
                 split: str = "train",
                 task: str = "share",
                 share_attr: str = "shape",
                 unique_attr: str = "texture",
                 synergy_attr: Tuple[str] = ("texture", "color"),
                 biased: bool = True,
                 max_size: int = 1e4,
                 num_per_combination: int = 3,
                 transform: Optional[Callable] = None,
                 seed: int = 42):

        assert task in {"share", "unique1", "unique2", "synergy"}
        assert share_attr in {"shape", "texture", "color"}
        assert unique_attr in {"shape", "texture", "color"}
        for attr in synergy_attr:
            assert attr in {"shape", "texture", "color"}

        super().__init__(root, split, num_per_combination=num_per_combination,
                         targets_format=int, transform=transform, seed=seed)
        self.task = task
        self.share_attr = share_attr
        self.unique_attr = unique_attr
        self.synergy_attr = synergy_attr
        self.max_size = int(max_size)
        self.biased = biased
        attrs = dict(color=self.color_to_index, shape=self.shape_to_index, texture=self.texture_to_index)
        synergy_attr_enc = (list(sorted(attrs[self.synergy_attr[0]].values())),
                            list(sorted(attrs[self.synergy_attr[1]].values())))
        self.correlated_feature_pairs = list(zip(synergy_attr_enc[0], self.rng.permutation(synergy_attr_enc[1])))
        self.idx_pairs = self._get_idx_pairs()

    def _get_idx_pairs(self):
        assert len(self.name_images) ** 2 < 1e7 # ensures reasonable size
        attr_to_id = dict(shape=0, color=1, texture=2)
        n = len(self.name_images)
        share = np.array([self.target_transform(n1)[attr_to_id[self.share_attr]] for n1 in self.name_images])

        # bool mask indicating which couples contain shared information
        share_eq = (share.reshape(n, 1) == share.reshape(1, n))
        # bool mask indicating which couples contains correlated features
        if self.biased:
            synergy = (np.array([self.target_transform(n1)[attr_to_id[self.synergy_attr[0]]] for n1 in self.name_images]),
                       np.array([self.target_transform(n1)[attr_to_id[self.synergy_attr[1]]] for n1 in self.name_images]))
            synergy_eq = np.zeros_like(share_eq, dtype=bool)
            for corr_pair in self.correlated_feature_pairs:  # takes only samples with correlated pairs
                m1 = np.repeat((synergy[0].reshape(n, 1) == corr_pair[0]), n, axis=1)
                m2 = np.repeat((synergy[1].reshape(1, n) == corr_pair[1]), n, axis=0)
                synergy_eq |= (m1 & m2)
        else:
            synergy_eq = np.ones_like(share_eq, dtype=bool)

        allowed_pairs = np.nonzero(share_eq & synergy_eq)
        n_allowed = len(allowed_pairs[0])
        if self.max_size > n_allowed:
            #print(f"Warning: `max_size` exceeds maximum dataset size, set to {n_allowed}")
            self.max_size = n_allowed
        subsampling = self.rng.choice(n_allowed, size=self.max_size, replace=False)
        return np.array(allowed_pairs).T[subsampling]

    def __getitem__(self, idx):
        idx1, idx2 = self.idx_pairs[idx]
        im1, target1 = super().__getitem__(idx1)
        im2, target2 = super().__getitem__(idx2)
        attr_to_id = dict(shape=0, color=1, texture=2)
        if self.task == "share":
            # sanity check
            assert target1[attr_to_id[self.share_attr]] == target2[attr_to_id[self.share_attr]]
            return [im1, im2], target1[attr_to_id[self.share_attr]]
        elif self.task == "unique1":
            return [im1, im2], target1[attr_to_id[self.unique_attr]]
        elif self.task == "unique2":
            return [im1, im2], target2[attr_to_id[self.unique_attr]]
        elif self.task == "synergy":
            corr_features = (target1[attr_to_id[self.synergy_attr[0]]], target2[attr_to_id[self.synergy_attr[1]]])
            y = int(corr_features in self.correlated_feature_pairs)
            return [im1, im2], y
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def __len__(self):
        return len(self.idx_pairs)

    def __repr__(self):
        return (f"{self.__class__.__name__}(split={self.split}, "
                f"num_per_combination={self.num_per_combination},"
                f"task={self.task},"
                f"max_size={self.max_size})")


class BimodalTrifeaturesCLIP(BimodalTrifeatures):
    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        return X


class BimodalTrifeaturesCrossSelf(BimodalTrifeatures):
    def __init__(self, *args, augment, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.img_transform = transform

    def __getitem__(self, i):
        (img1, img2), _ = super().__getitem__(i)
        aug11 = self.augment(img1)
        aug12 = self.augment(img1)
        aug21 = self.augment(img2)
        aug22 = self.augment(img2)
        if self.img_transform is not None:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
        return img1, img2, [aug11, aug12], [aug21, aug22]


class BimodalTrifeaturesMMSSL(BimodalTrifeatures):
    def __init__(self, *args, augment, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment

    def __getitem__(self, i):
        (img1, img2), _ = super().__getitem__(i)
        if isinstance(self.augment, list):
            aug11 = self.augment[0](img1)
            aug12 = self.augment[0](img1)
            aug21 = self.augment[1](img2)
            aug22 = self.augment[1](img2)
        else:
            aug11 = self.augment(img1)
            aug12 = self.augment(img1)
            aug21 = self.augment(img2)
            aug22 = self.augment(img2)

        return [aug11, aug21], [aug12, aug22]

