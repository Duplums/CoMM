import h5py
import numpy as np
from typing import Tuple
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset [1], adapted from
    https://github.com/stanford-iprl-lab/multimodal_representation/tree/master/multimodal/dataloaders


    [1] Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations
        for Contact-Rich Tasks, Lee, Zhu et al., ICRA 2019

    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        task: str = "ee_yaw_next",
        modalities: Tuple[str] = ("image", "force", "proprio"),
        test_ratio: float = 0.2,
        transform=None,
        episode_length=50,
        n_time_steps=1,
        action_dim=4
    ):
        """Initialize dataset.

        Args:
            root (str):
            split (str): "train", "val" or "test"
            task (str): "ee_yaw_next" for end-effector position prediction task (regression) or
                "contact_next" for next contact prediction (binary classifications)
            modalities: Tuple of modalities to return, in {"image", "force", "proprio"}
            test_ratio (float): test split ratio for train/test partition
            transform (fn, optional): Optional function to transform data. Defaults to None.
            episode_length (int, optional): Length of each episode. Defaults to 50.
            n_time_steps (int, optional): Number of time steps. Defaults to 1.
            action_dim (int, optional): Action dimension. Defaults to 4.
        """
        assert task in {"ee_yaw_next", "contact_next"}, f"Unknown task: {task}"

        self.root = root
        self.split = split
        self.task = task
        self.modalities = modalities
        self.test_ratio = test_ratio
        self.dataset_path = self._get_filenames(seed=42)
        self.transform = transform
        self.episode_length = episode_length
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim

    def _get_filenames(self, seed: int):
        """Get the filenames by split (reproducible with a fixed seed)"""
        filename_list = []
        for file in sorted(os.listdir(self.root)):
            if file.endswith(".h5"):
                filename_list.append(os.path.join(self.root, file))
        filename_list = np.array(filename_list)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(filename_list))
        rng.shuffle(idx)
        n_test = int(self.test_ratio * len(filename_list))
        n_train = len(filename_list) - n_test
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        if self.split == "train":
            return filename_list[train_idx]
        elif self.split in ["val", "test"]:
            return filename_list[test_idx]
        raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        """Get number of items in dataset."""
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):
        """Get item in dataset at index idx."""
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            dataset_index
        )

        # Filter modalities and returns required target
        X, y = [], None
        for m in self.modalities:
            assert m in sample, f"Unknown modality: {m}"
            X.append(sample[m])
        y = sample[self.task]

        return X, y

    def _get_single(
        self, dataset_name, dataset_index
    ):

        with h5py.File(dataset_name, "r", swmr=True, libver="latest") as dataset:

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            sample = {
                "image": Image.fromarray(image),
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    dataset["contact"][dataset_index + 1].sum() > 0
                ).astype(int)
            }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(n={len(self)}, split={self.split}, task={self.task})"


class ProcessForce(object):
    """Truncate a time series of force readings with a window size.
    Args:
        window_size (int): Length of the history window that is
            used to truncate the force readings
    """

    def __init__(self, window_size, key='force', tanh=False):
        """Initialize ProcessForce object.

        Args:
            window_size (int): Windows size
            key (str, optional): Key where data is stored. Defaults to 'force'.
            tanh (bool, optional): Whether to apply tanh to output or not. Defaults to False.
        """
        assert isinstance(window_size, int)
        self.window_size = window_size
        self.key = key
        self.tanh = tanh

    def __call__(self, sample):
        """Get data from sample."""
        force = sample[self.key]
        force = force[-self.window_size:]
        if self.tanh:
            force = np.tanh(force)  # remove very large force readings
        sample[self.key] = force
        return sample


class ProcessImage(object):
    """
        Transform numpy image (HWC format) to torch tensor (CHW format).
    """

    def __init__(self, size=128):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        new_dict = dict()
        for k, v in sample.items():
            if k == "image":
                new_dict[k] = self.tf(v)
            else:
                new_dict[k] = v
        return new_dict

