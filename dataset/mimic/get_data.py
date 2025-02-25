"""Implements dataloaders for generic MIMIC tasks."""
from typing import Callable, Optional, Union, Tuple
import numpy as np
from torch.utils.data import Dataset
import random
import pickle
import os


class MIMIC(Dataset):
    """MIMIC dataset (n=36k) pre-processed as part of MultiBench [1]
    It currently has 2 modalities:
        - static patient background information, tabular: shape (*, 5) where (*) is batch dimension
        - health recordings, time-series: shape (*, 24, 12)
    It comes with two main tasks: mortality prediction and ICD-9 code predictions.
    [1] MultiBench: Multiscale Benchmarks for Multimodal Representation Learning, Liang et al., NeurIPS Benchmarks 2021
    """
    def __init__(self, data_path: str = "im.pk",
                 split: str = "train",
                 task: int = -1,
                 modalities: Union[str, Tuple[str]] = ("tabular", "timeseries"),
                 flatten_time_series: bool = False,
                 transform_timeseries: Optional[Callable] = None,
                 transform_tabular: Optional[Callable] = None):
        """
         Args:
            data_path: Datafile location. Defaults to 'im.pk'.
            split: must be in {"train", "val", "test"}
            task: Integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task.
            modalities: Modalities to return. NB: the order is preserved.
            flatten_time_series: Whether to flatten time series data or not. Defaults to False.
            transform_timeseries: transformation(s) to apply to timeseries data.
            transform_tabular: transformation(s) to apply to tabular data.
        """

        self.transform_timeseries = transform_timeseries
        self.transform_tabular = transform_tabular
        self.modalities = modalities
        self.task = task
        self.split = split
        self.data_path = data_path

        if not os.path.exists(self.data_path):
            raise FileNotFoundError("MIMIC data require credentials from https://mimic.mit.edu. "
                                    "Once access is granted, send an email to yiweilyu@umich.edu and ask "
                                    "for im.pkl preprocessed file.")

        if isinstance(self.modalities, str):
            self.modalities = (self.modalities,)

        with open(data_path, 'rb') as f:
            datafile = pickle.load(f)

        X_t = datafile['ep_tdata']
        X_s = datafile['adm_features_all']

        X_t[np.isinf(X_t)] = 0
        X_t[np.isnan(X_t)] = 0
        X_s[np.isinf(X_s)] = 0
        X_s[np.isnan(X_s)] = 0

        X_s_avg = np.average(X_s, axis=0)
        X_s_std = np.std(X_s, axis=0)
        X_t_avg = np.average(X_t, axis=(0, 1))
        X_t_std = np.std(X_t, axis=(0, 1))

        for i in range(len(X_s)):
            X_s[i] = (X_s[i]-X_s_avg)/X_s_std
            for j in range(len(X_t[0])):
                X_t[i][j] = (X_t[i][j]-X_t_avg)/X_t_std

        timestep = len(X_t[0])
        series_dim = len(X_t[0][0])
        if flatten_time_series:
            X_t = X_t.reshape(len(X_t), timestep*series_dim)

        if task < 0:
            y = datafile['adm_labels_all'][:, 1]
            admlbl = datafile['adm_labels_all']
            le = len(y)
            for i in range(0, le):
                if admlbl[i][1] > 0:
                    y[i] = 1
                elif admlbl[i][2] > 0:
                    y[i] = 2
                elif admlbl[i][3] > 0:
                    y[i] = 3
                elif admlbl[i][4] > 0:
                    y[i] = 4
                elif admlbl[i][5] > 0:
                    y[i] = 5
                else:
                    y[i] = 0
        else:
            y = datafile['y_icd9'][:, task]

        X = []
        for mod in modalities:
            if mod == "tabular":
                X.append(X_s.astype(np.float32))
            elif mod == "timeseries":
                X.append(X_t.astype(np.float32))
            else:
                raise ValueError(f"Unknown modality: {mod}")
        datasets = list(zip(zip(*X), y))

        # Defines the split with a fixed random seed
        random.seed(10)
        random.shuffle(datasets)
        le = len(datasets)
        if split == "train":
            self.dataset = datasets[le//5:]
        elif split == "val":
            self.dataset = datasets[0:le//10]
        elif split == "test":
            self.dataset = datasets[le//10:le//5]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        for i, mod in enumerate(self.modalities):
            if mod == "tabular" and self.transform_tabular is not None:
                X[i] = self.transform_tabular(X[i])
            elif mod == "timeseries" and self.transform_timeseries is not None:
                X[i] = self.transform_timeseries(X[i])
        return X, y

    def __len__(self):
        return len(self.dataset)


