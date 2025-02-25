"""Implements dataloaders for AFFECT data."""
from typing import *
import pickle
import numpy as np
import torch
import gdown
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


class Affect(Dataset):
    """Affect dataset pre-processed as part of MultiBench [1]
    It implements 4 dataset in one class: CMU-MOSEI, CMU-MOSI, MUSTARD, UR-FUNNY.
    All 4 dataset have 3 modalities coming from audiovisual inputs (i.e. videos):
        - vision: shape (*, T, pv) where (*) is batch dimension
        - audio: shape (*, T, pa)
        - text: shape (*, T, pt)
        where T is sequence length (different for each sample) and pv, pa, pt the features size.
    It comes with one task for each dataset related to sentiments/emotions (e.g. humor, sarcasm, fear, etc.)
    [1] MultiBench: Multiscale Benchmarks for Multimodal Representation Learning, Liang et al., NeurIPS Benchmarks 2021"""

    FILE_IDS = {
        "sarcasm": "1EMBUmUL5B0PTncGx3L-sBElGOmjFBR_h",
        "mosi": "1_XdzdW8UNG1TTS6QcX10uhoS6N11OBit",
        "humor": "1L5slPmYyhEVtwGyM1kgcFMjeBpXLZGT0",
        "mosei": "180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz"
    }


    def __init__(self, 
                 data_path: str,
                 dataset: str,
                 split: str = "train",
                 modalities: Union[str, Tuple[str]] = ("vision", "audio", "text"),
                 task: str = "classification",
                 flatten_time_series: bool = False,
                 align: bool = True,
                 transform_vision: Optional[Callable] = None,
                 transform_audio: Optional[Callable] = None,
                 transform_text: Optional[Callable] = None,
                 z_norm: bool = False):
        """
        Args:
            data_path: Datafile location.
            dataset: Dataset to be loaded, in {"mosei", "mosi", "humor", "sarcasm"}.
                NB: "sarcasm" == MUSTARD, "humor" == UR-FUNNY
            split: in {"train", "val", "test"}
            modalities: Modalities to return. NB: the order is preserved.
            task: either "classification" or "regression".
                If "classification", label is binarized in {0, 1}, otherwise it is left unchanged.
            flatten_time_series: Whether to flatten time series data or not.
            align: Whether to align data or not across modalities
            transform_vision: Vision transformations to apply
            transform_audio: Audio transformations to apply
            transform_audio: Text transformations to apply
            z_norm: Whether to normalize data along the z dimension or not. Defaults to False.
        """
        self.data_path = data_path
        self.dataset = dataset
        self.split = split
        self.modalities = modalities
        self.task = task
        self.align = align
        self.flatten_time_series = flatten_time_series
        self.transform_vision = transform_vision
        self.transform_audio = transform_audio
        self.transform_text = transform_text
        self.z_norm = z_norm

        if isinstance(self.modalities, str):
            self.modalities = (self.modalities,)

        if not os.path.exists(data_path): # fetch the data
            self._download_file()

        with open(data_path, "rb") as f:
            data = pickle.load(f)
        split_ = split if split != "val" else "valid" # "val" -> "valid"
        data_split = data[split_]
        # Drop samples without text
        self.dataset = drop_entry(data_split)
        # Removes `-inf`
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def _download_file(self):
        """Download file from Google Drive using gdown."""
        print(f"Downloading {os.path.basename(self.data_path)} from Google Drive...")
        url = f"https://drive.google.com/uc?id={self.FILE_IDS[self.dataset]}"
        gdown.download(url, self.data_path, quiet=False)
        print(f"Download completed: {self.data_path}")

    def __getitem__(self, ind):
        vision = self.dataset['vision'][ind]
        audio = self.dataset['audio'][ind]
        text = self.dataset['text'][ind]

        if self.align:
            start = text.nonzero()[0][0]
            vision = vision[start:].astype(np.float32)
            audio = audio[start:].astype(np.float32)
            text = text[start:].astype(np.float32)
        else:
            vision = vision[vision.nonzero()[0][0]:].astype(np.float32)
            audio = audio[audio.nonzero()[0][0]:].astype(np.float32)
            text = text[text.nonzero()[0][0]:].astype(np.float32)

        # z-normalize data
        def z_normalize(x: np.array): # normalize along first axis
            return (x - x.mean(axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)

        if self.z_norm:
            vision = np.nan_to_num(z_normalize(vision))
            audio = np.nan_to_num(z_normalize(audio))
            text = np.nan_to_num(z_normalize(text))

        def _get_class(flag):
            if self.dataset != "humor":
                return [[1]] if flag > 0 else [[0]]
            else:
                return [flag]

        label = self.dataset['labels'][ind]
        label = _get_class(label) if self.task == "classification" else label

        modalities = dict(vision=vision, audio=audio, text=text)
        transforms = dict(vision=self.transform_vision, audio=self.transform_audio, text=self.transform_text)
        X, y = [], label
        for mod in self.modalities:
            if transforms[mod] is not None:
                modalities[mod] = transforms[mod](modalities[mod])
            if self.flatten_time_series:
                modalities[mod] = modalities[mod].flatten()
            X.append(modalities[mod])
        return X, y

    def __len__(self):
        """Get length of dataset."""
        return len(self.dataset['vision'])


def collate_fn_timeseries(inputs: List, max_seq_length: int = None):
    """Handles a list of timeseries data with eventually different lengths.
        Args:
             inputs: list of X where X is a list of modalities
             max_seq_length: if set, pads all timeseries to `max_seq_length`.
                Otherwise, all sequences are padded to the maximum sequence length in this batch.
        Output:
            X_ where X_ is a list of modalities with shape (*, T, p).
            If `max_seq_length` is set then T == `max_seq_length`.
    """
    X_padded = []  # List of padded modalities
    if len(inputs) > 0:
        n_mod = len(inputs[0])
        for i in range(n_mod):
            Xi = [torch.tensor(X[i]) for X in inputs]
            Xi_padded = pad_sequence(Xi, batch_first=True)  # shape (*, T, p)
            if max_seq_length is not None and max_seq_length > Xi_padded.shape[1]:
                Xi_padded = F.pad(Xi_padded, (0, 0, 0, max_seq_length - Xi_padded.shape[1]),
                                  "constant", 0)
            X_padded.append(Xi_padded)
    return X_padded


