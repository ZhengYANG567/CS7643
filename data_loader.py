import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
from glob import glob
import torchvision.transforms as transforms

class data_loader:
    def __init__(self, feature_path: str, label_path: str):
        self.feature_path = feature_path
        self.label_path = label_path
        self.file_names = [os.path.basename(f) for f in glob(f"{feature_path}/*.npy")]
        self.transform = lambda x: torch.tensor(np.transpose(x.astype(np.float32), (2, 0, 1)), dtype=torch.float32)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int):
        return self.transform(np.load(os.path.join(self.feature_path, self.file_names[idx]))), \
            self.transform(np.load(os.path.join(self.label_path, self.file_names[idx])))

    def check(self):
        s_feature_names = set(self.file_names)
        s_label_names = set([os.path.basename(f) for f in glob(f"{self.label_path}/*.npy")])
        assert(s_feature_names == s_label_names)

    class torch_wrapper:
        def __init__(self, data_loader):
            self.data_loader = data_loader
        def __len__(self):
            return len(self.data_loader)
        def __getitem__(self, idx: int):
            return self.data_loader[idx]

    def get_torch_loader(self, **kwargs):
        loader_args = {
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "batch_size": 32,
        }
        loader_args.update(kwargs)
        # return DataLoader(self.torch_wrapper(self), **loader_args)
        return DataLoader(self, **loader_args)
