import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch import LongTensor, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


class ImageDataset:
    def __init__(self, dir_name: str, batch_size: int = 16) -> None:
        super().__init__()
        self.dir_name = dir_name
        self.batch_size = batch_size

    def load_data(self, test_size: float = 0.3) -> Tuple[DataLoader, DataLoader]:
        cnt = 0
        train_images, train_labels, test_images, test_labels = [], [], [], []
        for dir_path, _, filenames in os.walk(self.dir_name):
            if len(filenames) == 0:
                continue
            tmp_images, tmp_labels = [], []
            for filename in filenames:
                with open(os.path.join(dir_path, filename), "rb") as pgmf:
                    im = plt.imread(pgmf)
                    tmp_images.append(im.flatten() / 255.0)
                    tmp_labels.append(cnt)
            cnt += 1
            tmp_images, tmp_labels = np.array(tmp_images), np.array(tmp_labels)
            train_ids, test_ids = train_test_split(
                range(len(tmp_images)), test_size=test_size, random_state=12
            )
            train_images += tmp_images[train_ids].tolist()
            train_labels += tmp_labels[train_ids].tolist()
            test_images += tmp_images[test_ids].tolist()
            test_labels += tmp_labels[test_ids].tolist()

        train_dataloader = DataLoader(
            TensorDataset(Tensor(train_images), LongTensor(train_labels)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        test_dataloader = DataLoader(
            TensorDataset(Tensor(test_images), LongTensor(test_labels)),
            batch_size=self.batch_size,
        )
        return train_dataloader, test_dataloader
