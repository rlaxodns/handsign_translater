import torch
import os
import pandas as pd
import glob

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, window_size, Folder_dir):
        self.data_list = []

        for filename in glob.glob(f'{Folder_dir}/**/*.csv', recursive=True):
            print(filename)

            data = pd.read_csv(filename, header = None).values

            features = torch.FloatTensor(data[:, 0:224])

            labels = torch.FloatTensor(data[:, 224])
            labels = labels.reshape(len(labels), -1)

            for i in range(len(features) - window_size):
                features_subset = features[i : i + window_size]

                labels_subset = labels[i]

                self.data_list.append([features_subset, labels_subset])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        return x, y