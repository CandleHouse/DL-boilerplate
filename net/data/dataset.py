from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging


class DatasetLoad(Dataset):
    def __init__(self, input_dir, label_dir):
        self.input_dir = input_dir
        self.label_dir = label_dir

        self.input_ids = sorted([splitext(file)[0] for file in listdir(input_dir) if not file.startswith('.')])
        self.label_ids = sorted([splitext(file)[0] for file in listdir(label_dir) if not file.startswith('.')])

        logging.info(f'Creating dataset with {len(self.input_ids)} examples')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        image = np.load(self.input_dir + self.input_ids[i] + '.npy')
        label = np.load(self.label_dir + self.label_ids[i] + '.npy')

        return {
            'image': torch.unsqueeze(torch.from_numpy(image), dim=0),
            'label': torch.unsqueeze(torch.from_numpy(label), dim=0),
        }