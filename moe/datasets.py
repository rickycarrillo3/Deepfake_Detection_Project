import os

import pandas as pd

import torch

from PIL import Image

from torch.utils.data import Dataset

import config 

class ExpertDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_root,
        transform=None
    ):

        self.df = pd.read_csv(csv_file)

        self.image_root = image_root

        self.transform = transform

        self.num_experts = len(self.df.columns) - 2

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.image_root,
            row.iloc[0]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            row.iloc[1],
            dtype=torch.float32
        )

        #expert_scores = torch.tensor(
        #    row.iloc[2:].values.astype("float32")
        #)
        
        expert_scores = torch.tensor(
            row[config.EXPERTS].values.astype("float32")
        )

        return (
            image,
            expert_scores,
            label
        )