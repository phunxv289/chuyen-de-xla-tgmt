"""
    Data loader and utilities for Age gender dataset
    For more information, please refer to
"""
import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from sources.config import *


class AgeGenderDataset(Dataset):
    def __init__(self, img_dataset, preprocess):
        self.preprocess = preprocess
        self.img_folder = img_folder
        self.label_df = self.load_data_file(label_file)
        self.img_ids = self.label_df['_id'].tolist()
        self.age_labels = self.label_df['age_label'].tolist()
        self.gender_labels = self.label_df['gender_label'].tolist()
        self.data_size = len(self.label_df)

    def load_data_file(self, label_file):
        df = pd.read_csv(label_file)
        df = df[df['gender_from_fb'].isin(['male', 'female'])]
        df['age_label'] = np.asarray(pd.cut(df['age'], AGE_BINS, labels=AGE_LABELS))
        df['age_label'] = df['age_label'].astype(int)
        df['gender_label'] = df['gender_from_fb'].map(GENDER_MAPPER)
        df['gender_label'] = df['gender_label'].astype(int)
        return df

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_file = os.path.join(self.img_folder, '{}.jpg'.format(img_id))
        img = Image.open(img_file)
        img = self.preprocess(img)
        age_label = self.age_labels[idx]
        gender_label = self.gender_labels[idx]

        return img, age_label, gender_label


if __name__ == '__main__':
    img_folder = r'C:\Users\phunx\Downloads\age_gender_fb'
    label_file = r'../resources/dataset/age_gender_train.csv'
    train_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_set = AgeGenderDataset(img_folder, label_file, train_preprocess)
    for _ in data_set:
        print()

    # data_loader = DataLoader(data_set, batch_size=32, shuffle=True)
    #
    # for _img, age_label, gender_label in data_loader:
    #     print()
