import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

from config import *


df = pd.read_pickle('svm_df.pkl')


# Below we create new dataframes, with each row containig a link to
# audio fragment image of whole audio file
df_train_frag, df_test_frag = train_test_split(
    df,
    test_size=0.2,
    stratify=df['depression.symptoms'],
    random_state=SEED)
df_train_frag, df_val_frag = train_test_split(
    df_train_frag,
    test_size=0.1,
    stratify=df_train_frag['depression.symptoms'],
    random_state=SEED)


def get_patient_fragments(row):
    fragments = []
    for audio in row.audio:
        name_prefix = audio.split('.')[0]
        for filename in os.listdir('spec_images'):
            if filename.startswith(name_prefix):
                fragments.append(filename)
    return fragments


for df_split in [df_train_frag, df_test_frag, df_val_frag]:
    df_split['audio.fragment'] = df_split.apply(get_patient_fragments, axis=1)

# drop redundant columns
drop_cols = ['audio', 'audio.narrative', 'audio.story', 'audio.instruction']
df_train_frag.drop(drop_cols, axis=1, inplace=True)
df_test_frag.drop(drop_cols, axis=1, inplace=True)
df_val_frag.drop(drop_cols, axis=1, inplace=True)

# create copy with patient-fragments correspondence for future inference
df_test_all_fragments = df_test_frag.copy(deep=True)

# create dataframe row for each audio fragment
df_train_frag = df_train_frag.explode('audio.fragment', ignore_index=True)
df_test_frag = df_test_frag.explode('audio.fragment', ignore_index=True)
df_val_frag = df_val_frag.explode('audio.fragment', ignore_index=True)

dataframes = dict(zip([TRAIN, TEST, VAL], [df_train_frag, df_test_frag, df_val_frag]))
for df_label in [TRAIN, TEST, VAL]:
    dataframes[df_label] = dataframes[df_label]

# Perform random oversampling until number of depressed and non-depressed patients is equal
def perform_oversampling(df, column_name):
    ros = RandomOverSampler(random_state=SEED)
    X_resamp, y_resamp = ros.fit_resample(df.drop(column_name, axis=1), df[column_name])
    res = X_resamp.merge(y_resamp, how='left', left_index=True, right_index=True)
    return res

dataframes[TRAIN] = perform_oversampling(df_train_frag, 'depressed')
dataframes[VAL] = perform_oversampling(df_val_frag, 'depressed')


class AudioDataset(Dataset):
    def __init__(self, label_data_df, img_dir, transform=None, target_transform=None):
        self.label_data_df = label_data_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_data_df)

    def __getitem__(self, idx):
        label = self.label_data_df.iloc[idx]['depression.symptoms']
        depressed = self.label_data_df.iloc[idx]['depressed']
        img_name = self.label_data_df.iloc[idx]['audio.fragment']
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            img = self.target_transform(img)
        patient_meta = torch.tensor(self.label_data_df.loc[:,['sex', 'age']].iloc[idx]).type(torch.float32)
        return img, patient_meta, label, depressed


data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Grayscale(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ]),
    TEST: transforms.Compose([
        transforms.Grayscale(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ]),
    VAL: transforms.Compose([
        transforms.Grayscale(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ]),
}

datasets = {
    TRAIN: AudioDataset(dataframes[TRAIN], 'spec_images', transform=data_transforms[TRAIN]),
    TEST: AudioDataset(dataframes[TEST], 'spec_images', transform=data_transforms[TEST]),
    VAL: AudioDataset(dataframes[VAL], 'spec_images', transform=data_transforms[VAL])
}

rng = torch.Generator().manual_seed(SEED)
dataloaders = {
    TRAIN: DataLoader(datasets[TRAIN], TRAIN_BATCH_SIZE, shuffle=True, generator=rng),
    TEST: DataLoader(datasets[TEST], TEST_BATCH_SIZE, shuffle=True, generator=rng),
    VAL: DataLoader(datasets[VAL], VAL_BATCH_SIZE, shuffle=True, generator=rng)
}

dataset_sizes = {
    TRAIN: len(datasets[TRAIN]),
    TEST: len(datasets[TEST]),
    VAL: len(datasets[VAL])
}

