import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

from config.config import *


def get_patient_audio(row, data_folder=os.path.join(DATA_PATH, 'wav files')):
    """
    Find patient's recordings in recordings folder
    """
    key = row.ID
    audio_files = []
    for filename in os.listdir(data_folder):
        if filename.find(key) != -1:
            audio_files.append(filename)
    return audio_files


def create_initial_dataframe():
    # participants_info
    participants = pd.read_excel(
        os.path.join(DATA_PATH, 'PsychiatricDiscourse_participant.data.xlsx')
    )

    depression_only = participants.loc[
        (participants['thought.disorder.symptoms'] == 0.) &
        (participants['depression.symptoms'] != 0.)
    ]
    control_group = participants.loc[
        (participants['depression.symptoms'] == 0.) &
        (participants['thought.disorder.symptoms'] == 0.)
    ]

    df = pd.concat([depression_only, control_group])
    df.drop(['education.level', 'diagnosis', 'thought.disorder.symptoms', 'group'], axis=1, inplace=True)
    df.sex.replace(['female', 'male'], [0, 1], inplace=True)
    df.age.fillna(df.age.mean(), inplace=True)
    df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

    df['audio'] = df.apply(get_patient_audio, axis=1)

    # exclude patients with no recordings
    df = df[df.audio.apply(len) > 0]

    df.reset_index(drop=True, inplace=True)
    df['depressed'] = pd.Series(df['depression.symptoms'] != 0).astype(int)

    return df


task_mapping = {
    'narrative': ['sportsman', 'adventure', 'winterday'],
    'story': ['present', 'trip', 'party'],
    'instruction': ['chair', 'table', 'bench']
}


def get_domain_audio(row, domain):
    """
    Usage: df['domain_name'] = df.apply(get_domain_audio, axis=1, domain=domain)
    """
    files = []
    for topic in task_mapping[domain]:
        for file_name in row.audio:
            if file_name.find(topic) != -1:
                files.append(file_name)

    return files[0] if len(files) else None


def get_patient_fragments(row):
    fragments = []
    for audio in row.audio:
        name_prefix = audio.split('.')[0]
        for filename in os.listdir(IMAGES_PATH):
            if filename.startswith(name_prefix):
                fragments.append(filename)
    return fragments


def process_dataframe(df):
    # get audio fragments
    df['audio.fragment'] = df.apply(get_patient_fragments, axis=1)

    # drop redundant columns
    drop_cols = ['audio']
    df = df.drop(drop_cols, axis=1)

    # create dataframe row for each audio fragment
    df = df.explode('audio.fragment', ignore_index=True)

    return df


def split_data(df):
    """
    Usage: df_train, df_test, df_val = split_data(df)
    """
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df['depression.symptoms'],
        random_state=SEED)
    df_train, df_val = train_test_split(
        df_train,
        test_size=0.1,
        stratify=df_train['depression.symptoms'],
        random_state=SEED)

    return df_train, df_test, df_val


# Perform random oversampling until number of depressed and non-depressed patients is equal
def perform_oversampling(df, column_name):
    """
    Usage: df = perform_oversampling(df, 'depressed')
    """
    ros = RandomOverSampler(random_state=SEED)
    X_resamp, y_resamp = ros.fit_resample(df.drop(column_name, axis=1), df[column_name])
    res = X_resamp.merge(y_resamp, how='left', left_index=True, right_index=True)
    return res


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
        try:
            img_path = os.path.join(self.img_dir, img_name)
        except:
            print(img_name)
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            img = self.target_transform(img)
        patient_meta = torch.tensor(self.label_data_df.loc[:,['sex', 'age']].iloc[idx]).type(torch.float32)
        return img, patient_meta, label, depressed


def create_train_test_eval_processors(dataframes, resize=(80, 80)):
    """
    Usage: datasets, dataloaders, dataset_sizes = create_train_test_eval_processors(dataframes)
    """
    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.Grayscale(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(resize),
        ]),
        TEST: transforms.Compose([
            transforms.Grayscale(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(resize),
        ]),
        VAL: transforms.Compose([
            transforms.Grayscale(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(resize),
        ]),
    }

    datasets = {
        TRAIN: AudioDataset(dataframes[TRAIN], IMAGES_PATH, transform=data_transforms[TRAIN]),
        TEST: AudioDataset(dataframes[TEST], IMAGES_PATH, transform=data_transforms[TEST]),
        VAL: AudioDataset(dataframes[VAL], IMAGES_PATH, transform=data_transforms[VAL])
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

    return datasets, dataloaders, dataset_sizes


def k_fold_split(df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    folds = defaultdict(dict)

    for i, (train_index, test_index) in enumerate(kf.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            stratify=train_df['depression.symptoms'],
            random_state=SEED)

        dataframes = dict(zip([TRAIN, TEST, VAL],
                              map(process_dataframe, [train_df, test_df, val_df])))
        datasets, dataloaders, dataset_sizes = create_train_test_eval_processors(dataframes)

        folds[i]['datasets'] = datasets
        folds[i]['dataloaders'] = dataloaders
        folds[i]['dataset_sizes'] = dataset_sizes

    return folds
