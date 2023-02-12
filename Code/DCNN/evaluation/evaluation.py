import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import os
import time
import numpy as np
from functools import partial, update_wrapper
from sklearn.metrics import accuracy_score, precision_score, f1_score
from imblearn.metrics import specificity_score

from data_processing import dataloaders, dataset_sizes, data_transforms, df_test_all_fragments
from model import dcnn, dcnn2, dcnn3

from config import *


def eval_model(model, criterion):
    """
    Evaluate single model accuracy

    Sample usage:
    eval_model(model, nn.CrossEntropyLoss())
    """
    since = time.time()
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[TEST]):
        print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.eval()

        images, meta, labels, depressed = data
        images = images.to(DEVICE)
        meta = meta.to(DEVICE)
        labels = labels.to(DEVICE)
        depressed = depressed.to(DEVICE)

        with torch.no_grad():
            outputs = model(images, meta)

        # For HuberLoss
        if isinstance(criterion, nn.HuberLoss):
            preds = torch.flatten(torch.round(outputs))
            outputs = torch.flatten(outputs).to(torch.float32)
            labels = labels.to(torch.float32)
            loss = criterion(outputs, labels)

        # For CrossEntropy
        if isinstance(criterion, nn.CrossEntropyLoss):
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

        # For BCELoss
        if isinstance(criterion, nn.BCELoss):
            preds = torch.flatten(torch.round(outputs))
            outputs = torch.flatten(outputs).to(torch.float32)
            depressed = depressed.to(torch.float32)
            loss = criterion(outputs, depressed)

        loss_test += loss.data.item()
        acc_test += torch.sum(preds == labels.data)

        del images, meta, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def maj_vote(outputs):
    """
    Return most common value based on smart majority vote
    """
    counts = np.bincount(outputs)
    # if more than one value is selected by majj vote return mean value
    if np.count_nonzero(counts == np.max(counts)) > 1:
        return np.rint(np.mean(outputs)).astype(int)
    # else return value given by maj vote
    else:
        return np.argmax(counts)


def predict_ensemble(data, *models):
    """
    Make prediction based on ensemble of models
    """
    images, meta = data

    with torch.no_grad():
        all_outputs = np.array([model(images, meta).data.cpu().numpy() for model in models])

    all_preds = np.argmax(all_outputs, axis=2)
    all_preds = np.swapaxes(all_preds, 0, 1)
    groupped_preds = np.apply_along_axis(maj_vote, axis=1, arr=all_preds)
    groupped_preds = torch.flatten(torch.tensor(groupped_preds)).to(DEVICE)

    return groupped_preds


def eval_ensemble(*models, metrics=[]):
    """
    Evaluate metrics for ensemble of models
    """
    assert metrics, "Specify metrics to calculate"

    since = time.time()

    y_true, y_pred = [], []

    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[TEST]):
        print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        for model in models:
            model.eval()

        images, meta, labels, depressed = data
        images, meta = map(lambda x: x.to(DEVICE), [images, meta])

        groupped_preds = predict_ensemble((images, meta), *models).cpu().numpy()

        y_pred.extend(groupped_preds)
        y_true.extend(labels)

        del images, meta, labels, groupped_preds
        torch.cuda.empty_cache()

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    for metric in metrics:
        print("{}: {:.4f}".format(metric.__name__, metric(y_true, y_pred)))
    print('-' * 10)


class FragmentAudioDataset(Dataset):
    def __init__(self, label_data_df, img_dir, transform=None, target_transform=None):
        self.label_data_df = label_data_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_data_df)

    def __getitem__(self, idx):
        """
        Return all fragments for each participant
        """
        label = self.label_data_df.iloc[idx]['depression.symptoms']
        depressed = self.label_data_df.iloc[idx]['depressed']
        frag_names = self.label_data_df.iloc[idx]['audio.fragment']

        # create array with data for all fragments [(img, meta), ,(...)]
        frag_data = []
        for frag_name in frag_names:
            frag_path = os.path.join(self.img_dir, frag_name)
            frag_img = read_image(frag_path)
            if self.transform:
                frag_img = self.transform(frag_img)
            frag_meta = torch.tensor(self.label_data_df.loc[:, ['sex', 'age']].iloc[idx]).type(torch.float32)
            frag_data.append((frag_img, frag_meta))

        if self.target_transform:
            label = self.target_transform(label)

        return frag_data, label, depressed


all_fragments_test_dataset = FragmentAudioDataset(
    df_test_all_fragments,
    'spec_images',
    transform=data_transforms[TEST]
)

all_fragments_test_dataloader = DataLoader(
    all_fragments_test_dataset,
    batch_size=1,   # setting to value other than 1 will lead to an error
    batch_sampler=None
)


def eval_ensemble_all_fragments(*models):
    """
    Evaluate accuracy for ensemble of models in predicting based on all patient's
    audio fragments

    Sample usage:
    eval_ensemble_all_fragments(dcnn, dcnn2, dcnn3)
    """
    since = time.time()
    acc_test = 0

    test_batches = len(all_fragments_test_dataloader)
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(all_fragments_test_dataloader):
        print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        for model in models:
            model.eval()

        frag_data, label, depressed = data
        label, depressed = map(lambda x: x.to(DEVICE), [label, depressed])

        frag_preds = []
        for frag_img, frag_meta in frag_data:
            frag_img, frag_meta = map(lambda x: x.to(DEVICE), [frag_img, frag_meta])
            frag_pred = predict_ensemble((frag_img, frag_meta), *models)
            frag_preds.append(frag_pred.item())
        audio_pred = torch.tensor(maj_vote(frag_preds)).to(DEVICE)

        acc_test += torch.sum(audio_pred == label.data)

        del frag_img, frag_meta, label, audio_pred
        torch.cuda.empty_cache()

    avg_acc = acc_test / test_batches

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def wrapped_partial(func, *args, **kwargs):
    """
    Helps initialize function's parameters and keep dunder attributes
    when passed to subsequent functions
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


accuracy = wrapped_partial(accuracy_score)
precision = wrapped_partial(precision_score, average='weighted')
f1 = wrapped_partial(f1_score, average='weighted')
specificity = wrapped_partial(specificity_score, average='weighted')

eval_ensemble(dcnn, dcnn2, dcnn3, metrics=[accuracy, precision, f1, specificity])
