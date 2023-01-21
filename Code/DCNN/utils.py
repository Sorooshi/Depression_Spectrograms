from torch.optim import Adam, lr_scheduler
import torch.nn as nn

from itertools import product
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model import train_model, DCNN
from evaluation import predict_ensemble

from config import *


def search(*args, **kwargs):
    criterion = kwargs.pop('criterion')()
    output_dim = 1 if isinstance(criterion, nn.HuberLoss) else 4

    model = DCNN(output_dim).to(DEVICE)
    optimizer_ft = Adam(model.parameters(), **kwargs)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)

    return train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)


def run_grid_serach(grid):
    """
    Sample grid:

    grid = {
        'lr': [1e-2, 1e-3],
        'weight_decay': [0.001, 0.0005, 0.0002],
        'criterion': [nn.CrossEntropyLoss, nn.HuberLoss]
    }
    """
    models = []
    for kwargs in product(*grid.values()):
        print("LR: {:.4f}; weight decay: {:.4f}; criterion: {}".format(kwargs[0], kwargs[1], kwargs[2].__name__))
        model, best_acc = search(**dict(zip(grid.keys(), kwargs)))
        models.append({
            'Model': model,
            'Best Acc': best_acc,
            'LR': kwargs[0],
            'Weight Decay': kwargs[1],
            'Criterion': kwargs[2].__name__
        })

    return models


def plot_confusion(*models):
    y_pred = []
    y_true = []

    # fill true and pred arrays
    for images, meta, labels, depressed in dataloaders[TEST]:
        images, meta, labels, depressed = map(lambda x: x.to(DEVICE),
                                              [images, meta, labels, depressed])

        preds = predict_ensemble((images, meta), *models)
        preds = preds.data.cpu().numpy()
        y_pred.extend(preds)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    classes = (0, 1, 2, 3)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100)
    ax = plt.subplot()
    sn.heatmap(df_cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')