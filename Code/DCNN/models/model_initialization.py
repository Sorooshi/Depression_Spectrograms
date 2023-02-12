import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from models.model import DCNN

from config.config import *


def setup_model(dataloaders):
    model = DCNN(output_dim=4).to(DEVICE)

    optimizer_ft = Adam(model.parameters(), lr=LR, weight_decay=0.0002)
    train_loop_params = {
        'model': model,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': optimizer_ft,
        'scheduler': lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5),
        'dataloaders': dataloaders,
        'num_epochs': 20
    }

    return train_loop_params
