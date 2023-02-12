import time
import copy
from tqdm import tqdm
from collections import defaultdict

import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from config import *


LR = 1e-3


class DCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.image_features_ = nn.Sequential(
            nn.Conv2d(1, 32, [7, 1]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d([4, 3], stride=[1, 3]),

            nn.Conv2d(32, 64, [7, 1], stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d([1, 3]),
        )
        self.meta_features_ = nn.Sequential(
            nn.Linear(2, 16)
        )
        self.mixed_features_ = nn.Sequential(
            nn.Linear(33280 + 16, 256),
            nn.Linear(256, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, output_dim),
            # nn.Sigmoid() # for BCELoss
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, images, meta):
        images = self.image_features_(images)
        images = torch.flatten(images, 1) # flatten all dimensions except batch
        meta = self.meta_features_(meta)
        mixed = torch.cat((images, meta), 1)
        mixed = self.mixed_features_(mixed)
        return mixed


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=20):
    since = time.time()

    patience = 3
    trigger_cnt = 0
    prev_val_loss = 100.0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0
    best_acc = 0

    dataset_sizes = dict(map(lambda item: (item[0], len(item[1].dataset)), dataloaders.items()))

    metrics = defaultdict(list)

    batch_num = {
        TRAIN: len(dataloaders[TRAIN]),
        VAL: len(dataloaders[VAL])
    }

    for epoch in tqdm(range(num_epochs)):
        metrics['Epoch'].append(epoch)

        for stage in [TRAIN, VAL]:
            model.train(stage == TRAIN)

            loss_epoch = 0
            acc_epoch = 0

            with torch.set_grad_enabled(stage == TRAIN):
                for i, data in enumerate(dataloaders[stage]):
                    print("\r Batch {}/{}".format(i, batch_num[stage]), end='', flush=True)

                    images, meta, labels, depressed = data

                    # For binary
                    # labels = depressed

                    images = images.to(DEVICE)
                    meta = meta.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(images, meta)

                    # For HuberLoss
                    if isinstance(criterion, nn.HuberLoss):
                        preds = torch.flatten(torch.round(outputs))
                        outputs = torch.flatten(outputs).to(torch.float32)
                        labels = labels.to(torch.float32)

                    # For CrossEntropy
                    if isinstance(criterion, nn.CrossEntropyLoss):
                        preds = torch.argmax(outputs, dim=1)

                    # For BCELoss
                    if isinstance(criterion, nn.BCELoss):
                        preds = torch.flatten(torch.round(outputs))
                        outputs = torch.flatten(outputs).to(torch.float32)
                        labels = labels.to(torch.float32)

                    loss = criterion(outputs, labels)

                    if stage == TRAIN:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    loss_epoch += loss.data.item()
                    acc_epoch += torch.sum(preds == labels.data).item()

                    del images, meta, labels, outputs, preds
                    torch.cuda.empty_cache()

            if stage == TRAIN:
                scheduler.step()

            avg_loss = loss_epoch / dataset_sizes[stage]
            avg_acc = acc_epoch / dataset_sizes[stage]

            if stage == VAL:
                if avg_loss <= best_loss:
                    best_acc = avg_acc
                    best_loss = avg_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if prev_val_loss <= avg_loss:
                    trigger_cnt += 1
                    print('Trigger', trigger_cnt)
                else:
                    trigger_cnt = 0
                    print('Trigger', trigger_cnt)
                prev_val_loss = avg_loss

        if trigger_cnt == patience:
            break

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


dcnn = DCNN(output_dim=4).to(DEVICE)
dcnn2 = DCNN(output_dim=4).to(DEVICE)
dcnn3 = DCNN(output_dim=4).to(DEVICE)

for model in (dcnn, dcnn2, dcnn3):
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = Adam(model.parameters(), lr=LR, weight_decay=0.0002)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)

    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)


