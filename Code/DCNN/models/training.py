import time
import copy
from tqdm import tqdm
from collections import defaultdict

import torch.nn as nn

from models.model_initialization import setup_model
from config.config import *


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
                else:
                    trigger_cnt = 0
                prev_val_loss = avg_loss

        if trigger_cnt == patience:
            print(f'Training finished due to early stopping at epoch {epoch}')
            break

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def train_k_fold(data_splits):
    models = []

    for data_split in data_splits.values():
        train_loop_args = setup_model(data_split['dataloaders'])
        model_res = train_model(**train_loop_args)

        models.append(model_res)

    return models