from DCNN.config import *

from DCNN.data_processing import create_initial_dataframe, \
    task_mapping, \
    get_domain_audio, \
    process_dataframe, \
    split_data, \
    create_train_test_eval_processors
from DCNN.model import DCNN, train_model

import torch.nn as nn
from torch.optim import Adam, lr_scheduler


def single_model_train_workflow():
    df = create_initial_dataframe()
    for domain in task_mapping:
        df[f'audio.{domain}'] = df.apply(get_domain_audio, axis=1, domain=domain)
    dataframes = dict(zip([TRAIN, TEST, VAL],
                          list(map(process_dataframe, split_data(df)))))

    # dataframes[TRAIN] = perform_oversampling(dataframes[TRAIN], 'depressed')
    # dataframes[VAL] = perform_oversampling(dataframes[VAL], 'depressed')

    _, dataloaders, _ = create_train_test_eval_processors(dataframes)

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

    return train_model(**train_loop_params)
