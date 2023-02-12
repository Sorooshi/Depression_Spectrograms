from data_processing.data_processing import \
    create_initial_dataframe, \
    k_fold_split
from models.training import train_k_fold
from evaluation.evaluation import eval_average, cross_val_metrics
from config.config import ModelTrainTerminate


def cross_validate_train_workflow(terminate_mode):
    # prepare data objects
    df = create_initial_dataframe()
    data_splits = k_fold_split(df)

    # train multiple models
    models = train_k_fold(data_splits)

    # execution result
    match terminate_mode:
        case ModelTrainTerminate.EVAL:
            dataloaders = [split['dataloaders'] for split in data_splits.values()]
            eval_average(models, dataloaders, cross_val_metrics())
        case ModelTrainTerminate.RET:
            return models
        case _:
            raise ValueError("This termination mode can't be used with cross validation. Use either 'ret' or 'eval'")


