from config.config import *

from data_processing.data_processing import create_initial_dataframe, \
    perform_oversampling, \
    process_dataframe, \
    split_data, \
    create_train_test_eval_processors
from models.training import train_model
from models.model_initialization import setup_model
from evaluation.evaluation import eval_model


def single_model_train_workflow(terminate_mode=ModelTrainTerminate.RET, oversample=False):
    # prepare data objects
    df = create_initial_dataframe()
    dataframes = dict(zip([TRAIN, TEST, VAL],
                          list(map(process_dataframe, split_data(df)))))

    if oversample:
        for domain in [TRAIN, VAL]:
            dataframes[domain] = perform_oversampling(dataframes[domain], 'depressed')

    _, dataloaders, _ = create_train_test_eval_processors(dataframes)

    # train model
    train_loop_params = setup_model(dataloaders)
    model_res = train_model(**train_loop_params)

    # execution result
    match terminate_mode:
        case ModelTrainTerminate.SAVE:
            torch.save(model_res.state_dict(), './model_weights.pt')
        case ModelTrainTerminate.EVAL:
            eval_model(model_res, train_loop_params['criterion'], train_loop_params['dataloaders'])
        case ModelTrainTerminate.Both:
            torch.save(model_res.state_dict(), './model_weights.pt')
            eval_model(model_res, train_loop_params['criterion'], train_loop_params['dataloaders'])
        case ModelTrainTerminate.RET:
            return model_res

