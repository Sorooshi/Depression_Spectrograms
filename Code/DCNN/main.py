import argparse

from workflows.single_model import single_model_train_workflow
from workflows.cross_validate_multiple_models import cross_validate_train_workflow
from config.config import ModelTrainTerminate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        choices=['single', 'cv', 'ensemble'],
        required=True,
        help='train single model | train ensemble | cross validate models')
    parser.add_argument(
        '-s',
        choices=list(ModelTrainTerminate),
        default=ModelTrainTerminate.RET,
        type=ModelTrainTerminate,
        help='save or evaluate the model or do both. Only used for single model train mode'
    )

    args = parser.parse_args()
    match args.m:
        case 'single':
            single_model_train_workflow(args.s)
        case 'cv':
            cross_validate_train_workflow(args.s)
        case 'ensemble':
            pass


if __name__ == '__main__':
    main()

