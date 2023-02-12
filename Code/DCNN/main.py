import argparse
from enum import Enum

from workflows.single_model import single_model_train_workflow


class ModelTrainTerminate(Enum):
    SAVE = 'save'
    EVAL = 'eval'
    BOTH = 'both'


def setup


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
        default=ModelTrainTerminate.BOTH,
        type=ModelTrainTerminate,
        help='save or evaluate the model or do both. Only used for single model train mode'
    )

    args = parser.parse_args()
    match args.m:
        case 'single':
            print(args.s)
            single_model_train_workflow(args.s)
        case 'cv':
            pass
        case 'ensemble':
            pass


if __name__ == '__main__':
    main()

