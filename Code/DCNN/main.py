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
        help='save | eval | both | ret. Save or evaluate the model or do both, ret used to pass the model further. '
             'Training types other than single only support ret and eval'
    )

    args = parser.parse_args()
    if args.m != 'single':
        assert args.s in [ModelTrainTerminate.RET, ModelTrainTerminate.EVAL], \
            "Only ret or eval terminations can be used for this training type"

    match args.m:
        case 'single':
            single_model_train_workflow(args.s)
        case 'cv':
            cross_validate_train_workflow(args.s)
        case 'ensemble':
            pass


if __name__ == '__main__':
    main()

