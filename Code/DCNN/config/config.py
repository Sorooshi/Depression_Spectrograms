import torch

TRAIN, TEST, VAL = 'Train', 'Test', 'Val'
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 16
VAL_BATCH_SIZE = 64

LR = 1e-3

DATA_PATH = '/Users/koldi/se/DepressionPrediction/Psychiatric-Disorders-Data/Notebooks/psychiatric.disorders.ML'
IMAGES_PATH = '/Users/koldi/se/DepressionPrediction/Depression_Spectrograms/Notebooks/spec_images_clean_6s_80freq'