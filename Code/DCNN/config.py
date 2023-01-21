import torch

TRAIN, TEST, VAL = 'Train', 'Test', 'Val'
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 16
VAL_BATCH_SIZE = 64