import torch


DATASET = "Cora"
DATA_DIR = "./data/"

POS_DIM = 128
HIDDEN_DIM = 128
NUM_HEADS = 4

NUM_LAYERS = 4
LR = 1e-3
KL_BETA = 0.0005
EPOCHS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VIS_PATH = "./visualizations"
TRAIN_VISUALIZATIONS = True

MODEL_PATH = "./model_weights/model.pt"
TRAIN_SAMPLES = 8000
TEST_SAMPLES = 200
VAL_SAMPLES = 1054
EXPERIMENTS = 20

LOAD = False
SAVE = True