import torch

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-4
LOAD_MODEL = True
SAVE_MODEL = True
NUM_WORKERS = 8
PIN_MEMORY =2 
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
LAMBDA1=100