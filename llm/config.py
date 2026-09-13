from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = (
    PROJECT_ROOT
    / "training_data"
    / "examples.jsonl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "llm"
    / "business_model.pt"
)

EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

MAX_SEQUENCE_LENGTH = 128
LEARNING_RATE = 3e-4
EPOCHS = 100
import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
