
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import densenet121


# Select the available device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Path to the trained model checkpoint
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "best_model.pth"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    # Recreate the same DenseNet121 architecture used during training
    model = densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features, 2
    )

    # Load the trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


model = load_model()
