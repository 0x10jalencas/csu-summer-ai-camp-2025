import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    Single-layer logistic model:
    outputs one raw logit which callers can pass through sigmoid.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)  # one neuron

    def forward(self, x):
        # Return a 1‑D tensor of logits (batch, )
        return self.fc(x).squeeze(1)


def model_fn(model_dir):
    # Load config
    import json
    import os

    with open(os.path.join(model_dir, "model_config.json")) as f:
        config = json.load(f)

    # Create model instance
    model = SimpleNet(input_size=config["input_size"])

    # Load model weights
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def input_fn(request_body, content_type="application/json"):
    """
    Expect JSON: {"features": [...] }
    Returns a FloatTensor shaped (batch, input_size)
    """
    import json
    import torch

    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    feat = torch.tensor(data["features"], dtype=torch.float32)

    # If caller sent a single row, add batch dim
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    return feat


def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
    return logits


def output_fn(prediction, content_type="application/json"):
    import json
    import torch

    if content_type == "application/json":
        probs = torch.sigmoid(prediction).tolist()
        return json.dumps({"probabilities": probs})
    raise ValueError(f"Unsupported content type: {content_type}")
