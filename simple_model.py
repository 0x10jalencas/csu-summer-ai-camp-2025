import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd   # <-- add this import
import numpy as np
from pathlib import Path
import sys

# ensure we can import ml/preprocess.py regardless of current working dir
ML_DIR = Path(__file__).resolve().parent / "ml"
sys.path.append(str(ML_DIR))
from preprocess import Encoder  # maps raw categorical -> numeric


class SimpleNet(nn.Module):
    """
    One-layer logistic model: outputs one raw logit.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)  # (batch,) raw logits


class CsvDataset(Dataset):
    """
    Loads a cleaned CSV produced by Encoder.
    Assumes:
      • all 22 feature columns come first
      • last column 'label' is the binary target (0 = ATTENDING, 1 = NOT ENROLLED)
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        label_col = "label"
        # drop rows without a label and coerce to float
        df = df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(float)
        # use shared Encoder to convert categorical strings to numeric codes
        # feature_schema.json lives in project_root/shared
        schema_path = Path(__file__).resolve().parent / "shared/feature_schema.json"
        enc = Encoder(schema_path)
        # apply shared Encoder row‑wise
        numeric_rows = df.apply(lambda row: enc.transform_row(row.to_dict()), axis=1).to_list()
        features_np = np.vstack(numeric_rows)
        # replace any nan that slipped through with 0.0
        features_np = np.nan_to_num(features_np, nan=0.0)
        self.X = torch.tensor(features_np, dtype=torch.float32)
        self.y = torch.tensor(df[label_col].values,
                              dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(csv_path: str, model_dir: str = "./model"):
    """Train the simple model and save it"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    input_size = 22          # encoder output length
    model = SimpleNet(input_size=input_size)
    model.to(device)

    # Create dataset and dataloader
    dataset = CsvDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save model configuration
    config = {
        "input_size": input_size,
        "model_type": "sigmoid_neuron"
    }

    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)

    print(f"Model configuration saved to {config_path}")

    return model

if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=False,
                        default="data/survey_22f_clean.csv",
                        help="Local path to cleaned 22‑feature CSV")
    parser.add_argument("--model-dir", default="./model")
    args = parser.parse_args()

    pathlib.Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    train_model(csv_path=args.csv_path, model_dir=args.model_dir)