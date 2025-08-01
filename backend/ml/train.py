"""
train.py  – SageMaker entry-point
Trains the SimpleNet (single-neuron sigmoid) on the 22 encoded features.

Run by SageMaker like:
    python train.py --epochs 20 --lr 1e-3 --batch_size 32 \
        --train /opt/ml/input/data/train/survey_22f_clean.csv
"""

import argparse, json, os, logging, io
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# --- logging ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- model -----------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)  # raw logit
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)  # (batch,)

# --- dataset ---------------------------------------------------------------
class CsvDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)
    def __len__(self):            return len(self.y)
    def __getitem__(self, idx):   return self.X[idx], self.y[idx]

# --- training loop ---------------------------------------------------------
def train_loop(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)

# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    # ensure args.train points to the CSV file
    if os.path.isdir(args.train):
        # SageMaker passes a *directory* containing the file
        files = [f for f in os.listdir(args.train) if f.endswith(".csv")]
        assert files, "no csv found in train channel"
        args.train = os.path.join(args.train, files[0])

    logger.info("loading %s", args.train)
    ds = CsvDataset(args.train)
    train_len = int(0.8 * len(ds))
    val_len = len(ds) - train_len
    ds_train, ds_val = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimpleNet(ds.X.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        loss = train_loop(model, train_loader, opt, loss_fn, device)
        logger.info("epoch %02d  loss %.4f", epoch+1, loss)

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

    with open(os.path.join(args.model_dir, "model_config.json"), "w") as f:
        json.dump({"input_size": ds.X.shape[1],
                   "hidden_size": 64,
                   "output_size": 1}, f)

    logger.info("saved model to %s", args.model_dir)

if __name__ == "__main__":
    main()