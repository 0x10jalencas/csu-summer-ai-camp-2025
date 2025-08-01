import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader

# Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Custom Dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=10):
        self.num_samples = num_samples
        self.input_size = input_size
        
        # Generate synthetic data
        self.X = torch.randn(num_samples, input_size)
        # Create a simple target: sum of first 3 features + noise
        self.y = torch.sum(self.X[:, :3], dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model_dir='./model'):
    """Train the simple model and save it"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleNet(input_size=10, hidden_size=64, output_size=1)
    model.to(device)
    
    # Create dataset and dataloader
    dataset = SimpleDataset(num_samples=1000, input_size=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
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
        'input_size': 10,
        'hidden_size': 64,
        'output_size': 1,
        'model_type': 'SimpleNet'
    }
    
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    print(f"Model configuration saved to {config_path}")
    
    return model

if __name__ == "__main__":
    train_model() 