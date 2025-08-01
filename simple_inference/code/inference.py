import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Neural Network (same as training)
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

def model_fn(model_dir):
    """
    Load the PyTorch model from the model_dir.
    This function is called by SageMaker to load the model.
    """
    try:
        logger.info("Loading model from directory: %s", model_dir)
        logger.info("Directory contents: %s", os.listdir(model_dir))
        
        # Load model configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info("Model config loaded: %s", config)
        
        # Create model instance
        model = SimpleNet(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size']
        )
        logger.info("Model instance created")
        
        # Load model weights
        model_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        logger.error("Model directory: %s", model_dir)
        if os.path.exists(model_dir):
            logger.error("Directory contents: %s", os.listdir(model_dir))
        raise e

def input_fn(request_body, request_content_type):
    """
    Deserialize the input data from the request body.
    This function is called by SageMaker to deserialize the input data.
    """
    logger.info("Deserializing the input data")
    
    if request_content_type == "application/json":
        data_dict = json.loads(request_body)
        logger.info("Input data deserialized successfully")
        return data_dict
    else:
        message = f"Unsupported content type: {request_content_type}"
        logger.error(message)
        raise ValueError(message)

def predict_fn(input_data, model):
    """
    Generate predictions using the loaded model.
    This function is called by SageMaker to generate predictions.
    """
    logger.info("Generating prediction")
    
    # Extract features from input data
    if isinstance(input_data, dict) and 'features' in input_data:
        features = input_data['features']
    elif isinstance(input_data, list):
        features = input_data
    else:
        raise ValueError("Input data must contain 'features' key or be a list of features")
    
    # Convert to tensor
    if isinstance(features, list):
        features = torch.tensor(features, dtype=torch.float32)
    else:
        features = torch.tensor([features], dtype=torch.float32)
    
    # Ensure correct shape (batch_size, input_size)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(features)
        prediction = prediction.numpy().tolist()
    
    logger.info(f"Prediction generated: {prediction}")
    return {"predictions": prediction}

def output_fn(prediction, content_type):
    """
    Serialize the prediction result.
    This function is called by SageMaker to serialize the prediction result.
    """
    logger.info("Serializing the prediction result")
    
    if content_type == "application/json":
        logger.info("Returning JSON string")
        return json.dumps(prediction)
    else:
        message = f"Unsupported content type: {content_type}"
        logger.error(message)
        raise ValueError(message) 