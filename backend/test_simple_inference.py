from simple_inference.code.inference import input_fn, output_fn, model_fn, predict_fn
import json
import numpy as np

# Test the simple model
def test_simple_model():
    # Sample input data (10 features)
    sample_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Convert to JSON string
    sample_json_string = json.dumps({
        "features": sample_features
    })
    
    model_dir = './model'
    
    print("Testing Simple PyTorch Model for SageMaker Deployment")
    print("=" * 50)
    
    # Test input_fn
    print("\n1. Testing input_fn...")
    deserialized_data = input_fn(sample_json_string, 'application/json')
    print(f"Deserialized Data: {deserialized_data}")
    
    # Test model_fn
    print("\n2. Testing model_fn...")
    try:
        model = model_fn(model_dir)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to run 'python simple_model.py' first to train and save the model")
        return
    
    # Test predict_fn
    print("\n3. Testing predict_fn...")
    prediction = predict_fn(deserialized_data, model)
    print(f"Prediction: {prediction}")
    
    # Test output_fn
    print("\n4. Testing output_fn...")
    serialized_data = output_fn(prediction, 'application/json')
    print(f"Serialized Data: {serialized_data}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("The model is ready for SageMaker deployment.")

if __name__ == "__main__":
    test_simple_model() 