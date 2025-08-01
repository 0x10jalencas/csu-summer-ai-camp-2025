import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import os
import shutil
from dotenv import load_dotenv
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

load_dotenv()
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"


def create_model_archive():
    """Create a tar.gz file containing the model artifacts for SageMaker"""
    print("Creating model artifacts...")
    # Create a temporary directory for the model artifacts
    model_artifacts_dir = 'model_artifacts'
    
    # Remove existing model_artifacts directory if it exists
    if os.path.exists(model_artifacts_dir):
        shutil.rmtree(model_artifacts_dir)
    
    os.makedirs(model_artifacts_dir, exist_ok=True)
    
    # Copy the inference code to the root of model artifacts
    shutil.copytree('simple_inference/code', os.path.join(model_artifacts_dir, 'code'))
    
    # Copy the trained model files to the root
    if os.path.exists('./model'):
        # Copy individual model files to root instead of nested directory
        for file in os.listdir('./model'):
            src_file = os.path.join('./model', file)
            dst_file = os.path.join(model_artifacts_dir, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
    else:
        print("Warning: Model directory not found. Please run 'python simple_model.py' first.")
        return None
    
    # Create tar.gz file
    import tarfile

    tar_filename = 'model_artifacts.tar.gz'
    with tarfile.open(tar_filename, "w:gz") as tar:
        tar.add(model_artifacts_dir, arcname='.')

    # Clean up temporary directory
    #shutil.rmtree(model_artifacts_dir)

    print(f"Model artifacts created: {tar_filename}")
    return tar_filename

def deploy_to_sagemaker(model_artifacts_path, role_arn=None):
    """Deploy the model to SageMaker"""
    print("Deploying model to SageMaker...")
    # Initialize SageMaker session
    session = sagemaker.Session()
    
    # Get the default role if not provided
    if role_arn is None:
        try:
            role_arn = get_execution_role()
        except:
            print("Error: Could not get execution role. Please provide a valid IAM role ARN.")
            return None
    
    print(f"Using IAM role: {role_arn}")
    
    # Upload model artifacts to S3
    model_data = session.upload_data(
        path=model_artifacts_path,
        bucket=session.default_bucket(),
        key_prefix='simple-pytorch-model'
    )
    
    print(f"Model artifacts uploaded to S3: {model_data}")
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role_arn,
        entry_point='inference.py',
        framework_version='1.9.0',
        py_version='py38'
    )
    
    # Deploy the model
    print("Deploying model to SageMaker...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        timeout=300
    )
    
    print(f"Model deployed successfully!")
    print(f"Endpoint name: {predictor.endpoint_name}")
    
    return predictor

def test_endpoint(predictor):
    """Test the deployed endpoint"""
    # Sample input data
    sample_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()
    
    # Create input data
    input_data = {
        "features": sample_features
    }

    
    print("\nTesting deployed endpoint...")
    print(f"Input: {input_data}")
    
    # Make prediction
    prediction = predictor.predict(input_data)
    print(f"Prediction: {prediction}")
    
    return prediction

def main():
    """Main function to train, package, and deploy the model"""
    
    print("Simple PyTorch Model - SageMaker Deployment")
    print("=" * 50)
    
    # Step 1: Train the model (if not already trained)
    if not os.path.exists('./model/model.pth'):
        print("\n1. Training model...")
        from simple_model import train_model
        train_model()
    else:
        print("\n1. Model already exists, skipping training...")
    
    # Step 2: Create model artifacts
    print("\n2. Creating model artifacts...")
    model_artifacts_path = create_model_archive()
    if model_artifacts_path is None:
        print("Failed to create model artifacts. Exiting.")
        return
    
    # Step 3: Deploy to SageMaker
    print("\n3. Deploying to SageMaker...")
    print("Note: This requires AWS credentials and appropriate permissions.")
    print("Make sure you have:")
    print("- AWS credentials configured")
    print("- SageMaker permissions")
    print("- S3 permissions")
    
    # Uncomment the following lines to actually deploy
    predictor = deploy_to_sagemaker(model_artifacts_path, role_arn="arn:aws:iam::202712152316:role/sagemaker-deployment")
    if predictor:
        print("Testing endpoint...")
        test_endpoint(predictor)

    print("\nDeployment script ready!")
    print("To deploy, uncomment the deployment lines in the script.")

if __name__ == "__main__":
    main() 