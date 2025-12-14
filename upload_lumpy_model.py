from huggingface_hub import login, upload_file
import os

def upload_lumpy_model():
    """
    Upload the lumpy disease model to Hugging Face Hub
    """
    try:
        # Login with your Hugging Face credentials
        print("Logging in to Hugging Face...")
        login()
        
        # Define the model file path
        model_file_path = os.path.join("models", "lumpy_disease_best.pt")
        
        # Check if the model file exists
        if not os.path.exists(model_file_path):
            print(f"Error: Model file not found at {model_file_path}")
            return
        
        print(f"Uploading {model_file_path} to Hugging Face...")
        
        # Upload the specific model file
        upload_file(
            path_or_fileobj=model_file_path,
            path_in_repo="lumpy_disease_best.pt",  # Name in the repository
            repo_id="gangurde/cattle_disease_model",
            repo_type="model",
            commit_message="Upload lumpy disease detection model"
        )
        
        print("✅ Model uploaded successfully!")
        print("You can view your model at: https://huggingface.co/gangurde/cattle_disease_model")
        
    except Exception as e:
        print(f"❌ Error uploading model: {str(e)}")

if __name__ == "__main__":
    upload_lumpy_model()