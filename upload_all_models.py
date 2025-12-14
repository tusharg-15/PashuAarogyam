from huggingface_hub import login, upload_file
import os

def upload_all_models():
    """
    Upload all disease detection models to Hugging Face Hub
    """
    try:
        # Login with your Hugging Face credentials
        print("Logging in to Hugging Face...")
        login()
        
        # Define all model files in the models directory
        model_files = [
            "lumpy_disease_best.pt",
            "cat_disease_best.pt", 
            "dog_disease_best.pt",
            "sheep_disease_model.pt"
        ]
        
        models_dir = "models"
        
        print(f"Uploading models from {models_dir} directory...")
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            
            # Check if the model file exists
            if os.path.exists(model_path):
                print(f"📤 Uploading {model_file}...")
                
                upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=f"models/{model_file}",  # Keep in models folder structure
                    repo_id="gangurde/cattle_disease_model",
                    repo_type="model",
                    commit_message=f"Upload {model_file.replace('_', ' ').replace('.pt', '')} model"
                )
                
                print(f"✅ {model_file} uploaded successfully!")
            else:
                print(f"⚠️  Warning: {model_file} not found in {models_dir}")
        
        print("\n🎉 All available models uploaded successfully!")
        print("You can view your models at: https://huggingface.co/gangurde/cattle_disease_model")
        
    except Exception as e:
        print(f"❌ Error uploading models: {str(e)}")

if __name__ == "__main__":
    upload_all_models()