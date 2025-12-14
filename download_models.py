"""
Download model files from Hugging Face at application startup
This prevents including large model files in the Git repository
"""
from huggingface_hub import hf_hub_download
import os

def download_models():
    """Download all required models from Hugging Face"""
    
    # Repository details
    REPO_ID = "gangurde/cattle_disease_model"
    
    # Models to download
    models = [
        "lumpy_disease_best.pt",
        "cat_disease_best.pt",
        "dog_disease_best.pt",
        "sheep_disease_model.pt"
    ]
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    for model_name in models:
        model_path = os.path.join("models", model_name)
        
        # Skip if model already exists
        if os.path.exists(model_path):
            print(f"✅ {model_name} already exists")
            continue
        
        try:
            print(f"📥 Downloading {model_name}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"models/{model_name}",
                local_dir=".",
                local_dir_use_symlinks=False
            )
            print(f"✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not download {model_name}: {str(e)}")
            print(f"   Please ensure the model is uploaded to {REPO_ID}")

if __name__ == "__main__":
    download_models()