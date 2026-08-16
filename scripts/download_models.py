import os
import argparse
from huggingface_hub import snapshot_download

# Mapping of standard testing models for the Universal Pipeline
MODELS = {
    "fp16": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "int4": "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
    "bitnet": "1bitLLM/bitnet_b1_58-large"
}

def download_model(model_id: str, output_dir: str):
    print(f"📥 Downloading {model_id} into {output_dir}...")
    
    # We only need safetensors/bin weights, configs, and tokenizer files
    allow_patterns = [
        "*.safetensors", "*.bin", "*.json", "*.model", "*.txt", "*.model"
    ]
    
    path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False
    )
    print(f"✅ Successfully downloaded to: {path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tenzo Model Downloader for Universal Engine Testing")
    parser.add_argument("--type", choices=["all", "fp16", "int4", "bitnet"], default="all",
                        help="Type of model to download")
    parser.add_argument("--dir", type=str, default="./models",
                        help="Output directory for models")
    
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    
    if args.type == "all":
        for t, model_id in MODELS.items():
            download_model(model_id, os.path.join(args.dir, t))
    else:
        download_model(MODELS[args.type], os.path.join(args.dir, args.type))

    print("🚀 All requested models are ready for testing.")
