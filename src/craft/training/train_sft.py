"""
Unified Swift training script for both R1 and Qwen models
"""

import os
import subprocess
import json
import argparse


def get_model_config(model_type):
    """Get model configuration based on model type"""
    configs = {
        "r1": {
            "model": "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
            "model_name": "R1Llama70B"
        },
        "qwen": {
            "model": "unsloth/QwQ-32B-unsloth-bnb-4bit", 
            "model_name": "QwQ32B"
        }
    }
    
    if model_type.lower() not in configs:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(configs.keys())}")
    
    return configs[model_type.lower()]


def validate_dataset(dataset_path):
    """Validate Alpaca format dataset"""
    print(f"=== Validating Dataset: {dataset_path} ===")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: Dataset should be a JSON array")
            return False
        
        if not data:
            print("Error: Dataset is empty")
            return False
        
        # Check format of first few examples
        required_keys = ['instruction', 'input', 'output']
        for i, example in enumerate(data[:3]):
            missing_keys = [key for key in required_keys if key not in example]
            if missing_keys:
                print(f"Warning: Example {i} missing keys: {missing_keys}")
        
        print(f"Dataset validated: {len(data)} examples")
        return True
        
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


def run_swift_training(model_type, dataset_path=None):
    """Run Swift training using command line interface with multi-GPU for speed"""
    
    # Get model configuration
    config = get_model_config(model_type)
    model = config["model"]
    model_name = config["model_name"]
    
    # Configuration
    if dataset_path is None:
        dataset_path = f"data/sft_data/{model_name.lower()}-sft.json"
    
    output_dir = f"outputs/{model_name.lower()}-sft"
    
    print(f"=== Training {model_name} Model ===")
    print(f"Model: {model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Validate dataset first
    if not validate_dataset(dataset_path):
        print("Dataset validation failed. Exiting.")
        return False
    
    # Build the swift sft command with optimal settings for 2500 tokens
    cmd = [
        "swift", "sft",
        "--model_type", "llama3",
        "--model_id", model,
        "--dataset", dataset_path,
        "--output_dir", output_dir,
        "--num_train_epochs", "3",
        "--max_samples", "100000",
        "--learning_rate", "2e-5",
        "--batch_size", "4",
        "--gradient_accumulation_steps", "8",
        "--max_new_tokens", "128",
        "--warmup_ratio", "0.1",
        "--save_steps", "500",
        "--eval_steps", "500",
        "--logging_steps", "100",
        "--save_total_limit", "3",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--gradient_checkpointing", "true",
        "--deepspeed", "zero2",
        "--eval_generation_config", '{"max_new_tokens": 128, "max_length": 1450}',
        "--template_type", "llama3",
        "--sft_type", "lora"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the training command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print(f"Output directory: {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Unified Swift Training Script')
    parser.add_argument('--model', type=str, choices=['r1', 'qwen'], required=True,
                       help='Model type to train (r1 or qwen)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to training dataset (optional, will use default if not provided)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset without training')
    
    args = parser.parse_args()
    
    print(f"Starting training for {args.model.upper()} model...")
    
    # Get model configuration
    config = get_model_config(args.model)
    dataset_path = args.dataset or f"data/sft_data/{config['model_name'].lower()}-sft.json"
    
    if args.validate_only:
        validate_dataset(dataset_path)
    else:
        success = run_swift_training(args.model, args.dataset)
        if success:
            print(f"Training completed successfully for {args.model.upper()} model!")
        else:
            print(f"Training failed for {args.model.upper()} model!")
            exit(1)


if __name__ == "__main__":
    main()
