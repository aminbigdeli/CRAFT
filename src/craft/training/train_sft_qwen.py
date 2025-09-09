"""
Simplified Swift training script using command line approach
"""

import os
import subprocess
import json


model = "unsloth/QwQ-32B-unsloth-bnb-4bit"
model_name = "QwQ32B"

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

def run_swift_training():
    """Run Swift training using command line interface with multi-GPU for speed"""
    
    # Configuration
    train_dataset_path = "./data/msmarco_boost_10_train.json"
    output_dir = f"./models/{model_name}"
    
    # Set environment variables for HuggingFace cache
    import os
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NPROC_PER_NODE'] = '4'
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1,2,3'
    
    # Setup Weights & Biases
    try:
        import wandb
        wandb.login(key="<your wandb key>")
        print("✅ Wandb login successful")
    except ImportError:
        print("⚠️  Wandb not installed. Install with: pip install wandb")
    except Exception as e:
        print(f"⚠️  Wandb login failed: {e}")
    
    # Validate dataset
    if not validate_dataset(train_dataset_path):
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the swift sft command with optimal settings for 2500 tokens
    cmd = [
        "swift", "sft",
        "--model", model,
        "--model_type", "qwq", 
        "--dataset", train_dataset_path,
        "--split_dataset_ratio", "0.02",
        "--output_dir", output_dir,
        "--num_train_epochs", "4",
        "--max_length", "1450",
        "--max_new_tokens", "128",
        "--per_device_train_batch_size", "6",
        "--gradient_accumulation_steps", "2",
        "--learning_rate", "5e-5",
        "--warmup_ratio", "0.05",
        "--save_steps", "100",
        "--eval_steps", "100",
        "--per_device_eval_batch_size", "4",
        "--logging_steps", "5",
        "--dataloader_num_workers", "1",
        "--dataset_num_proc", "4",
        "--bf16", "true",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--seed", "3407",
        "--deepspeed", "zero2_offload",
        "--eval_generation_config", '{"max_new_tokens": 128, "max_length": 1450}',
        "--report_to", "wandb",
    ]
    
    print("=== Starting Swift Training ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Dataset: {train_dataset_path}")
    print(f"Output: {output_dir}")
    
    try:
        # Run the training command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("=== Training completed successfully! ===")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False

if __name__ == "__main__":
    import torch
    
    print("=== Environment Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # Run training
    success = run_swift_training()
    
    if success:
        print(f"Training completed! Check ./models/{model_name}/ for results")
    else:
        print("Training failed. Check the error messages above.")
