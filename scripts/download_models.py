#!/usr/bin/env python3
"""
Model Download Script
=====================

This script downloads pre-trained CRAFT models from HuggingFace.

Usage:
    python scripts/download_models.py --model craft_qwen3 --output_dir models/
"""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def download_model(model_name, output_dir):
    """Download a model from HuggingFace."""
    print(f"Downloading {model_name} to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"radinrad/{model_name}")
        tokenizer.save_pretrained(output_dir)
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(f"radinrad/{model_name}")
        model.save_pretrained(output_dir)
        
        print(f"Successfully downloaded {model_name}")
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download CRAFT models')
    parser.add_argument('--model', type=str, 
                       choices=['craft_qwen3', 'craft_llama3.3'],
                       help='Model to download')
    parser.add_argument('--output_dir', type=str, default='models/',
                       help='Output directory')
    parser.add_argument('--all', action='store_true',
                       help='Download all available models')
    
    args = parser.parse_args()
    
    if args.all:
        models = ['craft_qwen3', 'craft_llama3.3']
        for model in models:
            model_dir = os.path.join(args.output_dir, model)
            download_model(model, model_dir)
    elif args.model:
        model_dir = os.path.join(args.output_dir, args.model)
        download_model(args.model, model_dir)
    else:
        print("Please specify --model or --all")


if __name__ == "__main__":
    main()
