#!/usr/bin/env python3
"""
Evaluation Script
=================

This script evaluates CRAFT models against baseline methods.

Usage:
    python scripts/evaluate.py --model models/craft_qwen3 --test_data data/test.json
"""

import argparse
import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import EvaluationMetrics
from craft.inference import CRAFTInference


def load_test_data(test_data_path):
    """Load test data from JSON file."""
    with open(test_data_path, 'r') as f:
        return json.load(f)


def evaluate_model(model_path, test_data, ranking_model="msmarco-MiniLM-L-12-v2"):
    """Evaluate a CRAFT model."""
    print(f"Evaluating model: {model_path}")
    
    # Initialize inference
    inference = CRAFTInference(
        model_path=model_path,
        ranking_model=ranking_model
    )
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    
    results = {
        'attack_performance': {},
        'content_fidelity': {},
        'quality_metrics': {}
    }
    
    for item in test_data:
        query = item['query']
        document = item['document']
        context_documents = item.get('context_documents', [])
        
        # Generate adversarial content
        adversarial_sentence = inference.generate(
            query=query,
            target_document=document,
            context_documents=context_documents
        )
        
        # Create adversarial document
        adversarial_document = document + " " + adversarial_sentence
        
        # Evaluate metrics
        attack_metrics = metrics.compute_attack_metrics(
            query, document, adversarial_document, inference
        )
        
        fidelity_metrics = metrics.compute_fidelity_metrics(
            document, adversarial_document
        )
        
        quality_metrics = metrics.compute_quality_metrics(
            adversarial_document
        )
        
        # Accumulate results
        for metric, value in attack_metrics.items():
            if metric not in results['attack_performance']:
                results['attack_performance'][metric] = []
            results['attack_performance'][metric].append(value)
        
        for metric, value in fidelity_metrics.items():
            if metric not in results['content_fidelity']:
                results['content_fidelity'][metric] = []
            results['content_fidelity'][metric].append(value)
        
        for metric, value in quality_metrics.items():
            if metric not in results['quality_metrics']:
                results['quality_metrics'][metric] = []
            results['quality_metrics'][metric].append(value)
    
    # Compute averages
    for category in results:
        for metric in results[category]:
            values = results[category][metric]
            results[category][metric] = {
                'values': values,
                'mean': sum(values) / len(values),
                'std': (sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CRAFT models')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model directory')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSON file')
    parser.add_argument('--ranking_model', type=str, 
                       default='msmarco-MiniLM-L-12-v2',
                       help='Neural ranking model to use')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load test data
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate model
    results = evaluate_model(args.model, test_data, args.ranking_model)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    
    for category, metrics in results.items():
        print(f"\n{category.upper()}:")
        for metric, stats in metrics.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
