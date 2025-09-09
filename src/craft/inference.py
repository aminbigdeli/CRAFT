"""
LiteLLM Inference Script with Ranking-based Best Response Selection

This script generates a pool of responses using LiteLLM and selects the best one
using the same ranking mechanism from dpo_data_builder_litellm.py.

Usage:
    python inference_litellm_ranked.py --host localhost --port 8000 --model QwQ32B --model-name QwQ32B-DPO-V1-Ch150-BertBaseUncased --num-responses 16 --batch-size 16 --max-workers 12 --scoring-gpus 0,1,2,3 --scoring-workers 4 --timeout 15 --max-attempts 1 --temperature 0.6 --max-tokens 128 --cross-encoder ./surrogates/S3
"""

import json
import pandas as pd
import re
from tqdm import tqdm
import argparse
import time
import os
import torch
from openai import OpenAI
from sentence_transformers import CrossEncoder, util
import numpy as np
import bisect
from concurrent.futures import ThreadPoolExecutor


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LiteLLM Inference Script with Ranking')
    parser.add_argument('--host', type=str, default='localhost', help='LiteLLM proxy host')
    parser.add_argument('--port', type=int, default=8000, help='LiteLLM proxy port')
    parser.add_argument('--model', type=str, default="R1Llama70B", help='Model name (as defined in LiteLLM config)')
    parser.add_argument('--model-name', type=str, required=True, help='Model name for output files')
    parser.add_argument('--max-attempts', type=int, default=3, help='Max retry attempts')
    parser.add_argument('--timeout', type=int, default=120, help='Request timeout')
    parser.add_argument('--device', type=int, default=0, help='Primary GPU ID for CrossEncoder')
    parser.add_argument('--scoring-gpus', type=str, default='0,1,2,3', help='GPUs for parallel CrossEncoder scoring (comma-separated)')
    parser.add_argument('--scoring-workers', type=int, default=4, help='Number of parallel CrossEncoder workers')
    parser.add_argument('--cross-encoder', type=str, default='./surrogates/S3', help='Path or name of CrossEncoder model (local dir or HF hub name)')
    parser.add_argument('--num-responses', type=int, default=5, help='Number of responses to generate per input')
    parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=1500, help='Maximum number of tokens per response')
    parser.add_argument('--max-workers', type=int, default=16, help='Number of concurrent requests')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of requests to send in each batch')
    parser.add_argument('--request-delay', type=float, default=0.0, help='Delay between request batches (seconds)')
    parser.add_argument('--data-file', type=str, default='./data/msmarco_boost_10_test.json', help='Input data file')
    parser.add_argument('--output-dir', type=str, default='./inferences', help='Output directory')
    return parser.parse_args()


def extract_json_response(text):
    """Extract the first JSON object's 'response' field from model output.

    - Prefer the first balanced {...} block after the instruction section.
    - Fallback to a non-greedy regex to locate the first valid JSON object.
    - Return None if no valid JSON object is found or it lacks 'response'.
    """
    if not text:
        return None

    # Focus on the segment after the guidance phrase if present
    segment = text.split("Do not include any additional text.")[-1]

    # 1) Try brace-balanced scan to capture the FIRST JSON object only
    start = segment.find('{')
    if start != -1:
        depth = 0
        end = None
        for idx in range(start, len(segment)):
            ch = segment[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end is not None and end > start:
            snippet = segment[start:end]
            try:
                obj = json.loads(snippet)
                return obj.get('response')
            except Exception:
                pass

    # 2) Fallback: non-greedy regex, take the first valid JSON match
    for m in re.finditer(r'({.*?})', segment, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            return obj.get('response')
        except Exception:
            continue

    return None



def initialize_cross_encoders(scoring_gpus, scoring_workers, model_name_or_path):
    """Initialize multiple CrossEncoder instances for parallel scoring.
    Model can be a local directory (e.g., 'surrogates/S3') or an HF hub ID.
    """
    tokens = [tok.strip() for tok in scoring_gpus.split(',') if tok.strip()]
    cross_encoders = []
    for i in range(scoring_workers):
        tok = tokens[i % len(tokens)] if tokens else 'cpu'
        use_cpu = (tok.lower() == 'cpu') or (not torch.cuda.is_available())
        device = 'cpu' if use_cpu else f"cuda:{int(tok)}"
        print(f"Loading CrossEncoder on {device}...")
        cross_encoder = CrossEncoder(
            model_name_or_path,
            max_length=512,
            device=device
        )
        cross_encoders.append((cross_encoder, device))
    print(f"Initialized {len(cross_encoders)} CrossEncoders across: {tokens if tokens else ['cpu']}, model={model_name_or_path}")
    return cross_encoders


def truncate_for_cross_encoder(query, doc_content, max_length=1500):
    """Truncate query and document to fit CrossEncoder max_length (1536 tokens)
    
    Args:
        query: The query text
        doc_content: The document content  
        max_length: Max tokens to use (leave buffer for special tokens)
    
    Returns:
        Truncated query and doc_content that fit within token limit
    """
    # Simple character-based truncation (rough approximation of tokens)
    # Allocate 1/3 to query, 2/3 to document
    query_limit = max_length // 3
    doc_limit = max_length - query_limit
    
    truncated_query = query[:query_limit] if len(query) > query_limit else query
    truncated_doc = doc_content[:doc_limit] if len(doc_content) > doc_limit else doc_content
    
    return truncated_query, truncated_doc


def calculate_similarity_scores_parallel(query, doc_contents, cross_encoders):
    """Calculate similarity scores using parallel CrossEncoders with proper truncation"""
    if not query or not doc_contents:
        return [-1.0] * len(doc_contents)
    
    # Distribute work across multiple CrossEncoders
    def score_batch(encoder_pair, batch_docs):
        cross_encoder, device = encoder_pair
        if not batch_docs:
            return []
        try:
            # Truncate all pairs to avoid tensor size errors
            pairs = []
            for doc in batch_docs:
                pairs.append((query, doc))
            
            scores = cross_encoder.predict(pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            print(f"Scoring error on {device}: {e}")
            # Return neutral scores for failed batch
            return [0.0] * len(batch_docs)
    
    # Split documents across available encoders
    batch_size = max(1, len(doc_contents) // len(cross_encoders))
    batches = [doc_contents[i:i + batch_size] for i in range(0, len(doc_contents), batch_size)]
    
    # Process batches in parallel
    all_scores = []
    with ThreadPoolExecutor(max_workers=len(cross_encoders)) as executor:
        futures = []
        for i, batch in enumerate(batches):
            if batch:  # Only submit non-empty batches
                encoder_idx = i % len(cross_encoders)
                future = executor.submit(score_batch, cross_encoders[encoder_idx], batch)
                futures.append(future)
        
        # Collect results in order
        for future in futures:
            try:
                batch_scores = future.result(timeout=30)
                all_scores.extend(batch_scores)
            except Exception as e:
                print(f"Parallel scoring failed: {e}")
                continue
    
    return all_scores


def get_rank(scores, target_score):
    """Get the rank of target_score in the sorted list of scores
    
    Matches the ranking logic in summarize_attack_results.py using bisect
    """
    if not scores:
        return 1
    # Sort scores in ascending order to match evaluation's ranking logic
    sorted_scores = sorted(scores)
    # Find the first position where target_score would be inserted to maintain order
    # This matches the bisect_right behavior in the evaluation script
    position = bisect.bisect_right(sorted_scores, target_score)
    # Calculate rank (1-based) from the end of the list
    rank = len(scores) + 1 - position
    return rank


def generate_single_response(client, model, prompt, temperature, top_p, timeout, max_tokens):
    """Generate a single response using the LiteLLM client"""
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout
        )
        return response.choices[0].text
    except Exception as e:
        print(f"Generation error: {e}")
        return None


def generate_multiple_responses(client, model, prompt, num_responses, temperature, top_p, timeout, max_tokens, max_workers, batch_size=4, request_delay=0.0):
    """Generate multiple responses using aggressive concurrent batching for maximum speed"""
    
    def generate_one():
        return generate_single_response(client, model, prompt, temperature, top_p, timeout, max_tokens)
    
    responses = []
    
    # For speed: Use fewer, larger batches
    if num_responses <= 8:
        # Small number of responses: do all at once
        batch_size = num_responses
        total_batches = 1
    else:
        # Large number: use optimized batching
        total_batches = (num_responses + batch_size - 1) // batch_size
    
    print(f"Generating {num_responses} responses in {total_batches} batch(es) with {max_workers} workers")
    
    for batch_num in range(total_batches):
        # Calculate requests in this batch
        requests_in_batch = min(batch_size, num_responses - len(responses))
        
        # Use ThreadPoolExecutor for maximum concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests simultaneously for maximum speed
            futures = [executor.submit(generate_one) for _ in range(requests_in_batch)]
            
            # Collect results as fast as possible
            batch_responses = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout + 5)  # Minimal timeout buffer
                    if result:
                        batch_responses.append(result)
                except Exception as e:
                    print(f"Batch {batch_num+1} request failed: {e}")
                    continue
            
            responses.extend(batch_responses)
            print(f"Batch {batch_num+1}/{total_batches}: Got {len(batch_responses)} responses")
        
        # Minimal delay for max speed
        if request_delay > 0 and batch_num < total_batches - 1:
            time.sleep(request_delay)
    
    return responses


def main():
    args = get_args()
    
    # Initialize LiteLLM client
    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key=os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")  # Use environment variable
    )
    
    print(f"Connecting to LiteLLM at http://{args.host}:{args.port}")
        
    models = client.models.list()
    for model in models.data:
        print(model.id)
    
    # Test connection
    try:
        response = client.completions.create(
            model=args.model,
            prompt="Hi",
            max_tokens=5,
            timeout=10
        )
        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1
    
    # Initialize CrossEncoders for ranking
    print("Initializing CrossEncoders for ranking...")
    cross_encoders = initialize_cross_encoders(args.scoring_gpus, args.scoring_workers, args.cross_encoder)
    
    # Load data
    with open(args.data_file, 'r') as f:
        test_data = pd.DataFrame(json.load(f))
    
    test_data["new_doc_content"] = None
    test_data["new_sentence"] = None
    test_data["best_response"] = None
    test_data["best_score"] = None
    test_data["best_rank"] = None
    test_data["all_responses"] = None
    test_data["all_scores"] = None
    
    print(f"Processing {len(test_data)} samples...")
    
    # Alpaca prompt template for non-thinking LLMs
    template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
#     # Alpaca prompt template for thinking LLMs
#     template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# /no_think
# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# Return ONLY a JSON object in the exact form:
# {{"response": "<final answer only>"}}
# Do not include chain-of-thought, analysis, rationale, or any tags like <think>/<thought>. Do not include any additional text.
# """
    
    successful = 0
    
    for i in tqdm(range(len(test_data))):
        if test_data.iloc[i]["new_sentence"] is not None:
            continue
            
        prompt = template.format(
            test_data.iloc[i]["instruction"],
            test_data.iloc[i]["input"]
        )
        
        for attempt in range(args.max_attempts):
            try:
                # Generate multiple responses
                responses = generate_multiple_responses(
                    client, args.model, prompt, args.num_responses,
                    args.temperature, args.top_p, args.timeout, args.max_tokens,
                    args.max_workers, args.batch_size, args.request_delay
                )
                
                if not responses:
                    print(f"No responses generated for sample {i}")
                    continue
                
                # Extract JSON responses
                extracted_responses = []
                for j, response in enumerate(responses):
                    extracted = extract_json_response(response)
                    if not extracted:
                        print(f"Failed to extract JSON from response {j} for sample {i}")
                        extracted = response
                    extracted_responses.append(extracted)
                
                if not extracted_responses:
                    print(f"No valid JSON responses for sample {i} out of {len(responses)} total responses")
                    continue

                
                # Get target document for ranking
                target_doc = test_data.iloc[i]["doc_content"]
                query = test_data.iloc[i]["query"]
                
                try:
                    
                    # Create documents by combining extracted responses with target doc
                    doc_contents = [resp + " " + target_doc for resp in extracted_responses]
                    
                    # Calculate similarity scores
                    scores = calculate_similarity_scores_parallel(query, doc_contents, cross_encoders)
                    
                    if not scores or len(scores) != len(extracted_responses):
                        print(f"Scoring failed for sample {i}")
                        continue
                    
                    # Find best response (highest score)
                    best_idx = np.argmax(scores)
                    best_response = extracted_responses[best_idx]
                    best_score = scores[best_idx]
                    best_rank = get_rank(scores, best_score)
                    
                    # Save results
                    test_data.at[i, "new_doc_content"] = best_response + " " + target_doc
                    test_data.at[i, "new_sentence"] = best_response
                    test_data.at[i, "best_response"] = best_response
                    test_data.at[i, "best_score"] = best_score
                    test_data.at[i, "best_rank"] = best_rank
                    test_data.at[i, "all_responses"] = json.dumps(extracted_responses)
                    test_data.at[i, "all_scores"] = json.dumps(scores)
                    
                    successful += 1
                    print(f"Sample {i}: Selected best response (rank={best_rank}, score={best_score:.4f}) from {len(extracted_responses)} candidates")
                    break
                    
                except Exception as e:
                    print(f"Ranking failed for sample {i}: {e}")
                    continue
                    
            except Exception as e:
                if attempt == args.max_attempts - 1:
                    print(f"Failed sample {i}: {e}")
                else:
                    time.sleep(2)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.model_name}_output"
    
    test_data.to_csv(f"{args.output_dir}/{output_file}.tsv", sep='\t', index=False)
    test_data.to_json(f"{args.output_dir}/{output_file}.json")
    
    # Calculate statistics
    valid_scores = test_data[test_data["best_score"].notna()]
    if len(valid_scores) > 0:
        avg_score = valid_scores["best_score"].mean()
        avg_rank = valid_scores["best_rank"].mean()
        stats_text = f"""Ranked Inference Statistics:
- Total samples: {len(test_data)}
- Successful samples: {successful}
- Success rate: {successful/len(test_data)*100:.2f}%
- Average best score: {avg_score:.4f}
- Average best rank: {avg_rank:.2f}
- Responses per sample: {args.num_responses}
- Model: {args.model_name}
- CrossEncoder: {args.cross_encoder}
- LiteLLM endpoint: {args.host}:{args.port}
"""
        with open(f"{args.output_dir}/{output_file}_stats.txt", 'w') as f:
            f.write(stats_text)
        print(f"\nStatistics:\n{stats_text}")
    
    print(f"\n✅ Completed: {successful}/{len(test_data)} successful")
    print(f"Results saved to {args.output_dir}/{output_file}.*")
    
    return 0


if __name__ == "__main__":
    exit(main())
