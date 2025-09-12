import sys
import json
import os
import time
from utils.azure_ai import get_azure_response, get_azure_responses_batch, get_azure_responses_parallel
from utils.rockbed import get_bedrock_response, get_bedrock_responses_parallel
from data.data_loader import load_tinygsm_questions
from data.data_utils import save_dataset, upload_to_huggingface, append_to_dataset, get_processed_count, get_processed_questions

def generate_solutions(questions, config, output_path, batch_size=10, use_batch=True, config_file=None):
    # Check for existing progress
    already_processed = get_processed_questions(output_path)
    processed_count = len(already_processed)
    
    # Filter out already processed questions
    remaining_questions = [q for q in questions if q not in already_processed]
    
    if processed_count > 0:
        print(f"Resuming from {processed_count} already processed questions")
        print(f"Remaining questions to process: {len(remaining_questions)}")
    
    if not remaining_questions:
        print("All questions already processed!")
        return processed_count
    
    total_questions = len(remaining_questions)
    times = []
    
    try:
        if use_batch and ('azure_deployments' in config or 'bedrock_models' in config):
            # Use batch processing for Azure or Bedrock
            print(f"Using batch processing with batch size {batch_size}")
            return generate_solutions_batch(remaining_questions, config, output_path, processed_count, batch_size, config_file)
        else:
            # Use sequential processing
            return generate_solutions_sequential(remaining_questions, config, output_path, processed_count, config_file)
    
    except Exception as e:
        if "RATE_LIMIT_EXCEEDED" in str(e):
            current_count = get_processed_count(output_path)
            print(f"\n🛑 Rate limit exceeded! Stopping gracefully.")
            print(f"✅ Progress saved: {current_count} questions processed")
            print(f"📝 To resume, run the same command again")
            return current_count
        elif "PERMISSION_DENIED" in str(e):
            print(f"\n❌ Permission denied: Your Bedrock API key doesn't have access to Nova Pro model")
            print(f"💡 Check your AWS IAM permissions for bedrock:InvokeModel on amazon.nova-pro-v1:0")
            raise e
        elif "UNAUTHORIZED" in str(e):
            print(f"\n❌ Unauthorized: Invalid API key or authentication failed")
            print(f"💡 Check your Bedrock API key in the config file")
            raise e
        else:
            raise e

def generate_solutions_batch(questions, config, output_path, processed_count, batch_size=10, config_file=None):
    """Generate solutions using batch processing for faster inference."""
    total_questions = len(questions)
    times = []
    
    # Process questions in batches
    for i in range(0, total_questions, batch_size):
        batch_start = time.time()
        batch_questions = questions[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_questions + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")
        
        # Prepare prompts for this batch
        prompts = [f"{question}\n\n{config['prompt']}" for question in batch_questions]
        
        # Get batch responses
        if 'azure_deployments' in config:
            # Use Azure batch API
            batch_config = config.get('batch_processing', {})
            max_workers = batch_config.get('max_workers', 5)
            responses = get_azure_responses_batch(prompts, config['deployment'], config, 
                                               batch_size=len(batch_questions), 
                                               max_workers=max_workers)
        elif 'bedrock_models' in config:
            # Use parallel processing for Bedrock
            batch_config = config.get('batch_processing', {})
            max_workers = batch_config.get('max_workers', 3)
            responses = get_bedrock_responses_parallel(prompts, config_file, max_workers)
        else:
            # Fallback to sequential for other APIs
            responses = []
            for prompt in prompts:
                response = get_azure_response(prompt, config['deployment'], config)
                responses.append(response)
        
        # Save each solution in the batch
        for j, (question, solution) in enumerate(zip(batch_questions, responses)):
            if solution is not None:
                current_total = processed_count + i + j + 1
                count = append_to_dataset(question, solution, output_path)
                print(f"  Saved entry {count} ({j+1}/{len(batch_questions)} in batch)")
            else:
                print(f"  Failed to process question {i + j + 1}")
        
        batch_time = time.time() - batch_start
        times.append(batch_time)
        
        # Calculate time estimates
        avg_time = sum(times) / len(times)
        remaining_batches = total_batches - batch_num
        estimated_remaining = avg_time * remaining_batches
        
        # Format time display
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        print(f"Batch {batch_num} completed in {format_time(batch_time)} | Avg: {format_time(avg_time)} | ETA: {format_time(estimated_remaining)}\n")
    
    return get_processed_count(output_path)

def generate_solutions_sequential(questions, config, output_path, processed_count, config_file=None):
    """Generate solutions using sequential processing (original method)."""
    total_questions = len(questions)
    times = []
    
    for i, question in enumerate(questions):
        iteration_start = time.time()
        current_total = processed_count + i + 1
        print(f"Processing {current_total}/{len(questions)}: {question[:50]}...")
        
        prompt = f"{question}\n\n{config['prompt']}"
        
        # Determine which API to use based on config
        if 'bedrock_models' in config:
            solution = get_bedrock_response(prompt, config_file)
        else:
            solution = get_azure_response(prompt, config['deployment'], config)
        
        iteration_time = time.time() - iteration_start
        times.append(iteration_time)
        
        # Calculate time estimates
        avg_time = sum(times) / len(times)
        remaining_questions_count = total_questions - (i + 1)
        estimated_remaining = avg_time * remaining_questions_count
        
        # Save each solution as it's generated
        count = append_to_dataset(question, solution, output_path)
        
        # Format time display
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        print(f"Saved entry {count} to {output_path} | Time: {format_time(iteration_time)} | Avg: {format_time(avg_time)} | ETA: {format_time(estimated_remaining)}")
    
    return count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic math problems using AI models')
    parser.add_argument('config_file', help='Path to configuration JSON file')
    parser.add_argument('--batch-size', type=int, help='Batch size for parallel processing (overrides config)')
    parser.add_argument('--no-batch', action='store_true', help='Disable batch processing and use sequential processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of concurrent workers (overrides config)')
    parser.add_argument('--limit', type=int, help='Limit the number of questions to process (for testing)')
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Get batch settings from config with defaults
    batch_config = config.get('batch_processing', {})
    default_batch_size = batch_config.get('batch_size', 10)
    default_max_workers = batch_config.get('max_workers', 5)
    default_batch_enabled = batch_config.get('enabled', True)
    
    print("Loading questions from dataset...")
    # Use config limit as default, command line can override
    limit = args.limit if args.limit is not None else config.get('limit')
    start_row = config.get('start_row', 0)
    questions = load_tinygsm_questions(limit=limit, start_row=start_row)
    print(f"Loaded {len(questions)} questions starting from row {start_row}")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    output_file = config.get('output_file', 'synthetic_dataset.json')
    output_path = os.path.join('output', output_file)
    
    # Determine final batch settings (command line overrides config)
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    max_workers = args.max_workers if args.max_workers is not None else default_max_workers
    use_batch = not args.no_batch and default_batch_enabled
    
    # Print processing configuration
    if not use_batch:
        print("Using sequential processing")
    else:
        print(f"Using batch processing with batch size {batch_size} and max workers {max_workers}")
    
    total_count = generate_solutions(questions, config, output_path, 
                                   batch_size=batch_size, 
                                   use_batch=use_batch,
                                   config_file=args.config_file)
    print(f"Completed! Saved {total_count} examples to {output_path}")
    
    if config.get('upload_to_hf', False):
        # Load final dataset for upload
        with open(output_path, 'r') as f:
            dataset = json.load(f)
        upload_to_huggingface(dataset, config.get('hf_repo'))

if __name__ == "__main__":
    main()