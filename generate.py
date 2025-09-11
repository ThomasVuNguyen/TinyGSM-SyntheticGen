import sys
import json
import os
import time
from utils.azure_ai import get_azure_response
from utils.rockbed import get_bedrock_response
from data.data_loader import load_tinygsm_questions
from data.data_utils import save_dataset, upload_to_huggingface, append_to_dataset, get_processed_count, get_processed_questions

def generate_solutions(questions, config, output_path):
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
        for i, question in enumerate(remaining_questions):
            iteration_start = time.time()
            current_total = processed_count + i + 1
            print(f"Processing {current_total}/{len(questions)}: {question[:50]}...")
            
            prompt = f"{question}\n\n{config['prompt']}"
            
            # Determine which API to use based on config
            if 'bedrock_models' in config:
                solution = get_bedrock_response(prompt, config['deployment'], config)
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
        
    except Exception as e:
        if "RATE_LIMIT_EXCEEDED" in str(e):
            current_count = get_processed_count(output_path)
            print(f"\n🛑 Rate limit exceeded! Stopping gracefully.")
            print(f"✅ Progress saved: {current_count} questions processed")
            print(f"📝 To resume, run the same command again")
            return current_count
        else:
            raise e

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate.py <config_file.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("Loading questions from dataset...")
    questions = load_tinygsm_questions(limit=config.get('limit'))
    print(f"Loaded {len(questions)} questions")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    output_file = config.get('output_file', 'synthetic_dataset.json')
    output_path = os.path.join('output', output_file)
    
    total_count = generate_solutions(questions, config, output_path)
    print(f"Completed! Saved {total_count} examples to {output_path}")
    
    if config.get('upload_to_hf', False):
        # Load final dataset for upload
        with open(output_path, 'r') as f:
            dataset = json.load(f)
        upload_to_huggingface(dataset, config.get('hf_repo'))

if __name__ == "__main__":
    main()