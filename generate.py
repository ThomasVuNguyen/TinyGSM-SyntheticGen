import sys
import json
import os
import time
from utils.azure_ai import get_azure_response
from data.data_loader import load_tinygsm_questions
from data.data_utils import save_dataset, upload_to_huggingface, append_to_dataset

def generate_solutions(questions, config, output_path):
    total_questions = len(questions)
    start_time = time.time()
    times = []
    
    for i, question in enumerate(questions):
        iteration_start = time.time()
        print(f"Processing {i+1}/{total_questions}: {question[:50]}...")
        
        prompt = f"{question}\n\n{config['prompt']}"
        solution = get_azure_response(prompt, config['deployment'], config)
        
        iteration_time = time.time() - iteration_start
        times.append(iteration_time)
        
        # Calculate time estimates
        avg_time = sum(times) / len(times)
        remaining_questions = total_questions - (i + 1)
        estimated_remaining = avg_time * remaining_questions
        
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