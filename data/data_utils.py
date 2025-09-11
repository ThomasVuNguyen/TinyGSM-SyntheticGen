import json
import os
from datasets import Dataset
from huggingface_hub import HfApi

def save_dataset(questions, solutions, filename="synthetic_dataset.json"):
    dataset = []
    for q, s in zip(questions, solutions):
        dataset.append({
            "user": q,
            "assistant": s
        })
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return dataset

def append_to_dataset(question, solution, filename):
    # Check if file exists and load existing data
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                dataset = json.load(f)
        except json.JSONDecodeError:
            dataset = []
    else:
        dataset = []
    
    # Append new entry
    dataset.append({
        "user": question,
        "assistant": solution
    })
    
    # Save updated dataset
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return len(dataset)

def get_processed_count(filename):
    """Get number of already processed questions"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                dataset = json.load(f)
            return len(dataset)
        except:
            return 0
    return 0

def get_processed_questions(filename):
    """Get list of already processed questions to avoid duplicates"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                dataset = json.load(f)
            return [entry['user'] for entry in dataset]
        except:
            return []
    return []

def upload_to_huggingface(dataset_data, repo_name):
    dataset = Dataset.from_list(dataset_data)
    dataset.push_to_hub(repo_name, private=False)