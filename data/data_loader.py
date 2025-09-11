from datasets import load_dataset
import os
import pickle
import hashlib

def load_tinygsm_questions(split='train', limit=None):
    print("Attempting to load TinyGSM dataset...")
    
    # Set minimal logging for datasets
    os.environ['DATASETS_VERBOSITY'] = 'error'
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on parameters
    cache_key = hashlib.md5(f"tinygsm_{split}_{limit}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"tinygsm_cache_{cache_key}.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                questions = pickle.load(f)
            print(f"Successfully loaded {len(questions)} questions from cache")
            return questions
        except Exception as e:
            print(f"Cache load failed: {e}, falling back to dataset loading")
    
    # Load from dataset
    questions = _load_from_dataset(split, limit)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(questions, f)
        print(f"Cached {len(questions)} questions for future use")
    except Exception as e:
        print(f"Cache save failed: {e}")
    
    return questions

def _load_from_dataset(split='train', limit=None):
    try:
        print("Trying optimized streaming approach...")
        dataset = load_dataset("TinyGSM/TinyGSM", split=split, streaming=True)
        
        # Use take() method for efficient limiting
        if limit:
            dataset = dataset.take(limit)
        
        # Convert to list efficiently
        questions = []
        print("Streaming dataset loaded, processing rows...")
        
        # Process in batches for better performance
        batch_size = 1000
        processed = 0
        
        for row in dataset:
            # Try different possible column names
            question = row.get('question') or row.get('user') or row.get('problem') or row.get('text')
            if question:
                questions.append(question)
                processed += 1
                
                # Progress reporting
                if processed % batch_size == 0:
                    print(f"Processed {processed} rows...")
                
        if questions:
            print(f"Successfully loaded {len(questions)} questions via streaming")
            return questions
    except Exception as e:
        print(f"Streaming failed: {e}")
        
    print("Trying regular download (this may take a while)...")
    dataset = load_dataset("TinyGSM/TinyGSM", split=split)
    print(f"Dataset downloaded successfully, size: {len(dataset)}")
    
    # Check what columns exist
    if len(dataset) > 0:
        print(f"Dataset columns: {list(dataset[0].keys())}")
        
    # Try different possible column names
    column_name = None
    for col in ['question', 'user', 'problem', 'text']:
        if col in dataset[0]:
            column_name = col
            print(f"Using column: {col}")
            break
            
    if column_name:
        questions = [row[column_name] for row in dataset]
        if limit:
            questions = questions[:limit]
        print(f"Successfully loaded {len(questions)} questions")
        return questions
    
    raise ValueError("Could not find question column in dataset")