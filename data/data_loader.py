from datasets import load_dataset
import os

def load_tinygsm_questions(split='train', limit=None):
    print("Attempting to load TinyGSM dataset...")
    
    # Set verbose logging for datasets
    os.environ['DATASETS_VERBOSITY'] = 'info'
    
    try:
        print("Trying streaming approach...")
        dataset = load_dataset("TinyGSM/TinyGSM", split=split, streaming=True)
        questions = []
        print("Streaming dataset loaded, processing rows...")
        
        for i, row in enumerate(dataset):
            if i == 0:
                print(f"First row columns: {list(row.keys())}")
            if limit and i >= limit:
                break
            if i % 100 == 0:
                print(f"Processed {i} rows...")
                
            # Try different possible column names
            question = row.get('question') or row.get('user') or row.get('problem') or row.get('text')
            if question:
                questions.append(question)
                
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