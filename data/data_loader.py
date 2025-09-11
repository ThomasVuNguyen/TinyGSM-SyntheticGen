from datasets import load_dataset

def load_tinygsm_questions(split='train', limit=None):
    try:
        # Try streaming first with correct column name
        dataset = load_dataset("TinyGSM/TinyGSM", split=split, streaming=True)
        questions = []
        for i, row in enumerate(dataset):
            if limit and i >= limit:
                break
            # Try different possible column names
            question = row.get('question') or row.get('user') or row.get('problem') or row.get('text')
            if question:
                questions.append(question)
        if questions:
            return questions
    except Exception as e:
        print(f"Streaming failed: {e}")
        
    # Try regular download
    dataset = load_dataset("TinyGSM/TinyGSM", split=split)
    # Check what columns exist
    if len(dataset) > 0:
        print(f"Dataset columns: {list(dataset[0].keys())}")
        
    # Try different possible column names
    column_name = None
    for col in ['question', 'user', 'problem', 'text']:
        if col in dataset[0]:
            column_name = col
            break
            
    if column_name:
        questions = [row[column_name] for row in dataset]
        if limit:
            questions = questions[:limit]
        return questions
    
    raise ValueError("Could not find question column in dataset")