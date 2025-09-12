import os
import json
from litellm import completion
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_ollama_response(prompt, config_file="ollama-config.json"):
    # Load config file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get model config
    model_config = config['ollama_models'][config['deployment']]
    
    # Set Ollama base URL if specified
    if 'base_url' in model_config:
        os.environ["OLLAMA_BASE_URL"] = model_config["base_url"]
    
    # Use model name from config
    model_name = f"ollama/{model_config['model_name']}"
    
    response = completion(
        model=model_name,
        messages=[{ "content": prompt, "role": "user"}],
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 2000)
    )
    return response.choices[0].message.content

def get_ollama_responses_parallel(prompts, config_file="ollama-config.json", max_workers=3):
    """Process multiple prompts in parallel for faster batch processing."""
    responses = [None] * len(prompts)
    
    def process_single_prompt(index, prompt):
        try:
            response = get_ollama_response(prompt, config_file)
            return index, response
        except Exception as e:
            print(f"Error processing prompt {index}: {e}")
            return index, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_prompt, i, prompt): i 
            for i, prompt in enumerate(prompts)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index, response = future.result()
            responses[index] = response
    
    return responses

if __name__ == "__main__":
    print("Chat with Ollama! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            response = get_ollama_response(user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
