import os
import json
from litellm import completion
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_bedrock_response(prompt, config_file="bedrock-llama33-70b.json"):
    # Load config file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get model config
    model_config = config['bedrock_models'][config['deployment']]
    
    # Set AWS credentials and region from config
    os.environ["AWS_ACCESS_KEY_ID"] = model_config["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = model_config["aws_secret_access_key"]
    os.environ["AWS_REGION_NAME"] = model_config["region"]
    
    # Use model ID from config
    model_id = f"bedrock/{model_config['model_id']}"
    
    response = completion(
        model=model_id,
        messages=[{ "content": prompt, "role": "user"}]
    )
    return response.choices[0].message.content

def get_bedrock_responses_parallel(prompts, config_file="bedrock-llama33-70b.json", max_workers=3):
    """Process multiple prompts in parallel for faster batch processing."""
    responses = [None] * len(prompts)
    
    def process_single_prompt(index, prompt):
        try:
            response = get_bedrock_response(prompt, config_file)
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
    print("Chat with Llama 3.3 70B! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            response = get_bedrock_response(user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")