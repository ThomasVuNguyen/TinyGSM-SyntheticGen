import requests
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_bedrock_response(prompt, model_name, config):
    model_config = config['bedrock_models'][model_name]
    
    # Bedrock API keys are used as bearer tokens in HTTP headers
    api_key = model_config["api_key"]
    region = model_config["region"]
    model_id = model_config["model_id"]
    
    # Bedrock API endpoint
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    body = {
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "temperature": model_config.get("temperature", 0.7),
            "maxTokens": model_config.get("max_tokens", 2000)
        }
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code != 200:
        raise Exception(f"Bedrock API error: {response.status_code} - {response.text}")
    
    response_data = response.json()
    return response_data['output']['message']['content'][0]['text']

if __name__ == "__main__":
    # Simple hardcoded config for testing
    test_config = {
        "bedrock_models": {
            "nova-pro": {
                "model_id": "amazon.nova-pro-v1:0",
                "api_key": "ABSKQmVkcm9ja0FQSUtleS15dmc5LWF0LTkzNTk4NzIyMzU5OTpCQ1NLem1NUHJhOWFFUzJpem5XcU5FdkQ0VTRVMEJRcDZFdi9NS09pd2g2NlJUaHBhLzRNZ3ZleXBjbz0=",
                "region": "us-east-1",
                "max_tokens": 2000,
                "temperature": 0.7
            }
        },
        "deployment": "nova-pro"
    }
    
    print("Chat with Nova Pro! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            response = get_bedrock_response(user_input, "nova-pro", test_config)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")