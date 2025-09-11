import boto3
import json
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_bedrock_response(prompt, model_name, config):
    model_config = config['bedrock_models'][model_name]
    
    # Initialize Bedrock client
    region = model_config["region"]
    model_id = model_config["model_id"]
    
    bedrock_client = boto3.client('bedrock-runtime', region_name=region)
    
    # Prepare the request body for Nova Pro
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
    
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']
        
    except Exception as e:
        raise Exception(f"Bedrock API error: {str(e)}")

if __name__ == "__main__":
    # Simple hardcoded config for testing
    test_config = {
        "bedrock_models": {
            "nova-pro": {
                "model_id": "amazon.nova-pro-v1:0",
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