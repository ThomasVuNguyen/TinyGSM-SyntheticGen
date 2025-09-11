import openai
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_azure_response(prompt, deployment_name, config):
    deployment = config['azure_deployments'][deployment_name]
    
    client = openai.AzureOpenAI(
        api_key=deployment["api_key"],
        api_version="2024-02-01",
        azure_endpoint=deployment["endpoint"]
    )
    
    response = client.chat.completions.create(
        model=deployment["model"],
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    response = get_azure_response("Hello, how are you?", "gpt4o-deployment")
    print(response)