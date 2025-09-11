import openai
import json
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_azure_response(prompt, deployment_name, config):
    deployment = config['azure_deployments'][deployment_name]
    
    client = openai.AzureOpenAI(
        api_key=deployment["api_key"],
        api_version="2024-02-01",
        azure_endpoint=deployment["endpoint"]
    )
    
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment["model"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        except openai.RateLimitError as e:
            print(f"Rate limit hit: {e}")
            raise Exception("RATE_LIMIT_EXCEEDED")
            
        except openai.APIError as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            print(f"API error: {e}. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    
    raise Exception(f"Failed after {max_retries} attempts")

if __name__ == "__main__":
    response = get_azure_response("Hello, how are you?", "gpt4o-deployment")
    print(response)