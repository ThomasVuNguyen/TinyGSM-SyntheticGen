import openai
import json
import sys
import os
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_azure_response(prompt, deployment_name, config):
    deployment = config['azure_deployments'][deployment_name]
    
    # Extract API version from endpoint or use default
    api_version = "2024-02-01"  # default
    if "api-version=" in deployment["endpoint"]:
        try:
            api_version = deployment["endpoint"].split("api-version=")[1].split("&")[0]
        except:
            pass  # use default if parsing fails
    
    client = openai.AzureOpenAI(
        api_key=deployment["api_key"],
        api_version=api_version,
        azure_endpoint=deployment["endpoint"]
    )
    
    max_retries = 5
    base_delay = 1
    
    # Prepare request parameters
    request_params = {
        "model": deployment["model"],
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # Add reasoning parameters for o4-mini model
    if "o4-mini" in deployment["model"] and "2025-04-01-preview" in api_version:
        request_params["reasoning_effort"] = "high"
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**request_params)
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

def get_azure_responses_batch(prompts, deployment_name, config, batch_size=10, max_workers=5):
    """
    Process multiple prompts in parallel batches for faster inference.
    
    Args:
        prompts: List of prompts to process
        deployment_name: Azure deployment name
        config: Configuration dictionary
        batch_size: Number of prompts to process in each batch
        max_workers: Maximum number of concurrent workers
    
    Returns:
        List of responses in the same order as input prompts
    """
    deployment = config['azure_deployments'][deployment_name]
    
    def process_single_prompt(prompt_data):
        prompt, index = prompt_data
        try:
            return get_azure_response(prompt, deployment_name, config), index
        except Exception as e:
            print(f"Error processing prompt {index}: {e}")
            return None, index
    
    results = [None] * len(prompts)
    prompt_data = [(prompt, i) for i, prompt in enumerate(prompts)]
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(prompt_data), batch_size):
        batch = prompt_data[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(prompt_data) + batch_size - 1)//batch_size} ({len(batch)} prompts)")
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            future_to_prompt = {executor.submit(process_single_prompt, prompt_data): prompt_data for prompt_data in batch}
            
            for future in as_completed(future_to_prompt):
                try:
                    response, index = future.result()
                    if response is not None:
                        results[index] = response
                except Exception as e:
                    prompt_data = future_to_prompt[future]
                    print(f"Error processing prompt {prompt_data[1]}: {e}")
    
    return results

async def get_azure_responses_async(prompts, deployment_name, config, max_concurrent=5):
    """
    Process multiple prompts asynchronously for maximum speed.
    
    Args:
        prompts: List of prompts to process
        deployment_name: Azure deployment name
        config: Configuration dictionary
        max_concurrent: Maximum number of concurrent requests
    
    Returns:
        List of responses in the same order as input prompts
    """
    deployment = config['azure_deployments'][deployment_name]
    
    async def process_single_prompt_async(session, prompt, index):
        try:
            headers = {
                "api-key": deployment["api_key"],
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": deployment["model"]
            }
            
            # Add reasoning parameters for o4-mini model
            if "o4-mini" in deployment["model"] and "2025-04-01-preview" in deployment["endpoint"]:
                data["reasoning_effort"] = "high"
            
            async with session.post(
                deployment["endpoint"],
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"], index
                else:
                    error_text = await response.text()
                    print(f"Error processing prompt {index}: {response.status} - {error_text}")
                    return None, index
        except Exception as e:
            print(f"Error processing prompt {index}: {e}")
            return None, index
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(session, prompt, index):
        async with semaphore:
            return await process_single_prompt_async(session, prompt, index)
    
    results = [None] * len(prompts)
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_with_semaphore(session, prompt, i) for i, prompt in enumerate(prompts)]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in completed_tasks:
            if isinstance(result, tuple) and len(result) == 2:
                response, index = result
                if response is not None:
                    results[index] = response
    
    return results

def get_azure_responses_parallel(prompts, deployment_name, config, max_workers=5):
    """
    Synchronous wrapper for parallel processing.
    """
    return asyncio.run(get_azure_responses_async(prompts, deployment_name, config, max_workers))

if __name__ == "__main__":
    # Test single response
    response = get_azure_response("Hello, how are you?", "gpt4o-deployment", {})
    print("Single response:", response)
    
    # Test batch processing
    test_prompts = [
        "What is 2+2?",
        "What is 3+3?", 
        "What is 4+4?"
    ]
    
    print("\nTesting batch processing...")
    batch_responses = get_azure_responses_batch(test_prompts, "gpt4o-deployment", {})
    for i, response in enumerate(batch_responses):
        print(f"Batch response {i+1}: {response}")