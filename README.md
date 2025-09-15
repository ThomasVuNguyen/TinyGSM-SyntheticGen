# TinyGSM-SyntheticGen

Synthetically generate TinyGSM dataset solutions using various AI providers (Azure OpenAI, AWS Bedrock, Ollama) with configurable JSON files.

## Features

- **Multi-Provider Support**: Azure OpenAI, AWS Bedrock, and Ollama
- **Batch Processing**: Parallel processing for faster generation
- **Resume Capability**: Automatically resumes from where it left off
- **Progress Tracking**: Real-time progress updates with ETA
- **Flexible Configuration**: JSON-based configuration for easy customization
- **HuggingFace Integration**: Direct upload to HuggingFace Hub

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create a configuration file:**
   - Copy one of the example configs from `examples/` directory
   - Edit the JSON file with your API credentials and settings
   - See configuration examples below for each provider

## Usage

### Basic Usage

Generate synthetic solutions using a JSON configuration file:

```bash
python generate.py your-config.json
```

### Command Line Options

```bash
python generate.py config.json [OPTIONS]

Options:
  --batch-size INT     Override batch size for parallel processing
  --no-batch          Disable batch processing (use sequential)
  --max-workers INT   Override maximum concurrent workers
  --limit INT         Limit number of questions to process (for testing)
```

### Examples

```bash
# Basic generation
python generate.py azure-config.json

# Test with limited questions
python generate.py azure-config.json --limit 5

# Override batch settings
python generate.py azure-config.json --batch-size 20 --max-workers 8

# Use sequential processing
python generate.py azure-config.json --no-batch
```

## Configuration Files

The tool uses JSON configuration files to specify AI provider settings, prompts, and generation parameters. Each provider has its own configuration format.

### Azure OpenAI Configuration

Create `azure-config.json`:

```json
{
  "azure_deployments": {
    "gpt4o-deployment": {
      "model": "gpt-4o",
      "endpoint": "https://your-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01",
      "api_key": "your-api-key"
    }
  },
  "deployment": "gpt4o-deployment",
  "limit": 100,
  "output_file": "tinygsm-azure.json",
  "upload_to_hf": false,
  "hf_repo": "your-username/tinygsm-azure",
  "batch_processing": {
    "enabled": true,
    "batch_size": 10,
    "max_workers": 5
  },
  "prompt": "Solve this math problem step by step..."
}
```

### AWS Bedrock Configuration

Create `bedrock-config.json`:

```json
{
  "bedrock_models": {
    "nova-pro": {
      "model_id": "amazon.nova-pro-v1:0",
      "aws_access_key_id": "your-access-key",
      "aws_secret_access_key": "your-secret-key",
      "region": "us-east-1",
      "max_tokens": 2000,
      "temperature": 0.7
    }
  },
  "deployment": "nova-pro",
  "limit": 100,
  "output_file": "tinygsm-bedrock.json",
  "upload_to_hf": false,
  "hf_repo": "your-username/tinygsm-bedrock",
  "prompt": "Solve this math problem step by step..."
}
```

### Ollama Configuration

Create `ollama-config.json`:

```json
{
  "ollama_models": {
    "llama3.1-8b": {
      "model_name": "llama3.1:8b",
      "base_url": "http://localhost:11434",
      "max_tokens": 2000,
      "temperature": 0.7
    }
  },
  "deployment": "llama3.1-8b",
  "limit": 100,
  "output_file": "tinygsm-ollama.json",
  "upload_to_hf": false,
  "hf_repo": "your-username/tinygsm-ollama",
  "prompt": "Solve this math problem step by step..."
}
```

## Configuration Options

### Common Options

| Option | Type | Description |
|--------|------|-------------|
| `deployment` | string | Which model/deployment to use |
| `limit` | integer | Number of questions to process (null = all) |
| `output_file` | string | Output filename in `output/` directory |
| `upload_to_hf` | boolean | Whether to upload to HuggingFace Hub |
| `hf_repo` | string | HuggingFace repository name |
| `prompt` | string | Custom prompt template |

### Batch Processing Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_processing.enabled` | boolean | true | Enable batch processing |
| `batch_processing.batch_size` | integer | 10 | Number of questions per batch |
| `batch_processing.max_workers` | integer | 5 | Maximum concurrent workers |

### Provider-Specific Options

#### Azure OpenAI
- `azure_deployments`: Dictionary of deployment configurations
- Each deployment needs: `model`, `endpoint`, `api_key`

#### AWS Bedrock
- `bedrock_models`: Dictionary of model configurations
- Each model needs: `model_id`, `aws_access_key_id`, `aws_secret_access_key`, `region`

#### Ollama
- `ollama_models`: Dictionary of model configurations
- Each model needs: `model_name`, `base_url`, `max_tokens`, `temperature`

## Output Format

Generated datasets are saved in the `output/` directory with the following format:

```json
[
  {
    "user": "What is 2 + 2?",
    "assistant": "The answer is 4. 2 + 2 = 4."
  },
  {
    "user": "If a train travels 60 mph for 2 hours, how far does it go?",
    "assistant": "The train travels 120 miles. Distance = Speed × Time = 60 mph × 2 hours = 120 miles."
  }
]
```

## Resume Capability

The tool automatically saves progress and can resume from where it left off. If the process is interrupted:

1. Run the same command again
2. The tool will detect already processed questions
3. It will continue from where it stopped

## Error Handling

The tool handles common errors gracefully:

- **Rate Limits**: Automatically stops and saves progress
- **Authentication Errors**: Clear error messages with troubleshooting tips
- **Network Issues**: Automatic retries with exponential backoff

## Examples

### Quick Start
```bash
# Copy an example config
cp examples/quick-start-azure.json my-config.json

# Edit the config with your credentials
# Then run generation
python generate.py my-config.json --limit 5
```

### Configuration Examples
See the `examples/` directory for complete configuration examples:

- **Quick Start**: `quick-start-*.json` - Simple configs for testing (10 questions)
- **Full Configs**: `*-config.json` - Production-ready configs with multiple models
- **Documentation**: `examples/README.md` - Detailed usage guide

### How to Edit Configuration Files

1. **Copy an example:**
   ```bash
   cp examples/quick-start-azure.json my-config.json
   ```

2. **Edit the JSON file** with your credentials:
   - **Azure OpenAI**: Replace `your-azure-openai-api-key` and endpoint URL
   - **AWS Bedrock**: Replace AWS credentials and region
   - **Ollama**: Ensure Ollama is running and model is pulled

3. **Customize settings:**
   - Change `limit` to control number of questions
   - Modify `output_file` for different output names
   - Adjust `batch_processing` settings for performance

### Example Commands
```bash
# Copy and edit a config first
cp examples/quick-start-azure.json my-config.json
# Edit my-config.json with your credentials

# Then run generation
python generate.py my-config.json

# Or use examples directly (after editing credentials)
python generate.py examples/azure-config.json
python generate.py examples/bedrock-config.json
python generate.py examples/ollama-config.json

# With custom settings
python generate.py my-config.json --batch-size 20 --max-workers 8
```

## Troubleshooting

### Common Issues

1. **Rate Limit Exceeded**: Reduce `batch_size` or `max_workers`
2. **Authentication Failed**: Check API keys and credentials
3. **Permission Denied**: Verify IAM permissions for Bedrock
4. **Ollama Connection Failed**: Ensure Ollama is running on the specified URL

### Getting Help

- Check the error messages for specific guidance
- Verify your configuration file format
- Ensure all required credentials are provided
- Test with a small `limit` first
