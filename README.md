# TinyGSM-SyntheticGen
Synthetically generate TinyGSM dataset with Azure OpenAI

## Setup

1. Update `config.py` with your Azure OpenAI credentials
2. Install dependencies:
```bash
pip install openai datasets huggingface_hub
```

## Usage

Generate synthetic solutions using the config file:

```bash
python generate.py config.json
```

This will:
1. Load questions from TinyGSM/TinyGSM dataset
2. Generate solutions using Azure OpenAI
3. Save results according to config settings

## Configuration

Create JSON config files to customize generation:

```json
{
  "deployment": "gpt4o-deployment",
  "limit": 10,
  "output_file": "synthetic_dataset_config1.json",
  "upload_to_hf": false,
  "hf_repo": "your-username/your-repo",
  "prompt": "Your custom prompt here..."
}
```

**Config options:**
- `deployment`: Which Azure deployment to use
- `limit`: Number of questions to process
- `output_file`: Output filename
- `upload_to_hf`: Whether to upload to HuggingFace
- `hf_repo`: HuggingFace repository name
- `prompt`: Custom prompt template

Edit `config.json` to customize your generation settings.
