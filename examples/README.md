# Configuration Examples

This directory contains example configuration files for different AI providers and use cases.

## Quick Start Examples

For testing the tool with a small number of questions:

- `quick-start-azure.json` - Azure OpenAI with GPT-4o-mini (cost-effective)
- `quick-start-bedrock.json` - AWS Bedrock with Claude 3.5 Sonnet
- `quick-start-ollama.json` - Ollama with Llama 3.1 8B (local)

## Full Configuration Examples

For production use with comprehensive settings:

- `azure-config.json` - Azure OpenAI with multiple model options
- `bedrock-config.json` - AWS Bedrock with multiple model options  
- `ollama-config.json` - Ollama with multiple model options

## Usage

1. **Copy an example file:**
   ```bash
   cp examples/quick-start-azure.json my-config.json
   ```

2. **Edit the configuration:**
   - Replace API keys and endpoints with your actual values
   - Adjust `limit` for testing (start with 10)
   - Modify `output_file` name if desired

3. **Run the generation:**
   ```bash
   python generate.py my-config.json
   ```

## Configuration Tips

### For Testing
- Use `limit: 10` to test quickly
- Use cheaper models (GPT-4o-mini, Llama 3.1 8B)
- Disable HuggingFace upload (`upload_to_hf: false`)

### For Production
- Set `limit: null` to process all questions
- Use more powerful models (GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B)
- Enable batch processing for faster generation
- Consider HuggingFace upload for sharing results

### Batch Processing
- Increase `batch_size` for faster processing (but watch rate limits)
- Adjust `max_workers` based on your API limits
- Azure: 5-10 workers recommended
- Bedrock: 3-5 workers recommended  
- Ollama: 3-5 workers recommended

## Model Recommendations

### Azure OpenAI
- **GPT-4o**: Best quality, higher cost
- **GPT-4o-mini**: Good quality, cost-effective
- **GPT-3.5-turbo**: Fast, lower cost

### AWS Bedrock
- **Nova Pro**: Amazon's latest model, good for math
- **Claude 3.5 Sonnet**: Excellent reasoning capabilities
- **Llama 3.1 70B**: Open source, good performance
- **Mixtral 8x7B**: Fast and efficient

### Ollama (Local)
- **Llama 3.1 70B**: Best quality, requires more RAM
- **Llama 3.1 8B**: Good balance of quality and speed
- **CodeLlama 7B**: Specialized for code generation
- **Mistral 7B**: Fast and efficient
- **Qwen 7B**: Good multilingual support

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API keys are correct
   - Check endpoint URLs
   - Ensure proper permissions

2. **Rate Limits**
   - Reduce `batch_size` and `max_workers`
   - Add delays between requests
   - Use sequential processing (`--no-batch`)

3. **Ollama Connection Issues**
   - Ensure Ollama is running: `ollama serve`
   - Check the model is pulled: `ollama pull llama3.1:8b`
   - Verify base URL is correct

4. **Memory Issues (Ollama)**
   - Use smaller models for limited RAM
   - Close other applications
   - Consider using cloud models instead

### Getting Help

- Check the main README.md for detailed documentation
- Test with small limits first
- Verify your configuration against the examples
- Check API provider documentation for specific requirements
