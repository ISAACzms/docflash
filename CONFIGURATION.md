# Doc Flash Configuration Guide

Doc Flash supports multiple LLM providers: **Azure OpenAI**, **OpenAI**, **Google Gemini**, and **Ollama** (local models). This guide explains how to configure your preferred provider.

## Quick Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your provider settings** (see sections below)

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

## Provider Configuration

### Option 1: Azure OpenAI (Default)

Best for enterprise users with Azure subscriptions.

**Required Environment Variables:**
```bash
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL_ID=gpt-4o-mini
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

**Authentication:** Uses Azure Default Credential (Azure CLI, Managed Identity, etc.)

### Option 2: OpenAI

Direct access to OpenAI models.

**Required Environment Variables:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_ID=gpt-4o-mini
```

**Optional:**
```bash
OPENAI_BASE_URL=https://api.openai.com/v1  # For custom endpoints
```

### Option 3: Google Gemini

Google's latest AI models.

**Required Environment Variables:**
```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_PROJECT_ID=your_google_project_id
GEMINI_MODEL_ID=gemini-1.5-flash
```

### Option 4: Ollama (Local Models)

Run models locally without API keys or internet connection.

**Prerequisites:**
1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows: Download from https://ollama.com
   ```

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   # Recommended starter model (1.3GB)
   ollama pull gemma2:2b
   
   # Other popular options
   ollama pull llama3.2:3b      # 2GB
   ollama pull mistral:7b       # 4.1GB
   ollama pull qwen2.5:7b       # 4.4GB
   ```

**Required Environment Variables:**
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL_ID=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434
```

**No API keys needed!** ⭐ Perfect for:
- Privacy-sensitive documents
- Offline processing
- Cost control
- Learning and experimentation

## OCR Configuration

OCR (Optical Character Recognition) can use the same provider as your main LLM or a different one. This allows you to optimize for different models - for example, using a cost-effective model for text extraction and a more powerful one for complex reasoning.

### OCR Provider Options

**Option 1: Use Same Provider as Main LLM (Default)**
```bash
# OCR_PROVIDER not set - inherits from LLM_PROVIDER
```

**Option 2: Azure OpenAI for OCR**
```bash
OCR_PROVIDER=azure_openai
OCR_AZURE_OPENAI_MODEL_ID=gpt-4o-mini
OCR_AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
OCR_TEMPERATURE=0.0
```

**Option 3: OpenAI for OCR**
```bash
OCR_PROVIDER=openai
OCR_OPENAI_MODEL_ID=gpt-4o-mini
OCR_TEMPERATURE=0.0
```

**Option 4: Google Gemini for OCR**
```bash
OCR_PROVIDER=gemini
OCR_GEMINI_MODEL_ID=gemini-1.5-flash
OCR_TEMPERATURE=0.0
```

**Option 5: Ollama for OCR**
```bash
OCR_PROVIDER=ollama
OCR_OLLAMA_MODEL_ID=gemma2:2b
OCR_TEMPERATURE=0.0
```

### Recommended Models for OCR

**Latest & Best Performance:**
- Azure OpenAI/OpenAI: `gpt-4.1` ⭐ **Newest**
- Google Gemini: `gemini-1.5-pro`

**Balanced Performance (Recommended):**
- Azure OpenAI/OpenAI: `gpt-4.1-mini` ⭐ **Newest & Cost-Effective**
- Google Gemini: `gemini-1.5-flash`

**Alternative High Performance:**
- Azure OpenAI/OpenAI: `gpt-4o`
- Azure OpenAI/OpenAI: `gpt-4o-mini`

**Cost Effective:**
- Azure OpenAI/OpenAI: `gpt-4-turbo`
- Google Gemini: `gemini-1.0-pro`

**Local/Privacy (Ollama):**
- **Best for OCR**: `gemma2:9b` ⭐ **Recommended**
- **Fastest**: `gemma2:2b` (Good accuracy, very fast)
- **Balanced**: `llama3.2:3b` (Good general performance)
- **Code-heavy docs**: `deepseek-coder:6.7b`

### OCR Features Supported

✅ **All Providers Support:**
- Multi-page PDF processing
- Text extraction with formatting
- Table detection and HTML conversion
- Handwriting recognition
- Signature detection
- Stamp and seal identification
- Image description generation

✅ **Advanced Features:**
- Orientation detection and correction
- Blank page skipping
- Concurrent processing for speed
- Error handling and retry logic

## Available Models

### Azure OpenAI Models
- `gpt-4.1` ⭐ **Latest**
- `gpt-4.1-mini` ⭐ **Latest** 
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-35-turbo`

### OpenAI Models
- `gpt-4.1` ⭐ **Latest**
- `gpt-4.1-mini` ⭐ **Latest**
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

### Google Gemini Models
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-1.0-pro`

### Ollama Models (Local)
- `gemma2:2b` ⭐ **Fastest** (1.3GB)
- `gemma2:9b` ⭐ **Recommended** (5.4GB)
- `gemma2:27b` (16GB)
- `llama3.2:1b` (1.3GB)
- `llama3.2:3b` (2.0GB)
- `llama3.1:8b` (4.7GB)
- `llama3.1:70b` (40GB)
- `mistral:7b` (4.1GB)
- `mistral-nemo:12b` (7.1GB)
- `qwen2.5:7b` (4.4GB)
- `qwen2.5:14b` (8.4GB)
- `deepseek-coder:6.7b` (3.8GB)
- `codellama:7b` (3.8GB)
- `phi3:3.8b` (2.3GB)

## Additional Settings

```bash
# LLM Temperature (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE=0.0

# LLM Max Tokens (adjust based on model and use case)
LLM_MAX_TOKENS=16000

# OCR Max Tokens (can be different from main LLM)
OCR_MAX_TOKENS=16000

# Flask Development Settings
FLASK_ENV=development
FLASK_DEBUG=True

# File Upload Limit (16MB default)
MAX_CONTENT_LENGTH=16777216
```

### Max Tokens Recommendations

**Different models have different token limits:**

**GPT-4.1 Series:**
- Input: 1,000,000 tokens (1M context window)
- Output: Up to 32,000 tokens
- Recommended: `LLM_MAX_TOKENS=32000`

**GPT-4o Series:**
- Input: 128,000 tokens  
- Output: Up to 16,384 tokens
- Recommended: `LLM_MAX_TOKENS=16000`

**GPT-4 Turbo:**
- Input: 128,000 tokens
- Output: Up to 4,096 tokens
- Recommended: `LLM_MAX_TOKENS=4000`

**Gemini Models:**
- Gemini 1.5 Pro: Up to 2,097,152 tokens
- Gemini 1.5 Flash: Up to 1,048,576 tokens
- Recommended: `LLM_MAX_TOKENS=16000` (for consistency)

**OCR Processing:**
- OCR typically needs higher token limits for document processing
- Recommended: `OCR_MAX_TOKENS=16000` for most use cases
- For very long documents: Consider `OCR_MAX_TOKENS=32000`

## Troubleshooting

### Azure OpenAI Issues
- Ensure you're logged into Azure CLI: `az login`
- Verify your deployment names match your Azure resource
- Check endpoint URL format

### OpenAI Issues
- Verify your API key is valid
- Check your usage limits and billing

### Gemini Issues
- Ensure your Google API key has Generative AI permissions
- Verify project ID is correct

### Ollama Issues
- Ensure Ollama service is running: `ollama serve`
- Check if model is downloaded: `ollama list`
- Verify base URL is accessible: `curl http://localhost:11434/api/tags`
- For GPU acceleration, ensure CUDA/Metal support is configured

### General Issues
- Check that required dependencies are installed
- Verify environment variables are loaded correctly
- Review application logs for specific error messages

## Security Notes

- Never commit `.env` files to version control
- Use environment-specific `.env` files for different deployments
- Regularly rotate API keys
- Use managed identities when possible (Azure)

## Example .env File

See `.env.example` for a complete template with all available options.