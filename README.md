# ⚡ Doc Flash - Content Intelligence Platform

A web application for extracting structured data from documents with AI-powered intelligence and feedback learning. Built on Google's LangExtract and Stanford's DSPy frameworks with an intuitive web interface.

## Features

### Document Processing
- **PDF Upload**: Convert PDFs to text using OCR
- **Text Input**: Process plain text content directly
- **Multiple Formats**: Handle various document types (contracts, invoices, forms)

### Extraction Modes
- **Extract Mode**: Find specific text spans from documents (names, dates, amounts)
- **Generate Mode**: Create interpreted content (summaries, classifications)

### AI Integration
- **Multiple Providers**: Azure OpenAI, OpenAI, Google Gemini, Ollama
- **Template System**: Save and reuse extraction configurations
- **Feedback Learning**: Improve results through user feedback and DSPy optimization

### Web Interface
- **Template Management**: Create and edit extraction schemas
- **Real-time Processing**: Live progress updates for document processing
- **Result Export**: Download results in HTML, JSON, or JSONL formats

## Installation

### Prerequisites
- Python 3.8+
- API access to one of: Azure OpenAI, OpenAI, Google Gemini, or Ollama

### Setup

1. **Clone and install**
```bash
git clone https://github.com/LM-150A/docflash.git
cd docflash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API credentials
```

3. **Run application**
```bash
python start_fastapi.py
```

Access the application at `http://localhost:5000`

## Configuration

### Environment Variables

```env
# Choose one provider
LLM_PROVIDER=azure_openai  # Options: azure_openai, openai, gemini, ollama

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_KEY=your-api-key

# OpenAI
OPENAI_API_KEY=your-openai-key

# Google Gemini  
GOOGLE_API_KEY=your-google-api-key

# Ollama (local)
OLLAMA_MODEL_ID=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Enable DSPy optimization
DSPY_ENABLED=true
```

## Usage

### 1. Define Schema
Create extraction attributes specifying what information to extract:

| Attribute | Description | Mode |
|-----------|-------------|------|
| client_name | Name of the client | Extract |
| contract_value | Total contract amount | Extract |
| summary | Brief contract summary | Generate |

### 2. Add Sample Documents
- Upload PDF files for OCR processing
- Copy/paste text content directly
- Provide multiple examples for better training

### 3. Generate Training Examples
The system creates training examples based on your schema and sample documents.

### 4. Process Documents
Upload new documents and run extraction with configurable settings:
- Number of extraction passes (1-3)
- Parallel processing workers (5-20)
- Temperature settings based on extraction modes

### 5. Review and Improve
- Rate generated examples to improve future results
- Use detailed feedback to guide AI improvements
- DSPy automatically optimizes prompts based on feedback

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main interface |
| `/upload_pdf` | POST | PDF upload and OCR |
| `/generate_examples` | POST | Create training examples |
| `/run_extraction` | POST | Process documents |
| `/register_template` | POST | Save templates |
| `/feedback/examples` | POST | Submit feedback |

## Architecture

```
Frontend (HTML/JS) ←→ Backend (FastAPI) ←→ AI Providers
    ↓                       ↓                    ↓
• Template UI          • LangExtract        • Azure OpenAI
• Document Upload      • OCR Pipeline       • OpenAI  
• Feedback System      • DSPy Integration   • Google Gemini
• Progress Tracking    • Template Storage   • Ollama
```

## Best Practices

### Schema Design
- Use Extract mode for factual data that appears verbatim
- Use Generate mode for analysis or interpreted content
- Write clear, specific attribute descriptions

### Training Data
- Provide 2-4 diverse sample documents
- Include variations and edge cases
- Ensure samples cover all schema attributes

### Feedback
- Rate examples regularly to improve performance
- Use detailed feedback for specific issues
- Feedback is isolated by document type

## Troubleshooting

### Common Issues

**API Configuration**
- Verify API credentials in `.env` file
- Check endpoint URLs and model names
- Ensure sufficient API quota/credits

**PDF Processing**
- Use clear, text-based PDFs (not scanned images)
- Check file size limits (typically 16MB max)
- Try alternative OCR if text extraction fails

**Poor Extraction Results**
- Review and improve schema descriptions
- Add more diverse training examples
- Increase extraction passes for better recall
- Provide feedback on generated examples

**DSPy Optimization**
- Set `DSPY_ENABLED=true` in environment
- Provide sufficient feedback (default: 10+ examples)
- Check logs for optimization triggers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Google LangExtract](https://github.com/google/langextract/blob/main/CITATION.cff) - Core extraction framework
- [Stanford DSPy](https://github.com/stanfordnlp/dspy) - Prompt optimization framework