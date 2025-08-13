# Doc Flash ⚡ - Intelligent Document Processing Platform

**Fast, intelligent document processing with the power of AI**

Doc Flash is a modern web application built on top of Google's LangExtract framework, providing an intuitive interface for creating document extraction templates and processing both **PDF documents** (via OCR) and **plain text content** (direct input) with AI-powered extraction capabilities.

## 🚀 Features

### 🎯 **Dual-Mode Extraction**
- **Extract Mode**: Pull exact text spans from documents (names, dates, amounts)
- **Generate Mode**: Create interpreted content (summaries, risk assessments, classifications)
- **Adaptive AI**: Automatically adjusts temperature and prompting strategies based on mode

### 📄 **Document Processing**
- **PDF OCR**: Upload PDFs and convert to markdown with real-time progress tracking
- **Direct Text Input**: Copy/paste text from any source (Word docs, emails, plain text files)
- **Flexible Input**: No OCR needed for text files - paste directly into the scratchpad
- **Multi-Provider Support**: Azure OpenAI, OpenAI, Google Gemini, Ollama (local models)
- **Template System**: Save and reuse extraction configurations
- **Batch Processing**: Handle multiple documents efficiently

### 📝 **Multiple Input Methods**
- **📄 PDF Upload**: Drag & drop PDFs for automatic OCR processing with real-time progress
- **📋 Text Paste**: Copy/paste content directly from .txt files, Word documents, emails, or web pages
- **⚡ Scratchpad**: Interactive workspace that persists content throughout your extraction session
- **🔄 Mixed Workflows**: Use OCR for some documents, direct text input for others

### 🧠 **Advanced AI Features**
- **DSPy Integration**: Automatic prompt optimization using Stanford DSPy framework
- **Real-time Learning**: AI learns from user feedback to improve future generations
- **Master Feedback System**: Rate entire example generations for better optimization
- **Document Class Isolation**: Feedback learning isolated by document type for precise improvements
- **Adaptive Optimization**: Automatic prompt refinement based on user satisfaction ratings
- **Template Management**: Document class dashboard with usage statistics and feedback tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   AI Providers  │
│   (HTML/JS)     │◄──►│   (FastAPI)      │◄──►│ Azure OpenAI    │
│                 │    │                  │    │ OpenAI          │
│ • Template UI   │    │ • LangExtract    │    │ Google Gemini   │
│ • PDF Upload    │    │ • OCR Pipeline   │    │ Ollama (Local)  │
│ • Feedback UI   │    │ • DSPy Pipeline  │    │                 │
│ • Real-time     │    │ • Template DB    │    │                 │
│   Progress      │    │ • WebSockets     │    │                 │
│ • Theme Support │    │ • Feedback Store │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Node.js (for optional frontend development)
- Azure OpenAI / OpenAI / Google Gemini API access OR Ollama (for local models)

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/LM-150A/doc-flash.git
cd doc-flash
```

2. **Create virtual environment**
```bash
python -m venv doc-flash_env
source doc-flash_env/bin/activate  # On Windows: doc-flash_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API credentials
```

5. **Run the application**
```bash
python start_fastapi.py
```

The application will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables (.env)

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_API_KEY=your-api-key  # Optional if using managed identity

# OpenAI Configuration (alternative)
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Google Gemini Configuration (alternative)
GOOGLE_API_KEY=your-google-api-key

# Ollama Configuration (local models - no API key needed)
OLLAMA_MODEL_ID=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434

# Provider Selection
LLM_PROVIDER=azure_openai  # Options: azure_openai, openai, gemini, ollama
DEFAULT_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.0

# DSPy Configuration (optional - enables advanced AI features)
DSPY_ENABLED=true
DSPY_OPTIMIZATION_FREQUENCY=10  # Trigger optimization every N feedback entries
```

## 📚 Usage Guide

### 1. **Define Schema** 📋
Create your extraction schema with dual-mode attributes:

**Example: Legal Contract Schema**
| Attribute | Description | Mode |
|-----------|-------------|------|
| client_name | Name of the client | Extract |
| contract_value | Total contract amount | Extract |
| risk_assessment | Overall contract risk level | Generate |
| key_terms_summary | Summary of important terms | Generate |

### 2. **Upload Sample Documents** 📄
- **PDF Files**: Upload PDFs for automatic OCR processing
- **Text Files**: Copy/paste text directly from .txt, Word docs, emails, or any text source
- **Scratchpad**: Interactive workspace for immediate text input and editing
- **Multiple Samples**: Add diverse example texts for better AI training

### 3. **Generate Examples** 🤖
Doc Flash automatically creates training examples with mode-aware prompting:
- **Extract fields**: Exact text spans with precise boundaries
- **Generate fields**: Interpreted content with contextual understanding

### 4. **Provide Feedback** 🧠
Rate generated examples to improve AI performance:
- **Quick Feedback**: 👍/👎 buttons for fast rating
- **Detailed Feedback**: Star ratings, specific issues, and comments
- **Real-time Learning**: DSPy optimizes prompts based on your feedback
- **Document Class Learning**: Feedback is isolated by document type for precise improvements

### 5. **Process Documents** ⚡
Run extraction on new documents with:
- Configurable passes (1-3) for improved recall
- Parallel processing (5-20 workers)
- Adaptive temperature based on schema modes
- **Enhanced with DSPy**: Improved prompts based on accumulated feedback

### 6. **Download Results** 📥
Get results in multiple formats:
- **HTML Visualization**: Interactive document view with highlighted extractions
- **JSONL**: Raw extraction data for further processing
- **Structured JSON**: Clean output matching your target schema

## 💡 Mode Examples

### **Extract Mode** 🎯
Perfect for pulling exact information:

```json
{
  "extraction_class": "contract_value",
  "extraction_text": "$25,000",
  "mode": "extract",
  "attributes": {"page_number": "1", "section": "terms"}
}
```

### **Generate Mode** 🧠
Ideal for interpreted content:

```json
{
  "extraction_class": "risk_assessment", 
  "extraction_text": "Low risk - standard terms, established client, reasonable duration",
  "mode": "generate",
  "attributes": {"page_number": "multiple", "confidence": "high"}
}
```


## 🔄 Workflow Example

### Processing a Services Agreement

1. **Input Document**: 
   - **PDF Option**: Drag services agreement PDF → OCR processes automatically
   - **Text Option**: Copy/paste contract text directly into scratchpad
2. **Define Schema**:
   ```
   • client_name (Extract) - Name of the client receiving services  
   • service_provider (Extract) - Company providing services
   • contract_duration (Extract) - Length of the service period
   • services_summary (Generate) - Brief summary of services provided
   • risk_level (Generate) - Assessment of contract risk factors
   ```

3. **Generate Examples**: AI creates mode-aware training examples
4. **Process Document**: Run extraction with adaptive temperature
5. **Get Results**:
   ```json
   {
     "contract_details": {
       "parties": {
         "client_name": {"result": "TechStart Inc.", "page_number": [1]},
         "service_provider": {"result": "ABC Marketing Solutions LLC", "page_number": [1]}
       },
       "analysis": {
         "services_summary": {"result": "Digital marketing consulting including market research, campaign development, and analytics", "page_number": [1]},
         "risk_level": {"result": "Low - standard consulting agreement with established scope", "page_number": [1]}
       }
     }
   }
   ```

## 🚀 Advanced Features

### **Real-time Progress Tracking**
WebSocket-powered live updates during OCR processing:
```javascript
// Real-time progress updates
websocket.onmessage = function(event) {
    const update = JSON.parse(event.data);
    updateProgressBar(update.progress);
};
```

### **Adaptive Temperature Control**
Intelligent temperature selection based on schema modes:
```python
# Automatic temperature optimization
has_generate_fields = any(attr.get('mode') == 'generate' for attr in schema)
optimal_temperature = 0.3 if has_generate_fields else 0.0
```

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main interface |
| `/configure` | GET | Template configuration |
| `/analyze` | GET | Document analysis interface |
| `/upload_pdf` | POST | PDF upload and OCR processing |
| `/generate_examples` | POST | AI example generation |
| `/run_extraction` | POST | Document extraction |
| `/register_template` | POST | Save extraction template |
| `/download/{session_id}/{file_type}` | GET | Download results |
| `/feedback/examples` | POST | Submit example feedback |
| `/feedback/examples/detailed` | POST | Submit detailed feedback |
| `/api/document_classes` | GET | Get document class statistics |
| `/rl/document_classes` | GET | Document class dashboard |

## 🏆 Best Practices

### **Schema Design**
- Use **Extract** for factual data that appears verbatim in documents
- Use **Generate** for analysis, summaries, or interpreted information
- Provide clear, specific descriptions for better AI understanding
- Include page numbers and section context in attributes

### **Sample Texts**
- Provide 2-4 diverse examples of your document type
- Include edge cases and variations
- Use real document excerpts when possible
- Cover all schema attributes in your samples

### **Feedback & Learning**
- **Rate examples regularly** for continuous AI improvement
- **Provide specific feedback** using the detailed feedback modal
- **Common feedback categories**:
  - Wrong text extraction
  - Incorrect field mapping
  - Missing important fields
  - Format or structure issues
  - Wrong text generation
- **Document class isolation**: Feedback only affects the specific document type
- **Regeneration**: Click regenerate after feedback to see immediate improvements

### **Advanced Feedback Memory System** 🧠
- **Positive Feedback Preservation**: When you approve a prompt with positive feedback, it stays unchanged on regeneration
- **Weighted Feedback History**: Recent feedback gets higher priority (1.0x) while older feedback maintains reduced influence (0.3x)
- **Institutional Memory**: System remembers critical lessons learned while prioritizing recent user intent
- **Quality Score Calculation**: Automatic 0.0-1.0 scoring based on feedback patterns helps DSPy make smarter optimization decisions
- **MIPROv2 Integration**: Enhanced training examples with chronological context and quality signals for better prompt evolution

### **Performance Optimization**
- Start with 1 extraction pass, increase only if needed
- Use 10 workers for balanced performance
- Adjust character buffer based on document complexity
- Monitor API costs with generate mode (higher temperature = more tokens)
- **Enable DSPy** for automatic prompt optimization based on user feedback

### **Template Management & Editing** ✏️
- **Save Templates**: Register templates for reuse across multiple documents
- **Edit Templates**: Modify existing templates with full state preservation
  - Navigate freely between "Define Schema" and "Generate Examples" steps
  - All examples, schema attributes, and prompt descriptions are preserved
  - No data loss when switching between steps during editing
- **Template Dashboard**: View all saved templates with usage statistics
- **Feedback Integration**: Templates automatically benefit from accumulated user feedback
- **Version Control**: Edit templates maintain complete compatibility with existing workflows

## 🔍 Troubleshooting

### **Common Issues**

**"Azure OpenAI endpoint is required"**
- Check your `.env` file has correct `AZURE_OPENAI_ENDPOINT`
- Verify API key or managed identity configuration

**"PDF processing failed"**
- Ensure file is a valid PDF
- Check file size limits (16MB max)
- Try OCR-friendly PDFs (clear text, good quality)

**"No extractions found"**
- Review your schema descriptions
- Add more diverse training examples
- Try increasing extraction passes
- Check if document matches your training examples
- **Provide feedback** on generated examples to improve future results

**"DSPy optimization not working"**
- Check `DSPY_ENABLED=true` in your `.env` file
- Ensure you have provided feedback on examples
- Wait for sufficient feedback (default: 10 entries) to trigger optimization
- Check console logs for DSPy-related messages

### **Performance Tips**
- Use Extract mode for simple factual data
- Reserve Generate mode for complex analysis
- Start with conservative worker settings
- Monitor temperature impact on costs

## 🤝 Contributing

We welcome contributions to Doc Flash! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and ensure they follow the project's coding standards
4. Add tests for new features or bug fixes
5. Run the existing test suite to ensure nothing is broken
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request with a clear description of your changes

### Development Setup
- Follow the installation instructions above
- Install development dependencies: `pip install -r requirements.txt`
- Run tests before submitting PR
- Follow PEP 8 coding standards

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Doc Flash in your research or project, please cite:

```bibtex
@software{doc_flash,
  title={Doc Flash: Intelligent Document Processing Platform},
  author={Lionel Martis},
  year={2025},
  url={https://github.com/LM-150A/doc-flash}
}
```

## 🙏 Acknowledgments

- **[Google LangExtract](https://github.com/google/langextract)**: Core extraction framework for structured data extraction from language models
- **[Stanford DSPy](https://github.com/stanfordnlp/dspy)**: Advanced framework for automatic prompt optimization and program synthesis

---

**Built with ❤️ using LangExtract, FastAPI, and modern web technologies**

*⚡ Flash-fast document processing for the AI age*