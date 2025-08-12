# Contributing to Doc Flash

Thank you for your interest in contributing to Doc Flash! 🎉

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/doc-flash.git
   cd doc-flash
   ```
   
   Or clone the main repository:
   ```bash
   git clone https://github.com/LM-150A/doc-flash.git
   cd doc-flash
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

## Code Quality Standards

We maintain high code quality standards using automated tools:

### Code Formatting
- **Black**: Automatic code formatting
- **isort**: Import statement sorting

### Linting
- **Flake8**: Style guide enforcement
- **Pylint**: Code analysis
- **MyPy**: Type checking (optional)

### Security
- **Bandit**: Security issue detection
- **Safety**: Dependency vulnerability scanning

### Running Quality Checks
```bash
# Format code
black docflash/
isort docflash/

# Check formatting
black --check docflash/
isort --check-only docflash/

# Lint code
flake8 docflash/
pylint docflash/

# Security scan
bandit -r docflash/
safety check
```

## Testing

```bash
# Test import structure
python -c "from docflash import app; print('✅ Import successful')"

# Run with different Python versions (if available)
python3.8 start_fastapi.py
python3.9 start_fastapi.py
python3.10 start_fastapi.py
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards
   - Add tests for new features
   - Update documentation if needed

3. **Run quality checks**
   ```bash
   black docflash/
   flake8 docflash/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: your descriptive commit message"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Contribution Guidelines

### Code Style
- Follow PEP 8 style guide
- Use descriptive variable and function names
- Add docstrings for public functions
- Keep functions focused and small

### Commit Messages
Use conventional commits format:
- `feat: add new extraction mode`
- `fix: resolve PDF upload issue`
- `docs: update API documentation`
- `refactor: improve error handling`

### Areas for Contribution

#### 🚀 **High Priority**
- New LLM provider integrations
- DSPy optimization improvements
- Performance optimizations
- UI/UX improvements
- Additional file format support
- Reinforcement learning enhancements

#### 🔧 **Medium Priority**
- Additional extraction templates
- Feedback system improvements
- Better error handling
- Documentation improvements
- Test coverage expansion
- Advanced DSPy module development

#### 📚 **Low Priority**
- Code refactoring
- Minor bug fixes
- Documentation typos

## LLM Provider Testing

When testing changes:

1. **Test with multiple providers**
   - Azure OpenAI
   - OpenAI
   - Google Gemini
   - Ollama (local models)

2. **Test different input methods**
   - PDF upload with OCR
   - Direct text paste
   - Mixed workflows

3. **Test various document types**
   - Contracts
   - Invoices
   - Reports
   - Emails

## Getting Help

- 📖 Check the [README.md](README.md) for basic setup
- 🔧 Review [CONFIGURATION.md](CONFIGURATION.md) for advanced setup
- 🐛 Open an issue for bugs or questions
- 💬 Start a discussion for feature ideas

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a positive environment

Thank you for contributing to Doc Flash! 🙏