# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | ✅ Yes             |
| 1.x.x   | ❌ No              |

## Reporting a Vulnerability

If you discover a security vulnerability in Doc Flash, please report it responsibly:

### 🔒 **Private Disclosure**
- **Email**: Send details to the repository owner via GitHub
- **Don't** open public issues for security vulnerabilities
- **Include**: Steps to reproduce, impact assessment, and suggested fixes

### 📋 **What to Include**
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and attack scenarios
- Your assessment of severity
- Any suggested mitigations or fixes

### ⚡ **Response Timeline**
- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next release cycle

## Security Considerations

### 🔐 **API Keys & Credentials**
- Never commit API keys to the repository
- Use environment variables for all sensitive data
- Rotate keys regularly
- Use managed identities when possible (Azure)

### 📄 **Document Processing**
- Files are processed temporarily and cleaned up
- No persistent storage of document content
- OCR processing happens locally or via secure APIs

### 🌐 **Network Security**
- All API communications use HTTPS
- Rate limiting implemented where applicable
- Input validation on all endpoints

### 🛡️ **Dependencies**
- Regular dependency updates via Dependabot
- Security scanning with `safety` and `bandit`
- Automated vulnerability detection in CI/CD

## Security Best Practices for Users

### 🏢 **Production Deployment**
- Use HTTPS for all connections
- Implement proper authentication
- Set up network firewalls
- Monitor for suspicious activity
- Regular backups and disaster recovery

### 🔑 **API Key Management**
- Store keys in secure key management systems
- Use principle of least privilege
- Monitor API usage and costs
- Set up alerts for unusual activity

### 📊 **Data Handling**
- Review documents before processing
- Be aware of data residency requirements
- Consider data classification levels
- Implement data retention policies

## Automated Security Measures

- **Bandit**: Static security analysis for Python code
- **Safety**: Dependency vulnerability scanning
- **GitHub Security Advisories**: Automated vulnerability detection
- **Dependabot**: Automated dependency updates

## Contact

For security-related questions or concerns, please contact the repository maintainers through GitHub's private vulnerability reporting feature.