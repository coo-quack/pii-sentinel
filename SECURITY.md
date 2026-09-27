# Security Policy

## Data Handling

pii-sentinel runs locally. Scanned text is not sent anywhere. Network access is used only to download the model
files and the base tokenizer from Hugging Face on first use.

## No Guarantee

Detection is statistical. The model misses some names, numbers and sensitive documents and flags some text that is
not personal information. Use the findings and sensitivity levels as a screening aid, not as a legal assessment.

## Limitations

- Trained on synthetic documents; real documents with unusual layouts may be harder
- A sensitive fact revealed only by a heading, or an online identifier without a name, is judged less reliably
- Addresses, dates of birth, SNS handles and passwords affect the document level but are not returned as findings

## Reporting Security Issues

If you discover a security vulnerability in pii-sentinel, please report it to:

- **Email:** dev@quack.jp
- **GitHub:** [Open a security advisory](https://github.com/coo-quack/pii-sentinel/security/advisories/new)

**Please do NOT:**
- Open public GitHub issues for security vulnerabilities
- Disclose the issue publicly before we've had a chance to address it

We aim to respond to security reports within 48 hours.

## License

pii-sentinel is released under the [MIT License](LICENSE). It is provided "AS IS" without warranty of any kind.
