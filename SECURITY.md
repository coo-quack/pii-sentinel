# Security

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
