---
license: mit
language: [ja, zh, ko, en, fr, it, de, es]
base_model: jhu-clsp/mmBERT-base
base_model_relation: finetune
pipeline_tag: token-classification
tags: [pii, personal-information, privacy, token-classification]
---

# mmBERT-pii-sentinel

A fine-tuned derivative of [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)
(revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`). All encoder layers are fine-tuned; three heads are added:

- span labelling (BIO) for person names, personal and generic e-mail addresses, personal and corporate phone numbers,
  national / government IDs, passport and driving licence numbers, card and bank account numbers, order / tracking
  and serial numbers (order, tracking and serial numbers are learnt but not reported)
- document sensitivity: none / low / high
- 19 document categories (person name, contact, address, date of birth, government ID, financial account, health,
  biometric or genetic, IP address of a person, SNS handle, employment, race or religion, political opinion or union,
  sex life or orientation, citizenship or immigration, precise location, credentials, private communications,
  HR or criminal record)

This model is not affiliated with or endorsed by the authors of mmBERT.

## Usage

The checkpoint has its own heads, so it is loaded with the pii-sentinel tool rather than a transformers pipeline:

```sh
git clone https://github.com/coo-quack/pii-sentinel.git && cd pii-sentinel && uv sync
uv run pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt
```

The model files are `model.safetensors` (weights) and `pii_sentinel.json` (label sets and training settings).

## Training data

Synthetic documents generated from templates in eight languages (ja, zh, ko, en, fr, it, de, es). Every person, number
and address is fictional, except famous historical figures used as public-figure examples. Labels are derived from the
templates. No real personal data and no outputs of other PII models are used.

## Evaluation

The test set has 320 documents, 40 in each of the eight languages, written for this project to a coverage table of
sensitivity categories, document formats and lengths. Each document was labelled independently twice, and
disagreements were adjudicated. A separate development set of the same size is used for error analysis.

Scores are for the released tool (the model, the regex rule set and the post-processing together) and count personal
values only.

| Metric | Test |
|---|---|
| Person names: recall / precision | 97.9% / 96.7% |
| Phone numbers: recall / precision | 94.7% / 90.0% |
| E-mail addresses: recall / precision | 91.7% / 84.6% |
| ID and account numbers: recall / precision | 83.1% / 90.1% |
| Sensitivity (none / low / high) accuracy | 90.0% |
| High documents judged low or none | 2 of 162 |
| Documents with personal information judged none | 7 of 246 |

## Limitations

- Trained only on synthetic text; real documents with unusual layouts may be harder.
- Only the formats of one representative country per language are covered (for example, Simplified Chinese only).
- The test set is also synthetic and small; a difference of one or two documents is within noise.
- The document level is weakest where only a heading reveals the sensitive fact (a member list of a religious
  community) and where a document holds an online identifier or an address without a name.

## License

MIT (see [LICENSE](LICENSE)). The base model's license terms are reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
