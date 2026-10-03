---
license: mit
language: [ja, zh, ko, en, fr, it, de, es]
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-base
base_model_relation: finetune
tags: [pii, personal-information, privacy, ner, token-classification, modernbert]
---

# mmBERT-pii-sentinel: Multilingual Personal-Information Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-pii--sentinel-black)](https://github.com/coo-quack/pii-sentinel)
[![Base model](https://img.shields.io/badge/🤗%20Base%20model-mmBERT--base-blue)](https://huggingface.co/jhu-clsp/mmBERT-base)

> TL;DR: A fine-tuned [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) that finds person names, contacts
> and ID numbers in eight languages, and judges how sensitive a document is.

mmBERT-pii-sentinel labels personal information in text as character spans: person names, e-mail addresses, phone
numbers, and ID, card and account numbers. It loads as a standard `ModernBertForTokenClassification` model, so the
transformers `pipeline` runs it as it is. Two more heads judge the whole document: a sensitivity level (`none`, `low`
or `high`) and 19 categories of personal information. Those are read by the
[pii-sentinel](https://github.com/coo-quack/pii-sentinel) tool, which also adds a regex rule set and runs everything on
your own machine.

## Table of Contents

- [Quick Start](#quick-start)
- [Model Description](#model-description)
- [Labels](#labels)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Training Data](#training-data)
- [Limitations](#limitations)
- [License](#license)

## Quick Start

### Installation

```bash
pip install torch transformers
```

Tested with transformers 5.18 and torch 2.14.

### Usage

```python
from transformers import pipeline

ner = pipeline("token-classification", model="coo-quack/mmBERT-pii-sentinel", aggregation_strategy="simple")
for entity in ner("Hi, this is Emily Carter. Call me at +1 415 555 0142 or emily.carter@example.org."):
    print(entity["entity_group"], entity["word"].strip(), round(float(entity["score"]), 3))
```

```text
PERSON Emily Carter 1.0
PHONE +1 415 555 0142 0.98
EMAIL emily.carter@example.org 1.0
```

The model was fine-tuned on windows of 512 tokens; split longer texts, or use the tool below, which does it for you.

### Full analysis with the pii-sentinel tool

For the document's sensitivity level and categories, the regex rules and the post-processing, use the command-line
tool. It needs [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uvx --from git+https://github.com/coo-quack/pii-sentinel@v0.2.0 \
  pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt
```

```text
document.txt: sensitivity high
  person_name: S…
  person_name: D…
  phone: 41…98
```

The scanned text never leaves the machine; the only network access is the one-time download of the model. See the
[README](https://github.com/coo-quack/pii-sentinel#readme) for the options, the JSON report and use in CI.

## Model Description

Intended for checking documents, messages, logs or datasets for personal information before they are shared,
published or sent to an external service, and for sorting documents by sensitivity so that the `high` ones get a
closer look.

It is a screening aid, not a guarantee: it will miss some personal information, and it should not be the only
safeguard for data that must not leak. It is not designed for anonymising text for legal purposes.

### Sensitivity levels

| Level | Meaning | Examples |
|:------|:--------|:---------|
| `none` | No personal information. | A release note, a product description, a company's switchboard number. |
| `low` | Personal information. | A name with an e-mail address, a customer's postal address, a date of birth, an IP address or a social media handle tied to a person. |
| `high` | Special categories of personal information. | Health, genetic or biometric data, religion, political opinions, sexual orientation, immigration status, ID and card numbers, HR and criminal records. |

The levels follow GDPR, Japan's Act on the Protection of Personal Information and the CPRA. A famous person who has
died is not counted as personal information; a living one is. The full rules are in the
[labelling policy](https://github.com/coo-quack/pii-sentinel/blob/main/docs/labeling-policy.md).

### Categories

Person name, e-mail or phone, postal address, date of birth, government ID, financial account, health, biometric or
genetic data, IP address of a person, social media handle, employment, race or religion, political opinion or union
membership, sex life or orientation, citizenship or immigration, precise location, credentials, private
communications, HR or criminal record.

## Labels

The token-classification head uses BIO tags over these entity groups:

| Entity group | What it marks |
|:-------------|:--------------|
| `PERSON` | A person's name: full name, surname or given name alone, a nickname that refers to a specific person. |
| `EMAIL` | A person's e-mail address. |
| `EMAIL_GENERIC` | A role address such as `support@` (not personal). |
| `PHONE` | A person's phone number. |
| `PHONE_CORPORATE` | A company's or department's number (not personal). |
| `MY_NUMBER` | Japan's individual number. |
| `NATIONAL_ID` | Another national or government ID number. |
| `PASSPORT_LICENCE` | A passport or driving licence number. |
| `CREDIT_CARD` | A payment card number. |
| `BANK_ACCOUNT` | A bank account number. |
| `ORDER_TRACKING` | An order or tracking number (not personal; learnt so that it is not mistaken for an ID). |
| `SERIAL` | A product serial number (not personal; same reason). |

## Evaluation

Measured on a held-out test set of 320 documents, 40 in each of the eight languages, written for this project to
cover the sensitivity categories, document formats and lengths. Each document was labelled independently twice and
disagreements were adjudicated. Only personal values are counted.

| Recall / precision | pii-sentinel tool | transformers pipeline |
|:-------------------|:------------------|:----------------------|
| Person names | 97.9% / 96.7% | 97.0% / 96.7% |
| Phone numbers | 94.7% / 90.0% | 94.7% / 90.0% |
| E-mail addresses | 91.7% / 84.6% | 87.5% / 63.6% |
| ID and account numbers | 83.1% / 90.1% | 76.6% / 50.0% |

The tool column is the model with the regex rules and the post-processing (which, for example, drops hashes and UUIDs
that look like numbers); the pipeline column is this model alone with `aggregation_strategy="simple"`.

| Document level (tool only) | Result |
|:---------------------------|:-------|
| Sensitivity (none / low / high) accuracy | 90.0% |
| `high` documents judged `low` or `none` | 2 of 162 |
| Documents with personal information judged `none` | 7 of 246 |

The test set is small: a difference of one or two documents is within noise.

## Model Architecture

| Parameter | Value |
|:----------|:------|
| Base model | [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), revision `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| Architecture | `ModernBertForTokenClassification`, plus document heads |
| Layers | 22 |
| Hidden size | 768 |
| Attention heads | 12 |
| Parameters | 308M |
| Vocabulary size | 256,000 (Gemma 2 tokenizer) |
| Fine-tuning window | 512 tokens |
| Languages trained and evaluated | ja, zh (Simplified), ko, en, fr, it, de, es |

All encoder layers are fine-tuned. The files are `model.safetensors` and `config.json` (the token-classification
model), the tokenizer files, and `document_heads.safetensors` with `pii_sentinel.json` (the sensitivity and category
heads, read by the pii-sentinel tool).

## Training Data

Synthetic documents generated from templates in the eight languages. Every person, number and address is fictional,
except famous historical figures used as public-figure examples. Labels come from what each template planted, not from
a model. No real personal data and no outputs of other PII models are used.

## Limitations

- Trained only on synthetic text; real documents with unusual layouts may be harder.
- Number and address formats are covered for one country per language (for example, Simplified Chinese only).
- mmBERT-base was pre-trained on many more languages, but this model was fine-tuned and evaluated on the eight above
  only.
- The document level is weakest where only a heading reveals the sensitive fact (a member list of a religious
  community) and where a document holds an online identifier or an address without a name.

## License

MIT (see [LICENSE](LICENSE)). The base model is also MIT; its notice is reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This model is not affiliated with or endorsed by the authors of
mmBERT.
