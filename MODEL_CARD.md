---
license: mit
language: [ja, zh, ko, en, fr, it, de, es]
base_model: jhu-clsp/mmBERT-base
base_model_relation: finetune
pipeline_tag: token-classification
tags: [pii, personal-information, privacy, token-classification]
---

# mmBERT-pii-sentinel

A multilingual model that finds personal information in text and tells you how sensitive a document is.

Given a document, it reports:

- **the sensitivity level**: `none`, `low` (names, contacts, other personal data) or `high` (health, ID numbers,
  card numbers and other special categories);
- **the values it found**, with their character positions: person names, e-mail addresses, phone numbers, and ID,
  card and account numbers;
- **19 categories** of personal information the document contains, each with a probability.

It covers Japanese, Chinese (Simplified), Korean, English, French, Italian, German and Spanish. It is the model of
the [pii-sentinel](https://github.com/coo-quack/pii-sentinel) tool, which runs it on your own machine together with a
set of regex rules.

## Quick start

The model has its own output heads, so it is run with the pii-sentinel tool, not with a transformers pipeline.
With [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uvx --from git+https://github.com/coo-quack/pii-sentinel@v0.1.2 \
  pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt
```

```text
document.txt: sensitivity high
  person_name: S…
  person_name: D…
  phone: 41…98
```

The model (about 1.2 GB) is downloaded once into the Hugging Face cache; after that the tool works offline and the
scanned text never leaves the machine. Values are masked unless you pass `--show-values`; `--json` gives a full report
and `--fail-on high` makes the command fail for use in CI. See the
[README](https://github.com/coo-quack/pii-sentinel#readme) for all options.

## Intended use

- Checking documents, messages, logs or datasets for personal information before they are shared, published or sent
  to an external service.
- Sorting documents by sensitivity so that the `high` ones get a closer look.

It is a screening aid, not a guarantee: it will miss some personal information, and its output should not be the
only safeguard for data that must not leak. It is not designed for anonymising text for legal purposes or for
languages other than the eight above.

## What it detects

### Sensitivity levels

| Level | Meaning | Examples |
|---|---|---|
| `none` | No personal information. | A release note, a product description, a company's switchboard number. |
| `low` | Personal information. | A name with an e-mail address, a customer's postal address, a date of birth, an IP address or a social media handle tied to a person. |
| `high` | Special categories of personal information. | Health, genetic or biometric data, religion, political opinions, sexual orientation, immigration status, ID and card numbers, HR and criminal records. |

The levels follow GDPR, Japan's Act on the Protection of Personal Information and the CPRA. A famous person who has
died is not counted as personal information; a living one is. The full rules are in the
[labelling policy](https://github.com/coo-quack/pii-sentinel/blob/main/docs/labeling-policy.md).

### Values

| Value | Notes |
|---|---|
| Person names | Full names, surnames or given names alone, nicknames that refer to a specific person. |
| E-mail addresses | Personal and role addresses (`support@`) are told apart. |
| Phone numbers | Personal and corporate numbers are told apart. |
| ID and account numbers | National IDs (such as Japan's My Number), passport and driving licence numbers, card and bank account numbers. |

The tool's regex rules add postal codes, public IP addresses and checksummed ID and card numbers the model missed, and
report secrets (API keys, tokens, passwords in connection strings) separately.

### Categories

Person name, e-mail or phone, postal address, date of birth, government ID, financial account, health, biometric or
genetic data, IP address of a person, social media handle, employment, race or religion, political opinion or union
membership, sex life or orientation, citizenship or immigration, precise location, credentials, private
communications, HR or criminal record.

## Evaluation

Measured on a held-out test set of 320 documents, 40 in each of the eight languages, written for this project to
cover the sensitivity categories, document formats and lengths. Each document was labelled independently twice and
disagreements were adjudicated. The scores are for the released tool (the model, the regex rules and the
post-processing together) and count personal values only.

| | Recall | Precision |
|---|---|---|
| Person names | 97.9% | 96.7% |
| Phone numbers | 94.7% | 90.0% |
| E-mail addresses | 91.7% | 84.6% |
| ID and account numbers | 83.1% | 90.1% |

| Document level | Result |
|---|---|
| Sensitivity (none / low / high) accuracy | 90.0% |
| `high` documents judged `low` or `none` | 2 of 162 |
| Documents with personal information judged `none` | 7 of 246 |

The test set is small: a difference of one or two documents is within noise.

## Limitations

- Trained only on synthetic text; real documents with unusual layouts may be harder.
- Number and address formats are covered for one country per language (for example, Simplified Chinese only).
- The document level is weakest where only a heading reveals the sensitive fact (a member list of a religious
  community) and where a document holds an online identifier or an address without a name.

## Model details

| | |
|---|---|
| Base model | [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), revision `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| Fine-tuning | All encoder layers, with three added heads: span labelling (BIO), document sensitivity, document categories |
| Parameters | 307M |
| Input | Text of any length, read in windows of 512 tokens |
| Files | `model.safetensors` (weights), `pii_sentinel.json` (label sets and settings) |

The span head also learns order, tracking and serial numbers so that they are not mistaken for ID numbers; the tool
does not report them.

## Training data

Synthetic documents generated from templates in the eight languages. Every person, number and address is fictional,
except famous historical figures used as public-figure examples. Labels come from what each template planted, not from
a model. No real personal data and no outputs of other PII models are used.

## License

MIT (see [LICENSE](LICENSE)). The base model is also MIT; its notice is reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This model is not affiliated with or endorsed by the authors of
mmBERT.
