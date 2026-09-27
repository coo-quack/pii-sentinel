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

## Training data

Synthetic documents generated from templates in eight languages (ja, zh, ko, en, fr, it, de, es). Every person, number
and address is fictional, except famous historical figures used as public-figure examples. Labels are derived from the
templates. No real personal data and no outputs of other PII models are used.

## Evaluation

Scores on a held-out multilingual test set are reported with each release.

## Limitations

- Trained only on synthetic text; real documents with unusual layouts may be harder.
- Only the formats of one representative country per language are covered (for example, Simplified Chinese only).

## License

MIT (see [LICENSE](LICENSE)). The base model's license terms are reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
