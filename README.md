# pii-sentinel

Multilingual personal-information detector. It finds names, personal contacts and identification numbers as
character spans, and judges how sensitive a document is (none / low / high) with 19 categories.

The model, **mmBERT-pii-sentinel**, is a fine-tuned derivative of
[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), published at
[coo-quack/mmBERT-pii-sentinel](https://huggingface.co/coo-quack/mmBERT-pii-sentinel). See [MODEL_CARD.md](MODEL_CARD.md).

Languages trained and evaluated: Japanese, Chinese (Simplified), Korean, English, French, Italian, German, Spanish.

## Use

```sh
uv run pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt   # values masked
uv run pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel --json --show-values - < document.txt
```

`--model` takes a Hugging Face model id (downloaded once into the local cache) or a local checkpoint directory.

The model's findings are combined with the regex rules of
[sensitive-canary](https://github.com/coo-quack/sensitive-canary) (`src/pii_sentinel/canary_rules.json`):

- a value the model did not mark but a rule recognises (a checksummed national ID or card number, an e-mail
  address, a phone number, a postal code, a public IP address) is added as a finding; a checksummed ID or card
  number also makes the document high;
- secrets (API keys, tokens, private keys, credentials in URLs and connection strings) are reported under
  `secrets`, apart from personal information, and do not change the sensitivity level.

## Develop

```sh
uv sync
uv run pytest
uv run python -m pii_sentinel.gen.generator --out data --train-docs 2200 --templates templates \
  --template-weight 110 \
  --exclude "$(ls eval/*.json eval/reference/*.json eval/work/raw/*.json | paste -sd, -)"
uv run python -m pii_sentinel.train --data data --out models/mmBERT-pii-sentinel
uv run python -m pii_sentinel.evaluate --model models/mmBERT-pii-sentinel eval/dev.json eval/test.json --out runs
# --no-rules scores the model alone, without the rule set
```

Training data is generated from templates; labels come from what each template planted, not from a model.
Names, brands and public figures that appear in the evaluation corpora are excluded from generation.
