# pii-sentinel

Multilingual personal-information detector. It finds names, personal contacts and identification numbers as
character spans, and judges how sensitive a document is (none / low / high) with 13 categories.

The model, **mmBERT-pii-sentinel**, is a fine-tuned derivative of
[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base). See [MODEL_CARD.md](MODEL_CARD.md).

Languages trained and evaluated: Japanese, Chinese (Simplified), Korean, English, French, Italian, German, Spanish.

## Use

```sh
uv run pii-sentinel scan --model models/<checkpoint> document.txt        # values masked
uv run pii-sentinel scan --model models/<checkpoint> --json --show-values - < document.txt
```

## Develop

```sh
uv sync
uv run pytest
uv run python -m pii_sentinel.gen.generator --out data --train-docs 2000 --templates templates \
  --exclude "$(ls eval/*.json eval/reference/*.json eval/work/raw/*.json | paste -sd, -)"
uv run python -m pii_sentinel.train --data data --out models/mmBERT-pii-sentinel
uv run python -m pii_sentinel.evaluate --model models/mmBERT-pii-sentinel eval/dev.json eval/test.json --out runs
```

Training data is generated from templates; labels come from what each template planted, not from a model.
Names, brands and public figures that appear in the evaluation corpora are excluded from generation.
