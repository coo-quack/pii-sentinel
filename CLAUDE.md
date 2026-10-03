# Project Rules

## Overview

Multilingual personal-information detector. A fine-tuned mmBERT-base labels spans (BIO) and judges document
sensitivity and categories; the sensitive-canary regex rules are combined with it at inference.

## Tech Stack

- Python 3.12, uv
- PyTorch, transformers (mmBERT-base at a pinned revision)
- Linter/Formatter: ruff
- Tests: pytest

## Commands

- `uv run pytest` — Run all tests
- `uv run ruff check src tests` / `uv run ruff format src tests` — Lint and format
- `uv run pii-sentinel scan --model <model> FILE` — Run the CLI
- `uv run python -m pii_sentinel.evaluate --model <model> eval/dev.json eval/test.json --out runs` — Evaluate
- `uv run python -m pii_sentinel.gen.generator --out data ...` — Generate training data (see CONTRIBUTING.md)
- `uv run python -m pii_sentinel.train --data data --out <dir>` — Train (needs a GPU)

## Project Structure

```
src/pii_sentinel/
  model.py        # Encoder with span, sensitivity and category heads; save/load
  data.py         # Windows, BIO labels, span decoding and trimming
  predict.py      # Inference, post-processing, rule combination
  rules.py        # sensitive-canary rule engine (canary_rules.json)
  evaluate.py     # Scoring against eval corpora
  train.py        # Training loop
  cli.py          # pii-sentinel scan / serve
  server.py       # serve: HTTP on a Unix socket or loopback address
  testset.py      # Building and labelling the evaluation set
  gen/            # Synthetic training data generator and slot templates
templates/        # Slot templates (hand-written documents with placeholders)
eval/             # dev.json (committed), coverage.json, reference sets; test.json is not committed
docs/             # Labelling policy and guides
```

## Key Constraints

- Gold labels follow `docs/labeling-policy.md`; never change a label to match the model's output.
- `eval/test.json` is used for aggregate scores only; error analysis uses `eval/dev.json`.
- Names that appear in evaluation text are excluded from training data generation.
- Documentation and code comments are written in English.
