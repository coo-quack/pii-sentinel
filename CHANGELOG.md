# Changelog

## v0.1.0 (2026-09-28)

- First release of mmBERT-pii-sentinel: a fine-tuned mmBERT-base that finds person names, personal and company
  contacts and ID and account numbers as character spans, and judges document sensitivity (none / low / high)
  with 19 categories, in Japanese, Chinese, Korean, English, French, Italian, German and Spanish.
- The sensitive-canary regex rules are combined with the model: values the model missed are added, and secrets are
  reported separately under `secrets`.
- `pii-sentinel scan` CLI with masked values by default, `--json`, `--show-values` and `--fail-on`.
- Test set results are in `MODEL_CARD.md`.
