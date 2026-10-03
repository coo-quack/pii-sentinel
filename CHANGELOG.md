# Changelog

## v0.1.2 (2026-10-02)

Documentation only; the model weights and the tool's behaviour are unchanged from v0.1.1.

- The README and the model card (the Hugging Face page) are rewritten for users: what the tool reports, a one-line
  `uvx` quick start, the options, use in CI and offline, the meaning of the levels, findings and JSON fields.
- The training and evaluation commands move to CONTRIBUTING.md.

## v0.1.1 (2026-09-28)

The model weights are unchanged from v0.1.0; this release fixes inference and the tooling around it.

- A checksummed ID or card number makes the document high also when the model, not a rule, found it. Before, the
  rule's level was lost whenever the model had marked the same value, and when two rules matched one value the
  first one decided the level.
- A model finding inside a secret is dropped (the password before the "@" of a connection string was reported as
  an e-mail address).
- `sensitivity.probabilities` is the judgement of the window that decided the level, so its most likely class
  matches the model's level; before, it was the maximum of each class over all windows.
- Windows go through the encoder in batches, so a very long input no longer needs memory for all windows at once.
- A stride that would not advance through the text is rejected instead of looping forever.
- The CLI hides a one-character value entirely when masking.
- Test set: person-name precision 96.7% (was 96.6%); every other score is unchanged.

Development:

- The evaluation prints rates from the counts (a recall of 621/634 was printed as 98.0% instead of 97.9%).
- Training data generation: the same `--seed` now gives the same data in every run; evaluation names are matched
  as whole words in Latin script, so fewer training names are left out; a person known only by an identifier keeps
  the `person_name` category; a missing `--templates` directory stops the run; generator strings that matched
  evaluation text were rewritten; a password placeholder is allowed only in high templates; long documents now
  often exceed one window.
- Training: with per-window document pooling, each document's sensitivity and categories are trained on the
  maximum over its windows, so a window without the deciding fact is not trained to report it.
- Input checks raise errors instead of using `assert`, and the generator has tests.

## v0.1.0 (2026-09-28)

- First release of mmBERT-pii-sentinel: a fine-tuned mmBERT-base that finds person names, personal and company
  contacts and ID and account numbers as character spans, and judges document sensitivity (none / low / high)
  with 19 categories, in Japanese, Chinese, Korean, English, French, Italian, German and Spanish.
- The sensitive-canary regex rules are combined with the model: values the model missed are added, and secrets are
  reported separately under `secrets`.
- `pii-sentinel scan` CLI with masked values by default, `--json`, `--show-values` and `--fail-on`.
- Test set results are in `MODEL_CARD.md`.
