# Changelog

## v0.5.0 (2026-10-18)

The model is retrained (21 span types, 1024-token windows, label smoothing) and the post-processing changed.

- The model marks six more kinds of personal information as findings: `postal_address`, `date_of_birth`,
  `employment_info` (employer, department, job title or a title next to the name), `sns_handle` (handle or
  login name), `ip_address` with `pii` telling a person's address from a server's, and `number` with
  `number_type: member_or_customer`. Company addresses, brand accounts and server IPs are reported with
  `pii: false`.
- A personal postal address, date of birth or handle makes the document at least `low`, as a personal contact or
  number already did.
- `categories` is documented: the values are probabilities, and the README says which ones to read and at what
  threshold.
- A checkpoint carries its own span label set, so the published model (12 span types) and the new one (21) both
  load.
- The evaluation leaves out predicted finding types and number types that the corpus does not label, and reports
  them as `unscored`.
- The training data generator labels the new span types, with new slot templates for addresses, dates of birth,
  workplaces, handles, IPs and member numbers in all eight languages, and larger job title lists.
- The generator draws its distribution from `src/pii_sentinel/gen/coverage.json` (sensitivity, length and format
  shares, minimum span and category rates) and reports the deviations after a run.
- An address is reported as personal only when a person is named within a few lines of it; an e-mail address
  with a role word in its local part (hr-team, services.rh) is reported as not personal.
- Training: `--label-smoothing` (default 0.1), `--select-on eval/dev.json` to choose the checkpoint by the
  acceptance criteria on the development set, and `python -m pii_sentinel.average` to average checkpoints.
- The labelling policy gains a section on attribute and identifier spans (company addresses, deceased persons,
  public figures' offices, login names, partial addresses).

## v0.4.0 (2026-10-10)

The model weights are unchanged from v0.2.0.

- When values are masked (the default), a secret is reported with the 1-based `line` it starts on instead of its
  `start` and `end`, which revealed its length. With `--show-values` or `"show_values": true` a secret still has
  `start` and `end`. Readers of the JSON report that used the offsets of masked secrets need to change.
- `serve --verbose` escapes control characters in the request line it logs again, as the standard library does, so
  a local client can no longer send terminal escape sequences or forge log lines.
- Update transformers to 5.19.0.

## v0.3.0 (2026-10-03)

The model weights are unchanged from v0.2.0.

- `pii-sentinel serve` keeps the model loaded and scans text sent over HTTP, on a Unix socket (mode 600) or a
  loopback address. It refuses foreign `Host` headers and non-JSON bodies, never logs the text, and answers with the
  report of `scan --json`.
- `--threads` limits the CPU threads PyTorch uses.

## v0.2.0 (2026-10-03)

The model is published in the layout of a transformers `ModernBertForTokenClassification` model. The weights are
the same as in v0.1.x, converted: the development and test sets give the same result for every document.

- The Hub repository has `config.json` and the tokenizer files, so `pipeline("token-classification")` loads the
  model directly, without `trust_remote_code` and without fetching mmBERT-base at run time.
- The sensitivity and category heads move to `document_heads.safetensors`, read by this tool only.
- The model card follows the layout of the mmBERT cards and reports the scores of the transformers pipeline alone
  next to those of the tool.
- The tool still reads checkpoints in the earlier layout.

## v0.1.2 (2026-10-03)

The model weights and the tool's output are unchanged from v0.1.1: the development and test sets give the same
result for every document.

- The README and the model card (the Hugging Face page) are rewritten for users: what the tool reports, a one-line
  `uvx` quick start, the options, use in CI and offline, the meaning of the levels, findings and JSON fields.
- The training and evaluation commands move to CONTRIBUTING.md.
- Dependencies: torch 2.14.1, transformers 5.18.0, regex 2026.9.29.

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
