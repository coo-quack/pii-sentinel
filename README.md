# pii-sentinel

Find personal information in text before you share it.

pii-sentinel reads a document and tells you:

- **how sensitive it is**: `none`, `low` (names, contacts, other personal data) or `high` (health, ID numbers,
  card numbers and other special categories);
- **what it found and where**: person names, personal e-mail addresses and phone numbers, ID, card and account
  numbers, with their character positions;
- **which secrets it contains**: API keys, tokens, private keys and passwords in URLs or connection strings.

It works on Japanese, Chinese (Simplified), Korean, English, French, Italian, German and Spanish, and runs on your
own machine: the text you scan is never sent anywhere. The only network access is the one-time download of the
model ([coo-quack/mmBERT-pii-sentinel](https://huggingface.co/coo-quack/mmBERT-pii-sentinel), about 1.2 GB) from
Hugging Face.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs a suitable Python (3.12 or
later) by itself.

```sh
uvx --from git+https://github.com/coo-quack/pii-sentinel@v0.1.2 \
  pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt
```

The first run downloads the model and takes about a minute; later runs start in a few seconds.

```text
document.txt: sensitivity high
  person_name: S…
  person_name: D…
  phone: 41…98
```

Values are masked by default so that the report itself does not leak them. Add `--show-values` to see them in full.

To run it often, clone the repository instead, so that each run does not resolve the package again:

```sh
git clone https://github.com/coo-quack/pii-sentinel.git && cd pii-sentinel && uv sync
uv run pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt
```

## Usage

```sh
pii-sentinel scan --model MODEL [options] [FILE ...]
```

Pass one or more files, or `-` (or nothing) to read from standard input. Scanning many files in one run is much
faster than one run per file, because the model is loaded only once.

| Option | Meaning |
|---|---|
| `--model MODEL` | A Hugging Face model id (`coo-quack/mmBERT-pii-sentinel`) or a local model directory. Required. |
| `--json` | Print a JSON report instead of text. |
| `--show-values` | Show the found values in full instead of masked. |
| `--fail-on low\|high` | Exit with status 2 when any document is at this level or above. |
| `--device DEVICE` | `cuda`, `mps` or `cpu`. By default the GPU is used when there is one (including Apple silicon). |

### Block sensitive files in CI or a Git hook

```sh
pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel --fail-on high $(git diff --cached --name-only --diff-filter=d)
```

Only text files can be scanned; a binary file stops the run. The command exits with 0 when every document is below the level and with 2 when one reaches it. Note that a wrong
command line also exits with 2.

### Use it offline

After the first download the model stays in the Hugging Face cache (`~/.cache/huggingface`). Set
`HF_HUB_OFFLINE=1` to run without any network access:

```sh
HF_HUB_OFFLINE=1 uv run pii-sentinel scan --model coo-quack/mmBERT-pii-sentinel document.txt
```

## What it reports

### Sensitivity levels

| Level | Meaning | Examples |
|---|---|---|
| `none` | No personal information. | A release note, a product description, a company's switchboard number. |
| `low` | Personal information. | A name with an e-mail address, a customer's postal address, a date of birth, an IP address or a social media handle tied to a person. |
| `high` | Special categories of personal information. | Health, genetic or biometric data, religion, political opinions, sexual orientation, immigration status, ID and card numbers, HR and criminal records. |

The levels follow GDPR, Japan's Act on the Protection of Personal Information and the CPRA. A famous person who has
died is not counted as personal information; a living one is. The full rules are in
[docs/labeling-policy.md](docs/labeling-policy.md).

### Findings

| `type` | What it is | `number_type` |
|---|---|---|
| `person_name` | A person's name. | |
| `email` | An e-mail address. | |
| `phone` | A phone number. | |
| `number` | An ID, card or account number. | `my_number`, `national_id`, `driver_licence_or_passport`, `credit_card`, `bank_account`, `business_registration` |
| `postal_code` | A postal code. | |
| `ip_address` | A public IP address. | |

Each finding has `pii`: `true` when it points to a person, `false` when it does not (a company's hotline, an address
such as `support@`). In the text report the latter are marked `[not PII]`.

Secrets are listed separately under `secrets` with the rule that matched them. They do not change the sensitivity
level, which is about personal information only.

### JSON report

`--json` prints one object per document (`categories` trimmed here; the report lists all 19 with their
probabilities):

```json
{
  "source": "document.txt",
  "sensitivity": {
    "level": "high",
    "probabilities": { "none": 0.0000079, "low": 0.0000080, "high": 0.9999841 }
  },
  "categories": { "person_name": 0.9995, "email_or_phone": 0.9963, "health_info": 0.9982 },
  "findings": [
    { "type": "person_name", "value": "S…", "start": 3, "end": 8, "pii": true, "label": "PERSON" },
    { "type": "person_name", "value": "D…", "start": 30, "end": 42, "pii": true, "label": "PERSON" },
    { "type": "phone", "value": "41…98", "start": 117, "end": 129, "pii": true, "label": "PHONE" }
  ],
  "secrets": [],
  "windows": 1
}
```

- `start` and `end` are character offsets into the text (`end` is exclusive).
- `sensitivity.probabilities` is the model's own judgement. `level` can be higher than its most likely class,
  because the rules below can raise it.
- `windows` is the number of 512-token pieces the text was split into; any length of text can be scanned.

## How it works

A fine-tuned [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) model marks names, contacts and numbers in
the text and judges the document's level and categories. The regex rules of
[sensitive-canary](https://github.com/coo-quack/sensitive-canary) then check its work:

- a value the model missed but a rule recognises (a national ID or card number with a valid checksum, an e-mail
  address, a phone number, a postal code, a public IP address) is added;
- an ID or card number with a valid checksum makes the document `high`, whichever of the two found it;
- a personal contact or number makes the document at least `low`;
- secrets are found by the rules alone.

## Accuracy

Measured on a held-out test set of 320 documents, 40 in each language, with the model and the rules together:

| | Recall | Precision |
|---|---|---|
| Person names | 97.9% | 96.7% |
| Phone numbers | 94.7% | 90.0% |
| E-mail addresses | 91.7% | 84.6% |
| ID and account numbers | 83.1% | 90.1% |

The sensitivity level is right for 90.0% of the documents. 2 of the 162 `high` documents were judged lower, and 7 of
the 246 documents with personal information were judged `none`. See the [model card](MODEL_CARD.md) for details.

## Limitations

- pii-sentinel will miss some personal information. Use it as one check among others, not as a guarantee.
- It was trained on synthetic text only. Real documents with unusual layouts may be harder.
- Number and address formats are covered for one country per language (for example, Simplified Chinese only).
- Languages other than the eight above are not evaluated.
- A document whose only sensitive fact is in a heading (a member list of a religious community) is often judged too
  low.

## Contributing

To train or evaluate the model, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. The model is a derivative of mmBERT-base, which is also MIT; its notice is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project is not affiliated with the authors of mmBERT.
