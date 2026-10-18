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
uvx --from git+https://github.com/coo-quack/pii-sentinel@v0.5.0 \
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
Masked secrets are reported with their line number instead of their offsets (see [JSON report](#json-report)).

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
| `--show-values` | Show the found values in full, and the offsets of secrets, instead of masked. |
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

### Keep it running as a local server

Each `scan` loads the model again, which takes a few seconds. To scan many texts from another program, keep the
model loaded with `serve`:

```sh
uv run pii-sentinel serve --model coo-quack/mmBERT-pii-sentinel --socket /tmp/pii-sentinel.sock
```

```sh
curl --unix-socket /tmp/pii-sentinel.sock -H 'Content-Type: application/json' \
  -d '{"text": "Hi, this is Emily Carter."}' http://localhost/scan
```

The answer is the JSON report of one document, as `scan --json` prints it. Pass `"show_values": true` for the
values in full (with the offsets of secrets) and `"rules": false` for the model alone.
`GET /health` tells whether the server is up.

- It listens on a Unix socket that only you can open (mode 600), or with `--host 127.0.0.1 --port 8765` on a
  loopback address; other addresses are refused. Over TCP, any program on the machine can call it.
- It never logs or stores the text it receives; `--verbose` logs only the method, path and status.
- One request is handled at a time. Bodies above 1 MB are refused (`--max-bytes`).
- `--threads N` limits the CPU threads PyTorch uses (also for `scan`).
- A socket path can be at most 103 bytes long.

### Use the model from transformers

The model is a standard `ModernBertForTokenClassification`, so it also runs without this tool:

```python
from transformers import pipeline

ner = pipeline("token-classification", model="coo-quack/mmBERT-pii-sentinel", aggregation_strategy="simple")
print(ner("Hi, this is Emily Carter. Call me at +1 415 555 0142."))
```

This gives the names, contacts and numbers only, without the sensitivity level, the rules or the post-processing,
and its precision for e-mail addresses and numbers is lower (see the
[model card](https://huggingface.co/coo-quack/mmBERT-pii-sentinel#evaluation)). Texts longer than 1024 tokens should
be split first.

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
| `number` | An ID, card, account, member or customer number. | `my_number`, `national_id`, `driver_licence_or_passport`, `credit_card`, `bank_account`, `member_or_customer`, `business_registration` |
| `postal_code` | A postal code. | |
| `ip_address` | An IP address. | |
| `postal_address` | A street-level postal address. | |
| `date_of_birth` | A date of birth. | |
| `employment_info` | A person's employer, department, job title or role. | |
| `sns_handle` | A social-media handle or login name. | |

Each finding has `pii`: `true` when it points to a person, `false` when it does not (a company's hotline, an address
such as `support@`, a company's postal address, a brand's handle, a server's IP address). In the text report the
latter are marked `[not PII]`.

Secrets are listed separately under `secrets` with the rule that matched them. They do not change the sensitivity
level, which is about personal information only.

### Categories

`categories` (JSON report and server only) gives, for each of 19 kinds of personal information, the model's
probability that the document contains it. The values are independent of each other (they do not add up to 1),
come from the model alone (the rules do not adjust them) and are the highest value over the 1024-token pieces of
the text. Read a category by comparing its value with a threshold: `postal_address >= 0.5` means "this document
probably contains someone's postal address".

For a postal address, a date of birth, a workplace, a handle or a person's IP address, prefer the findings: they
carry the value and its position, and a `postal_address`, `date_of_birth` or `sns_handle` finding with
`pii: true` also raises the level to at least `low`. Measured per document on the same 320 held-out documents
(a document counts as found when the model reports a finding of that type with `pii: true`):

| Finding | Recall | Precision | Documents containing it |
|---|---|---|---|
| `date_of_birth` | 90% | 86% | 20 |
| `postal_address` | 94% | 70% | 17 |
| `ip_address` (pii) | 4 of 4 | 4 of 6 | 4 |
| `sns_handle` | 3 of 4 | 3 of 6 | 4 |
| `employment_info` | 55% | 88% | 112 |

To block on a date of birth or a postal address, use those findings. An address is reported as personal only
when the text ties it to a person (a name, a personal contact or a member number within a few lines); an address
with no person near it, or under a company name, is reported with `pii: false`. The handle and IP-address types
were measured on too few documents to rely on. `employment_info` still misses about half of the workplaces written inside
long records and minutes, so it cannot block on its own, but what it reports is usually right. The
category probabilities of these five kinds are less accurate than the findings; the other categories
(`person_name`, `email_or_phone`, `government_id`, `financial_account`, the `high` categories) overlap with the
findings and the sensitivity level, which are measured under [Accuracy](#accuracy).

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
| Person names | 97.5% | 98.6% |
| Phone numbers | 94.7% | 94.7% |
| E-mail addresses | 83.3% | 76.9% |
| ID and account numbers | 88.3% | 93.2% |

The sensitivity level is right for 90.9% of the documents. 4 of the 162 `high` documents were judged lower, and 7 of
the 246 documents with personal information were judged `none`. The attribute findings are measured under
[Categories](#categories). See the [model card](MODEL_CARD.md) for details.

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
