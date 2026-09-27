# Labelling guide

You label documents for a multilingual personal-information detector. Someone else wrote them; you have not seen their labels.
The rules are in `docs/labeling-policy.md` (Japanese). Follow them exactly.
Do not read anything else in the repository except your input file and this guide.

## Input and output

Your input file (`eval/work/blind/<name>.json`) has `{"tests": [{"id", "lang", "text"}]}`.
Write `eval/work/labels/<name>.json` with one entry per input document, in the same order:

```json
{"tests": [{"id": "...", "language_ok": true, "expected": {"sensitivity": "none|low|high", "findings": [ ... ]}}]}
```

- `language_ok`: false if the text is not written in the language given by `lang`.
- `sensitivity`: none = no personal information of a living person; low = personal information; high = sensitive personal information, as defined in the policy (a heading or the kind of document can reveal it; a living public figure is low; a deceased person is none).

Findings list every distinct value of these types, each once:

- `{"type": "person_name", "value": "<exactly as written, without honorifics or titles>", "pii": true}`. Include surname-only, given-name-only, nickname and initials mentions of people, misspelt names, and names of public figures and deceased persons. Do not list companies, products, places, headings, job titles or common words.
- `{"type": "email", "value": "...", "pii": true}` for a person's address (also when written like "taro [at] example [dot] com"); `"pii": false` for generic or company addresses.
- `{"type": "phone", "value": "...", "pii": true}` for a person's number; `"pii": false` for company switchboards, hotlines and toll-free numbers.
- `{"type": "number", "value": "...", "pii": true, "number_type": "my_number|national_id|driver_licence_or_passport|credit_card|bank_account"}` for personal ID and financial numbers. Do not list invoice, order, tracking, case, ticket or serial numbers, dates, amounts or postal codes.

Every `value` must be an exact substring of the text.

## Working method

Read each document yourself and decide its labels yourself. Do not label with code, regular expressions or automatic extraction. Use Python only to write your decisions into the JSON file (about 10 documents at a time) and to check that the file loads, that every input id is present, and that every value is a substring of its text.
