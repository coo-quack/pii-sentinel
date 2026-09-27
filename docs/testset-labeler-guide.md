# Labelling guide

You label documents for a multilingual personal-information detector. Someone else wrote them; you have not seen their labels.
The rules are in `docs/labeling-policy.md`. Read it in full first and follow it exactly.
Do not read anything else in the repository except your input file and this guide.

## Input and output

Your input file (`eval/work/blind/<name>.json`) has `{"tests": [{"id", "lang", "text"}]}`.
Write your output file (the path is given in your instructions) with one entry per input document, in the same order:

```json
{"tests": [{"id": "...", "language_ok": true, "expected": {"sensitivity": "none|low|high", "findings": [ ... ]}}]}
```

- `language_ok`: false if the text is not written in the language given by `lang`.
- `sensitivity`: none = no personal information of a living person; low = personal information; high = sensitive personal information, as defined in the policy. A heading or the kind of document can reveal a sensitive fact (a patient register, a list of staff on sick leave, a church member list); a living public figure in a public role is low; a deceased person alone is none.

Findings list every distinct value of these types, each once. A missing value is as wrong as a wrong one.

- `{"type": "person_name", "value": "<exactly as written, without honorifics or titles>", "pii": true}` for every mention of a person:
  - full names, surname-only, given-name-only, nicknames, initials that stand for a person ("M.K."), misspelt names and names in romanisation;
  - every name in rosters, tables, registers, attendee lists, signatures, e-mail headers and quoted messages, however many there are;
  - public figures, historical figures and deceased persons (they are detected as names; only the sensitivity treats them differently);
  - a name written surname first with a comma ("Moreau, Nathalie") is one value.
  Leave out honorifics and titles ("周教授" → "周", "Dr. Weber" → "Weber"). Do not list companies, products, places, headings, job titles, common words, closing phrases ("敬具", "Best regards"), or party placeholders in contracts ("甲", "乙", "甲方", "乙方", "Party A").
- `{"type": "email", "value": "...", "pii": true}` for a person's address (also when written like "taro [at] example [dot] com"); `"pii": false` for generic, role or company addresses (info@, support@, research@, a department's address). List both kinds.
- `{"type": "phone", "value": "...", "pii": true}` for a person's number (including a person's direct line or mobile); `"pii": false` for company switchboards, hotlines and toll-free numbers. List both kinds.
- `{"type": "number", "value": "...", "pii": true, "number_type": "my_number|national_id|driver_licence_or_passport|credit_card|bank_account"}` for personal ID and financial numbers (national ID, social security, tax ID, passport, driving licence, residence card, health insurance, card and account numbers, IBAN). Use `"pii": false` for a company's account or registration number. Do not list invoice, order, tracking, case, ticket, patient-record, employee or serial numbers, dates, amounts or postal codes.

Every `value` must be an exact substring of the text, written as it appears there (keep its spaces and hyphens).

## Working method

Read each document yourself, from beginning to end, and decide its labels yourself. Long documents often have names and numbers near the end; check the whole text before you finish a document. Do not label with code, regular expressions or automatic extraction. Use Python only to write your decisions into the JSON file (about 10 documents at a time) and to check that the file loads, that every input id is present, and that every value is a substring of its text.
