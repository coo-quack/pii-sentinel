# Writing guide

You write evaluation documents for a multilingual personal-information detector.
The labelling rules are in `docs/labeling-policy.md` (Japanese). Follow them exactly.
Do not read anything else in the repository except your order file and this guide.

## One document per order

Your order file (`eval/work/orders/<lang>.json`) lists 80 orders; you are given a range of them. Write exactly one document per order in your range and keep its `id`, `lang` and `split`.
Each order fixes:

- `sensitivity` and `cell`: what the document must be about (definitions below). The gold sensitivity you give must follow the policy, even if it ends up different from the order.
- `format`: the kind of document (email, chat, form, csv_or_table, log, meeting_minutes, contract, medical_record, roster, sns_post, code, ocr_scan, news_article, letter, email_thread, report).
- `chars`: the allowed length in characters as counted by Python `len(text)`: [min, max). Check it.
- `position` (only for medium and longer documents): where the fact that decides the sensitivity appears — the first fifth, the middle, or the last fifth of the text.
- `overlays`: extra features the document must contain (definitions below).

Write natural, native text in the order's language. A `ko` document is Korean, a `fr` document is French, and so on. Every document must be a different, realistic scene. Do not reuse names across documents.

## Cells

High (sensitive personal information about an identifiable living person):

- race_or_ethnic_origin, political_opinion, religion_or_belief, trade_union, sex_life_or_orientation
- health: diagnosis, treatment, medication, disability, pregnancy or maternity leave, sick leave, health check results, being a patient of a clinic
- genetic_or_biometric: genetic test results, fingerprints, face or iris templates used to identify someone
- criminal_record_or_victim: arrests, charges, convictions, court cases, being a victim of a crime
- citizenship_or_immigration: nationality, residence status, visa, asylum
- government_id_number: my number (ja), national ID, social security number, passport, driver's licence, residence card
- financial_account_or_credentials: credit card number, bank account or IBAN, a person's login password
- precise_location: GPS coordinates or live location of a person
- private_messages_of_others: a copied or collected private chat or e-mail of someone else
- formal_hr_record: disciplinary action, dismissal, harassment complaint, formal performance review

Low (personal information, nothing sensitive):

- name_only, contact_without_name (a personal phone or e-mail with no name), postal_address, date_of_birth, employer_or_job
- online_identifier: a person's IP address, device ID, cookie ID or SNS handle
- living_public_figure: a real, currently living, widely known public figure (head of government, CEO, athlete, musician) in a neutral public role. Never attach sensitive facts to a real person.
- business_paperwork: invoice, receipt, payslip, delivery note, rental contract about a named person
- neutral_roster: a list of names whose heading reveals nothing sensitive (meeting attendees, passengers, club members of a hobby club)

None (no personal information of a living person):

- deceased_person: a historical person, real or invented, whose death is clearly in the past
- company_contact_only: only company or generic contacts (info@, support lines)
- sensitive_topic_in_general: health, religion, crime, immigration etc. discussed without any identifiable person
- name_like_product_or_place: product, brand or place names that look like personal names
- code_log_or_config: source code, logs or configuration without personal data (service accounts are fine)
- statistics_or_news_without_people

## Overlays

- distractor_numbers: include invoice, order, case, ticket or serial numbers that are not personal (do not list them as findings)
- code_switching: mix in words or a sentence from another language
- surname_only / given_name_only / nickname / initials (e.g. "T. Yamada", "J.M.") / title_or_honorific: refer to at least one person this way
- typo_in_name: misspell a person's name once (both spellings are findings)
- spaced_or_dotted_digits: write a phone or ID number with unusual spacing or dots ("0 9 0 1 2 3 4", "090.1234.5678")
- obfuscated_email: write an e-mail address like "taro [at] example [dot] com"
- implicit_from_heading_or_doc_type: the sensitive fact is never stated in a sentence; only the heading or the kind of document reveals it (e.g. a list titled for a dialysis unit with only names)
- self_disclosure: the person writes about their own sensitive information (label it high, as the policy currently says)
- full_width_characters (ja, zh, ko): full-width digits or Latin letters in a phone, e-mail or ID
- romanized_name (ja, zh, ko): a person's name written in Latin letters
- name_without_diacritics (fr, it, de, es): a name written without its accents or umlauts ("Muller", "Nunez")

## Labels

Use this schema for each document:

```json
{"id": "...", "lang": "...", "split": "...", "order": {<the order copied unchanged>},
 "format": "<format>", "text": "...",
 "expected": {"sensitivity": "none|low|high", "findings": [ ... ]}}
```

Findings list every distinct value of these types, each once:

- `{"type": "person_name", "value": "<exactly as written, without honorifics or titles>", "pii": true, "honorific": "<the honorific or title if attached>"}`. Include surname-only, given-name-only, nickname and initials mentions of people. For deceased persons add `"public_figure": true`; for living public figures add `"public_figure": true` too. Do not list companies, products or places.
- `{"type": "email", "value": "...", "pii": true}` for a person's address; `"pii": false` for generic or company addresses. An obfuscated address is listed as written.
- `{"type": "phone", "value": "...", "pii": true}` for a person's number; `"pii": false` for company switchboards, hotlines and toll-free numbers.
- `{"type": "number", "value": "...", "pii": true, "number_type": "my_number|national_id|driver_licence_or_passport|credit_card|bank_account"}` for personal ID and financial numbers. Do not list invoice, order, tracking, case, ticket or serial numbers, dates, amounts or postal codes.

Every `value` must be an exact substring of `text`. Real newlines in the text are fine (JSON-encode them); never write the two characters backslash and n into the text itself.
Names, numbers and addresses must be invented and obviously fake-but-realistic (16-digit card numbers, national IDs in the real format of the country, phones in national format), except for the living public figures.

## Working method

Write every document yourself, one by one, as a person would. Do not create texts or labels with code, templates, loops, random choices or automatic extraction, and never pad a document with repeated or meaningless sentences to reach the length. A long document is long because the scene has that much real content (a full meeting record, a multi-page contract, a long e-mail thread).
Label the findings yourself by reading your own text. Company names, headings, common words and job titles are not person names.

Use Python only to put your hand-written documents into the JSON file (write about 10 documents at a time) and to run the check. When finished, run from the repository root:

```
.venv/bin/python -m pii_sentinel.testset check eval/coverage.json eval/work/raw/<your file>.json
```

Fix every problem it prints and run it again until it prints `0 problems`. Orders outside your range are reported as not written; ignore that line only.
