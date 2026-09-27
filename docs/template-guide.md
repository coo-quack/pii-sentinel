# Writing guide for slot templates (training data)

You write realistic documents with the personal values left as placeholders. A program later fills the placeholders with invented values many times, and knows exactly where each value is. So the quality that matters is the writing: natural, varied, native text that looks like a real document of its kind.

Read `docs/labeling-policy.md` (Japanese) to understand what is personal information and what is sensitive. Do not read anything else in the repository except your brief file and this guide.

## One template per brief

Your brief file lists briefs; you are given a range of them. Write one template per brief in your range. Each brief fixes:

- `sensitivity` and `cell`: what the document must contain (see below). The document must really contain that kind of information about the people in it, and nothing more sensitive.
- `domain` and `format`: the setting (a hotel, a dental clinic, a trade union office ...) and the kind of document (e-mail, form, meeting notes ...).
- `length`: short (under 200 characters), medium (200 to 600) or long (600 to 1100). For ja, zh and ko halve these limits. Count the template text as written, placeholders included.
- `implicit` (only for high): when true, never state the sensitive fact in a sentence; let only the heading, the kind of document or the setting reveal it (a sign-in sheet of a dialysis unit, a list of members of a union branch, a DNA sample consent form).

## Placeholders

Never write a real or invented person's name, e-mail address, phone number, ID number, card or account number yourself. Use placeholders:

| Placeholder | Filled with |
|---|---|
| `{P1}` … `{P4}` | the full name of person 1 … 4 |
| `{P1.s}` `{P1.g}` | person 1's surname / given name only |
| `{P1.i}` | person 1's initial and surname ("J. Smith"; in ja, zh, ko the surname) |
| `{P1.rev}` | "Surname, Given" (Latin-script languages; otherwise the full name) |
| `{P1.email}` `{P1.phone}` | person 1's own e-mail address / mobile phone |
| `{P1.addr}` `{P1.dob}` `{P1.job}` | person 1's home address / date of birth / job title |
| `{P1.govid}` `{P1.passport}` | person 1's national ID (my number in ja) / passport or driving licence number |
| `{P1.card}` `{P1.bank}` | person 1's credit card number / bank account number |
| `{P1.handle}` `{P1.ip}` | person 1's SNS handle / IP address |
| `{ORG}` `{ORG.email}` `{ORG.phone}` | a company name, its generic address (info@...), its switchboard or hotline |
| `{PUB}` `{PUB.s}` | a living public figure (CEO, mayor, athlete, singer) |
| `{DEAD}` `{DEAD.born}` `{DEAD.died}` | a historical person who died long ago, and the years |
| `{DATE}` `{PLACE}` `{CODE}` `{AMOUNT}` `{GPS}` `{PW}` `{PRODUCT}` | a date, a city, a document or order number, a sum of money, GPS coordinates, a password, a product name |

The same placeholder always means the same person or value inside one template. Write honorifics and titles around the placeholder as the language needs ("{P1.s}さん", "Dr. {P2}", "{P3} 님", "Frau {P1.s}"). Use several name forms for the same person across the document. Everything that is not a placeholder must be free of personal values; numbers in the text other than placeholders must be harmless (quantities, times, room numbers, short reference numbers).

## Cells

High:
race_or_ethnic_origin, political_opinion, religion_or_belief, trade_union, health (diagnosis, treatment, medication, disability, pregnancy, sick leave, health check results, being a patient), sex_life_or_orientation, genetic_or_biometric, criminal_record_or_victim, citizenship_or_immigration (nationality, residence status, visa, asylum), government_id_number (use `.govid` or `.passport`), financial_account_or_credentials (use `.card`, `.bank` or `{PW}`), precise_location (use `{GPS}` for a person), private_messages_of_others (someone's copied private chat or mail), formal_hr_record (disciplinary action, dismissal, harassment complaint, formal review).

Low (personal but not sensitive):
names_in_everyday_work, contact_details, address_or_birthday, online_identifier (`.handle` or `.ip`), living_public_figure (`{PUB}` in a neutral public role), business_paperwork (invoice, payslip, delivery note, rental contract about a person), neutral_roster (a list of names under a heading that reveals nothing sensitive).

None (no living person identifiable): deceased_person (`{DEAD}`), company_contact_only (`{ORG.email}`, `{ORG.phone}`), sensitive_topic_in_general, name_like_product_or_place, code_log_or_config, statistics_or_news_without_people. None templates must not use `{P1}` … `{P4}` or `{PUB}` or any personal placeholder.

## Output

Write `templates/<your file>.json`:

```json
{"templates": [{"id": "<brief id>", "lang": "...", "sensitivity": "...", "cell": "...", "text": "..."}]}
```

Real newlines in the text are fine (JSON-encode them); never write the two characters backslash and n into the text.

## Working method

Write every template yourself, one by one. Do not generate text with code, loops or random choices and never pad. Use Python only to put your templates into the JSON file (about 10 at a time) and then run:

```
.venv/bin/python -m pii_sentinel.gen.slots check templates/<your file>.json --briefs <directory of your brief file>
```

Fix every problem it prints and run it again until it prints `0 problems`. Briefs outside your range are reported as missing; ignore that line only.
