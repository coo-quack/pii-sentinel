# Labelling policy

Gold labels for pii-sentinel's training and evaluation data follow this policy.
It was set by comparing the definitions in the following standards (checked against the original texts or their official commentary on 2026-09-27):

- EU General Data Protection Regulation (GDPR), Article 4(1), Article 9, Article 10 and Recital 27
- US NIST SP 800-122 (Guide to Protecting the Confidentiality of PII)
- California CCPA/CPRA (Civil Code 1798.140(ae))
- Japan's Act on the Protection of Personal Information and the Personal Information Protection Commission's general guidelines
- ISO/IEC 29100 (privacy framework)

This is not a legal assessment; it is a policy for making the detector's gold labels consistent.

## What personal information is

**Personal information**: information about a specific living individual that identifies that person directly or indirectly.

The GDPR lists as means of identification "a name, an identification number, location data, an online identifier or one or more factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity" of the person[^gdpr4].
NIST SP 800-122 likewise treats as PII information that can be used to distinguish or trace an individual, and information that is linked or linkable to an individual[^nist21].
Both include information that does not identify anyone on its own but does when combined with other information.

Under this definition, the following count as personal information:

- names (surname only, given name only or a nickname too, when they refer to a specific person; living celebrities included)
- a person's e-mail address, phone number, home address and date of birth
- employer, job title and affiliation
- online identifiers (IP address, cookie ID, device ID, SNS handle)[^online]
- location data and records of behaviour tied to a specific person, such as purchase or browsing history

The following do not count as personal information:

- information about companies, organisations and departments (a switchboard number, a generic address such as info@)
- product and brand names, even when they look like personal names
- statistics and general discussion not tied to a person, and system logs that only contain service accounts
- **information about deceased persons**: the GDPR does not apply to the personal data of deceased persons[^recital27], and Japanese law defines personal information as information about a living individual. Names of historical figures are still detected as names, but they do not raise the document's sensitivity.

## Three sensitivity levels

| Level | Meaning |
|---|---|
| none | contains no personal information |
| low | contains personal information |
| high | contains sensitive personal information |

Low does not mean "not personal information".
A document with only a name or a contact detail must still be detected as containing personal information.
Low and high differ in how much harm a leak would cause.
NIST SP 800-122 gives as a low-impact example the inconvenience of changing a phone number, and as moderate or higher impacts financial loss from identity theft, public humiliation, discrimination and blackmail[^nist-impact].

## What makes a document high (sensitive personal information)

A document is high when it contains any of the following about a specific person.
The list is the union of each standard's categories that need special handling.

| Category | Basis |
|---|---|
| racial or ethnic origin | GDPR Art. 9, CPRA, Japan's special care-required personal information (race) |
| political opinions, religious or philosophical beliefs, trade union membership | GDPR Art. 9, CPRA, special care-required personal information (creed) |
| health (medical history, diagnosis, treatment, disability, health check results, pregnancy, sick leave) | GDPR Art. 9, CPRA, special care-required personal information (medical history, disability, health check results) |
| sex life or sexual orientation | GDPR Art. 9, CPRA |
| genetic data and biometric data used to identify a person (fingerprints, face templates, iris) | GDPR Art. 9, CPRA, Japan's individual identification codes |
| criminal records, arrests, prosecutions and other criminal proceedings, being a victim of crime | GDPR Art. 10, special care-required personal information |
| social status, citizenship or immigration status | special care-required personal information (social status), CPRA (citizenship or immigration status) |
| government identification numbers (my number, social security number, passport, driving licence, residence card, health insurance number) | Japan's individual identification codes, CPRA |
| financial account numbers, credit card numbers, login credentials | CPRA, NIST SP 800-122 (SSNs and financial accounts are usually rated moderate impact or higher) |
| precise geolocation | CPRA |
| someone else's private messages (e-mail, chat) that were copied or collected | CPRA (when the business is not the intended recipient) |
| formal HR records (disciplinary action, dismissal, harassment complaints, formal performance reviews) | organisational policy; NIST SP 800-122 lists information that could harm employability or reputation among sensitive topics[^census] |

**Equal treatment**: every category above is judged by the same standard. The only question is whether the document reveals that fact about a specific person; no category is judged more or less strictly than another. A fact revealed only by the heading or the kind of document counts the same as a fact stated outright (a church member list, a roster of an LGBTQ support group, the attendees or chair of a trade union meeting and the speakers at a party rally are all equally high).

The test is whether the document shows that the person belongs to the group or holds the belief. A document that only discusses religion, politics or another topic, or that names members of a body outside these categories (a works council, a research seminar), does not reveal it. A living public figure's public office (a politician's party or government post) is part of the public role and stays low.

**Identified by an identifier alone**: a person is identifiable even without a name when the document has an identifier that singles them out, such as an employee or member number, a patient or case number, an account ID, an SNS handle, a device ID, a person's IP address or a government ID number. A sensitive fact attached to such an identifier makes the document high; the identifier alone makes it low.

**Marriage and partnership records**: registrations, certificates, contracts and other records of a marriage or partnership are high, whatever the genders of the couple, as information from which sex life or sexual orientation can be inferred. Everyday mentions of a wife, husband or partner do not count.

**Rosters**: a list of names alone is high when its heading or context reveals one of the facts above ("patients of X clinic", "staff going on maternity leave", "church member list").
A roster under a neutral heading (meeting attendees, department contacts) is low.

## Examples

| Document | Level | Reason |
|---|---|---|
| "田中美咲さんがプロジェクトに参加しました" | low | a name only |
| "Call me at 06 12 34 56 78" | low | a personal phone number, no name |
| "IP: 192.168.1.10（佐藤さんの自宅の PC）" | low | an online identifier tied to a person |
| "お問い合わせは support@example.com まで" | none | a generic address |
| "夏目漱石は日本の作家です" | none | a deceased person (still detected as a name) |
| "CEO の Tim Cook が新製品を発表した" | low | a living public figure |
| "I use Claude and iPhone daily" | none | product names |
| "産休・育休予定者リスト：村田由子 090-…" | high | a roster revealing pregnancy (health) |
| "停職処分の社員：A、B、C" | high | a formal disciplinary record |
| "婚姻届の受理証明：夫 佐藤一郎、妻 佐藤花子" | high | a marriage record (whatever the genders of the couple) |
| "社員番号 E-204518：うつ病のため休職" | high | health information about a person identified by an identifier, no name |
| "山田太郎のマイナンバー：1234 5678 9012" | high | a government identification number |

[^gdpr4]: GDPR Article 4(1). https://gdpr-info.eu/art-4-gdpr/
[^nist21]: NIST SP 800-122, section 2.1. https://csrc.nist.gov/pubs/sp/800/122/final
[^online]: The Court of Justice of the EU has held that a dynamic IP address can be personal data when a third party holds the additional information needed to identify the person (per gdpr-info.eu's commentary on "Personal Data"). Japanese law treats cookies and browsing history that cannot be matched with other information separately, as "personally referable information"; this policy follows the GDPR and counts them as personal information.
[^recital27]: GDPR Recital 27: "This Regulation does not apply to the personal data of deceased persons." https://gdpr-info.eu/recitals/no-27/
[^nist-impact]: NIST SP 800-122, section 3.1, on impact levels.
[^census]: The US Census Bureau policy cited in footnote 35 of NIST SP 800-122, section 3.2.3.
