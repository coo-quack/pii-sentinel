"""Slot templates: hand-written documents whose personal values are placeholders filled by the generator.

python -m pii_sentinel.gen.slots check templates/ja_1.json [more.json ...] [--briefs <dir>]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PERSON_ATTRS = {
    "",
    "s",
    "g",
    "i",
    "rev",
    "email",
    "phone",
    "addr",
    "dob",
    "job",
    "govid",
    "passport",
    "card",
    "bank",
    "handle",
    "ip",
}
OTHER = {
    "ORG",
    "ORG.email",
    "ORG.phone",
    "PUB",
    "PUB.s",
    "DEAD",
    "DEAD.born",
    "DEAD.died",
    "DATE",
    "PLACE",
    "CODE",
    "AMOUNT",
    "GPS",
    "PW",
    "PRODUCT",
}
PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.]+)\}")
PERSON = re.compile(r"^P([1-4])(?:\.([a-z]+))?$")
HIGH_SLOTS = {"govid", "passport", "card", "bank"}
LIMITS = {"short": (0, 200), "medium": (200, 600), "long": (600, 1100)}
CJK = {"ja", "zh", "ko"}

# Categories implied by the cell (the fact the template was written to contain).
CELL_CATEGORY = {
    "race_or_ethnic_origin": "race_or_religion",
    "religion_or_belief": "race_or_religion",
    "political_opinion": "political_or_union",
    "trade_union": "political_or_union",
    "health": "health_info",
    "sex_life_or_orientation": "sex_life_or_orientation",
    "genetic_or_biometric": "biometric_or_genetic",
    "criminal_record_or_victim": "hr_or_criminal_record",
    "formal_hr_record": "hr_or_criminal_record",
    "citizenship_or_immigration": "citizenship_or_immigration",
    "precise_location": "precise_location",
    "private_messages_of_others": "private_communications",
}
# Categories implied by the placeholders used.
SLOT_CATEGORY = {
    "email": "email_or_phone",
    "phone": "email_or_phone",
    "addr": "postal_address",
    "dob": "date_of_birth",
    "job": "employment_info",
    "govid": "government_id",
    "passport": "government_id",
    "card": "financial_account",
    "bank": "financial_account",
    "ip": "ip_address_of_a_person",
    "handle": "sns_handle",
}


def placeholders(text):
    return PLACEHOLDER.findall(text)


def person_slots(text):
    out = []
    for ph in placeholders(text):
        m = PERSON.match(ph)
        if m:
            out.append((int(m.group(1)), m.group(2) or ""))
    return out


def problems(t, brief=None):
    out, text = [], t["text"]
    for ph in placeholders(text):
        m = PERSON.match(ph)
        if not (m and (m.group(2) or "") in PERSON_ATTRS) and ph not in OTHER:
            out.append(f"unknown placeholder {{{ph}}}")
    if "\\n" in text:
        out.append("literal backslash-n")
    stripped = PLACEHOLDER.sub("", text)
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", stripped):
        out.append("an e-mail address written out instead of a placeholder")
    if re.search(r"\d(?:[\s\-.]?\d){6,}", stripped):
        out.append("a long number written out instead of a placeholder")
    persons = person_slots(text)
    if t["sensitivity"] == "none" and (persons or "{PUB" in text):
        out.append("a none template uses a personal placeholder")
    if t["sensitivity"] != "none" and not persons and "{PUB" not in text:
        out.append("a low or high template needs a person placeholder")
    if t["cell"] == "government_id_number" and not any(a in ("govid", "passport") for _, a in persons):
        out.append("government_id_number needs .govid or .passport")
    if t["cell"] == "financial_account_or_credentials" and not (
        any(a in ("card", "bank") for _, a in persons) or "{PW}" in text
    ):
        out.append("financial_account_or_credentials needs .card, .bank or {PW}")
    if t["sensitivity"] != "high" and any(a in HIGH_SLOTS for _, a in persons):
        out.append("an ID or account number makes the document high")
    if brief:
        lo, hi = LIMITS[brief["length"]]
        if t["lang"] in CJK:
            lo, hi = lo // 2, hi // 2
        if not lo <= len(text) < hi:
            out.append(f"length {len(text)} outside {brief['length']} [{lo}, {hi})")
        if (t["sensitivity"], t["cell"]) != (brief["sensitivity"], brief["cell"]):
            out.append("sensitivity or cell differs from the brief")
    sentences = [s.strip() for s in re.split(r"[。．.!?！？\n]+", text) if len(s.strip()) > 15]
    if any(n > 1 for n in Counter(sentences).values()):
        out.append("repeated sentence")
    return out


def cmd_check(a):
    total, briefs_seen = 0, {}
    for path in a.files:
        lang = path.stem.split("_")[0]
        briefs = {}
        if a.briefs:
            briefs = {b["id"]: b for b in json.loads((a.briefs / f"{lang}.json").read_text())}
        briefs_seen.setdefault(lang, set())
        for t in json.loads(path.read_text())["templates"]:
            briefs_seen[lang].add(t["id"])
            for p in problems(t, briefs.get(t["id"])):
                print(f"!! {t['id']}: {p}")
                total += 1
    for lang, seen in briefs_seen.items() if a.briefs else ():
        n = len(json.loads((a.briefs / f"{lang}.json").read_text())) - len(seen)
        if n:
            print(f"{lang}: {n} briefs not written (fine if outside your range)")
    print(f"{total} problems")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+", type=Path)
    c.add_argument(
        "--briefs", type=Path, help="directory of brief files; checks lengths and cells against them"
    )
    a = ap.parse_args(argv)
    cmd_check(a)


if __name__ == "__main__":
    main()
