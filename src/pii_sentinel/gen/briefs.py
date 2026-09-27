"""Writing briefs for slot templates (training data), independent of the test set's coverage table.

python -m pii_sentinel.gen.briefs --out <brief dir> --per-language 120 [--seed 11]
"""

import argparse
import json
import random
from pathlib import Path

LANGS = ["ja", "zh", "ko", "en", "fr", "it", "de", "es"]
CELLS = {
    "high": {
        "race_or_ethnic_origin": 3,
        "political_opinion": 4,
        "religion_or_belief": 5,
        "trade_union": 4,
        "health": 10,
        "sex_life_or_orientation": 3,
        "genetic_or_biometric": 4,
        "criminal_record_or_victim": 5,
        "citizenship_or_immigration": 5,
        "government_id_number": 4,
        "financial_account_or_credentials": 4,
        "precise_location": 3,
        "private_messages_of_others": 2,
        "formal_hr_record": 4,
    },
    "low": {
        "names_in_everyday_work": 8,
        "contact_details": 6,
        "address_or_birthday": 4,
        "online_identifier": 4,
        "living_public_figure": 4,
        "business_paperwork": 6,
        "neutral_roster": 4,
    },
    "none": {
        "deceased_person": 5,
        "company_contact_only": 5,
        "sensitive_topic_in_general": 6,
        "name_like_product_or_place": 3,
        "code_log_or_config": 3,
        "statistics_or_news_without_people": 2,
    },
}
DOMAINS = [
    "logistics company",
    "primary school",
    "dental clinic",
    "city hall",
    "hotel",
    "football club",
    "law firm",
    "software start-up",
    "supermarket chain",
    "university lab",
    "charity",
    "car dealership",
    "airline",
    "insurance company",
    "construction site",
    "restaurant",
    "gym",
    "real-estate agency",
    "hospital ward",
    "recruiting agency",
    "online game community",
    "bank branch",
    "museum",
    "farm cooperative",
    "film studio",
    "police station",
    "nursing home",
    "language school",
    "church or temple",
    "trade union office",
    "embassy or consulate",
    "political party office",
    "research biobank",
    "marathon organisers",
]
FORMATS = [
    "e-mail",
    "chat messages",
    "form with labelled fields",
    "table or CSV rows",
    "meeting notes",
    "letter",
    "memo",
    "incident report",
    "ticket or case note",
    "SNS post",
    "notice or announcement",
    "log lines",
    "receipt or invoice",
    "list with a heading",
    "interview transcript",
    "diary or blog entry",
]
LENGTHS = {"short": 0.45, "medium": 0.45, "long": 0.10}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-language", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    slots = [(s, c) for s, cells in CELLS.items() for c, n in cells.items() for _ in range(n)]
    assert len(slots) == a.per_language, len(slots)
    a.out.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        rng.shuffle(slots)
        briefs = []
        for i, (s, c) in enumerate(slots, 1):
            briefs.append(
                {
                    "id": f"t_{lang}_{i:03d}",
                    "lang": lang,
                    "sensitivity": s,
                    "cell": c,
                    "domain": rng.choice(DOMAINS),
                    "format": rng.choice(FORMATS),
                    "length": rng.choices(list(LENGTHS), weights=list(LENGTHS.values()))[0],
                    "implicit": s == "high"
                    and c not in ("government_id_number", "financial_account_or_credentials")
                    and rng.random() < 0.35,
                }
            )
        (a.out / f"{lang}.json").write_text(json.dumps(briefs, ensure_ascii=False, indent=1) + "\n")
        print(lang, len(briefs))


if __name__ == "__main__":
    main()
