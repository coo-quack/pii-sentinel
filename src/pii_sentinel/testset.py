"""Build and check a coverage-driven test set.

    python -m pii_sentinel.testset orders eval/coverage.json --out eval/work/orders [--seed 5]
    python -m pii_sentinel.testset check eval/coverage.json eval/work/raw/*.json
    python -m pii_sentinel.testset compare eval/work/raw/ja_1.json eval/work/labels/ja_1.json --out eval/work/disagreements/ja_1.json
    python -m pii_sentinel.testset finalize ja_1 [more names] --root eval/work
    python -m pii_sentinel.testset split eval/work/final/*.json --out eval

`orders` turns the coverage table into one writing order per document (sensitivity cell, format, length,
position of the key fact and overlays). `check` reports how the written documents cover the table, without
printing any text. `compare` lists where two independent labellings disagree. `split` writes dev.json and
test.json from the adjudicated files.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CJK = {"ja", "zh", "ko"}
NUMBER_TYPES = {"my_number", "national_id", "driver_licence_or_passport", "credit_card", "bank_account"}
# Cells whose short and medium documents only make sense in some formats.
CELL_FORMATS = {
    "code_log_or_config": ["code", "log"],
    "neutral_roster": ["roster", "csv_or_table"],
    "business_paperwork": ["form", "csv_or_table", "letter", "ocr_scan"],
    "company_contact_only": ["email", "letter", "sns_post", "form"],
    "online_identifier": ["log", "sns_post", "chat", "email"],
    "private_messages_of_others": ["chat", "email", "report"],
}


def build_orders(cov, lang, rng):
    slots = [
        {"sensitivity": s, "cell": c}
        for s, cells in cov["primary"].items()
        for c, n in cells.items()
        for _ in range(n)
    ]
    if len(slots) != cov["per_language"]:
        raise ValueError(f"{lang}: the coverage table has {len(slots)} slots, not {cov['per_language']}")
    rng.shuffle(slots)

    # Lengths: long formats get the long buckets, everything else short or medium.
    buckets = [b for b, (_, _, n) in cov["length"].items() for _ in range(n)]
    rng.shuffle(buckets)
    for s, b in zip(slots, buckets, strict=True):
        s["length"] = b
    longs = [s for s in slots if s["length"] in ("long", "very_long")]
    others = [s for s in slots if s["length"] not in ("long", "very_long")]
    for i, s in enumerate(longs):
        s["format"] = cov["long_formats"][i % len(cov["long_formats"])]
    formats = [f for f in cov["formats"] if f not in ("email_thread", "report")]
    rng.shuffle(formats)
    for i, s in enumerate(others):
        s["format"] = (
            rng.choice(CELL_FORMATS[s["cell"]]) if s["cell"] in CELL_FORMATS else formats[i % len(formats)]
        )
    pos = cov["position_of_key_fact"]
    for i, s in enumerate(x for x in slots if x["length"] != "short" and x["sensitivity"] != "none"):
        s["position"] = pos[i % len(pos)]

    for s in slots:
        s["overlays"] = []
    ov = cov["overlays"]
    pools = {
        "any": slots,
        "with_people": [s for s in slots if s["sensitivity"] != "none"],
        "high_only": [s for s in slots if s["sensitivity"] == "high"],
        "cjk_only": [s for s in slots if s["sensitivity"] != "none"] if lang in CJK else [],
        "latin_only": [s for s in slots if s["sensitivity"] != "none"] if lang not in CJK else [],
    }
    for group, tags in ov.items():
        for tag, n in tags.items():
            pool = pools[group]
            if tag == "implicit_from_heading_or_doc_type":
                pool = [s for s in pool if s["cell"] not in cov["implicit_excluded"]]
            if not pool:
                continue
            # Spread overlays: prefer documents that carry the fewest so far.
            for s in sorted(rng.sample(pool, len(pool)), key=lambda s: len(s["overlays"]))[:n]:
                s["overlays"].append(tag)

    # Half of every cell to dev, half to test (odd cells alternate which side gets the extra one).
    for i, s in enumerate(sorted(slots, key=lambda s: (s["sensitivity"], s["cell"]))):
        s["split"] = "test" if i % 2 == 0 else "dev"
    for i, s in enumerate(slots, 1):
        s["id"] = f"ts_{lang}_{i:03d}"
        s["lang"] = lang
        lo, hi, _ = cov["length"][s["length"]]
        s["chars"] = [lo, hi]
    return slots


def length_bucket(cov, n):
    for b, (lo, hi, _) in cov["length"].items():
        if lo <= n < hi:
            return b
    return "very_long"


def cmd_orders(a):
    cov = json.loads(a.coverage.read_text())
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    for lang in cov["languages"]:
        orders = build_orders(cov, lang, rng)
        (a.out / f"{lang}.json").write_text(json.dumps(orders, ensure_ascii=False, indent=1) + "\n")
        print(lang, Counter(o["sensitivity"] for o in orders), Counter(o["split"] for o in orders))


def cmd_check(a):
    cov = json.loads(a.coverage.read_text())
    problems, totals = [], Counter()
    by_lang = defaultdict(dict)
    for path in a.files:
        for t in json.loads(path.read_text())["tests"]:
            by_lang[path.stem.split("_")[0]][t["id"]] = t
    for lang, by in sorted(by_lang.items()):
        orders = {o["id"]: o for o in json.loads((a.orders / f"{lang}.json").read_text())}
        missing = sorted(set(orders) - set(by))
        if missing:
            problems.append(f"{lang}: {len(missing)} orders not written (fine if outside your range)")
        shingles = {}
        for oid, t in by.items():
            x = re.sub(r"\s+", " ", t["text"])
            shingles[oid] = {x[i : i + 12] for i in range(0, max(1, len(x) - 12), 6)}
            sentences = [s.strip() for s in re.split(r"[。．.!?！？\n]+", t["text"]) if len(s.strip()) > 15]
            repeated = [s for s, n in Counter(sentences).items() if n > 1]
            if repeated:
                problems.append(f"{oid}: repeated sentence ({len(repeated)})")
        ids = sorted(shingles)
        for i, p in enumerate(ids):
            for q in ids[:i]:
                sp, sq = shingles[p], shingles[q]
                if sp and sq and len(sp & sq) / min(len(sp), len(sq)) > 0.5:
                    problems.append(f"{p}: near-duplicate of {q}")
        for oid, t in by.items():
            o = orders.get(oid)
            if not o:
                problems.append(f"{oid}: not in orders")
                continue
            n = len(t["text"])
            if length_bucket(cov, n) != o["length"]:
                problems.append(f"{oid}: length {n} outside {o['length']}")
            if "\\n" in t["text"]:
                problems.append(f"{oid}: literal backslash-n")
            for f in t["expected"]["findings"]:
                if f["value"] not in t["text"]:
                    problems.append(f"{oid}: finding value not in text ({f['type']})")
            if lang == "ko" and not re.search(r"[가-힣]", t["text"]):
                problems.append(f"{oid}: no Hangul")
            if lang == "ja" and not re.search(r"[぀-ヿ]", t["text"]):
                problems.append(f"{oid}: no kana")
            if lang == "zh" and re.search(r"[぀-ヿ가-힣]", t["text"]):
                problems.append(f"{oid}: kana or Hangul in zh")
            totals[(o["split"], t["expected"]["sensitivity"])] += 1
            totals[("target_vs_label", o["sensitivity"], t["expected"]["sensitivity"])] += 1
    for p in problems:
        print("!!", p)
    for k, v in sorted(totals.items(), key=str):
        print(k, v)
    print(f"{len(problems)} problems")


FIELDS = ("type", "pii", "number_type")


def key(f):
    v = f["value"].strip()
    if f["type"] in ("phone", "number"):
        v = re.sub(r"[\s\-().]", "", v)
    return (f["type"], v.lower(), f["pii"], f.get("number_type"))


def cmd_compare(a):
    first = {t["id"]: t for t in json.loads(a.first.read_text())["tests"]}
    second = {t["id"]: t for t in json.loads(a.second.read_text())["tests"]}
    rows, agree = [], Counter()
    for tid, t in first.items():
        u = second.get(tid)
        if not u:
            rows.append({"id": tid, "missing_in_second": True})
            continue
        s1, s2 = t["expected"]["sensitivity"], u["expected"]["sensitivity"]
        k1 = {key(f): f for f in t["expected"]["findings"]}
        k2 = {key(f): f for f in u["expected"]["findings"]}
        agree["sensitivity"] += s1 == s2
        agree["docs"] += 1
        agree["findings_both"] += len(set(k1) & set(k2))
        agree["findings_any"] += len(set(k1) | set(k2))
        if s1 != s2 or set(k1) != set(k2) or u.get("language_ok") is False:
            rows.append(
                {
                    "id": tid,
                    "sensitivity": [s1, s2],
                    "only_first": [k1[k] for k in set(k1) - set(k2)],
                    "only_second": [k2[k] for k in set(k2) - set(k1)],
                    "language_ok": u.get("language_ok", True),
                }
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    print(
        f"{a.first.stem}: docs {agree['docs']}  sensitivity agreement {agree['sensitivity'] / max(1, agree['docs']):.1%}"
        f"  finding agreement (Jaccard) {agree['findings_both'] / max(1, agree['findings_any']):.1%}"
        f"  disagreements {len(rows)}"
    )


def cmd_finalize(a):
    # Agreed documents keep the shared labels; disputed ones take the adjudicated labels.
    r = a.root
    for name in a.names:
        raw = json.loads((r / "raw" / f"{name}.json").read_text())["tests"]
        second = {t["id"]: t for t in json.loads((r / a.labels / f"{name}.json").read_text())["tests"]}
        disputed = {row["id"] for row in json.loads((r / a.disagreements / f"{name}.json").read_text())}
        decided = {}
        if disputed:
            decided = {
                t["id"]: t for t in json.loads((r / a.decisions / f"{name}.json").read_text())["tests"]
            }
        missing = sorted(disputed - set(decided))
        if missing:
            raise SystemExit(f"{name}: no decision for {missing}")
        out, changed = [], Counter()
        for t in raw:
            source = decided[t["id"]] if t["id"] in disputed else second[t["id"]]
            expected = source["expected"]
            for f in expected["findings"]:
                if f["value"] not in t["text"]:
                    raise SystemExit(f"{t['id']}: {f['value']!r} is not in the text")
            changed[t["id"] in disputed] += 1
            out.append({**{k: v for k, v in t.items() if k != "expected"}, "expected": expected})
        (r / a.final).mkdir(exist_ok=True)
        (r / a.final / f"{name}.json").write_text(
            json.dumps({"tests": out}, ensure_ascii=False, indent=1) + "\n"
        )
        print(f"{name}: agreed {changed[False]}, adjudicated {changed[True]}")


def apply_corrections(t, fix):
    exp = t["expected"]
    exp["findings"] = [f for f in exp["findings"] if f["type"] not in fix.get("remove_types", [])]
    exp["findings"] = [f for f in exp["findings"] if f["value"] not in fix.get("remove_values", [])]
    for f in fix.get("add", []):
        if f["value"] not in t["text"]:
            raise SystemExit(f"{t['id']}: {f['value']!r} is not in the text")
        exp["findings"].append(f)
    for change in fix.get("set_pii", []):
        for f in exp["findings"]:
            if f["type"] == "email" and f["value"] == change["value"]:
                f["pii"] = change["pii"]
    if "sensitivity" in fix:
        exp["sensitivity"] = fix["sensitivity"]


def cmd_split(a):
    out = defaultdict(list)
    fixes = {}
    if a.corrections and a.corrections.exists():
        fixes = {c["id"]: c for c in json.loads(a.corrections.read_text())["corrections"]}
    valid = {"person_name", "email", "phone", "number"}
    for path in a.files:
        for t in json.loads(path.read_text())["tests"]:
            if t["id"] in fixes:
                apply_corrections(t, fixes.pop(t["id"]))
            t["expected"]["findings"] = [
                f
                for f in t["expected"]["findings"]
                if f["type"] in valid and (f["type"] != "number" or f.get("number_type") in NUMBER_TYPES)
            ]
            for f in t["expected"]["findings"]:
                if f["type"] == "person_name":
                    f["pii"] = (
                        True  # names are always reported as personal, public and deceased people included
                    )
            t["expected"]["findings"] = [
                f
                for f in t["expected"]["findings"]
                if not (f["type"] == "person_name" and (f["value"].startswith("@") or "_" in f["value"]))
            ]  # SNS handles are online identifiers, not names
            out[t["split"]].append(t)
    if fixes:
        raise SystemExit(f"corrections for unknown ids: {sorted(fixes)}")
    for split, tests in out.items():
        (a.out / f"{split}.json").write_text(
            json.dumps({"description": a.description, "tests": tests}, ensure_ascii=False, indent=1) + "\n"
        )
        print(split, len(tests), Counter(t["expected"]["sensitivity"] for t in tests))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orders")
    o.add_argument("coverage", type=Path)
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--seed", type=int, default=5)
    c = sub.add_parser("check")
    c.add_argument("coverage", type=Path)
    c.add_argument("files", nargs="+", type=Path)
    c.add_argument("--orders", type=Path, default=Path("eval/work/orders"))
    m = sub.add_parser("compare")
    m.add_argument("first", type=Path)
    m.add_argument("second", type=Path)
    m.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("names", nargs="+")
    f.add_argument("--root", type=Path, default=Path("eval/work"))
    f.add_argument("--labels", default="labels", help="labelling used for agreed documents")
    f.add_argument("--disagreements", default="disagreements")
    f.add_argument("--decisions", default="decisions")
    f.add_argument("--final", default="final")
    s = sub.add_parser("split")
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--description", default="")
    s.add_argument("--corrections", type=Path, default=Path("eval/work/corrections.json"))
    a = ap.parse_args(argv)
    {
        "orders": cmd_orders,
        "check": cmd_check,
        "compare": cmd_compare,
        "finalize": cmd_finalize,
        "split": cmd_split,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
