"""Score a checkpoint on evaluation corpora ({"tests": [{id, lang, text, expected: {sensitivity, findings}}]}).

Matching follows the corpora's convention: a prediction counts only when type, normalised value and pii agree.
The acceptance check is person-name recall >= 98%, no document labelled high judged low or none, and no
document labelled low or high judged none.

    python -m pii_sentinel.evaluate --model models/mmBERT-pii-sentinel eval/dev.json [more.json ...] [--out runs]
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from . import model as M
from .predict import analyse

HONORIFICS = [
    "さん",
    "様",
    "氏",
    "君",
    "殿",
    "先生",
    "部長",
    "課長",
    "社長",
    "医師",
    "先生",
    "女士",
    "小姐",
    "经理",
    "老师",
    "医生",
    "씨",
    "님",
    "선생님",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Dr.",
    "Prof.",
    "Mr",
    "Ms",
    "Mrs",
    "Dr",
    "M.",
    "Mme",
    "Mlle",
    "Herr",
    "Frau",
    "Sig.",
    "Sig.ra",
    "Dott.",
    "Dott.ssa",
    "Sr.",
    "Sra.",
    "Srta.",
    "Don",
    "Doña",
]
RANK = {"none": 0, "low": 1, "high": 2}


def normalise(value, ftype):
    if ftype in ("phone", "number"):
        return re.sub(r"[\s\-().]", "", value)
    if ftype == "email":
        return value.lower()
    v = value.strip()
    changed = True
    while changed:
        changed = False
        for h in HONORIFICS:
            if v.endswith(h) and len(v) > len(h):
                v, changed = v[: -len(h)].strip(), True
            if v.startswith(h + " "):
                v, changed = v[len(h) :].strip(), True
    return re.sub(r"\s+", " ", v)


def match(predicted, expected):
    used, rows = set(), []
    for exp in expected:
        hit = None
        for i, p in enumerate(predicted):
            if (
                i not in used
                and p["type"] == exp["type"]
                and p["pii"] == exp["pii"]
                and normalise(p["value"], p["type"]) == normalise(exp["value"], exp["type"])
            ):
                hit = i
                break
        if hit is None:
            rows.append({"type": exp["type"], "expected": exp, "predicted": None})
        else:
            used.add(hit)
            rows.append({"type": exp["type"], "expected": exp, "predicted": predicted[hit]})
    rows += [
        {"type": p["type"], "expected": None, "predicted": p}
        for i, p in enumerate(predicted)
        if i not in used
    ]
    return rows


def score(corpus, model, tok, device):
    per_type = defaultdict(Counter)
    # Personal values only: predictions and gold with pii=false (company contacts) are left out, so
    # gaps in how completely non-personal contacts were labelled do not count against precision.
    personal = defaultdict(Counter)
    person_by_lang = defaultdict(Counter)
    confusion = Counter()
    results, started = [], time.perf_counter()
    for t in corpus["tests"]:
        res = analyse(model, tok, t["text"], device)
        seen, preds = set(), []
        for f in res["findings"]:
            key = (f["type"], normalise(f["value"], f["type"]), f["pii"])
            if key not in seen:
                seen.add(key)
                preds.append(f)
        rows = match(preds, t["expected"]["findings"])
        for r in rows:
            c = per_type[r["type"]]
            outcome = "tp" if r["expected"] and r["predicted"] else "fn" if r["expected"] else "fp"
            c[outcome] += 1
            e, p = r["expected"], r["predicted"]
            if e and p and e["pii"]:
                personal[r["type"]]["tp"] += 1
            elif e and not p and e["pii"]:
                personal[r["type"]]["fn"] += 1
            elif p and not e and p["pii"]:
                personal[r["type"]]["fp"] += 1
            if r["type"] == "person_name":
                person_by_lang[t["lang"]][outcome] += 1
        exp_s, pred_s = t["expected"]["sensitivity"], res["sensitivity"]["level"]
        confusion[(exp_s, pred_s)] += 1
        results.append(
            {
                "id": t["id"],
                "lang": t["lang"],
                "text": t["text"],
                "sensitivity": {"expected": exp_s, "predicted": pred_s},
                "categories": res["categories"],
                "rows": rows,
            }
        )
    elapsed = time.perf_counter() - started

    def prf(c):
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / max(1e-9, p + r), 4),
            **c,
        }

    high_missed = [
        r["id"]
        for r in results
        if r["sensitivity"]["expected"] == "high" and r["sensitivity"]["predicted"] != "high"
    ]
    under = [
        r["id"] for r in results if RANK[r["sensitivity"]["predicted"]] < RANK[r["sensitivity"]["expected"]]
    ]
    personal_missed = [
        r["id"]
        for r in results
        if r["sensitivity"]["expected"] != "none" and r["sensitivity"]["predicted"] == "none"
    ]
    person_recall = prf(per_type["person_name"])["recall"]
    summary = {
        "documents": len(results),
        "seconds": round(elapsed, 1),
        "types": {k: prf(v) for k, v in sorted(per_type.items())},
        "personal_types": {k: prf(v) for k, v in sorted(personal.items())},
        "person_by_lang": {k: prf(v) for k, v in sorted(person_by_lang.items())},
        "sensitivity_accuracy": round(
            sum(v for (e, p), v in confusion.items() if e == p) / max(1, len(results)), 4
        ),
        "confusion": {f"{e}->{p}": v for (e, p), v in sorted(confusion.items())},
        "high_missed": high_missed,
        "personal_missed": personal_missed,
        "under_estimated": under,
        "acceptance": {
            "person_recall_ok": person_recall >= 0.98,
            "high_missed_ok": not high_missed,
            "personal_missed_ok": not personal_missed,
            "pass": person_recall >= 0.98 and not high_missed and not personal_missed,
        },
    }
    return summary, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("corpora", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    model, tok, _ = M.load(a.model, device)
    for path in a.corpora:
        summary, results = score(json.loads(path.read_text()), model, tok, device)
        t = summary["types"]
        line = "  ".join(f"{k} P{v['precision']:.1%} R{v['recall']:.1%}" for k, v in t.items())
        print(f"== {path.name}: {line}")
        pt = summary["personal_types"]
        print(
            "   pii only: "
            + "  ".join(f"{k} P{v['precision']:.1%} R{v['recall']:.1%}" for k, v in pt.items())
        )
        print(
            "   person recall by lang: "
            + " ".join(f"{k} {v['recall']:.0%}" for k, v in summary["person_by_lang"].items())
        )
        print(
            f"   sensitivity {summary['sensitivity_accuracy']:.1%}  high missed {len(summary['high_missed'])} "
            f"{summary['high_missed']}  personal->none {len(summary['personal_missed'])} "
            f"{summary['personal_missed']}  under {len(summary['under_estimated'])}  {summary['seconds']}s  "
            f"=> {'PASS' if summary['acceptance']['pass'] else 'FAIL'}"
        )
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"{path.stem}.json").write_text(
                json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1)
            )


if __name__ == "__main__":
    main()
