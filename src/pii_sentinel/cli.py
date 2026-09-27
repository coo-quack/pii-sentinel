"""pii-sentinel scan: find personal information in files or stdin.

pii-sentinel scan --model <model directory> report.txt [--json] [--show-values] [--fail-on low|high]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from . import model as M
from .predict import analyse

RANK = {"none": 0, "low": 1, "high": 2}


def mask(value, ftype=""):
    # Keep only enough to tell findings apart; short values and names keep a single character.
    if ftype == "person_name" or len(value) < 8:
        return f"{value[0]}…"
    return f"{value[:2]}…{value[-2:]}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pii-sentinel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scan files (or stdin) for personal information")
    scan.add_argument("files", nargs="*", type=Path)
    scan.add_argument("--model", required=True, help="checkpoint directory or Hugging Face model id")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--show-values", action="store_true")
    scan.add_argument("--fail-on", choices=["low", "high"])
    scan.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    model, tok, meta = M.load(a.model, device)
    sources = [("stdin", sys.stdin.read()) if str(p) == "-" else (str(p), p.read_text()) for p in a.files]
    sources = sources or [("stdin", sys.stdin.read())]
    reports, worst = [], 0
    for name, text in sources:
        res = analyse(
            model,
            tok,
            text,
            device,
            max_length=meta.get("max_length", 512),
            doc_pooling=meta.get("doc_pooling", "per_window"),
        )
        for f in res["findings"] + res["secrets"]:
            if not a.show_values:
                f["value"] = mask(f["value"], f["type"])
        reports.append({"source": name, **res})
        worst = max(worst, RANK[res["sensitivity"]["level"]])
    if a.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for r in reports:
            print(f"{r['source']}: sensitivity {r['sensitivity']['level']}")
            for f in r["findings"]:
                extra = f" ({f['number_type']})" if "number_type" in f else ""
                print(f"  {f['type']}{extra}{'' if f['pii'] else ' [not PII]'}: {f['value']}")
            for f in r["secrets"]:
                print(f"  secret ({f['rule']}): {f['value']}")
    if a.fail_on and worst >= RANK[a.fail_on]:
        sys.exit(2)


if __name__ == "__main__":
    main()
