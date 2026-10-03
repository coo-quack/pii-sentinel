"""pii-sentinel: find personal information in text.

pii-sentinel scan --model <model> report.txt [--json] [--show-values] [--fail-on low|high]
pii-sentinel serve --model <model> [--socket PATH | --host 127.0.0.1 --port 8765]
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
    # Keep only enough to tell findings apart; short values and names keep a single character, and a
    # one-character value (a surname such as 李) is hidden entirely.
    if len(value) <= 1:
        return "…"
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
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--show-values", action="store_true")
    scan.add_argument("--fail-on", choices=["low", "high"])
    serve = sub.add_parser("serve", help="keep the model loaded and scan text sent over HTTP")
    where = serve.add_mutually_exclusive_group()
    where.add_argument("--socket", help="listen on this Unix socket (created with mode 600)")
    where.add_argument(
        "--host", default="127.0.0.1", help="loopback address to listen on (default 127.0.0.1)"
    )
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--max-bytes", type=int, default=1_000_000, help="largest request body accepted")
    serve.add_argument(
        "--verbose", action="store_true", help="log requests (method, path, status; never the text)"
    )
    for p in (scan, serve):
        p.add_argument("--model", required=True, help="checkpoint directory or Hugging Face model id")
        p.add_argument(
            "--device",
            default="cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
        )
        p.add_argument("--threads", type=int, help="CPU threads for PyTorch (default: all cores)")
    a = ap.parse_args(argv)
    if a.command == "serve" and a.socket is None:
        from .server import LOOPBACK_HOSTS

        if a.host not in LOOPBACK_HOSTS:
            ap.error(f"--host must be a loopback address ({', '.join(sorted(LOOPBACK_HOSTS))}), got {a.host}")
    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    model, tok, meta = M.load(a.model, device)
    if a.command == "serve":
        from .server import Scanner, serve

        scanner = Scanner(model, tok, meta, device, mask)
        serve(
            scanner, socket_path=a.socket, host=a.host, port=a.port, max_bytes=a.max_bytes, verbose=a.verbose
        )
        return
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
