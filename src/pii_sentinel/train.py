"""Fine-tune mmBERT-base into mmBERT-pii-sentinel.

python -m pii_sentinel.train --data data --out <output dir> [--epochs 3] [--device cuda]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import model as M
from .data import IGNORE, encode, read_jsonl
from .labels import BIO


def batches(items, size, rng):
    # Group by length so padding stays small, then shuffle the batches.
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    groups = [order[i : i + size] for i in range(0, len(order), size)]
    rng.shuffle(groups)
    for g in groups:
        yield [items[i] for i in g]


def collate(batch, pad_id, device):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(b["labels"])
    sens = torch.tensor([b["sensitivity"] for b in batch])
    cats = torch.tensor([b["categories"] for b in batch])
    return ids.to(device), mask.to(device), labels.to(device), sens.to(device), cats.to(device)


def losses(out, labels, sens, cats):
    span = F.cross_entropy(out["span"].flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE)
    return (
        span,
        F.cross_entropy(out["sensitivity"].float(), sens),
        F.binary_cross_entropy_with_logits(out["categories"].float(), cats),
    )


@torch.no_grad()
def evaluate(model, items, pad_id, device, dtype, batch_size=32):
    model.eval()
    tp = fp = fn = 0
    sens_ok = cat_ok = cat_n = n = 0
    for i in range(0, len(items), batch_size):
        ids, mask, labels, sens, cats = collate(items[i : i + batch_size], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            out = model(ids, mask)
        pred = out["span"].argmax(-1)
        valid = labels != IGNORE
        tp += ((pred == labels) & valid & (labels != 0)).sum().item()
        fp += ((pred != labels) & valid & (pred != 0)).sum().item()
        fn += ((pred != labels) & valid & (labels != 0)).sum().item()
        sens_ok += (out["sensitivity"].argmax(-1) == sens).sum().item()
        cat_ok += ((out["categories"] > 0).float() == cats).sum().item()
        cat_n += cats.numel()
        n += len(sens)
    model.train()
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "token_f1": 2 * p * r / max(1e-9, p + r),
        "sensitivity_acc": sens_ok / n,
        "category_acc": cat_ok / cat_n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only N training documents (smoke tests)")
    a = ap.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        ap.error(f"{a.out} is not empty")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device(a.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok = M.load_tokenizer()
    docs = read_jsonl(a.data / "train.jsonl")
    train = [encode(tok, d, a.max_length) for d in (docs[: a.limit] if a.limit else docs)]
    valid = [encode(tok, d, a.max_length) for d in read_jsonl(a.data / "valid.jsonl")]
    model = M.new_model().to(device)
    model.train()
    steps_per_epoch = math.ceil(len(train) / a.batch_size)
    total = int(steps_per_epoch * a.epochs)
    warm = int(total * a.warmup)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm))
    )
    print(f"train {len(train)} valid {len(valid)} steps {total} device {device} {dtype}", flush=True)

    history, best, step, started = [], None, 0, time.perf_counter()
    meta = {
        "epochs": a.epochs,
        "lr": a.lr,
        "batch_size": a.batch_size,
        "max_length": a.max_length,
        "train_docs": len(train),
        "total_steps": total,
    }

    def run_eval():
        nonlocal best
        rep = evaluate(model, valid, tok.pad_token_id, device, dtype)
        if device.type == "mps":
            torch.mps.empty_cache()
        rep.update(step=step, elapsed_s=round(time.perf_counter() - started, 1))
        history.append(rep)
        print(json.dumps(rep), flush=True)
        score = rep["token_f1"] + rep["sensitivity_acc"] + rep["category_acc"]
        if best is None or score > best:
            best = score
            M.save(model, a.out, {**meta, "step": step, "valid": rep})
            print(f"saved best to {a.out}", flush=True)

    while step < total:
        for batch in batches(train, a.batch_size, rng):
            if step >= total:
                break
            ids, mask, labels, sens, cats = collate(batch, tok.pad_token_id, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                out = model(ids, mask)
            span, sl, cl = losses(out, labels, sens, cats)
            loss = span + sl + cl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": round(loss.item(), 4),
                            "span": round(span.item(), 4),
                            "sens": round(sl.item(), 4),
                            "cats": round(cl.item(), 4),
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
            if step % a.eval_every == 0:
                run_eval()
    run_eval()
    (a.out / "training_history.json").write_text(json.dumps(history, indent=1) + "\n")
    print(f"done in {time.perf_counter() - started:.0f}s; labels {len(BIO)}", flush=True)


if __name__ == "__main__":
    main()
