"""Fine-tune mmBERT-base into mmBERT-pii-sentinel.

python -m pii_sentinel.train --data data --out <output dir> [--epochs 3] [--device cuda]
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import model as M
from .data import IGNORE, encode_windows, read_jsonl
from .labels import BIO


def groups_by_length(items, size, max_tokens):
    """Batches of documents of similar length, at most `size` windows and `max_tokens` padded tokens each,
    so a batch of long documents holds fewer of them."""

    def width(i):
        return max(len(w["input_ids"]) for w in items[i]["windows"])

    order = sorted(range(len(items)), key=lambda i: (len(items[i]["windows"]), width(i)))
    groups, cur, windows, wide = [], [], 0, 0
    for i in order:
        n, w = len(items[i]["windows"]), width(i)
        if cur and (windows + n > size or (windows + n) * max(wide, w) > max_tokens):
            groups.append(cur)
            cur, windows, wide = [], 0, 0
        cur.append(i)
        windows, wide = windows + n, max(wide, w)
    if cur:
        groups.append(cur)
    return groups


def batches(items, groups, rng):
    groups = list(groups)
    rng.shuffle(groups)
    for g in groups:
        yield [items[i] for i in g]


def collate(batch, pad_id, device):
    """Windows of all documents in the batch as rows, and for each row the document it belongs to."""
    windows = [(d, w) for d, b in enumerate(batch) for w in b["windows"]]
    width = max(len(w["input_ids"]) for _, w in windows)
    ids = torch.full((len(windows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), width), dtype=torch.long)
    labels = torch.full((len(windows), width), IGNORE, dtype=torch.long)
    for i, (_, w) in enumerate(windows):
        n = len(w["input_ids"])
        ids[i, :n] = torch.tensor(w["input_ids"])
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(w["labels"])
    doc_index = torch.tensor([d for d, _ in windows])
    sens = torch.tensor([b["sensitivity"] for b in batch])
    cats = torch.tensor([b["categories"] for b in batch])
    return (
        ids.to(device),
        mask.to(device),
        labels.to(device),
        sens.to(device),
        cats.to(device),
        doc_index.to(device),
    )


def losses(out, labels, sens, cats):
    span = F.cross_entropy(out["span"].flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE)
    return (
        span,
        F.cross_entropy(out["sensitivity"].float(), sens),
        F.binary_cross_entropy_with_logits(out["categories"].float(), cats),
    )


def per_doc_max(x, doc_index, n_docs):
    init = torch.full((n_docs, x.shape[-1]), float("-inf"), dtype=x.dtype, device=x.device)
    return init.scatter_reduce(0, doc_index.unsqueeze(-1).expand_as(x), x, reduce="amax")


def forward(model, ids, mask, sens, cats, doc_index, doc_pooling):
    """window_max: the document heads see all windows of a document pooled together. per_window: every
    window is judged on its own, and the document is scored on the feature-wise maximum of its windows'
    logits (as inference takes the most sensitive window), so a window without the deciding fact is not
    trained to report it."""
    if doc_pooling == "window_max":
        return model(ids, mask, doc_index, len(sens)), sens, cats
    out = model(ids, mask)
    for k in ("sensitivity", "categories"):
        out[k] = per_doc_max(out[k], doc_index, len(sens))
    return out, sens, cats


@torch.no_grad()
def evaluate(model, items, pad_id, device, dtype, doc_pooling, batch_size=32):
    model.eval()
    tp = fp = fn = 0
    sens_ok = cat_ok = cat_n = n = 0
    for i in range(0, len(items), batch_size):
        ids, mask, labels, sens, cats, doc_index = collate(items[i : i + batch_size], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            out, sens, cats = forward(model, ids, mask, sens, cats, doc_index, doc_pooling)
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
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="padded tokens per batch (long documents get smaller batches)",
    )
    ap.add_argument("--base", default=M.BASE_MODEL, help="Hugging Face encoder to fine-tune")
    ap.add_argument("--base-revision", default=M.BASE_REVISION)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--span-weight",
        type=float,
        default=1.0,
        help="weight of the span loss against the two document losses",
    )
    ap.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    ap.add_argument(
        "--ema",
        type=float,
        default=0,
        help="decay of a moving average of the weights to score as well (0: off)",
    )
    ap.add_argument("--doc-pooling", choices=["per_window", "window_max"], default="per_window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only N training documents (smoke tests)")
    a = ap.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        ap.error(f"{a.out} is not empty")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device(a.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok = M.load_tokenizer(a.base, a.base_revision)
    docs = read_jsonl(a.data / "train.jsonl")
    docs = docs[: a.limit] if a.limit else docs
    train = [encode_windows(tok, d, a.max_length) for d in docs]
    valid = [encode_windows(tok, d, a.max_length) for d in read_jsonl(a.data / "valid.jsonl")]
    model = M.new_model(a.base, a.base_revision).to(device)
    model.train()
    groups = groups_by_length(train, a.batch_size, a.max_tokens)
    steps_per_epoch = len(groups)
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
        "max_tokens": a.max_tokens,
        "seed": a.seed,
        "ema": a.ema,
        "doc_pooling": a.doc_pooling,
        "span_weight": a.span_weight,
        "train_docs": len(train),
        "total_steps": total,
    }

    # Moving average of the weights, kept in fp32 next to the trained ones; each evaluation also scores it.
    live = [v for v in model.state_dict().values() if v.dtype.is_floating_point]
    ema = [v.detach().clone().float() for v in live] if a.ema else []

    def run_eval():
        nonlocal best
        candidates = [("raw", None)]
        if ema:
            candidates.append(("ema", ema))
        for name, weights in candidates:
            backup = None
            if weights:
                backup = [v.detach().clone() for v in live]
                torch._foreach_copy_(live, weights)
            rep = evaluate(model, valid, tok.pad_token_id, device, dtype, a.doc_pooling)
            if device.type == "mps":
                torch.mps.empty_cache()
            rep.update(step=step, weights=name, elapsed_s=round(time.perf_counter() - started, 1))
            history.append(rep)
            print(json.dumps(rep), flush=True)
            score = rep["token_f1"] + rep["sensitivity_acc"] + rep["category_acc"]
            if best is None or score > best:
                best = score
                M.save(model, a.out, {**meta, "step": step, "valid": rep}, tok, a.base, a.base_revision)
                print(f"saved best ({name}) to {a.out}", flush=True)
            if backup:
                torch._foreach_copy_(live, backup)

    while step < total:
        for batch in batches(train, groups, rng):
            if step >= total:
                break
            ids, mask, labels, sens, cats, doc_index = collate(batch, tok.pad_token_id, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                out, sens, cats = forward(model, ids, mask, sens, cats, doc_index, a.doc_pooling)
            span, sl, cl = losses(out, labels, sens, cats)
            loss = a.span_weight * span + sl + cl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if ema:
                with torch.no_grad():
                    torch._foreach_mul_(ema, a.ema)
                    torch._foreach_add_(ema, [v.float() for v in live], alpha=1 - a.ema)
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
