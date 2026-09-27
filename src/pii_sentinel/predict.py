"""Inference: findings and document sensitivity for arbitrary-length text."""

import re

import torch

from .data import decode_spans, token_char_spans
from .labels import BIO, CATEGORIES, REPORT, SENSITIVITY

ID_TO_TAG = dict(enumerate(BIO))
# Order, tracking and serial numbers are labelled in training so the model can tell them apart from IDs,
# but they are not personal information and are not reported.
UNREPORTED = {"ORDER_TRACKING", "SERIAL"}
MIN_NUMBER_DIGITS = 6


def _ascii_alnum(c):
    return c.isascii() and c.isalnum()


# A contact value is sometimes tagged as several touching spans of different types
# ("support" + "@example." + "kr"); those pieces are joined into one span.
FAMILY = {"EMAIL": "email", "EMAIL_GENERIC": "email", "PHONE": "phone", "PHONE_CORPORATE": "phone"}
AT = re.compile(r"[@＠]|\[at\]|\(at\)|\bat\b", re.IGNORECASE)
MIN_PHONE_DIGITS = 7


def merge_pieces(text, spans):
    out = []
    for s, e, kind in spans:
        prev = out[-1] if out else None
        if (
            prev
            and kind in FAMILY
            and FAMILY.get(prev[2]) == FAMILY[kind]
            and s - prev[1] <= 1
            and not text[prev[1] : s].strip()
        ):
            # The type covering more characters wins.
            prev[3][kind] = prev[3].get(kind, 0) + (e - s)
            prev[1] = e
            prev[2] = max(prev[3], key=prev[3].get)
        else:
            out.append([s, e, kind, {kind: e - s}])
    return [(s, e, kind) for s, e, kind, _ in out]


def reportable(text, s, e, kind):
    if kind in UNREPORTED:
        return False
    value = text[s:e]
    if FAMILY.get(kind) == "phone":
        return sum(c.isdigit() for c in value) >= MIN_PHONE_DIGITS
    if FAMILY.get(kind) == "email":
        return bool(AT.search(value))
    if REPORT[kind][0] != "number":
        return True
    if sum(c.isdigit() for c in value) < MIN_NUMBER_DIGITS:
        return False
    # A piece of a longer code (INV-2026-0847, 2026나45678) is not an ID number of its own.
    before, after = text[s - 1 : s], text[e : e + 1]
    if _ascii_alnum(before) or _ascii_alnum(after) or value[0] in "-/" or value[-1] in "-/":
        return False
    if before in ("-", "/", "#") and _ascii_alnum(text[s - 2 : s - 1]):
        return False
    return not (after in ("-", "/") and _ascii_alnum(text[e + 1 : e + 2]))


@torch.no_grad()
def analyse(model, tok, text, device, max_length=512, stride=128):
    enc = tok(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        out = model(ids, mask)
    tags = out["span"].argmax(-1).cpu().tolist()
    # Overlapping windows: each token keeps the prediction from the window where it sits most centrally.
    best = {}
    for w, offsets in enumerate(enc["offset_mapping"].tolist()):
        offs = token_char_spans(offsets, text)
        n = int(mask[w].sum())
        for i in range(n):
            s, e = offs[i]
            if s == e:
                continue
            centrality = min(i, n - 1 - i)
            if (s, e) not in best or centrality > best[(s, e)][0]:
                best[(s, e)] = (centrality, tags[w][i])
    keys = sorted(best)
    spans = merge_pieces(text, decode_spans(text, keys, [best[k][1] for k in keys], ID_TO_TAG))
    findings = []
    for s, e, kind in spans:
        if not reportable(text, s, e, kind):
            continue
        ftype, pii, number_type = REPORT[kind]
        f = {"type": ftype, "value": text[s:e], "start": s, "end": e, "pii": pii, "label": kind}
        if number_type:
            f["number_type"] = number_type
        findings.append(f)
    sens = out["sensitivity"].float().softmax(-1).cpu()
    level = max(int(p.argmax()) for p in sens)
    cats = out["categories"].float().sigmoid().max(0).values.cpu().tolist()
    return {
        "sensitivity": {
            "level": SENSITIVITY[level],
            "probabilities": dict(zip(SENSITIVITY, sens.max(0).values.tolist(), strict=True)),
        },
        "categories": {c: round(p, 4) for c, p in zip(CATEGORIES, cats, strict=True)},
        "findings": findings,
        "windows": int(ids.shape[0]),
    }
