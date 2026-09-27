"""Inference: findings and document sensitivity for arbitrary-length text."""

import re

import torch

from . import rules
from .data import decode_spans, is_hanzi_or_hangul, token_char_spans, windows
from .labels import BIO, CATEGORIES, REPORT, SENSITIVITY

ID_TO_TAG = dict(enumerate(BIO))
# Order, tracking and serial numbers are labelled in training so the model can tell them apart from IDs,
# but they are not personal information and are not reported.
UNREPORTED = {"ORDER_TRACKING", "SERIAL"}
MIN_NUMBER_DIGITS = 6
# Chosen on the development set: raises name recall and precision over plain argmax.
O_THRESHOLD = 0.8
# Machine identifiers that look like long numbers but belong to no person.
MACHINE_ID = re.compile(
    r"^(?:[0-9a-f]{32,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?:[0-9a-f]{2}[:-]){4,}[0-9a-f]{2})$",
    re.IGNORECASE,
)


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
    if kind == "PERSON":
        # A lone Latin letter or a string without letters is not a name; a single Chinese or Korean
        # character can be a surname (李, 이).
        letters = [c for c in value if c.isalpha()]
        return bool(letters) and (len(value.strip()) > 1 or is_hanzi_or_hangul(value.strip()))
    if FAMILY.get(kind) == "phone":
        return sum(c.isdigit() for c in value) >= MIN_PHONE_DIGITS
    if FAMILY.get(kind) == "email":
        return bool(AT.search(value))
    if REPORT[kind][0] != "number":
        return True
    if sum(c.isdigit() for c in value) < MIN_NUMBER_DIGITS or MACHINE_ID.match(value):
        return False
    # A piece of a longer code (INV-2026-0847, 2026나45678) is not an ID number of its own.
    before, after = text[s - 1 : s], text[e : e + 1]
    if _ascii_alnum(before) or _ascii_alnum(after) or value[0] in "-/" or value[-1] in "-/":
        return False
    if before in ("-", "/", "#") and _ascii_alnum(text[s - 2 : s - 1]):
        return False
    return not (after in ("-", "/") and _ascii_alnum(text[e + 1 : e + 2]))


# How a personal-information rule from the sensitive-canary rule set is reported when the model found
# nothing at that place: (finding type, pii, number_type, raises the document to).
RULE_REPORT = {
    "pii-email": ("email", True, None, "none"),
    "pii-credit-card": ("number", True, "credit_card", "high"),
    "pii-ssn": ("number", True, "national_id", "high"),
    "pii-mynumber-jp": ("number", True, "my_number", "high"),
    "pii-nir-fr": ("number", True, "national_id", "high"),
    "pii-codice-fiscale-it": ("number", True, "national_id", "high"),
    "pii-steuer-id-de": ("number", True, "national_id", "high"),
    "pii-dni-nie-es": ("number", True, "national_id", "high"),
    "pii-rrn-kr": ("number", True, "national_id", "high"),
    "pii-resident-id-cn": ("number", True, "national_id", "high"),
    "pii-brn-kr": ("number", False, "business_registration", "none"),
    "pii-postal-jp": ("postal_code", True, None, "none"),
    "pii-postal-code": ("postal_code", True, None, "none"),
    "pii-postal-cn": ("postal_code", True, None, "none"),
    "pii-ipv4-public": ("ip_address", True, None, "none"),
    "pii-ipv6": ("ip_address", True, None, "none"),
    **{
        f"pii-phone-{c}": ("phone", True, None, "none")
        for c in ("us", "jp", "fr", "it", "de", "es", "kr", "cn")
    },
}
# Role addresses (a department, not a person) in the eight languages.
GENERIC_LOCAL = re.compile(
    r"^(?:info|support|contact|kontakt|contacto|contatto|help|hilfe|ayuda|aiuto|sales|vertrieb|ventas|vendite"
    r"|admin|office|buero|oficina|ufficio|noreply|no-reply|service|servicio|servizio|hr|rrhh|personal|press"
    r"|presse|prensa|stampa|team|hello|research|coordina\w*|recruit\w*|jobs|careers|karriere|empleo|lavoro"
    r"|billing|accounts?|marketing|news|privacy|datenschutz|security|legal|webmaster|postmaster|dpo|compliance"
    r"|customer\w*|kundenservice|booking|reservations?|events?|orders?|secretar\w*|sekretariat|segreteria"
    r"|direccion|direzione|redaktion|redazione)\d*@",
    re.IGNORECASE,
)
RANK = {"none": 0, "low": 1, "high": 2}


def _overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def add_rule_findings(text, findings):
    """Values the model did not mark but the rule set recognises (checksummed IDs, contact formats), and
    secrets, which are reported apart from personal information. Returns (secrets, level floor).

    Every matching rule raises the floor, whether or not the model already found the value. Findings inside a
    secret are dropped: the password before the "@" of a connection string is not an e-mail address."""
    matches = rules.scan(text)
    secrets = [
        {"type": "secret", "value": m.value, "start": m.start, "end": m.end, "rule": m.rule}
        for m in matches
        if m.category == "secret"
    ]
    findings[:] = [f for f in findings if not any(_overlaps(f, x) for x in secrets)]
    floor = "none"
    for m in matches:
        span = {"start": m.start, "end": m.end}
        if m.rule not in RULE_REPORT or any(_overlaps(span, x) for x in secrets):
            continue
        ftype, pii, number_type, level = RULE_REPORT[m.rule]
        if ftype == "email" and GENERIC_LOCAL.match(m.value):
            pii, level = False, "none"
        floor = max(floor, level, key=RANK.get)
        if any(_overlaps(span, f) for f in findings):
            continue
        f = {"type": ftype, "value": m.value, **span, "pii": pii, "label": m.rule}
        if number_type:
            f["number_type"] = number_type
        findings.append(f)
    findings.sort(key=lambda f: f["start"])
    return secrets, floor


@torch.no_grad()
def analyse(
    model,
    tok,
    text,
    device,
    max_length=512,
    stride=None,
    o_threshold=O_THRESHOLD,
    use_rules=True,
    doc_pooling="per_window",
    batch_size=16,
):
    """With use_rules, the sensitive-canary rule set adds what the model missed and reports secrets. doc_pooling is the
    checkpoint's "doc_pooling": "per_window" judges each window and takes the most sensitive, "window_max"
    pools all windows before the document heads. Windows go through the encoder batch_size at a time.

    The reported probabilities are the model's judgement that decided the level (the most sensitive window);
    the rules and a personal contact or number can raise the level above it."""
    parts = windows(tok, text, max_length, stride)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tags, window_probs, pooled_all = [], [], []
    for b in range(0, len(parts), batch_size):
        chunk = parts[b : b + batch_size]
        width = max(len(w) for w, _ in chunk)
        ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, (w, _) in enumerate(chunk):
            ids[i, : len(w)] = torch.tensor(w)
            mask[i, : len(w)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            states, pooled = model.encode(ids, mask)
            span = model.span_head(states)
            if doc_pooling == "window_max":
                pooled_all.append(pooled.float())
            else:
                window_probs.append(
                    torch.cat(
                        [
                            model.sensitivity_head(pooled).float().softmax(-1),
                            model.category_head(pooled).float().sigmoid(),
                        ],
                        -1,
                    )
                )
        probs = span.float().softmax(-1)
        if o_threshold is None:
            chunk_tags = probs.argmax(-1)
        else:
            # Recall-oriented decoding: a token is an entity unless the model is at least this sure it is O.
            best_entity = probs[..., 1:].argmax(-1) + 1
            chunk_tags = torch.where(probs[..., 0] >= o_threshold, 0, best_entity)
        chunk_tags = chunk_tags.cpu().tolist()
        tags += [row[: len(w)] for row, (w, _) in zip(chunk_tags, chunk, strict=True)]
    if doc_pooling == "window_max":
        pooled = torch.cat(pooled_all).amax(0, keepdim=True).to(next(model.parameters()).dtype)
        window_probs = [
            torch.cat(
                [
                    model.sensitivity_head(pooled).float().softmax(-1),
                    model.category_head(pooled).float().sigmoid(),
                ],
                -1,
            )
        ]
    doc = torch.cat(window_probs).cpu()
    sens, cats = doc[:, : len(SENSITIVITY)], doc[:, len(SENSITIVITY) :]
    # Overlapping windows: each token keeps the prediction from the window where it sits most centrally.
    best = {}
    for w, (_, offsets) in enumerate(parts):
        offs = token_char_spans(offsets, text)
        n = len(tags[w])
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
    # The most sensitive window decides; between windows of the same level, the more confident one.
    decisive = max(range(len(sens)), key=lambda w: (int(sens[w].argmax()), float(sens[w].max())))
    level = SENSITIVITY[int(sens[decisive].argmax())]
    cats = cats.max(0).values.tolist()
    secrets = []
    if use_rules:
        secrets, floor = add_rule_findings(text, findings)
        level = max(level, floor, key=RANK.get)
    # A document with a personal contact or number contains personal information, whatever the document
    # head said (names are left out: a deceased person's name is detected but does not count).
    if any(f["pii"] and f["type"] in ("email", "phone", "number") for f in findings):
        level = max(level, "low", key=RANK.get)
    return {
        "sensitivity": {
            "level": level,
            "probabilities": dict(zip(SENSITIVITY, sens[decisive].tolist(), strict=True)),
        },
        "categories": {c: round(p, 4) for c, p in zip(CATEGORIES, cats, strict=True)},
        "findings": findings,
        "secrets": secrets,
        "windows": len(parts),
    }
