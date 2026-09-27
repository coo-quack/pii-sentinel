"""Turning labelled documents into token-level tensors and back."""

import json
from pathlib import Path

from .labels import BIO, BIO_INDEX, CATEGORIES

IGNORE = -100


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def token_char_spans(offsets, text):
    """Offsets with leading whitespace trimmed (SentencePiece pieces often start with the space)."""
    out = []
    for start, end in offsets:
        while start < end and text[start].isspace():
            start += 1
        out.append((start, end))
    return out


def bio_labels(offsets, spans):
    """BIO tag per token. Special tokens (empty offsets) are ignored by the loss."""
    tags = []
    for start, end in offsets:
        if start == end:
            tags.append(IGNORE)
            continue
        tag = "O"
        for s, e, kind in spans:
            if start < e and end > s:
                tag = f"{'B' if start <= s else 'I'}-{kind}"
                break
        tags.append(BIO_INDEX[tag] if tag != "O" else 0)
    # A run must start with B-. Whitespace-only tokens are ignored by the loss, so look past them to the
    # previous labelled token: "152 3385" stays one span even though a bare space token sits in between.
    prev = 0
    for i, t in enumerate(tags):
        if t == IGNORE:
            continue
        if t != 0:
            kind = BIO[t][2:]
            if prev == 0 or BIO[prev][2:] != kind:
                tags[i] = BIO_INDEX["B-" + kind]
        prev = t
    return tags


def encode(tokenizer, doc, max_length):
    enc = tokenizer(doc["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = token_char_spans(enc["offset_mapping"], doc["text"])
    return {
        "input_ids": enc["input_ids"],
        "labels": bio_labels(offsets, doc["spans"]),
        "sensitivity": doc["sensitivity"],
        "categories": [1.0 if c in doc["categories"] else 0.0 for c in CATEGORIES],
    }


def decode_spans(text, offsets, tag_ids, id_to_tag):
    """Group B-/I- runs of the same type into character spans."""
    spans, cur = [], None
    for (start, end), t in zip(offsets, tag_ids, strict=True):
        if start == end:
            continue
        tag = id_to_tag[t]
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        prefix, kind = tag[0], tag[2:]
        if cur and kind == cur[2] and (prefix == "I" or start <= cur[1]):
            cur[1] = end
        else:
            if cur:
                spans.append(cur)
            cur = [start, end, kind]
    if cur:
        spans.append(cur)
    out = []
    for s, e, k in spans:
        s = trim_leading(text, s, e)
        if k == "PERSON":
            e = trim_particles(text, s, e)
        if text[s:e].strip():
            out.append((s, e, k))
    return out


LEADING = set(",，、:：;；.。")
# Particles the tokenizer can glue onto the end of a name. Japanese and Chinese ones are removed only after a
# kanji or katakana (「まこと」 keeps its と); Korean keeps single-syllable particles because they collide with
# name syllables (지은), and Hangul is usually split per syllable anyway.
CJK_PARTICLES = [
    "から",
    "まで",
    "より",
    "の",
    "は",
    "が",
    "を",
    "に",
    "と",
    "も",
    "へ",
    "で",
    "や",
    "的",
    "是",
    "说",
]
HONORIFIC_TAIL = [
    "ちゃん",
    "くん",
    "さん",
    "先生",
    "様",
    "氏",
    "君",
    "殿",
    "에게서",
    "에게",
    "께서",
    "한테",
    "씨",
    "님",
]


def is_kanji_or_katakana(ch):
    return "\u4e00" <= ch <= "\u9fff" or "\u30a0" <= ch <= "\u30ff"


def trim_leading(text, s, e):
    while s < e and text[s] in LEADING:
        s += 1
    return s


def trim_particles(text, s, e):
    changed = True
    while changed:
        changed = False
        for p in HONORIFIC_TAIL:
            if text[s:e].endswith(p) and e - len(p) - s >= 2:
                e, changed = e - len(p), True
                break
        else:
            for p in CJK_PARTICLES:
                if (
                    text[s:e].endswith(p)
                    and e - len(p) - s >= 2
                    and is_kanji_or_katakana(text[e - len(p) - 1])
                ):
                    e, changed = e - len(p), True
                    break
    return e
