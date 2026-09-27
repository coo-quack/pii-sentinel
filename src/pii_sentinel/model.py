"""mmBERT-pii-sentinel: an mmBERT-base encoder with span, sensitivity and category heads."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .labels import BIO, CATEGORIES, SENSITIVITY

BASE_MODEL = "jhu-clsp/mmBERT-base"
BASE_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
FORMAT = "pii-sentinel/1"
MODEL_NAME = "mmBERT-pii-sentinel"
META_FILE = "pii_sentinel.json"
WEIGHTS_FILE = "model.safetensors"


class PiiSentinel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.span_head = nn.Linear(hidden, len(BIO))
        self.sensitivity_head = nn.Linear(hidden, len(SENSITIVITY))
        self.category_head = nn.Linear(hidden, len(CATEGORIES))

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        states = self.dropout(states)
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
        return {
            "span": self.span_head(states),
            "sensitivity": self.sensitivity_head(pooled),
            "categories": self.category_head(pooled),
        }


def load_tokenizer(revision=BASE_REVISION):
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision)


def new_model():
    """The base encoder with freshly initialised heads (for training)."""
    return PiiSentinel(AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32))


def save(model, out: Path, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / WEIGHTS_FILE))
    full = {
        "format": FORMAT,
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "bio": BIO,
        "categories": CATEGORIES,
        "sensitivity": SENSITIVITY,
        **meta,
    }
    (out / META_FILE).write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n")


def load(path: Path, device="cpu"):
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{path} is not a {FORMAT} checkpoint")
    if meta["bio"] != BIO or meta["categories"] != CATEGORIES:
        raise RuntimeError(f"{path} was trained with a different label set")
    config = AutoConfig.from_pretrained(meta["base_model"], revision=meta["base_revision"])
    model = PiiSentinel(AutoModel.from_config(config))
    model.load_state_dict(load_file(str(path / WEIGHTS_FILE)))
    return model.to(device).eval(), load_tokenizer(meta["base_revision"]), meta
