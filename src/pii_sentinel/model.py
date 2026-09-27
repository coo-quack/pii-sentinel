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

    def forward(self, input_ids, attention_mask, doc_index=None, n_docs=None):
        """One row per window. With doc_index, the windows of each document are pooled together (the
        feature-wise maximum of their mean-pooled states), so the document heads see the whole document even
        when a fact and the person it concerns fall in different windows."""
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        states = self.dropout(states)
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
        if doc_index is not None:
            index = doc_index.unsqueeze(-1).expand_as(pooled)
            init = torch.full(
                (n_docs, pooled.shape[-1]), float("-inf"), dtype=pooled.dtype, device=pooled.device
            )
            pooled = init.scatter_reduce(0, index, pooled, reduce="amax")
        return {
            "span": self.span_head(states),
            "sensitivity": self.sensitivity_head(pooled),
            "categories": self.category_head(pooled),
        }


def load_tokenizer(base=BASE_MODEL, revision=BASE_REVISION):
    return AutoTokenizer.from_pretrained(base, revision=revision)


def new_model(base=BASE_MODEL, revision=BASE_REVISION):
    """The base encoder with freshly initialised heads (for training)."""
    return PiiSentinel(AutoModel.from_pretrained(base, revision=revision, dtype=torch.float32))


def save(model, out: Path, meta: dict, base=BASE_MODEL, revision=BASE_REVISION):
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / WEIGHTS_FILE))
    full = {
        "format": FORMAT,
        "model_name": MODEL_NAME,
        "base_model": base,
        "base_revision": revision,
        "bio": BIO,
        "categories": CATEGORIES,
        "sensitivity": SENSITIVITY,
        **meta,
    }
    (out / META_FILE).write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n")


def resolve(model):
    """A local checkpoint directory, or a Hugging Face model id (downloaded once into the local cache)."""
    path = Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model), allow_patterns=[META_FILE, WEIGHTS_FILE]))


def load(path, device="cpu"):
    path = resolve(path)
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{path} is not a {FORMAT} checkpoint")
    if meta["bio"] != BIO or meta["categories"] != CATEGORIES:
        raise RuntimeError(f"{path} was trained with a different label set")
    config = AutoConfig.from_pretrained(meta["base_model"], revision=meta["base_revision"])
    model = PiiSentinel(AutoModel.from_config(config))
    model.load_state_dict(load_file(str(path / WEIGHTS_FILE)))
    return model.to(device).eval(), load_tokenizer(meta["base_model"], meta["base_revision"]), meta
