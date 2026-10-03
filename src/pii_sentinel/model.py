"""mmBERT-pii-sentinel: an mmBERT-base encoder with span, sensitivity and category heads.

A checkpoint is laid out like a transformers ModernBertForTokenClassification model (config.json, tokenizer files and
model.safetensors), so `pipeline("token-classification")` loads the span labelling as it is. The document heads are
kept apart in document_heads.safetensors, which only this package reads, together with pii_sentinel.json."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.initialization import no_init_weights
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from .labels import BIO, CATEGORIES, SENSITIVITY

BASE_MODEL = "jhu-clsp/mmBERT-base"
BASE_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
FORMAT = "pii-sentinel/2"
LEGACY_FORMAT = "pii-sentinel/1"
MODEL_NAME = "mmBERT-pii-sentinel"
META_FILE = "pii_sentinel.json"
WEIGHTS_FILE = "model.safetensors"
DOC_HEADS_FILE = "document_heads.safetensors"
DOC_HEADS = ("sensitivity_head.", "category_head.")
CHECKPOINT_FILES = [
    META_FILE,
    WEIGHTS_FILE,
    DOC_HEADS_FILE,
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def token_classification_config(config):
    """The base config set up for ModernBertForTokenClassification with the BIO labels. The prediction head is
    linear (dense, then LayerNorm) so that it can reproduce the span classifier exactly; see sync_head."""
    config.num_labels = len(BIO)
    config.id2label = dict(enumerate(BIO))
    config.label2id = {tag: i for i, tag in enumerate(BIO)}
    config.classifier_activation = "linear"
    config.architectures = ["ModernBertForTokenClassification"]
    return config


class PiiSentinel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.model = encoder
        config = encoder.config
        hidden = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = ModernBertPredictionHead(config)
        self.head.requires_grad_(False)
        self.classifier = nn.Linear(hidden, len(BIO))
        self.sensitivity_head = nn.Linear(hidden, len(SENSITIVITY))
        self.category_head = nn.Linear(hidden, len(CATEGORIES))
        # Training scores the span classifier on the encoder states directly; a saved or loaded checkpoint goes
        # through the head, which sync_head sets to give the same result.
        self.use_head = False

    @torch.no_grad()
    def sync_head(self):
        """Make head(states) equal to states. The encoder ends with a LayerNorm without bias (weight g), so the
        states are g * z with z of zero mean and unit variance; dense = diag(1/g) recovers z and the head's
        LayerNorm (weight g) turns it back into g * z, up to its epsilon."""
        g = self.model.final_norm.weight
        self.head.dense.weight.copy_(torch.diag(1.0 / g))
        self.head.norm.weight.copy_(g)
        self.use_head = True

    def span_logits(self, states):
        return self.classifier(self.head(states) if self.use_head else states)

    def encode(self, input_ids, attention_mask):
        """Token states and the mean-pooled state of each window."""
        states = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        states = self.dropout(states)
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        return states, (states * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, doc_index=None, n_docs=None):
        """One row per window. With doc_index, the windows of each document are pooled together (the
        feature-wise maximum of their mean-pooled states), so the document heads see the whole document even
        when a fact and the person it concerns fall in different windows."""
        states, pooled = self.encode(input_ids, attention_mask)
        if doc_index is not None:
            index = doc_index.unsqueeze(-1).expand_as(pooled)
            init = torch.full(
                (n_docs, pooled.shape[-1]), float("-inf"), dtype=pooled.dtype, device=pooled.device
            )
            pooled = init.scatter_reduce(0, index, pooled, reduce="amax")
        return {
            "span": self.span_logits(states),
            "sensitivity": self.sensitivity_head(pooled),
            "categories": self.category_head(pooled),
        }


def load_tokenizer(base=BASE_MODEL, revision=BASE_REVISION):
    return AutoTokenizer.from_pretrained(base, revision=revision)


def new_model(base=BASE_MODEL, revision=BASE_REVISION):
    """The base encoder with freshly initialised heads (for training)."""
    encoder = AutoModel.from_pretrained(base, revision=revision, dtype=torch.float32)
    token_classification_config(encoder.config)
    return PiiSentinel(encoder)


def save(model, out: Path, meta: dict, tokenizer, base=BASE_MODEL, revision=BASE_REVISION):
    out.mkdir(parents=True, exist_ok=True)
    use_head = model.use_head
    model.sync_head()
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    model.use_head = use_head
    heads = {k: v for k, v in state.items() if k.startswith(DOC_HEADS)}
    tc = {k: v for k, v in state.items() if k not in heads}
    save_file(tc, str(out / WEIGHTS_FILE), metadata={"format": "pt"})
    save_file(heads, str(out / DOC_HEADS_FILE), metadata={"format": "pt"})
    model.model.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
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

    return Path(snapshot_download(repo_id=str(model), allow_patterns=CHECKPOINT_FILES))


def load(path, device="cpu"):
    path = resolve(path)
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("format") not in (FORMAT, LEGACY_FORMAT):
        raise RuntimeError(f"{path} is not a {FORMAT} checkpoint")
    if meta["bio"] != BIO or meta["categories"] != CATEGORIES:
        raise RuntimeError(f"{path} was trained with a different label set")
    state = load_file(str(path / WEIGHTS_FILE))
    if meta["format"] == FORMAT:
        state |= load_file(str(path / DOC_HEADS_FILE))
        config = AutoConfig.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        # Releases up to 0.1.x: encoder.* and span_head.*, with the config and tokenizer of the base model.
        config = token_classification_config(
            AutoConfig.from_pretrained(meta["base_model"], revision=meta["base_revision"])
        )
        tokenizer = load_tokenizer(meta["base_model"], meta["base_revision"])
        renames = {"encoder.": "model.", "span_head.": "classifier."}
        state = {
            next((new + k[len(old) :] for old, new in renames.items() if k.startswith(old)), k): v
            for k, v in state.items()
        }
    # The weights are loaded right after, so skip the random initialisation (about 25 s on a laptop CPU).
    with no_init_weights():
        model = PiiSentinel(AutoModel.from_config(config))
    if meta["format"] == FORMAT:
        model.load_state_dict(state)
        model.use_head = True
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("head.") for k in missing):
            raise RuntimeError(f"{path}: unexpected weights {unexpected}, missing {missing}")
        model.sync_head()
    return model.to(device).eval(), tokenizer, meta
