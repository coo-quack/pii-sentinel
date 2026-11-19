import json

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import (
    ModernBertConfig,
    ModernBertForTokenClassification,
    ModernBertModel,
    PreTrainedTokenizerFast,
)

from pii_sentinel import model as M
from pii_sentinel.labels import SPAN_TYPES


def tiny_tokenizer():
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, **{f"w{i}": i + 4 for i in range(60)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )


def tiny_model(bio=M.BIO):
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    torch.manual_seed(0)
    encoder = ModernBertModel(M.token_classification_config(config, bio))
    with torch.no_grad():
        encoder.final_norm.weight.uniform_(0.5, 3.0)
    return M.PiiSentinel(encoder, bio=bio).eval()


def test_synced_head_reproduces_the_span_classifier():
    model = tiny_model()
    ids = torch.randint(0, 64, (2, 12))
    mask = torch.ones_like(ids)
    states, _ = model.encode(ids, mask)
    direct = model.span_logits(states)
    model.sync_head()
    assert torch.allclose(model.span_logits(states), direct, atol=1e-4)


def test_weights_load_as_modernbert_token_classification():
    model = tiny_model()
    model.sync_head()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(M.DOC_HEADS)}
    standard = ModernBertForTokenClassification(model.model.config).eval()
    standard.load_state_dict(state, strict=True)
    ids = torch.randint(0, 64, (1, 10))
    mask = torch.ones_like(ids)
    states, _ = model.encode(ids, mask)
    assert torch.allclose(
        standard(input_ids=ids, attention_mask=mask).logits, model.span_logits(states), atol=1e-4
    )


def test_checkpoint_keeps_its_own_span_labels(tmp_path):
    bio = ["O"] + [f"{p}-{t}" for t in SPAN_TYPES[:12] for p in ("B", "I")]
    model = tiny_model(bio)
    M.save(model, tmp_path, {}, tiny_tokenizer())
    loaded, _, meta = M.load(tmp_path)
    assert meta["bio"] == bio
    assert loaded.bio == bio
    assert loaded.classifier.out_features == 25
    ids = torch.randint(0, 64, (1, 8))
    states, _ = loaded.encode(ids, torch.ones_like(ids))
    assert loaded.span_logits(states).shape[-1] == 25


def test_checkpoint_with_unknown_span_type_is_refused(tmp_path):
    M.save(tiny_model(), tmp_path, {}, tiny_tokenizer())
    meta_file = tmp_path / M.META_FILE
    meta = json.loads(meta_file.read_text())
    meta["bio"] = ["O", "B-FOO", "I-FOO"]
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="B-FOO"):
        M.load(tmp_path)


def test_default_model_is_fetched_at_the_release_tag(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, revision=None, allow_patterns=None):
        calls.append((repo_id, revision))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    monkeypatch.setattr(M, "released_revision", lambda: "v9.9.9")
    M.resolve(M.DEFAULT_MODEL)
    M.resolve("someone/other-model")
    M.resolve(M.DEFAULT_MODEL, "main")
    assert calls == [(M.DEFAULT_MODEL, "v9.9.9"), ("someone/other-model", None), (M.DEFAULT_MODEL, "main")]


def test_a_release_tag_missing_on_the_hub_falls_back_to_main(monkeypatch, tmp_path):
    import httpx2
    from huggingface_hub.errors import RevisionNotFoundError

    calls = []

    def fake_download(repo_id, revision=None, allow_patterns=None):
        calls.append(revision)
        if revision is not None:
            raise RevisionNotFoundError(
                "no such tag", response=httpx2.Response(404, request=httpx2.Request("GET", "https://hf.co"))
            )
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    monkeypatch.setattr(M, "released_revision", lambda: "v9.9.9-dev")
    assert M.resolve(M.DEFAULT_MODEL) == tmp_path
    assert calls == ["v9.9.9-dev", None]


def test_a_local_directory_is_used_as_it_is(tmp_path):
    assert M.resolve(str(tmp_path)) == tmp_path
