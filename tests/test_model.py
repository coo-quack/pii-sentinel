import torch
from transformers import ModernBertConfig, ModernBertForTokenClassification, ModernBertModel

from pii_sentinel import model as M


def tiny_model():
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
    encoder = ModernBertModel(M.token_classification_config(config))
    with torch.no_grad():
        encoder.final_norm.weight.uniform_(0.5, 3.0)
    return M.PiiSentinel(encoder).eval()


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
