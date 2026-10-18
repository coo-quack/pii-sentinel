import pytest
import torch
from test_model import tiny_model, tiny_tokenizer

from pii_sentinel import average
from pii_sentinel import model as M
from pii_sentinel.labels import SPAN_TYPES


def test_average_is_the_mean_of_the_checkpoints(tmp_path):
    first, second, out = tmp_path / "s0", tmp_path / "s1", tmp_path / "avg"
    M.save(tiny_model(), first, {}, tiny_tokenizer())
    other = tiny_model()
    # tiny_model() is seeded, so the two checkpoints differ only in the classifier weights here. Their final norms
    # match, so the synced heads match too and the head parameters are means as well.
    with torch.no_grad():
        other.classifier.weight.normal_(generator=torch.Generator().manual_seed(1))
    M.save(other, second, {}, tiny_tokenizer())

    average.main([str(first), str(second), "--out", str(out)])

    states = [M.load(p)[0].state_dict() for p in (first, second)]
    avg, _, meta = M.load(out)
    for key, value in avg.state_dict().items():
        expected = (states[0][key].float() + states[1][key].float()) / 2
        assert torch.allclose(value, expected, atol=1e-6), key
    assert meta["averaged_from"] == [str(first), str(second)]
    assert meta["step"] == 0
    assert meta["valid"] is None


def test_checkpoints_with_different_span_labels_are_refused(tmp_path):
    bio = ["O"] + [f"{p}-{t}" for t in SPAN_TYPES[:12] for p in ("B", "I")]
    M.save(tiny_model(), tmp_path / "a", {}, tiny_tokenizer())
    M.save(tiny_model(bio), tmp_path / "b", {}, tiny_tokenizer())
    with pytest.raises(RuntimeError, match="span labels"):
        average.main([str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "avg")])
