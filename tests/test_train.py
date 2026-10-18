import torch

from pii_sentinel.data import IGNORE
from pii_sentinel.labels import BIO, CATEGORIES, SENSITIVITY
from pii_sentinel.train import losses, selection_key


def report(f1, sens, cat, high=0, personal=0):
    return {
        "token_f1": f1,
        "sensitivity_acc": sens,
        "category_acc": cat,
        "dev_high_missed": high,
        "dev_personal_missed": personal,
    }


def test_label_smoothing_changes_only_the_span_loss():
    torch.manual_seed(0)
    out = {
        "span": torch.randn(2, 6, len(BIO)),
        "sensitivity": torch.randn(2, len(SENSITIVITY)),
        "categories": torch.randn(2, len(CATEGORIES)),
    }
    labels = torch.randint(0, len(BIO), (2, 6))
    labels[0, 0] = IGNORE
    sens = torch.tensor([0, 2])
    cats = torch.rand(2, len(CATEGORIES)).round()

    plain = losses(out, labels, sens, cats, 0.0)
    smoothed = losses(out, labels, sens, cats, 0.1)
    assert not torch.isclose(plain[0], smoothed[0])
    assert torch.equal(plain[1], smoothed[1])
    assert torch.equal(plain[2], smoothed[2])


def test_without_a_development_corpus_the_synthetic_score_is_the_key():
    # Nothing is ineligible early in training, and the penalties are not applied.
    rep = report(0.9, 1.0, 0.8, high=5, personal=5)
    assert selection_key(rep, step=0, total=100, dev_based=False) == 0.9 + 1.0 + 0.8


def test_an_early_step_is_not_eligible_with_a_development_corpus():
    rep = report(0.9, 1.0, 0.8)
    assert selection_key(rep, step=59, total=100, dev_based=True) is None
    assert selection_key(rep, step=60, total=100, dev_based=True) is not None


def test_fewer_high_misses_win_when_the_penalty_outweighs_the_score():
    many_misses = report(0.95, 1.0, 1.0, high=4)  # synthetic 2.95, key 2.95 - 0.08 = 2.87
    few_misses = report(0.90, 1.0, 1.0, high=0)  # synthetic 2.90, key 2.90
    assert selection_key(many_misses, 80, 100, True) < selection_key(few_misses, 80, 100, True)


def test_the_synthetic_score_decides_when_the_counts_are_equal():
    better = report(0.95, 1.0, 1.0, high=1, personal=1)
    worse = report(0.90, 1.0, 1.0, high=1, personal=1)
    assert selection_key(better, 80, 100, True) > selection_key(worse, 80, 100, True)
