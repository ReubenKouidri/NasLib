import math

import numpy as np
import pytest
import torch

from dnasty.utils.metrics import (
    accuracy,
    auroc,
    compute_metric,
    macro_auroc,
    macro_f1,
    per_class_auroc,
)


def test_auroc_perfect_reversed_and_chance():
    targets = [0, 0, 1, 1]
    assert auroc([0.1, 0.2, 0.8, 0.9], targets) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], targets) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], targets) == 0.5  # all tied


def test_auroc_hand_computed():
    # positives 0.8, 0.4; negatives 0.6, 0.3, 0.1 -> pairs won: 3 + 2 of 6
    assert auroc([0.8, 0.4, 0.6, 0.3, 0.1], [1, 1, 0, 0, 0]) == pytest.approx(5 / 6)
    # a tie between a positive and a negative counts half
    assert auroc([0.8, 0.6, 0.6, 0.1], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_auroc_undefined_without_both_classes():
    assert math.isnan(auroc([0.1, 0.9], [1, 1]))
    assert math.isnan(auroc([0.1, 0.9], [0, 0]))


def test_macro_auroc_skips_undefined_classes():
    scores = np.array([[0.9, 0.2, 0.3], [0.1, 0.8, 0.4], [0.8, 0.1, 0.5]])
    targets = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    per_class = per_class_auroc(scores, targets)
    assert per_class[0] == 1.0 and per_class[1] == 1.0 and math.isnan(per_class[2])
    assert macro_auroc(scores, targets) == 1.0
    assert math.isnan(macro_auroc(scores, np.zeros_like(targets)))


def test_macro_f1_hand_computed():
    # class 0: tp 1 fp 1 fn 0 -> 2/3; class 1: tp 1 fp 0 fn 1 -> 2/3
    preds = np.array([[1, 0], [1, 1], [0, 0]])
    targets = np.array([[1, 0], [0, 1], [0, 1]])
    assert macro_f1(preds, targets) == pytest.approx(2 / 3)
    # integer labels: class 0 tp 1 fn 1 -> 2/3, class 1 tp 1 fp 1 -> 2/3,
    # class 2 absent from both is skipped
    assert macro_f1([0, 1, 1], [0, 1, 0], num_classes=3) == pytest.approx(2 / 3)


def test_accuracy_from_logits():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    assert accuracy(logits, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)


def test_compute_metric_single_and_multilabel():
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [1.0, 0.5]])
    labels = torch.tensor([0, 1, 1])
    assert compute_metric(
        "accuracy", logits, labels, multilabel=False
    ) == pytest.approx(2 / 3)
    assert 0.0 <= compute_metric("macro_auroc", logits, labels, multilabel=False) <= 1.0
    multi = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert compute_metric("macro_auroc", logits, multi, multilabel=True) == 1.0
    # logits > 0 -> [[1, 0], [0, 1], [1, 1]] == targets, so F1 is perfect
    assert compute_metric("macro_f1", logits, multi, multilabel=True) == 1.0
    with pytest.raises(ValueError):
        compute_metric("accuracy", logits, multi, multilabel=True)
    with pytest.raises(ValueError):
        compute_metric("nope", logits, labels, multilabel=False)


def test_matches_sklearn_when_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(200, 4))
    targets = (rng.random((200, 4)) < 0.3).astype(int)
    targets[0] = 1  # make sure every class has a positive
    targets[1] = 0
    expected = sklearn_metrics.roc_auc_score(targets, scores, average="macro")
    assert macro_auroc(scores, targets) == pytest.approx(expected)
    preds = (scores > 0).astype(int)
    expected_f1 = sklearn_metrics.f1_score(targets, preds, average="macro")
    assert macro_f1(preds, targets) == pytest.approx(expected_f1)
