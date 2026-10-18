"""Regression tests against a real checkpoint. They run only when a local checkpoint is available offline.

Set PII_SENTINEL_MODEL to a checkpoint directory; without it the whole module is skipped:

    PII_SENTINEL_MODEL=models/attrs8-s2 uv run pytest tests/test_regression.py
"""

import os

import pytest
import torch

from pii_sentinel import model as M
from pii_sentinel.predict import analyse

CHECKPOINT = os.environ.get("PII_SENTINEL_MODEL")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="set PII_SENTINEL_MODEL to a checkpoint directory")


@pytest.fixture(scope="module")
def scan():
    device = torch.device("cpu")
    model, tok, meta = M.load(CHECKPOINT, device)

    def run(text):
        return analyse(
            model,
            tok,
            text,
            device,
            max_length=meta.get("max_length", 512),
            use_rules=True,
            doc_pooling=meta.get("doc_pooling", "per_window"),
        )

    return run


def has(res, ftype, value=None, pii=None, number_type=None):
    return any(
        f["type"] == ftype
        and (value is None or f["value"] == value)
        and (pii is None or f["pii"] is pii)
        and (number_type is None or f.get("number_type") == number_type)
        for f in res["findings"]
    )


def listing(res):
    # The findings in the assertion message, so a failure shows what the model returned.
    return [(f["type"], f["value"], f["pii"]) for f in res["findings"]]


def test_office_title_is_employment_and_the_name_is_a_person(scan):
    res = scan("Bürgermeister Klaus Weber eröffnete die Sitzung.")
    assert has(res, "employment_info", value="Bürgermeister"), listing(res)
    assert has(res, "person_name", value="Klaus Weber"), listing(res)


def test_address_beside_a_person_is_personal(scan):
    res = scan("Shipping to: Taro Yamada\n1-2-3 Chiyoda, Tokyo")
    assert has(res, "postal_address", pii=True), listing(res)


def test_company_address_is_not_personal(scan):
    res = scan("Acme GmbH\nParadeplatz 55, 68161 Mannheim")
    assert has(res, "postal_address", pii=False), listing(res)


def test_role_mailbox_is_not_personal(scan):
    res = scan("Contact: hr-team@company.com")
    assert has(res, "email", pii=False), listing(res)


def test_member_id_is_a_member_or_customer_number(scan):
    res = scan("Member ID: M-48213 (Anna Rossi)")
    assert has(res, "number", number_type="member_or_customer"), listing(res)
