from pii_sentinel.data import IGNORE, bio_labels, decode_spans
from pii_sentinel.evaluate import match, normalise
from pii_sentinel.labels import BIO


def test_bio_round_trip():
    text = "Call Jane Doe at 090-1234-5678."
    offsets = [(0, 0), (0, 4), (5, 9), (10, 13), (14, 16), (17, 30), (30, 31), (0, 0)]
    spans = [(5, 13, "PERSON"), (17, 30, "PHONE")]
    tags = bio_labels(offsets, spans)
    assert tags[0] == IGNORE and tags[-1] == IGNORE
    assert [BIO[t] for t in tags[1:-1]] == ["O", "B-PERSON", "I-PERSON", "O", "B-PHONE", "O"]
    ids = [0 if t == IGNORE else t for t in tags]
    assert decode_spans(text, offsets, ids, dict(enumerate(BIO))) == [(5, 13, "PERSON"), (17, 30, "PHONE")]


def test_normalise_strips_honorifics_and_separators():
    assert normalise("鈴木美咲様", "person_name") == "鈴木美咲"
    assert normalise("Dr. Lisa Anderson", "person_name") == "Lisa Anderson"
    assert normalise("+81 (90) 1234-5678", "phone") == "+819012345678"


def test_match_requires_pii_agreement():
    exp = [{"type": "email", "value": "info@example.com", "pii": False}]
    rows = match([{"type": "email", "value": "info@example.com", "pii": True}], exp)
    assert [bool(r["expected"]) and bool(r["predicted"]) for r in rows] == [False, False]


def test_number_fragments_are_not_reported():
    from pii_sentinel.predict import reportable

    text = "請求書 No.INV-2026-0847 口座 1234567 カード 4111 1111 1111 1111 事件番号 2026나45678"
    s = text.index("2026-0847")
    assert not reportable(text, s, s + len("2026-0847"), "BANK_ACCOUNT")
    s = text.index("1234567")
    assert reportable(text, s, s + 7, "BANK_ACCOUNT")
    s = text.index("4111")
    assert reportable(text, s, s + 19, "CREDIT_CARD")
    s = text.index("45678")
    assert not reportable(text, s, s + 5, "NATIONAL_ID")
    assert not reportable(text, 0, 3, "ORDER_TRACKING")


def test_contact_pieces_are_merged():
    from pii_sentinel.predict import merge_pieces, reportable

    text = "連絡先 support@example.kr または +1-202-555-0173、月給 5000"
    a = text.index("support")
    spans = [(a, a + 7, "EMAIL_GENERIC"), (a + 7, a + 16, "EMAIL"), (a + 16, a + 18, "EMAIL_GENERIC")]
    assert merge_pieces(text, spans) == [(a, a + 18, "EMAIL_GENERIC")]
    b = text.index("+1")
    assert merge_pieces(text, [(b, b + 1, "PHONE"), (b + 1, b + 15, "PHONE_CORPORATE")]) == [
        (b, b + 15, "PHONE_CORPORATE")
    ]
    c = text.index("5000")
    assert not reportable(text, c, c + 4, "PHONE")
    assert not reportable(text, a, a + 7, "EMAIL_GENERIC")


def test_mask_keeps_little_of_short_values():
    from pii_sentinel.cli import mask

    assert mask("山田太郎", "person_name") == "山…"
    assert mask("李明", "person_name") == "李…"
    assert mask("090-1234-5678", "phone") == "09…78"
    assert mask("1234567", "number") == "1…"


def test_machine_ids_are_not_reported_as_numbers():
    from pii_sentinel.predict import reportable

    for value in [
        "d41d8cd98f00b204e9800998ecf8427e",
        "a4f29c31-7e2b-41d9-8a3c-5f8b2a91d4e6",
        "00:1A:2B:3C:4D:5E",
    ]:
        text = f"id {value} end"
        assert not reportable(text, 3, 3 + len(value), "NATIONAL_ID")


def test_titles_after_single_character_surnames_are_trimmed():
    from pii_sentinel.data import trim_particles

    for text, name in [("周教授", "周"), ("이 박사", "이"), ("田中先生", "田中"), ("王思敏", "王思敏")]:
        assert text[: trim_particles(text, 0, len(text))] == name


def test_surname_first_names_match_either_way():
    joined = [{"type": "person_name", "value": "Moreau, Nathalie", "pii": True}]
    split = [
        {"type": "person_name", "value": "Moreau", "pii": True},
        {"type": "person_name", "value": "Nathalie", "pii": True},
    ]
    assert all(r["expected"] and r["predicted"] for r in match(joined, split))
    assert all(r["expected"] and r["predicted"] for r in match(split, joined))


def test_rules_add_what_the_model_missed_and_report_secrets_apart():
    from pii_sentinel.predict import add_rule_findings

    text = "Key AKIADUMMYKEY00000000, card 4539148803436467, mail kim@example.kr"
    findings = [{"type": "email", "value": "kim@example.kr", "start": 55, "end": 69, "pii": True}]
    secrets, floor = add_rule_findings(text, findings)
    assert [s["value"] for s in secrets] == ["AKIADUMMYKEY00000000"]
    assert [f["value"] for f in findings] == ["4539148803436467", "kim@example.kr"]
    assert floor == "high"
    role = []
    assert add_rule_findings("Contacto: coordinador@empresa.com", role) == ([], "none")
    assert role[0]["pii"] is False


def test_lone_letters_and_digits_are_not_names():
    from pii_sentinel.predict import reportable

    text = "Pbro. 1 李 Ann"
    assert not reportable(text, 0, 1, "PERSON")
    assert not reportable(text, 6, 7, "PERSON")
    assert reportable(text, 8, 9, "PERSON")
    assert reportable(text, 10, 13, "PERSON")


def test_windows_cover_the_whole_text():
    from pii_sentinel import model as M
    from pii_sentinel.data import windows

    tok = M.load_tokenizer()
    text = " ".join(f"Person{i} Surname{i} works in room {i}." for i in range(400))
    parts = windows(tok, text, 128, 32)
    assert all(len(ids) <= 128 for ids, _ in parts)
    covered = {o for _, offs in parts for o in offs if o != (0, 0)}
    everything = {
        tuple(o) for o in tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    }
    assert covered == everything
    assert windows(tok, "short text", 128)[0][0] == tok("short text")["input_ids"]


def test_windows_reject_a_stride_that_would_not_advance():
    import pytest

    from pii_sentinel import model as M
    from pii_sentinel.data import windows

    tok = M.load_tokenizer()
    with pytest.raises(ValueError):
        windows(tok, "text " * 400, 128, 126)


def test_mask_hides_one_character_values():
    from pii_sentinel.cli import mask

    assert mask("李", "person_name") == "…"
    assert mask("") == "…"


def test_printed_rates_come_from_the_counts():
    from pii_sentinel.evaluate import pr

    assert pr({"tp": 621, "fp": 22, "fn": 13}) == "P96.6% R97.9%"
    assert pr({"tp": 3}) == "P100.0% R100.0%"


def _report_with_secret(text):
    # A report as analyse() returns it: PII with a name, and a secret that starts on line 3.
    name = text.index("Emily Carter")
    secret = text.index("hunter22")
    return {
        "sensitivity": {"level": "high", "probabilities": {"none": 0.0, "low": 0.0, "high": 1.0}},
        "categories": {},
        "findings": [
            {"type": "person_name", "value": "Emily Carter", "start": name, "end": name + 12, "pii": True}
        ],
        "secrets": [
            {
                "type": "secret",
                "value": "hunter22",
                "start": secret,
                "end": secret + 8,
                "rule": "generic-password",
            }
        ],
        "windows": 1,
    }


def test_masked_secret_has_line_and_no_offsets():
    from pii_sentinel.cli import masked_report

    text = "Hello, Emily Carter.\nSecond line\npassword=hunter22\n"
    res = _report_with_secret(text)
    out = masked_report(res, text)
    assert out["secrets"] == [{"type": "secret", "value": "…", "line": 3, "rule": "generic-password"}]
    name = text.index("Emily Carter")
    assert out["findings"] == [
        {"type": "person_name", "value": "E…", "start": name, "end": name + 12, "pii": True}
    ]
    # The report that was passed in is left as it was, so a later caller still sees the raw values.
    assert res["secrets"][0]["value"] == "hunter22" and res["secrets"][0]["start"] == text.index("hunter22")


def test_masked_multiline_secret_reports_its_start_line():
    from pii_sentinel.cli import masked_report

    # A private key block that starts on line 2 and ends on line 4.
    text = "Header\n-----BEGIN PRIVATE KEY-----\nMIIFAKEKEYDATA\n-----END PRIVATE KEY-----\n"
    start = text.index("-----BEGIN")
    end = text.index("-----END") + len("-----END PRIVATE KEY-----")
    res = {
        "findings": [],
        "secrets": [
            {"type": "secret", "value": text[start:end], "start": start, "end": end, "rule": "private-key"}
        ],
    }
    out = masked_report(res, text)
    assert out["secrets"] == [{"type": "secret", "value": "…", "line": 2, "rule": "private-key"}]


def test_masked_secret_at_offset_zero_reports_line_one():
    from pii_sentinel.cli import masked_report

    text = "hunter22\nSecond line\n"
    res = {
        "findings": [],
        "secrets": [
            {"type": "secret", "value": "hunter22", "start": 0, "end": 8, "rule": "generic-password"}
        ],
    }
    out = masked_report(res, text)
    assert out["secrets"] == [{"type": "secret", "value": "…", "line": 1, "rule": "generic-password"}]


def test_cli_masks_secrets_unless_show_values(tmp_path, monkeypatch, capsys):
    import json

    from pii_sentinel import cli

    text = "Hello, Emily Carter.\nSecond line\npassword=hunter22\n"
    path = tmp_path / "doc.txt"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(cli.M, "load", lambda *args: (None, None, {}))
    monkeypatch.setattr(cli, "analyse", lambda model, tok, text, device, **kwargs: _report_with_secret(text))

    cli.main(["scan", "--model", "x", "--device", "cpu", "--json", str(path)])
    (report,) = json.loads(capsys.readouterr().out)
    assert report["secrets"] == [{"type": "secret", "value": "…", "line": 3, "rule": "generic-password"}]
    assert report["findings"][0]["value"] == "E…" and report["findings"][0]["start"] == text.index("Emily")

    cli.main(["scan", "--model", "x", "--device", "cpu", "--show-values", "--json", str(path)])
    (report,) = json.loads(capsys.readouterr().out)
    secret = text.index("hunter22")
    assert report["secrets"] == [
        {
            "type": "secret",
            "value": "hunter22",
            "start": secret,
            "end": secret + 8,
            "rule": "generic-password",
        }
    ]
    assert report["findings"][0]["value"] == "Emily Carter"

    cli.main(["scan", "--model", "x", "--device", "cpu", str(path)])
    assert "  secret (generic-password) on line 3: …" in capsys.readouterr().out


def test_employment_pieces_join_like_contact_pieces():
    from pii_sentinel.predict import merge_pieces

    text = "東京商事営業部へ"
    assert merge_pieces(text, [(0, 4, "EMPLOYMENT"), (4, 7, "EMPLOYMENT")]) == [(0, 7, "EMPLOYMENT")]


def test_server_ip_labelled_non_personal_suppresses_the_rule():
    from pii_sentinel.predict import add_rule_findings

    # A public address, because the rule skips reserved ranges such as the documentation networks.
    text = "ip host: 8.8.8.8"
    start = text.index("8.8.8.8")
    alone = []
    add_rule_findings(text, alone)
    assert [f["value"] for f in alone] == ["8.8.8.8"]
    labelled = [
        {
            "type": "ip_address",
            "value": "8.8.8.8",
            "start": start,
            "end": start + 7,
            "pii": False,
            "label": "IP_SERVER",
        }
    ]
    add_rule_findings(text, labelled)
    assert [f["label"] for f in labelled] == ["IP_SERVER"]


def test_birth_dates_and_addresses_raise_a_none_document_to_low():
    from pii_sentinel.predict import personal_floor

    assert personal_floor([{"type": "date_of_birth", "pii": True}]) == "low"
    assert personal_floor([{"type": "postal_address", "pii": True}]) == "low"
    assert personal_floor([{"type": "date_of_birth", "pii": False}]) == "none"
    assert personal_floor([{"type": "person_name", "pii": True}]) == "none"


def test_attribute_values_must_have_the_shape_of_their_kind():
    from pii_sentinel.predict import reportable

    text = "ip 203.0.113.9 ref E-12345 x"
    s = text.index("203")
    assert reportable(text, s, s + 11, "IP_SERVER")
    assert not reportable(text, s, s + 4, "IP_SERVER")
    m = text.index("E-12345")
    assert reportable(text, m, m + 7, "MEMBER_ID")
    assert not reportable(text, len(text) - 1, len(text), "EMPLOYMENT")


def test_predicted_types_the_corpus_never_labels_are_unscored(monkeypatch):
    from pii_sentinel import evaluate

    corpus = {
        "tests": [
            {
                "id": "t1",
                "lang": "en",
                "text": "Kim Lee mail x@example.com",
                "expected": {
                    "sensitivity": "low",
                    "findings": [{"type": "person_name", "value": "Kim Lee", "pii": True}],
                },
            }
        ]
    }
    predicted = [
        {"type": "person_name", "value": "Kim Lee", "start": 0, "end": 7, "pii": True},
        {"type": "email", "value": "x@example.com", "start": 16, "end": 29, "pii": True},
    ]
    monkeypatch.setattr(
        evaluate,
        "analyse",
        lambda *args, **kwargs: {"findings": predicted, "sensitivity": {"level": "low"}, "categories": {}},
    )
    summary, _ = evaluate.score(corpus, None, None, "cpu")
    assert summary["unscored"] == {"email": 1}
    assert "email" not in summary["types"]
    assert summary["types"]["person_name"]["tp"] == 1


def test_number_types_the_corpus_never_labels_are_unscored(monkeypatch):
    from pii_sentinel import evaluate

    corpus = {
        "tests": [
            {
                "id": "t1",
                "lang": "en",
                "text": "card 4111 1111 1111 1111 member 12345678",
                "expected": {
                    "sensitivity": "high",
                    "findings": [
                        {
                            "type": "number",
                            "value": "4111111111111111",
                            "pii": True,
                            "number_type": "credit_card",
                        }
                    ],
                },
            }
        ]
    }
    predicted = [
        {
            "type": "number",
            "value": "4111111111111111",
            "start": 5,
            "end": 24,
            "pii": True,
            "number_type": "credit_card",
        },
        {
            "type": "number",
            "value": "12345678",
            "start": 32,
            "end": 40,
            "pii": True,
            "number_type": "member_or_customer",
        },
    ]
    monkeypatch.setattr(
        evaluate,
        "analyse",
        lambda *args, **kwargs: {"findings": predicted, "sensitivity": {"level": "high"}, "categories": {}},
    )
    summary, _ = evaluate.score(corpus, None, None, "cpu")
    assert summary["unscored"] == {"number:member_or_customer": 1}
    assert summary["types"]["number"]["tp"] == 1
    assert summary["types"]["number"].get("fp", 0) == 0
    assert summary["personal_types"]["number"].get("fp", 0) == 0


def test_address_beside_a_person_stays_personal():
    from pii_sentinel.predict import address_context

    text = "Shipping to: Taro Yamada\n1-2-3 Chiyoda, Tokyo"
    findings = [
        {
            "type": "person_name",
            "value": "Taro Yamada",
            "start": 13,
            "end": 24,
            "pii": True,
            "label": "PERSON",
        },
        {
            "type": "postal_address",
            "value": "1-2-3 Chiyoda, Tokyo",
            "start": 25,
            "end": 45,
            "pii": True,
            "label": "ADDRESS",
        },
    ]
    address_context(text, findings)
    assert findings[1]["pii"] is True


def test_company_address_with_only_a_hotline_is_not_personal():
    from pii_sentinel.predict import address_context

    text = "Acme GmbH\nParadeplatz 55, 68161 Mannheim\nTel 0800 123456"
    findings = [
        {
            "type": "postal_address",
            "value": "Paradeplatz 55, 68161 Mannheim",
            "start": 10,
            "end": 40,
            "pii": True,
            "label": "ADDRESS",
        },
        {
            "type": "phone",
            "value": "0800 123456",
            "start": 45,
            "end": 56,
            "pii": False,
            "label": "PHONE_CORPORATE",
        },
    ]
    address_context(text, findings)
    assert findings[0]["pii"] is False


def test_company_address_with_two_names_far_above_is_not_personal():
    from pii_sentinel.predict import address_context

    text = "Anna Rossi\nMarco Bianchi\n\n\n\nMainkai 15, 60311 Frankfurt"
    findings = [
        {"type": "person_name", "value": "Anna Rossi", "start": 0, "end": 10, "pii": True, "label": "PERSON"},
        {
            "type": "person_name",
            "value": "Marco Bianchi",
            "start": 11,
            "end": 24,
            "pii": True,
            "label": "PERSON",
        },
        {
            "type": "postal_address",
            "value": "Mainkai 15, 60311 Frankfurt",
            "start": 28,
            "end": 55,
            "pii": True,
            "label": "ADDRESS",
        },
    ]
    address_context(text, findings)
    assert findings[2]["pii"] is False


def test_letter_with_one_signing_name_keeps_the_address_personal():
    from pii_sentinel.predict import address_context

    text = "Dear Sir,\n...\n...\n...\nMainkai 15, 60311 Frankfurt\n\nAnna Rossi"
    findings = [
        {
            "type": "postal_address",
            "value": "Mainkai 15, 60311 Frankfurt",
            "start": 22,
            "end": 49,
            "pii": True,
            "label": "ADDRESS",
        },
        {
            "type": "person_name",
            "value": "Anna Rossi",
            "start": 51,
            "end": 61,
            "pii": True,
            "label": "PERSON",
        },
    ]
    address_context(text, findings)
    assert findings[0]["pii"] is True


def test_company_marker_far_above_blocks_the_one_person_fallback():
    from pii_sentinel.predict import address_context

    text = "Acme Inc.\nLine two\nLine three\nLine four\nMainkai 15, 60311 Frankfurt\n\nAnna Rossi"
    findings = [
        {
            "type": "postal_address",
            "value": "Mainkai 15, 60311 Frankfurt",
            "start": 40,
            "end": 67,
            "pii": True,
            "label": "ADDRESS",
        },
        {
            "type": "person_name",
            "value": "Anna Rossi",
            "start": 69,
            "end": 79,
            "pii": True,
            "label": "PERSON",
        },
    ]
    address_context(text, findings)
    assert findings[0]["pii"] is False


def test_role_addresses_are_not_personal():
    from pii_sentinel.predict import role_addresses

    findings = [
        {"type": "email", "value": "rh@entreprise.com", "start": 0, "end": 17, "pii": True, "label": "EMAIL"},
        {
            "type": "email",
            "value": "ops [at] services.fr",
            "start": 20,
            "end": 40,
            "pii": True,
            "label": "EMAIL",
        },
        {
            "type": "email",
            "value": "anna.rossi@mail.example",
            "start": 50,
            "end": 73,
            "pii": True,
            "label": "EMAIL",
        },
    ]
    findings.append(
        {
            "type": "email",
            "value": "hr-team@company.com",
            "start": 80,
            "end": 99,
            "pii": True,
            "label": "EMAIL",
        }
    )
    role_addresses(findings)
    assert [f["pii"] for f in findings] == [False, False, True, False]


NEW_TYPES_TEXT = (
    "Shipping to: Taro Yamada\n"
    "1-2-3 Chiyoda, Tokyo\n"
    "Born 1985-04-12, handle @taro_y, Bürgermeister, server 192.0.2.10, member M-48213\n"
)


def _report_with_new_types(text):
    # A report as analyse() returns it, with the finding types beyond names and secrets: an address, a birth
    # date, a job, a handle, an address that is not personal, and a member number.
    def finding(ftype, value, pii, **extra):
        start = text.index(value)
        return {"type": ftype, "value": value, "start": start, "end": start + len(value), "pii": pii, **extra}

    return {
        "sensitivity": {"level": "high", "probabilities": {"none": 0.0, "low": 0.0, "high": 1.0}},
        "categories": {},
        "findings": [
            finding("postal_address", "1-2-3 Chiyoda, Tokyo", True, label="ADDRESS"),
            finding("date_of_birth", "1985-04-12", True, label="DATE_OF_BIRTH"),
            finding("employment_info", "Bürgermeister", True, label="EMPLOYMENT"),
            finding("sns_handle", "@taro_y", True, label="HANDLE"),
            finding("ip_address", "192.0.2.10", False, label="IP_SERVER"),
            finding("number", "M-48213", True, label="MEMBER_ID", number_type="member_or_customer"),
        ],
        "secrets": [],
        "windows": 1,
    }


def test_cli_text_report_marks_non_personal_lines_and_number_types(tmp_path, monkeypatch, capsys):
    from pii_sentinel import cli

    path = tmp_path / "doc.txt"
    path.write_text(NEW_TYPES_TEXT, encoding="utf-8")
    monkeypatch.setattr(cli.M, "load", lambda *args: (None, None, {}))
    monkeypatch.setattr(
        cli, "analyse", lambda model, tok, text, device, **kwargs: _report_with_new_types(text)
    )

    cli.main(["scan", "--model", "x", "--device", "cpu", str(path)])
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if "[not PII]" in line] == ["  ip_address [not PII]: 19…10"]
    assert "  number (member_or_customer): M…" in lines
    assert "  postal_address: 1-…yo" in lines


def test_cli_masked_json_masks_each_new_value_and_keeps_the_kind(tmp_path, monkeypatch, capsys):
    import json

    from pii_sentinel import cli

    path = tmp_path / "doc.txt"
    path.write_text(NEW_TYPES_TEXT, encoding="utf-8")
    monkeypatch.setattr(cli.M, "load", lambda *args: (None, None, {}))
    monkeypatch.setattr(
        cli, "analyse", lambda model, tok, text, device, **kwargs: _report_with_new_types(text)
    )

    cli.main(["scan", "--model", "x", "--device", "cpu", "--json", str(path)])
    (report,) = json.loads(capsys.readouterr().out)
    assert [(f["type"], f["value"], f["pii"]) for f in report["findings"]] == [
        ("postal_address", "1-…yo", True),
        ("date_of_birth", "19…12", True),
        ("employment_info", "Bü…er", True),
        ("sns_handle", "@…", True),
        ("ip_address", "19…10", False),
        ("number", "M…", True),
    ]
    assert report["findings"][-1]["number_type"] == "member_or_customer"
    assert report["findings"][-1]["label"] == "MEMBER_ID"
