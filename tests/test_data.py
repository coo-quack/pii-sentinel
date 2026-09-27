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

    text = "Key AKIA3QF7TZ9KLMN2PQRS, card 4532015112830366, mail kim@example.kr"
    findings = [{"type": "email", "value": "kim@example.kr", "start": 55, "end": 69, "pii": True}]
    secrets, floor = add_rule_findings(text, findings)
    assert [s["value"] for s in secrets] == ["AKIA3QF7TZ9KLMN2PQRS"]
    assert [f["value"] for f in findings] == ["4532015112830366", "kim@example.kr"]
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
