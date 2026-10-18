import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from pii_sentinel.gen import generator as G
from pii_sentinel.gen.slots import problems

ROOT = Path(__file__).resolve().parents[1]
SMALL = [
    "--train-docs",
    "30",
    "--valid-docs",
    "3",
    "--langs",
    "en,ja",
    "--templates",
    str(ROOT / "templates"),
]


def generate(out, hash_seed):
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    cmd = [sys.executable, "-m", "pii_sentinel.gen.generator", "--out", str(out), "--seed", "7", *SMALL]
    subprocess.run(cmd, check=True, env=env, capture_output=True)
    return (out / "train.jsonl").read_bytes()


def test_same_seed_gives_the_same_data_whatever_the_hash_seed(tmp_path):
    assert generate(tmp_path / "a", 1) == generate(tmp_path / "b", 2)


def test_main_can_run_twice_in_one_process(tmp_path):
    G.main(["--out", str(tmp_path / "a"), "--seed", "7", *SMALL])
    G.main(["--out", str(tmp_path / "b"), "--seed", "7", *SMALL])
    assert (tmp_path / "a" / "train.jsonl").read_bytes() == (tmp_path / "b" / "train.jsonl").read_bytes()


def test_names_clash_only_as_whole_words_in_latin_script():
    assert G.clashes("Kim", "Kim Lee")
    assert G.clashes("Kim Lee", "Kim")
    assert G.clashes("Harris", "Harris")
    assert not G.clashes("Kimberly", "Kim")
    assert not G.clashes("Harris", "Harrison")
    assert not G.clashes("John", "Johnson")


def test_names_clash_anywhere_in_cjk_but_not_on_one_character():
    assert G.clashes("李明", "李明华")
    assert G.clashes("李明华", "李明")
    assert G.clashes("李", "李")
    assert not G.clashes("李明", "李")


def test_missing_template_directory_stops_the_run(tmp_path):
    with pytest.raises(SystemExit, match="no \\*.json"):
        G.load_slot_templates(tmp_path / "missing")


def test_a_password_makes_a_template_high():
    t = {
        "cell": "numbers_of_no_person",
        "sensitivity": "none",
        "lang": "en",
        "text": "Key: {PW}\nBuild: {HASH}",
    }
    assert any("{PW}" in p for p in problems(t))


def test_identifier_only_slot_documents_keep_person_name():
    lang = G.Lang(json.loads((G.HERE / "resources" / "en.json").read_text()), [], [])
    template = {
        "cell": "identifier_only",
        "sensitivity": "low",
        "lang": "en",
        "text": "Member {P1.id} renewed.",
    }
    saved = dict(G.SLOT_T)
    G.SLOT_T["en"] = {"train": [template], "valid": [template]}
    try:
        doc = G.Doc()
        G.slot_doc(doc, lang, "train")
    finally:
        G.SLOT_T.clear()
        G.SLOT_T.update(saved)
    assert doc.personal and "person_name" in doc.cats


def test_attribute_and_identifier_spans_appear_in_training_data(tmp_path):
    G.main(
        [
            "--out",
            str(tmp_path),
            "--seed",
            "7",
            "--train-docs",
            "300",
            "--valid-docs",
            "3",
            "--langs",
            "en,ja",
            "--templates",
            str(ROOT / "templates"),
        ]
    )
    docs = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    kinds = Counter(span[2] for d in docs for span in d["spans"])
    for kind in (
        "ADDRESS",
        "ADDRESS_CORPORATE",
        "DATE_OF_BIRTH",
        "EMPLOYMENT",
        "HANDLE",
        "IP_PERSONAL",
        "IP_SERVER",
        "MEMBER_ID",
    ):
        assert kinds[kind] > 0, kind
    # Role labels ("Manager: <name>") are checked per language; a language whose resources do not have
    # "role_labels" yet is skipped after the other checks have run.
    missing = []
    for lang in ("en", "ja"):
        res = json.loads((G.HERE / "resources" / f"{lang}.json").read_text())
        if "role_labels" not in res:
            missing.append(lang)
            continue
        found = {
            d["text"][s:e] for d in docs if d["lang"] == lang for s, e, k in d["spans"] if k == "EMPLOYMENT"
        }
        assert found & set(res["role_labels"]), lang
    if missing:
        pytest.skip(f"no role_labels in resources yet: {', '.join(missing)}")


def test_every_language_has_title_name_carriers():
    assert set(G.TITLE_NAME_T) == set(G.LANGS)
    assert all(G.TITLE_NAME_T[lang] for lang in G.LANGS)


def test_a_job_title_written_beside_a_name_is_an_employment_span(tmp_path):
    G.main(
        [
            "--out",
            str(tmp_path),
            "--seed",
            "7",
            "--train-docs",
            "300",
            "--valid-docs",
            "3",
            "--langs",
            "en,ja",
            "--templates",
            str(ROOT / "templates"),
        ]
    )
    docs = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    adjacent = 0
    for lang in ("en", "ja"):
        res = json.loads((G.HERE / "resources" / f"{lang}.json").read_text())
        titles = set(res.get("title_suffixes") or []) | set(res["job_titles"])
        for d in docs:
            if d["lang"] != lang:
                continue
            people = [(s, e) for s, e, k in d["spans"] if k == "PERSON"]
            for s, e, k in d["spans"]:
                if k != "EMPLOYMENT" or d["text"][s:e] not in titles:
                    continue
                for ps, pe in people:
                    before = ps >= e and d["text"][e:ps] in ("", " ")
                    after = pe <= s and d["text"][pe:s] in ("", " ")
                    adjacent += before or after
    assert adjacent > 0


def _lang_for_test(code):
    return G.Lang(json.loads((G.HERE / "resources" / f"{code}.json").read_text()), [], [])


def _docs_from(fn, lang, n=20):
    # The caller seeds G.rng, so the documents are the same on every run.
    out = []
    for _ in range(n):
        doc = G.Doc()
        fn(doc, lang, "train")
        out.append(doc)
    return out


@pytest.mark.parametrize("code", ["en", "ja"])
def test_minutes_name_employment_roles_for_attendees(monkeypatch, code):
    monkeypatch.setattr(G, "rng", random.Random(3))
    docs = _docs_from(G.minutes, _lang_for_test(code))
    assert any(kind == "EMPLOYMENT" for d in docs for _, _, kind in d.spans)


@pytest.mark.parametrize("code", ["en", "ja"])
def test_record_is_high_and_names_its_staff(monkeypatch, code):
    monkeypatch.setattr(G, "rng", random.Random(3))
    docs = _docs_from(G.record, _lang_for_test(code))
    assert all(d.high for d in docs)
    assert all(any(kind == "EMPLOYMENT" for _, _, kind in d.spans) for d in docs)


def test_every_public_role_template_takes_a_title_and_a_name():
    for lang, templates in G.SENS["PUBLIC_ROLE_T"].items():
        for t in templates:
            assert "{TITLE}" in t and "{P}" in t, (lang, t)


def test_coverage_shares_of_a_small_run_follow_the_table(tmp_path):
    cov = json.loads(G.COVERAGE_PATH.read_text())
    G.main(
        [
            "--out",
            str(tmp_path),
            "--seed",
            "7",
            "--train-docs",
            "200",
            "--valid-docs",
            "20",
            "--langs",
            "en,ja",
            "--templates",
            str(ROOT / "templates"),
        ]
    )
    train = json.loads((tmp_path / "stats.json").read_text())["coverage"]["pools"]["train"]
    for name, share in cov["sensitivity"].items():
        assert abs(train["sensitivity"][name]["actual"] - share) < 0.1, name
    for name, (_, _, share) in cov["length"].items():
        assert abs(train["length"][name]["actual"] - share) < 0.1, name


def test_long_documents_sometimes_have_a_section_heading(monkeypatch):
    monkeypatch.setattr(G, "rng", random.Random(3))
    lang = _lang_for_test("en")
    heads = set(G.SENS["NEUTRAL_HEADS"]["en"]) | set(G.SENS["FORM_HEADS_LOW"]["en"])
    heads.add(G.MINUTES_LABELS["en"]["agenda"])
    docs = _docs_from(G.long_doc, lang, 40)
    assert any(line in heads for d in docs for line in d.text.split("\n"))
