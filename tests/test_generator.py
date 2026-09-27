import json
import os
import subprocess
import sys
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
