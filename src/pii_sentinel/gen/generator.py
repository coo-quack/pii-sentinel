"""Synthetic multilingual training data with labels derived from the templates that built each document.

Every value planted in a document is recorded as a character span with its type, and the document gets a
sensitivity level and category set from the facts it contains. Nothing is labelled by a model.

    python -m pii_sentinel.gen.generator --out data [--train-docs 2000] [--valid-docs 150]
        [--langs ja,zh,ko,en,fr,it,de,es] [--seed 1] [--exclude eval/a.json,eval/b.json] [--templates templates]
        [--coverage coverage.json] [--pool-factor 2.0]
"""

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..labels import CATEGORIES, SENSITIVITY, SPAN_TYPES

HERE = Path(__file__).parent
LANGS = ["ja", "zh", "ko", "en", "fr", "it", "de", "es"]
TABLES = json.loads((HERE / "tables.json").read_text())
SENS = json.loads((HERE / "sensitive.json").read_text())
LOOKALIKE = json.loads((HERE / "resources" / "lookalikes.json").read_text())
NAMELESS = json.loads((HERE / "resources" / "nameless.json").read_text())
COVERAGE_PATH = HERE / "coverage.json"
COVERAGE = json.loads(COVERAGE_PATH.read_text())  # main() replaces it with --coverage

rng = random.Random(1)


def pick(xs):
    return xs[rng.randrange(len(xs))]


def chance(p):
    return rng.random() < p


def rint(lo, hi):
    return rng.randint(lo, hi)


def digits(n):
    return "".join(str(rng.randrange(10)) for _ in range(n))


def group(s, sizes, sep=" "):
    out, i = [], 0
    for n in sizes:
        out.append(s[i : i + n])
        i += n
    if i < len(s):
        out.append(s[i:])
    return sep.join(x for x in out if x)


# ---------------------------------------------------------------- passports and licences (hand-written)
PASSPORT_T = {
    "ja": [
        "{P}様のパスポート番号：{PASS}",
        "旅券番号 {PASS}（{P}）",
        "{P}さんの運転免許証番号は{PASS}です。",
    ],
    "zh": ["{P}的护照号码：{PASS}", "护照号 {PASS}（持有人：{P}）", "{P}的驾驶证编号是{PASS}。"],
    "ko": ["{P} 님의 여권번호: {PASS}", "여권 번호 {PASS} ({P})", "{P} 씨의 운전면허번호는 {PASS}입니다."],
    "en": [
        "Passport No. {PASS} ({P})",
        "{P}'s passport number is {PASS}.",
        "Driver's license number for {P}: {PASS}",
    ],
    "fr": [
        "Numéro de passeport de {P} : {PASS}",
        "Passeport n° {PASS} ({P})",
        "Permis de conduire de {P} : n° {PASS}",
    ],
    "it": [
        "Passaporto di {P}: n. {PASS}",
        "Numero di passaporto {PASS} ({P})",
        "Patente di guida di {P}: {PASS}",
    ],
    "de": ["Reisepassnummer von {P}: {PASS}", "Pass-Nr. {PASS} ({P})", "Führerscheinnummer von {P}: {PASS}"],
    "es": [
        "Pasaporte de {P}: {PASS}",
        "Número de pasaporte {PASS} ({P})",
        "Permiso de conducir de {P}: n.º {PASS}",
    ],
}
PASSPORT_LABEL = {
    "ja": ["旅券番号", "パスポート番号", "運転免許証番号"],
    "zh": ["护照号码", "驾驶证编号"],
    "ko": ["여권번호", "운전면허번호"],
    "en": ["Passport No.", "Driver's license"],
    "fr": ["N° de passeport", "Permis de conduire"],
    "it": ["N. passaporto", "Patente"],
    "de": ["Reisepass-Nr.", "Führerschein-Nr."],
    "es": ["N.º de pasaporte", "Permiso de conducir"],
}


def letters(n):
    return "".join(chr(65 + rng.randrange(26)) for _ in range(n))


def passport(lang):
    return {
        "ja": lambda: pick([letters(2) + digits(7), digits(12)]),
        "zh": lambda: pick(["E" + digits(8), "G" + digits(8)]),
        "ko": lambda: pick(["M" + digits(8), "M" + digits(3) + letters(1) + digits(4)]),
        "en": lambda: pick([digits(9), letters(1) + digits(8), "D" + digits(7)]),
        "fr": lambda: digits(2) + letters(2) + digits(5),
        "it": lambda: letters(2) + digits(7),
        "de": lambda: pick(
            ["C" + "".join(pick("CFGHJKLMNPRTVWXYZ0123456789") for _ in range(8)), digits(10)]
        ),
        "es": lambda: pick([letters(3) + digits(6), digits(8)]),
    }[lang]()


# ---------------------------------------------------------------- resources and names
def reserved_names(paths):
    names, texts = [], []
    for p in paths:
        for t in json.loads(Path(p).read_text()).get("tests", []):
            texts.append(t["text"])
            names += [f["value"] for f in t["expected"]["findings"] if f["type"] == "person_name"]
    return names, texts


@dataclass
class Person:
    surname: str
    given: str
    full: str
    gender: str
    foreign: bool


CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")


def seen_in(name, texts):
    """Whether a name occurs in evaluation text (whole words for Latin script, any position for CJK)."""
    if len(name) < 2:
        return False
    if CJK_CHAR.search(name):
        return any(name in t for t in texts)
    pattern = re.compile(rf"(?<![^\W\d_]){re.escape(name)}(?![^\W\d_])")
    return any(pattern.search(t) for t in texts)


def clashes(name, reserved):
    """Whether a generator name is, or is part of, an evaluation name, or contains one ("Kim" is part of
    "Kim Lee" but not of "Kimberly")."""
    return name == reserved or seen_in(name, [reserved]) or seen_in(reserved, [name])


class Lang:
    def __init__(self, res, reserved, excluded_log, texts=()):
        self.r = res
        self.code = res["lang"]

        def split(xs, label):
            uniq = list(dict.fromkeys(xs))
            kept = [n for n in uniq if not any(clashes(n, r) for r in reserved) and not seen_in(n, texts)]
            if len(kept) < len(uniq):
                excluded_log.append(f"{self.code}.{label}: " + " ".join(n for n in uniq if n not in kept))
            return {
                "train": [n for i, n in enumerate(kept) if i % 7 != 3],
                "valid": [n for i, n in enumerate(kept) if i % 7 == 3],
            }

        self.sur = split(res["surnames"], "surnames")
        self.gm = split(res["given_male"], "given_male")
        self.gf = split(res["given_female"], "given_female")
        self.foreign = split(res.get("foreign_full_names", []), "foreign")

    @property
    def cjk(self):
        return self.r["name_joiner"] == ""

    def person(self, pool):
        if self.foreign[pool] and chance(0.1):
            full = pick(self.foreign[pool])
            parts = [p for p in re.split(r"[・·\s]", full) if p]
            return Person(parts[-1], parts[0], full, pick("mf"), True)
        gender = pick("mf")
        surname = pick(self.sur[pool])
        given = pick(self.gm[pool] if gender == "m" else self.gf[pool])
        order = [surname, given] if self.r["name_order"] == "surname_first" else [given, surname]
        return Person(surname, given, self.r["name_joiner"].join(order), gender, False)

    def mention(self, p, form="auto"):
        if form == "auto":
            form = "full" if chance(0.7) else "surname" if self.cjk else pick(["surname", "given"])
        if p.foreign or form == "full":
            name = p.full
            if not self.cjk and not p.foreign:
                # Written forms seen in forms and records: initials, surname first, capitalised surname.
                name = rng.choices(
                    [
                        name,
                        f"{p.given[0]}. {p.surname}",
                        f"{p.surname}, {p.given}",
                        f"{p.surname} {p.given}",
                        f"{p.surname.upper()} {p.given}",
                    ],
                    weights=[86, 4, 4, 4, 2],
                )[0]
        else:
            name = p.surname if form == "surname" else p.given
        if self.code in ("fr", "it", "de", "es") and chance(0.05):
            name = strip_accents(name)
        return name

    def honorific(self, p):
        xs = self.r["honorific_prefix_male" if p.gender == "m" else "honorific_prefix_female"]
        return pick(xs) if xs else ""


# ---------------------------------------------------------------- values
def strip_accents(s):
    s = s.replace("ß", "ss")
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    )


FULL_WIDTH = str.maketrans("0123456789-+()@.", "０１２３４５６７８９－＋（）＠．")


def phone_variant(lang, number):
    # Writing variants seen in real text: full-width digits (CJK) and digits spaced one by one.
    if lang in ("ja", "zh", "ko") and chance(0.06):
        return number.translate(FULL_WIDTH)
    if chance(0.03):
        return " ".join(c for c in number if c.isdigit())
    return number


def email_variant(address):
    if chance(0.04):
        local, domain = address.split("@", 1)
        dot = pick([" [dot] ", " (dot) ", " dot "])
        return f"{local} {pick(['[at]', '(at)', 'at'])} " + domain.replace(".", dot)
    return address


def luhn(prefix, length):
    s = prefix + digits(length - len(prefix) - 1)
    total = 0
    for i, ch in enumerate(reversed(s)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return s + str((10 - total % 10) % 10)


def personal_phone(lang):
    if chance(0.35):
        return personal_phone_variant(lang)
    d4 = lambda: digits(4)  # noqa: E731
    return {
        "ja": lambda: (
            f"0{pick(['90', '80', '70'])}-{d4()}-{d4()}" if chance(0.85) else f"0{rint(3, 6)}-{d4()}-{d4()}"
        ),
        "zh": lambda: (lambda m: pick([m, group(m, [3, 4, 4], "-"), group(m, [3, 4, 4])]))(
            f"1{pick('3589')}{digits(9)}"
        ),
        "ko": lambda: f"010-{d4()}-{d4()}",
        "en": lambda: pick(
            [
                f"({rint(201, 989)}) {rint(200, 989)}-{d4()}",
                f"{rint(201, 989)}-{rint(200, 989)}-{d4()}",
                f"+1 {rint(201, 989)} {rint(200, 989)} {d4()}",
            ]
        ),
        "fr": lambda: (
            lambda m: group(m, [2, 2, 2, 2, 2]) if chance(0.8) else f"+33 {m[1]} {group(m[2:], [2, 2, 2, 2])}"
        )(f"0{pick('67')}{digits(8)}"),
        "it": lambda: (
            f"3{digits(2)} {digits(3)} {d4()}" if chance(0.8) else f"+39 3{digits(2)} {digits(3)} {d4()}"
        ),
        "de": lambda: (
            f"01{pick(['51', '52', '60', '62', '70', '71', '76'])} {digits(8)}"
            if chance(0.7)
            else f"+49 1{pick(['51', '60', '70'])} {digits(8)}"
        ),
        "es": lambda: (
            f"6{digits(2)} {digits(3)} {digits(3)}"
            if chance(0.8)
            else f"+34 6{digits(2)} {digits(3)} {digits(3)}"
        ),
    }[lang]()


def personal_phone_variant(lang):
    a, b = digits(4), digits(4)
    if lang == "ja":
        p = pick(["90", "80", "70"])
        return pick(
            [
                f"+81-{p}-{a}-{b}",
                f"+81 {p} {a} {b}",
                f"0{p}({a}){b}",
                f"(0{p}){a}-{b}",
                f"0{p}{a}{b}",
                f"050-{a}-{b}",
                f"050{a}{b}",
            ]
        )
    if lang == "zh":
        m = f"1{pick('3589')}{digits(9)}"
        return pick(
            [f"+86 {group(m, [3, 4, 4])}", f"+86-{group(m, [3, 4, 4], '-')}", f"+86{m}", f"(+86) {m}"]
        )
    if lang == "ko":
        return pick([f"+82-10-{a}-{b}", f"+82 10 {a} {b}", f"010 {a} {b}", f"010{a}{b}", f"010.{a}.{b}"])
    if lang == "en":
        return pick(
            [
                f"+1 ({rint(201, 989)}) {rint(200, 989)}-{a}",
                f"{rint(201, 989)}.{rint(200, 989)}.{a}",
                f"+44 7{digits(3)} {digits(6)}",
                f"07{digits(3)} {digits(6)}",
                f"+44 20 {a} {b}",
                f"020-{a}-{b}",
            ]
        )
    if lang == "fr":
        m = f"0{pick('67')}{digits(8)}"
        return pick(
            [
                group(m, [2, 2, 2, 2, 2], "."),
                group(m, [2, 2, 2, 2, 2], "-"),
                f"+33 (0){m[1]} {group(m[2:], [2, 2, 2, 2])}",
                f"+33{m[1:]}",
            ]
        )
    if lang == "it":
        return pick(
            [
                f"+39 3{digits(2)} {digits(7)}",
                f"3{digits(2)}-{digits(7)}",
                f"3{digits(9)}",
                f"+39 3{digits(2)}-{digits(3)}-{a}",
            ]
        )
    if lang == "de":
        p = f"01{pick(['51', '52', '60', '62', '70', '71', '76'])}"
        return pick(
            [f"+49 (0){p[1:]} {digits(8)}", f"{p}/{digits(8)}", f"{p}-{digits(8)}", f"+49{p[1:]}{digits(8)}"]
        )
    return pick(
        [
            f"+34 6{digits(2)} {digits(2)} {digits(2)} {digits(2)}",
            f"6{digits(8)}",
            f"+34 6{digits(2)}-{digits(3)}-{digits(3)}",
            f"(+34) 6{digits(8)}",
        ]
    )


def corporate_phone(lang):
    if chance(0.25):
        return {
            "ja": lambda: pick(
                [
                    f"(03){digits(4)}-{digits(4)}",
                    f"03({digits(4)}){digits(4)}",
                    f"+81-3-{digits(4)}-{digits(4)}",
                    f"0120{digits(6)}",
                ]
            ),
            "zh": lambda: pick(
                [f"+86 10 {digits(4)} {digits(4)}", f"(010) {digits(4)}-{digits(4)}", f"400{digits(7)}"]
            ),
            "ko": lambda: pick(
                [f"+82-2-{digits(4)}-{digits(4)}", f"(02) {digits(4)}-{digits(4)}", f"1588-{digits(4)}"]
            ),
            "en": lambda: pick(
                [
                    f"+1 800 {digits(3)} {digits(4)}",
                    f"+44 20 {digits(4)} {digits(4)}",
                    f"0800 {digits(3)} {digits(4)}",
                ]
            ),
            "fr": lambda: pick(
                [f"+33 1 {group(digits(8), [2, 2, 2, 2])}", f"01.{group(digits(8), [2, 2, 2, 2], '.')}"]
            ),
            "it": lambda: pick([f"+39 02 {digits(4)} {digits(4)}", f"02-{digits(8)}"]),
            "de": lambda: pick([f"+49 (0)30 {digits(7)}", f"030/{digits(7)}", f"+49 89 {digits(6)}"]),
            "es": lambda: pick(
                [f"+34 91 {digits(3)} {digits(2)} {digits(2)}", f"91-{digits(3)}-{digits(4)}"]
            ),
        }[lang]()
    return {
        "ja": lambda: pick(
            [
                f"0120-{digits(3)}-{digits(3)}",
                f"03-{digits(4)}-{digits(4)}",
                f"06-{digits(4)}-{digits(4)}",
                f"0570-{digits(3)}-{digits(3)}",
            ]
        ),
        "zh": lambda: pick(
            [f"400-{digits(3)}-{digits(4)}", f"010-{digits(4)}-{digits(4)}", f"021-{digits(4)}-{digits(4)}"]
        ),
        "ko": lambda: pick(
            [f"1588-{digits(4)}", f"02-{digits(4)}-{digits(4)}", f"031-{digits(3)}-{digits(4)}"]
        ),
        "en": lambda: pick(
            [
                f"1-800-{digits(3)}-{digits(4)}",
                f"({rint(201, 989)}) 555-{digits(4)}",
                f"1-888-{digits(3)}-{digits(4)}",
            ]
        ),
        "fr": lambda: pick(
            [
                f"01 {group(digits(8), [2, 2, 2, 2])}",
                f"0800 {group(digits(6), [2, 2, 2])}",
                f"04 {group(digits(8), [2, 2, 2, 2])}",
            ]
        ),
        "it": lambda: pick(
            [f"02 {digits(4)} {digits(4)}", f"800 {digits(3)} {digits(3)}", f"06 {digits(4)} {digits(4)}"]
        ),
        "de": lambda: pick([f"030 {digits(7)}", f"0800 {digits(3)} {digits(4)}", f"089 {digits(6)}"]),
        "es": lambda: pick(
            [
                f"91 {digits(3)} {digits(2)} {digits(2)}",
                f"900 {digits(3)} {digits(3)}",
                f"93 {digits(3)} {digits(4)}",
            ]
        ),
    }[lang]()


DNI = "TRWAGMYFPDXBNJZSQVHLCKE"


def gov_id(lang):
    value, kind = _gov_id(lang)
    return number_variant(value), kind


def number_variant(value):
    # Numbers are sometimes written digit by digit or with dots instead of the usual separators.
    if chance(0.03):
        return " ".join(c for c in value if not c.isspace())
    if chance(0.02):
        return re.sub(r"[\s-]", ".", value)
    return value


def nie():
    first, n = pick("XYZ"), rint(0, 9999999)
    return f"{first}{n:07d}{DNI[('XYZ'.index(first) * 10**7 + n) % 23]}"


def _gov_id(lang):
    # Other official formats of the same countries, so the model is not tied to one shape per language.
    if chance(0.25):
        other = {
            "ja": lambda: pick(
                [f"{letters(2)}{digits(8)}{letters(2)}", digits(12)]
            ),  # residence card, licence
            "zh": lambda: pick([f"E{digits(8)}", f"{digits(6)}{rint(1960, 2004)}{digits(8)}"]),
            "ko": lambda: f"{digits(3)}-{digits(2)}-{digits(5)}",  # licence-like
            "en": lambda: pick(
                [
                    f"{letters(2)} {digits(2)} {digits(2)} {digits(2)} {pick('ABCD')}",
                    f"{digits(3)} {digits(3)} {digits(3)}",
                ]
            ),  # UK NI number, Canadian SIN
            "fr": lambda: pick(
                [f"{digits(2)}.{digits(2)}.{digits(2)}-{digits(3)}.{digits(2)}", f"{digits(12)}"]
            ),
            "it": lambda: pick([f"{letters(2)}{digits(7)}{letters(1)}", f"CA{digits(5)}{letters(2)}"]),
            "de": lambda: f"{letters(1)}{digits(2)}{letters(1)}{digits(2)}{letters(1)}{digits(3)}",  # ID card
            "es": lambda: pick(
                [
                    nie(),
                    f"{rint(5, 25)}.{digits(3)}.{digits(3)}-{pick('0123456789K')}",
                ]
            ),  # NIE, RUT
        }[lang]
        return other(), "NATIONAL_ID"
    if lang == "ja":
        d = digits(12)
        return (d if chance(0.5) else group(d, [4, 4, 4], pick([" ", "-"]))), "MY_NUMBER"
    if lang == "zh":
        return (
            f"{rint(11, 65)}{digits(4)}{rint(1960, 2004)}{rint(1, 12):02d}{rint(1, 28):02d}{digits(3)}"
            f"{'X' if chance(0.1) else digits(1)}"
        ), "NATIONAL_ID"
    if lang == "ko":
        return f"{rint(60, 99)}{rint(1, 12):02d}{rint(1, 28):02d}-{pick('12')}{digits(6)}", "NATIONAL_ID"
    if lang == "en":
        return f"{rint(100, 665)}-{digits(2)}-{digits(4)}", "NATIONAL_ID"
    if lang == "fr":
        return (
            f"{pick('12')} {digits(2)} {rint(1, 12):02d} {digits(2)} {digits(3)} {digits(3)} {digits(2)}",
            "NATIONAL_ID",
        )
    if lang == "it":
        return (
            f"{letters(6)}{digits(2)}{letters(1)}{digits(2)}{letters(1)}{digits(3)}{letters(1)}",
            "NATIONAL_ID",
        )
    if lang == "de":
        d = f"{rint(1, 9)}{digits(10)}"
        return (d if chance(0.5) else group(d, [2, 3, 3, 3])), "NATIONAL_ID"
    n = rint(10000000, 99999999)
    return f"{n}{DNI[n % 23]}", "NATIONAL_ID"


def card(lang):
    prefix = "62" if lang == "zh" else "35" if lang == "ja" and chance(0.4) else pick("45")
    c = luhn(prefix, 16)
    if chance(0.03):
        return " ".join(c)
    return group(c, [4, 4, 4, 4], " " if chance(0.7) else "-") if chance(0.6) else c


IBAN_LEN = {"fr": 27, "it": 27, "de": 22, "es": 24}


def bank(lang):
    if lang in IBAN_LEN and chance(0.6):
        body = (letters(1) + digits(IBAN_LEN[lang] - 5)) if lang == "it" else digits(IBAN_LEN[lang] - 4)
        return group(f"{lang.upper()}{digits(2)}{body}", [4] * 7)
    if lang == "ja":
        return digits(7) if chance(0.5) else f"{digits(3)}-{digits(7)}"
    return digits(pick([10, 11, 12, 13, 14]))


def date_str(r, birth):
    y = rint(1950, 2005) if birth else rint(2024, 2027)
    m, d = rint(1, 12), rint(1, 28)
    fmt = pick(r["date_formats"])
    pad = any(s in fmt for s in (".{m}", "/{m}", "-{m}"))
    month = r["month_names"][m - 1] if r["month_names"] else str(m)
    return (
        fmt.replace("{y}", str(y))
        .replace("{m}", f"{m:02d}" if pad else str(m))
        .replace("{d}", str(d))
        .replace("{month}", month)
    )


def ip():
    return pick(
        [
            f"192.168.{rint(0, 255)}.{rint(2, 254)}",
            f"10.{rint(0, 255)}.{rint(0, 255)}.{rint(2, 254)}",
            f"203.0.113.{rint(2, 254)}",
            f"198.51.100.{rint(2, 254)}",
        ]
    )


def ascii_name(s):
    s = unicodedata.normalize("NFD", s).replace("ß", "ss")
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


def person_email(L, who):
    # A person's own address: usually a personal mailbox, sometimes their work address at a company domain.
    domains = L.r["email_domains_company"] if chance(0.25) else L.r["email_domains_personal"]
    return f"{email_local(L, who)}@{pick(domains)}"


def email_local(L, who):
    g, f = ascii_name(who.given), ascii_name(who.surname)
    if L.cjk or who.foreign or not g or not f or chance(0.25):
        return pick(L.r["email_locals"])
    return pick([f"{g}.{f}", f"{g[0]}.{f}", f"{f}.{g}", f"{g}{f}{rint(1, 99)}", f"{g[0]}{f}"])


def handle(r):
    # A social-media handle is usually written with "@", sometimes bare.
    local = re.sub(r"[.-]", "_", pick(r["email_locals"]))
    return "@" + local if chance(0.7) else local


def nick():
    return f"{pick(TABLES['NICK_A'])}{pick(TABLES['NICK_B'])}{rint(1, 2026) if chance(0.6) else ''}"


# Technical lines with numbers and addresses but no person: machine IDs, server hosts, service accounts.
# {IP} is a server address (IP_SERVER) and {ID} a plain machine identifier; the rest come from TECH_VALUES.
TECH_NEG = [
    "request_id={ID} session={ID} status=200 latency_ms={NUM}",
    "device:\n  mac: {ID}\n  build: {ID}\n  channel: stable",
    "Invoice payable to {CO}, account {CO_ACCOUNT}, {CO_REG}",
    "function validateEmail(email) {\n  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;\n  return pattern.test(email);\n}",
    "{TS} | IP: {IP} | User: {SVC} | Action: {ACTION} | Status: success",
    "server:\n  host: {IP}\n  port: {PORT}\n  admin_email: {ROLE}@example.com\n  timeout: {TIMEOUT}",
    "SELECT id, name, email FROM users WHERE created_at > '{YEAR}-01-01' LIMIT {LIMIT};",
    "const phoneRegex = /^0\\d{1,4}-\\d{1,4}-\\d{4}$/; // matches numbers like 0X-XXXX-XXXX",
    # Service names and credential-like values: snake_case, but not handles.
    "export DB_USER=app_service_{NUM}\nexport DB_PASS={PW}\nexport API_HOST={IP}",
    "config:\n  owner: platform_team\n  bucket: backups_{YEAR}\n  key_id: {ID}",
    "[INFO] worker_pool_{NUM} started by scheduler_bot at {TS}",
]
TECH_VALUES = {
    "PW": lambda: password(),
    "ID": lambda: machine_id(),
    "NUM": lambda: str(rint(3, 900)),
    "TS": lambda: (
        f"{rint(2024, 2026)}-0{rint(1, 9)}-1{rint(0, 9)} 1{rint(0, 9)}:{rint(10, 59)}:{rint(10, 59)}"
    ),
    "CO": lambda: pick(["Nordwind GmbH", "Atelier Sud SARL", "Tecnova S.L.", "Blue Harbor Ltd"]),
    "CO_ACCOUNT": lambda: company_account(pick(["de", "fr", "es", "en"])),
    "CO_REG": lambda: company_registration(),
    "SVC": lambda: pick(["svc_backup", "cron", "deploy-bot", "healthcheck", "system"]),
    "ACTION": lambda: pick(["connect", "sync", "rotate", "ping"]),
    "PORT": lambda: str(rint(1024, 9999)),
    "ROLE": lambda: pick(["ops", "admin", "noreply", "alerts"]),
    "TIMEOUT": lambda: str(rint(5, 60)),
    "YEAR": lambda: str(rint(2024, 2026)),
    "LIMIT": lambda: str(rint(10, 500)),
}


def tech_neg(doc):
    last, line = 0, pick(TECH_NEG)
    for m in re.finditer(r"\{([A-Z_]+)\}", line):
        doc.add(line[last : m.start()])
        last = m.end()
        if m.group(1) == "IP":
            doc.span(ip(), "IP_SERVER")
        else:
            doc.add(TECH_VALUES[m.group(1)]())
    doc.add(line[last:])


# ---------------------------------------------------------------- documents
@dataclass
class Doc:
    text: str = ""
    spans: list = field(default_factory=list)
    cats: set = field(default_factory=set)
    high: bool = False
    personal: bool = False

    def add(self, s):
        self.text += s

    def span(self, value, kind):
        start = len(self.text)
        self.text += value
        self.spans.append([start, len(self.text), kind])

    def person(self, name, public=False):
        self.span(name, "PERSON")
        self.cats.add("person_name")
        if not public:
            self.personal = True


FACT_CATS = {
    "person_mention": [],
    "member_id": [],
    "email_personal": ["email_or_phone"],
    "phone_personal": ["email_or_phone"],
    "address_personal": ["postal_address"],
    "dob": ["date_of_birth"],
    "gov_id": ["government_id"],
    "passport": ["government_id"],
    "credit_card": ["financial_account"],
    "bank_account": ["financial_account"],
    "health": ["health_info"],
    "biometric": ["biometric_or_genetic"],
    "ip_personal": ["ip_address_of_a_person"],
    "sns_personal": ["sns_handle"],
    "employment": ["employment_info"],
    "role_label": ["employment_info"],
    "title_name": ["employment_info"],
    "religion": ["race_or_religion"],
    "hr_record": ["hr_or_criminal_record"],
    **{k: [v["cat"]] for k, v in SENS["FACTS"].items()},
}
HIGH_FACTS = {
    "gov_id",
    "passport",
    "credit_card",
    "bank_account",
    "health",
    "biometric",
    "religion",
    "hr_record",
    *SENS["FACTS"],
}
PERSON_FACTS = list(FACT_CATS)
PERSON_WEIGHTS = {
    "person_mention": 8,
    "email_personal": 5,
    "phone_personal": 5,
    "employment": 5,
    "role_label": 3,
    "title_name": 5,
    "address_personal": 3,
    "dob": 3,
    "ip_personal": 2,
    "sns_personal": 2,
    "member_id": 2,
    # Attribute facts added in this branch had diluted the high-sensitivity share; these weights restore it.
    "gov_id": 2,
    "passport": 1.5,
    "credit_card": 1.5,
    "bank_account": 1.5,
    "health": 2,
    "biometric": 1,
    "religion": 1,
    "hr_record": 1,
    **{k: 0.5 for k in SENS["FACTS"]},
}
NEG_FACTS = [
    "email_generic",
    "phone_corporate",
    "address_company",
    "date_nonbirth",
    "health_general",
    "biometric_general",
    "ip_server",
    "sns_brand",
    "job_posting",
    "religion_general",
    "hr_policy",
    "order_number",
    "tracking_number",
    "serial_number",
    "place_distractor",
    "filler",
    "brand_mention",
    "public_mention",
    "general_sensitive",
]
NEG_WEIGHTS = {
    "filler": 3,
    "place_distractor": 2,
    "brand_mention": 2,
    "general_sensitive": 2,
    "address_company": 3,
}


def weighted(keys, weights):
    return rng.choices(keys, weights=[weights.get(k, 1) for k in keys])[0]


def person_weights(L):
    # A language without "role_labels" in its resources gets no role_label documents yet.
    if "role_labels" in L.r:
        return PERSON_WEIGHTS
    return {**PERSON_WEIGHTS, "role_label": 0}


# A name written with a job title beside it. A shape is the name-title block, filled by fill(): {P} is the
# name, {PS} the surname and {TITLE} the title. A carrier is the sentence or line around the block, where
# {NAME} stands for the block and {DAY} is filled by fill().
# A place in parentheses after the titled name (Mayor Weber (Berlin)) is plain text, not a role.
TITLE_SHAPE = {
    "ja": ["{P}{TITLE}", "{PS}{TITLE}", "{TITLE}の{P}", "{P}{TITLE}（{PLACE}）"],
    "zh": ["{P}{TITLE}", "{PS}{TITLE}", "{TITLE}{P}", "{P}{TITLE}（{PLACE}）"],
    "ko": ["{P} {TITLE}", "{PS} {TITLE}", "{TITLE} {P}", "{P} {TITLE} ({PLACE})"],
    "en": ["{TITLE} {P}", "{P}, {TITLE}", "{P} ({TITLE})", "{TITLE} {P} ({PLACE})"],
    "fr": ["{TITLE} {P}", "{P}, {TITLE}", "{P} ({TITLE})", "{TITLE} {P} ({PLACE})"],
    "it": ["{TITLE} {P}", "{P}, {TITLE}", "{P} ({TITLE})", "{TITLE} {P} ({PLACE})"],
    "de": ["{TITLE} {P}", "{P}, {TITLE}", "{P} ({TITLE})", "{TITLE} {P} ({PLACE})"],
    "es": ["{TITLE} {P}", "{P}, {TITLE}", "{P} ({TITLE})", "{TITLE} {P} ({PLACE})"],
}
TITLE_NAME_T = {
    "ja": ["{NAME}が会議を開会しました。", "出席：{NAME}", "{NAME}が{DAY}までに回答します。"],
    "zh": ["{NAME}主持了会议。", "出席：{NAME}", "{NAME}将在{DAY}前回复。"],
    "ko": ["{NAME} 주재로 회의가 열렸습니다.", "참석: {NAME}", "{NAME}에게 {DAY}까지 답변 부탁드립니다."],
    "en": ["{NAME} opened the meeting.", "Present: {NAME}", "{NAME}, will reply by {DAY}."],
    "fr": ["{NAME} a ouvert la réunion.", "Présents : {NAME}", "Réponse attendue : {DAY}."],
    "it": ["{NAME} ha aperto la riunione.", "Presenti: {NAME}", "Scadenza: {DAY}."],
    "de": ["{NAME} eröffnete die Sitzung.", "Anwesend: {NAME}", "Frist: {DAY}."],
    "es": ["{NAME} abrió la reunión.", "Asistentes: {NAME}", "Fecha límite: {DAY}."],
}


def job_title_list(L):
    # Title suffixes ("Mayor", "部長") are preferred; a language without them uses the job titles.
    return L.r.get("title_suffixes") or L.r["job_titles"]


def template_for(L, key):
    lang = L.code
    if key == "role_label":
        return "{ROLE}" + colon(L) + "{P}"
    if key == "title_name":
        return pick(TITLE_NAME_T[lang]).replace("{NAME}", pick(TITLE_SHAPE[lang]))
    if key == "brand_mention":
        return pick(TABLES["BRAND_T"][lang])
    if key == "public_mention":
        return pick(TABLES["PUBLIC_T"][lang] + TABLES["PUBLIC_BIO"][lang])
    if key == "passport":
        return pick(PASSPORT_T[lang])
    if key in SENS["FACTS"]:
        return pick(SENS["FACTS"][key]["t"][lang])
    if key == "general_sensitive":
        return pick(SENS["GENERAL"][lang])
    return pick(L.r["templates"][key])


def surname_org(L, pool):
    return pick(TABLES["SURNAME_ORG"][L.code]).replace("{s}", pick(L.sur[pool]))


def doc_code():
    y, m, d = rint(2023, 2027), rint(1, 12), rint(1, 28)
    pfx = pick(
        ["INV", "FAC", "RG", "FT", "ORD", "CMD", "REF", "TKT", "DOC", "BL", "PO", "AZ", "EXP", "PED", "No"]
    )
    return pick(
        [
            f"{pfx}-{y}{m:02d}{d:02d}-{digits(pick([3, 4]))}",
            f"{pfx}-{y}-{digits(4)}",
            f"#{pfx}-{y}{m:02d}{d:02d}-{digits(3)}",
            f"#{digits(pick([5, 6, 8]))}",
            f"{y}-{pfx}-{digits(5)}",
            f"{pfx}{digits(pick([6, 8, 10]))}",
        ]
    )


def gps():
    lat, lon = rng.uniform(-45, 60), rng.uniform(-120, 145)
    return pick(
        [
            f"{lat:.4f}, {lon:.4f}",
            f"{lat:.6f},{lon:.6f}",
            f"N{abs(lat):.4f} E{abs(lon):.4f}",
            f"lat {lat:.5f} / lon {lon:.5f}",
        ]
    )


def personal_identifier():
    # Employee, member, patient or case numbers: they single out a person, so they are labelled MEMBER_ID.
    return pick(["E-", "M", "PT-", "No.", "ID ", "#", "C-", "EMP", ""]) + (
        digits(pick([5, 6, 7])) if chance(0.7) else letters(2) + digits(pick([4, 5]))
    )


def company_account(lang):
    # A company's own bank account: a real account number format, but no person's.
    iban = {"fr": "FR", "it": "IT", "de": "DE", "es": "ES", "en": "GB"}.get(lang)
    if iban and chance(0.7):
        return group(
            f"{iban}{digits(2)}{digits({'fr': 23, 'it': 23, 'de': 18, 'es': 20, 'en': 18}[lang])}", [4] * 7
        )
    return digits(pick([7, 10, 12]))


def company_registration():
    return pick(
        [
            f"HRB {digits(5)}",
            f"RCS {digits(3)} {digits(3)} {digits(3)}",
            f"CIF B{digits(8)}",
            f"{digits(4)}-{digits(2)}-{digits(6)}",
            f"Company No. {digits(8)}",
        ]
    )


def machine_id():
    hexs = "0123456789abcdef"
    return pick(
        [
            "".join(pick(hexs) for _ in range(32)),
            "-".join("".join(pick(hexs) for _ in range(n)) for n in (8, 4, 4, 4, 12)),
            ":".join("".join(pick(hexs.upper()) for _ in range(2)) for _ in range(6)),
            "".join(pick(hexs) for _ in range(40)),
        ]
    )


def password():
    word = pick(["sakura", "Tiger", "blue", "Luna", "sunny", "Mango", "rocket", "Kimchi", "Paris", "moon"])
    return word + str(rint(1, 2027)) + pick(["!", "#", "$", "", "@", "?"]) + pick(["", letters(2).lower()])


def amount(lang):
    return (
        pick(SENS["CURRENCY"][lang])
        .replace("{n}", str(rint(1, 980)))
        .replace("{m}", f"{rint(0, 999):03d}")
        .replace("{c}", f"{rint(0, 99):02d}")
    )


def fill(doc, L, pool, key, who, tpl=None, public=False):
    r = L.r
    tpl = tpl or template_for(L, key)
    for c in FACT_CATS.get(key, []):
        doc.cats.add(c)
    if key in HIGH_FACTS:
        doc.high = True
    if key == "member_id":
        # A member number alone identifies a person.
        doc.personal = True
    if key == "public_role":
        # A living public figure's office is employment information; its {TITLE} span is the office.
        doc.cats.add("employment_info")
    has_hon = "{HON}" in tpl
    pm = (
        L.mention(who, "surname")
        if has_hon and chance(0.5)
        else L.mention(who, "auto" if key == "person_mention" else "full")
    )
    born = rint(1500, 1925)
    years = {"B": born, "D": born + rint(38, 88)}
    last = 0
    for m in re.finditer(r"\{([A-Z0-9]+)\}", tpl):
        doc.add(tpl[last : m.start()])
        last = m.end()
        ph = m.group(1)
        if ph == "P":
            doc.person(pm, public=public)
        elif ph in ("P2", "P3"):
            doc.person(L.mention(L.person(pool)))
        elif ph == "PUB":
            doc.person(pick(TABLES["PUBLIC"][L.code]), public=True)
        elif ph == "HON":
            doc.add(L.honorific(who))
        elif ph == "EMAIL":
            doc.span(email_variant(person_email(L, who)), "EMAIL")
        elif ph == "GEMAIL":
            doc.span(f"{pick(r['generic_locals'])}@{pick(r['email_domains_company'])}", "EMAIL_GENERIC")
        elif ph == "PHONE":
            doc.span(phone_variant(L.code, personal_phone(L.code)), "PHONE")
        elif ph == "CPHONE":
            doc.span(corporate_phone(L.code), "PHONE_CORPORATE")
        elif ph == "GOVID":
            doc.span(*gov_id(L.code))
        elif ph == "PASS":
            doc.span(passport(L.code), "PASSPORT_LICENCE")
        elif ph == "CARD":
            doc.span(card(L.code), "CREDIT_CARD")
        elif ph == "BANK":
            doc.span(bank(L.code), "BANK_ACCOUNT")
        elif ph == "ORDER":
            doc.span(digits(pick([10, 11, 12])), "ORDER_TRACKING")
        elif ph == "TRACK":
            doc.span(pick([digits(12), group(digits(12), [4, 4, 4], "-")]), "ORDER_TRACKING")
        elif ph == "SERIAL":
            doc.span(pick([digits(10), f"{digits(4)}-{digits(4)}-{digits(4)}"]), "SERIAL")
        elif ph == "CODE":
            doc.span(doc_code(), "ORDER_TRACKING")
        elif ph in ("B", "D"):
            doc.add(str(years[ph]))
        elif ph == "NICK":
            doc.span(nick(), "HANDLE")
            doc.cats.add("sns_handle")
            doc.personal = True
        elif ph == "MID":
            doc.span(personal_identifier(), "MEMBER_ID")
        elif ph == "PS":
            doc.person(L.mention(who, "surname"))
        elif ph == "TITLE":
            doc.span(pick(job_title_list(L)), "EMPLOYMENT")
        elif ph == "ROLE":
            doc.span(pick(r["role_labels"]), "EMPLOYMENT")
        elif ph == "ADDR":
            doc.span(pick(r["addresses"]), "ADDRESS_CORPORATE" if key in NEG_FACTS else "ADDRESS")
        elif ph == "DATE":
            if key == "dob":
                doc.span(date_str(r, True), "DATE_OF_BIRTH")
            else:
                doc.add(date_str(r, False))
        elif ph == "IP":
            doc.span(ip(), "IP_PERSONAL" if key == "ip_personal" else "IP_SERVER")
        elif ph == "HANDLE":
            doc.span(handle(r), "HANDLE_BRAND" if key == "sns_brand" else "HANDLE")
        else:
            value = {
                "DIAG": lambda: pick(r["diagnoses"]),
                "ORG": lambda: surname_org(L, pool) if chance(0.25) else pick(r["orgs"]),
                "DEPT": lambda: pick(r["depts"]),
                "JOB": lambda: pick(r["job_titles"]),
                "REL": lambda: pick(r["religions"]),
                "PRODUCT": lambda: pick(r["products"]),
                "PLACE": lambda: pick(r["places"]),
                "AMB": lambda: pick(r["ambiguous_surname_places"]),
                "WORD": lambda: pick(r["common_words"]),
                "DAY": lambda: pick(r["days"]),
                "BRAND": lambda: pick(TABLES["BRANDS"]),
                "GPS": gps,
                "PW": password,
                "AMT": lambda: amount(L.code),
            }[ph]()
            if key == "employment" and ph in ("ORG", "DEPT", "JOB"):
                doc.span(value, "EMPLOYMENT")
            else:
                doc.add(value)
    doc.add(tpl[last:])


def sep(L):
    return "" if L.cjk and L.code != "ko" else " "


def titled_name(doc, L, who, form="full"):
    # The title goes after the name in CJK (no space in ja and zh) and before it in the other languages.
    title = pick(job_title_list(L))
    if L.cjk:
        doc.person(L.mention(who, form))
        doc.add(sep(L))
        doc.span(title, "EMPLOYMENT")
    else:
        doc.span(title, "EMPLOYMENT")
        doc.add(" ")
        doc.person(L.mention(who, form))
    doc.cats.add("employment_info")


def join_next(doc, L):
    if re.search(r"[。．.!?！？」）)]\s*$", doc.text) and not chance(0.2):
        doc.add(sep(L))
    else:
        doc.add("\n")


def lab(L, k):
    return pick(PASSPORT_LABEL[L.code]) if k == "passport" else pick(L.r["labels"][k])


def colon(L):
    return pick(["：", ":"]) if L.code == "zh" else pick(["：", ": "]) if L.code == "ja" else ": "


def narrative(doc, L, pool):
    who = L.person(pool)
    facts = set()
    n = rint(1, 3)
    while len(facts) < n:
        facts.add(weighted(PERSON_FACTS, person_weights(L)))
    facts = sorted(facts) + [weighted(NEG_FACTS, NEG_WEIGHTS) for _ in range(rint(0, 2))]
    rng.shuffle(facts)
    for i, k in enumerate(facts):
        if i:
            join_next(doc, L)
        fill(doc, L, pool, k, who)


def negative(doc, L, pool):
    for i in range(rint(2, 4)):
        if i:
            join_next(doc, L)
        fill(doc, L, pool, weighted(NEG_FACTS, NEG_WEIGHTS), L.person(pool))


def email_doc(doc, L, pool):
    who = L.person(pool)
    fill(doc, L, pool, "greeting", who)
    doc.add("\n\n")
    body = (
        [weighted(NEG_FACTS, NEG_WEIGHTS)]
        if chance(0.5)
        else [
            weighted([k for k in PERSON_FACTS if k != "employment"], person_weights(L)),
            weighted(NEG_FACTS, NEG_WEIGHTS),
        ]
    )
    for i, k in enumerate(body):
        if i:
            join_next(doc, L)
        fill(doc, L, pool, k, who)
    doc.add("\n\n")
    fill(doc, L, pool, "closing", who)
    doc.add("\n")
    doc.person(L.mention(who, "full"))
    if chance(0.7):
        doc.add("\n")
        doc.span(
            pick(L.r["orgs"]) + " " + (pick(L.r["depts"]) if chance(0.5) else pick(L.r["job_titles"])),
            "EMPLOYMENT",
        )
        doc.cats.add("employment_info")
    if chance(0.6):
        doc.add(f"\n{lab(L, pick(['phone', 'mobile']))}{colon(L)}")
        doc.span(personal_phone(L.code), "PHONE")
        doc.cats.add("email_or_phone")
    if chance(0.5):
        doc.add(f"\n{lab(L, 'email')}{colon(L)}")
        doc.span(person_email(L, who), "EMAIL")
        doc.cats.add("email_or_phone")
    if chance(0.3):
        # A company footer: its address and switchboard are corporate, so they add no category.
        doc.add("\n" + pick(L.r["orgs"]) + " · ")
        doc.span(pick(L.r["addresses"]), "ADDRESS_CORPORATE")
        doc.add(" · ")
        doc.span(corporate_phone(L.code), "PHONE_CORPORATE")


def _field(doc, L, k, who):
    if k == "phone":
        doc.span(phone_variant(L.code, personal_phone(L.code)), "PHONE")
    elif k == "email":
        doc.span(email_variant(person_email(L, who)), "EMAIL")
    elif k == "gov_id":
        doc.span(*gov_id(L.code))
    elif k == "passport":
        doc.span(passport(L.code), "PASSPORT_LICENCE")
    elif k == "card":
        doc.span(card(L.code), "CREDIT_CARD")
    elif k == "bank":
        doc.span(bank(L.code), "BANK_ACCOUNT")
    elif k == "address":
        doc.span(pick(L.r["addresses"]), "ADDRESS")
    elif k == "dob":
        doc.span(date_str(L.r, True), "DATE_OF_BIRTH")
    elif k == "dept":
        doc.span(pick(L.r["depts"]), "EMPLOYMENT")
    elif k == "job":
        doc.span(pick(L.r["job_titles"]), "EMPLOYMENT")
    elif k == "ip":
        doc.span(ip(), "IP_PERSONAL")
    elif k == "sns":
        doc.span(handle(L.r), "HANDLE")
    else:
        doc.add(pick(L.r["diagnoses"]))


FIELD_CAT = {
    "phone": "email_or_phone",
    "email": "email_or_phone",
    "address": "postal_address",
    "dob": "date_of_birth",
    "dept": "employment_info",
    "job": "employment_info",
    "gov_id": "government_id",
    "passport": "government_id",
    "card": "financial_account",
    "bank": "financial_account",
    "diagnosis": "health_info",
    "ip": "ip_address_of_a_person",
    "sns": "sns_handle",
}
HIGH_FIELDS = {"gov_id", "passport", "card", "bank", "diagnosis"}


def form(doc, L, pool):
    who = L.person(pool)
    nl = (
        "\n" if chance(0.7) else pick([" / ", ", ", " "])
    )  # one-line forms occur in OCR and flattened exports
    heading = None
    if chance(0.4):
        if chance(0.5):
            heading = pick(sorted(SENS["FORM_HEADS_HIGH"]))
            doc.add(pick(SENS["FORM_HEADS_HIGH"][heading]["titles"][L.code]) + "\n")
            doc.cats.add(SENS["FORM_HEADS_HIGH"][heading]["cat"])
            doc.high = True
        else:
            doc.add(pick(SENS["FORM_HEADS_LOW"][L.code]) + "\n")
    doc.add(f"{lab(L, 'name')}{colon(L)}")
    if chance(0.2):
        titled_name(doc, L, who)
    else:
        doc.person(L.mention(who, "full"))
    if chance(0.3):
        # A staff line under the name, such as "Attending physician: <name>" on a patient form.
        doc.add(nl)
        doc.span(pick(L.r["role_labels"]), "EMPLOYMENT")
        doc.add(colon(L))
        doc.person(L.mention(L.person(pool), "full"))
        doc.cats.add("employment_info")
    if chance(0.2):
        staff = f"{pick(['E-', 'M', '', 'No.'])}{digits(pick([5, 6, 7]))}"
        doc.add(f"{nl}{pick(SENS['STAFF_ID'][L.code])}{colon(L)}")
        doc.span(staff, "MEMBER_ID")
    if chance(0.12):
        labels, countries = SENS["NATIONALITY"][L.code]
        doc.add(f"{nl}{pick(labels)}{colon(L)}{pick(countries)}")
        doc.cats.add("citizenship_or_immigration")
        doc.high = True
    basic = [k for k in FIELD_CAT if k not in HIGH_FIELDS]
    fields = (
        (rng.sample(basic, rint(0, 3)) + rng.sample(sorted(HIGH_FIELDS), 1))
        if chance(0.35)
        else rng.sample(basic, rint(1, 4))
    )
    for k in fields:
        doc.add(f"{nl}{lab(L, k)}{colon(L)}")
        _field(doc, L, k, who)
        doc.cats.add(FIELD_CAT[k])
        if k in HIGH_FIELDS:
            doc.high = True


def csv(doc, L, pool):
    patients = chance(0.3)
    if chance(0.7):
        doc.add(pick(L.r["labels"]["patient_list_titles" if patients else "attendee_list_titles"]) + "\n")
    cols = list(
        dict.fromkeys(
            rng.sample(["dept", "job", "email", "phone"] + (["diagnosis"] if patients else []), rint(0, 2))
        )
    )
    if chance(0.25):
        # Tables of card, account or ID numbers for several people.
        cols.append(pick(["card", "bank", "gov_id", "passport"]))
    s = pick([",", "\t", " | "])
    doc.add(s.join([lab(L, "name")] + [lab(L, c) for c in cols]))
    for _ in range(rint(2, 4)):
        doc.add("\n")
        who = L.person(pool)
        # Without a job column, a name cell sometimes carries the title.
        if "job" not in cols and chance(0.2):
            titled_name(doc, L, who)
        else:
            doc.person(L.mention(who, "full"))
        for c in cols:
            doc.add(s)
            _field(doc, L, c, who)
    for c in cols:
        doc.cats.add(FIELD_CAT[c])
    if set(cols) & {"card", "bank", "gov_id", "passport"}:
        doc.high = True
    if patients or "diagnosis" in cols:
        doc.cats.add("health_info")
        doc.high = True


def chat(doc, L, pool):
    people = [L.person(pool), L.person(pool)]
    for i in range(rint(2, 4)):
        if i:
            doc.add("\n")
        who = people[i % 2]
        form = "surname" if L.cjk else pick(["given", "full"])
        # Sometimes the speaker label carries the job title as well.
        if chance(0.3):
            titled_name(doc, L, who, form)
        else:
            doc.person(L.mention(who, form))
        doc.add(colon(L))
        k = (
            weighted(NEG_FACTS, NEG_WEIGHTS)
            if chance(0.75)
            else pick(["person_mention", "phone_personal", "email_personal"])
        )
        fill(doc, L, pool, k, who)


def boundary(doc, L, pool):
    lang, who = L.code, L.person(pool)

    def filler():
        if chance(0.5):
            join_next(doc, L)
            fill(doc, L, pool, "filler", who)

    kind = pick(
        [
            "public",
            "public",
            "deceased",
            "deceased",
            "public_role",
            "public_role",
            "brand",
            "brand",
            "tech",
            "bare",
            "bare",
            "bare",
            "review",
            "invoice",
        ]
    )
    if kind == "public_role":
        # A living person in a public role (CEO, mayor, athlete) is still personal information.
        fill(doc, L, pool, "public_role", who, pick(SENS["PUBLIC_ROLE_T"][lang]))
        filler()
    elif kind == "deceased":
        # Historical people: names are detected, but the dead are not data subjects (GDPR recital 27).
        fill(doc, L, pool, "deceased", L.person(pool), pick(SENS["DECEASED_T"][lang]), public=True)
        filler()
    elif kind == "public":
        fill(doc, L, pool, "public", who, pick(TABLES["PUBLIC_T"][lang] + TABLES["PUBLIC_BIO"][lang]))
        filler()
    elif kind == "brand":
        fill(doc, L, pool, "brand", who, pick(TABLES["BRAND_T"][lang]))
        filler()
    elif kind == "tech":
        tech_neg(doc)
    elif kind == "bare":
        fill(doc, L, pool, "bare", who, pick(TABLES["BARE_T"][lang]))
        doc.cats.add("email_or_phone")
        doc.personal = True
    elif kind == "review":
        fill(doc, L, pool, "review", who, pick(TABLES["REVIEW_T"][lang]))
    else:
        title, billed, total = TABLES["INVOICE"][lang]
        doc.add(f"{title}\n{billed}{colon(L)}")
        doc.person(L.mention(who, "full"))
        doc.add(f"\n{lab(L, 'email')}{colon(L)}")
        doc.span(person_email(L, who), "EMAIL")
        doc.add(f"\n{lab(L, 'phone')}{colon(L)}")
        doc.span(personal_phone(lang), "PHONE")
        doc.add(f"\n{total}{rint(12, 980)},{rint(0, 999):03d}")
        doc.cats.add("email_or_phone")


def heading_list(doc, L, pool):
    # Only the heading reveals the sensitive fact; the rows are plain names (and sometimes contacts).
    neutral = chance(0.35)
    kind = pick(sorted(SENS["HEAD_LISTS"]))
    spec = SENS["HEAD_LISTS"][kind]
    title = pick(SENS["NEUTRAL_HEADS"][L.code]) if neutral else pick(spec["titles"][L.code])
    doc.add(title + pick([":\n", "\n", "：\n", " "]))
    extra = pick([None, None, "phone", "dept", "job"])
    inline = chance(0.3)
    # Decided once per list: either every row carries its job title or none does.
    titled = chance(0.3)
    for i in range(rint(2, 5)):
        if i:
            doc.add(pick([", ", "、" if L.cjk else "; "]) if inline else "\n")
        who = L.person(pool)
        if not inline and chance(0.4):
            doc.add(f"{i + 1}. ")
        if titled:
            titled_name(doc, L, who)
        else:
            doc.person(L.mention(who, "full"))
        if extra == "phone":
            doc.add(" ")
            doc.span(personal_phone(L.code), "PHONE")
            doc.cats.add("email_or_phone")
        elif extra in ("dept", "job"):
            doc.add(" (")
            doc.span(pick(L.r["depts" if extra == "dept" else "job_titles"]), "EMPLOYMENT")
            doc.add(")")
            doc.cats.add("employment_info")
    if not neutral:
        doc.cats.add(spec["cat"])
        doc.high = True


# Fixed labels of the meeting minutes, per language.
MINUTES_LABELS = {
    "ja": {"date": "日時", "place": "場所", "attendees": "出席者", "chair": "司会", "agenda": "議題"},
    "zh": {"date": "日期", "place": "地点", "attendees": "出席人员", "chair": "主持人", "agenda": "议题"},
    "ko": {"date": "일시", "place": "장소", "attendees": "참석자", "chair": "사회", "agenda": "안건"},
    "en": {"date": "Date", "place": "Place", "attendees": "Attendees", "chair": "Chair", "agenda": "Agenda"},
    "fr": {
        "date": "Date",
        "place": "Lieu",
        "attendees": "Participants",
        "chair": "Président de séance",
        "agenda": "Ordre du jour",
    },
    "it": {
        "date": "Data",
        "place": "Luogo",
        "attendees": "Presenti",
        "chair": "Moderatore",
        "agenda": "Ordine del giorno",
    },
    "de": {
        "date": "Datum",
        "place": "Ort",
        "attendees": "Teilnehmer",
        "chair": "Leitung",
        "agenda": "Tagesordnung",
    },
    "es": {
        "date": "Fecha",
        "place": "Lugar",
        "attendees": "Asistentes",
        "chair": "Moderador",
        "agenda": "Orden del día",
    },
}


def minutes(doc, L, pool):
    # Meeting minutes: a title, the date and place, the attendees and chair, then agenda lines. An agenda line
    # can carry a personal fact about an attendee, so the minutes take the fact's level.
    ml = MINUTES_LABELS[L.code]
    if chance(0.3):
        kind = pick(sorted(SENS["HEAD_LISTS"]))
        spec = SENS["HEAD_LISTS"][kind]
        doc.add(pick(spec["titles"][L.code]))
        doc.cats.add(spec["cat"])
        doc.high = True
    else:
        doc.add(pick(SENS["NEUTRAL_HEADS"][L.code]))
    doc.add(f"\n{ml['date']}{colon(L)}{date_str(L.r, False)}")
    doc.add(f"\n{ml['place']}{colon(L)}{pick(L.r['places'])}")
    attendees = [L.person(pool) for _ in range(rint(2, 4))]
    doc.add(f"\n{ml['attendees']}{colon(L)}")
    for i, who in enumerate(attendees):
        if i:
            doc.add(pick([", ", "、" if L.cjk else "; "]))
        shape = pick(["plain", "titled", "dept", "org_job"])
        if shape == "titled":
            titled_name(doc, L, who)
            continue
        doc.person(L.mention(who, "full"))
        if shape == "dept":
            doc.add(" (")
            doc.span(pick(L.r["depts"]), "EMPLOYMENT")
            doc.add(")")
            doc.cats.add("employment_info")
        elif shape == "org_job":
            doc.add(" (")
            doc.span(pick(L.r["orgs"]), "EMPLOYMENT")
            doc.add(", ")
            doc.span(pick(L.r["job_titles"]), "EMPLOYMENT")
            doc.add(")")
            doc.cats.add("employment_info")
    if chance(0.5):
        doc.add(f"\n{ml['chair']}{colon(L)}")
        chair = L.person(pool)
        if chance(0.5):
            titled_name(doc, L, chair)
        else:
            doc.person(L.mention(chair, "full"))
    doc.add(f"\n{ml['agenda']}{colon(L).rstrip()}")
    for _ in range(rint(1, 3)):
        doc.add("\n")
        fill(doc, L, pool, weighted(PERSON_FACTS, person_weights(L)), pick(attendees))


def record(doc, L, pool):
    # A record about one person under a heading that names a sensitive kind: the subject's fields, the staff
    # who handled the case, and the facts of that kind about the subject.
    kind = pick(sorted(SENS["FORM_HEADS_HIGH"]))
    spec = SENS["FORM_HEADS_HIGH"][kind]
    doc.add(pick(spec["titles"][L.code]) + "\n")
    doc.cats.add(spec["cat"])
    doc.high = True
    who = L.person(pool)
    doc.add(f"{lab(L, 'name')}{colon(L)}")
    doc.person(L.mention(who, "full"))
    basic = [k for k in FIELD_CAT if k not in HIGH_FIELDS]
    for k in rng.sample(basic, rint(1, 3)):
        doc.add(f"\n{lab(L, k)}{colon(L)}")
        _field(doc, L, k, who)
        doc.cats.add(FIELD_CAT[k])
    staff = ["role", "titled"]
    rng.shuffle(staff)
    for s in staff[: rint(1, 2)]:
        doc.add("\n")
        if s == "role":
            doc.span(pick(L.r["role_labels"]), "EMPLOYMENT")
            doc.add(colon(L))
            doc.person(L.mention(L.person(pool), "full"))
            doc.cats.add("employment_info")
        else:
            titled_name(doc, L, L.person(pool))
    if kind == "health":
        fact = "health"
    elif kind == "hr":
        fact = "hr_record"
    else:
        same_cat = [k for k, v in SENS["FACTS"].items() if v["cat"] == spec["cat"]]
        fact = pick(same_cat) if same_cat else "person_mention"
    for _ in range(rint(1, 2)):
        doc.add("\n")
        fill(doc, L, pool, fact, who)


def transaction(doc, L, pool):
    # Everyday business paperwork about a person: personal, but nothing sensitive.
    who = L.person(pool)
    tpl = pick(SENS["TRANSACTION"][L.code])
    fill(doc, L, pool, "transaction", who, tpl)
    doc.personal = True
    if "{EMAIL}" in tpl:
        doc.cats.add("email_or_phone")
    if "{ADDR}" in tpl:
        doc.cats.add("postal_address")
    if chance(0.3):
        join_next(doc, L)
        fill(doc, L, pool, weighted(NEG_FACTS, NEG_WEIGHTS), who)


def self_disclosure(doc, L, pool):
    # People writing about their own health; the signature identifies them.
    who = L.person(pool)
    if chance(0.5):
        fill(doc, L, pool, "greeting", L.person(pool))
        doc.add("\n\n")
    fill(doc, L, pool, "self", who, pick(SENS["SELF_T"][L.code]))
    doc.cats.add("health_info")
    doc.high = True


def topic_article(doc, L, pool):
    # Sensitive topics discussed without any identifiable person.
    for i in range(rint(2, 4)):
        if i:
            join_next(doc, L)
        key = pick(
            ["general_sensitive", "general_sensitive", "health_general", "religion_general", "hr_policy"]
        )
        fill(doc, L, pool, key, L.person(pool))


def long_doc(doc, L, pool):
    # Several blocks in one document, so the deciding fact can sit anywhere in a long text. The text grows until
    # it reaches a target length from the table's long or very_long bucket. Half the documents are sectioned: a
    # heading line sits between blocks.
    lo, hi = draw_long_bucket(L.code)
    target = rint(lo, hi - 1)
    sectioned = chance(0.5)
    heads = (
        SENS["NEUTRAL_HEADS"][L.code] + SENS["FORM_HEADS_LOW"][L.code] + [MINUTES_LABELS[L.code]["agenda"]]
    )
    while len(doc.text) < target:
        if doc.text:
            doc.add("\n\n")
            if sectioned:
                doc.add(pick(heads) + "\n")
        pick(BLOCKS)(doc, L, pool)


def sensitive_list(doc, L, pool):
    if chance(0.5):
        return heading_list(doc, L, pool)
    kind = pick(["complaint", "disciplinary", "religion", "arrest"])
    titles, facts = TABLES["SENS_LISTS"][L.code][kind]
    doc.add(pick(titles) + pick([":\n", "\n", "：\n"]))
    style = rint(0, 2)
    for i in range(rint(2, 4)):
        if i:
            doc.add("\n")
        fact, name = pick(facts), L.mention(L.person(pool), "full")
        if style == 0:
            doc.add("- ")
            doc.person(name)
            doc.add(f"{colon(L)}{fact}")
        elif style == 1:
            doc.person(name)
            doc.add(f" ({fact})")
        else:
            doc.add(f"{i + 1}. ")
            doc.person(name)
            doc.add(f", {fact}")
    doc.cats.add(TABLES["SENS_GATE"][kind])
    doc.high = True


SLOT_WEIGHT = 0
SLOT_T = {}  # lang -> {"train": [...], "valid": [...]} of hand-written slot templates
SLOT_RE = re.compile(r"\{([A-Za-z0-9_.]+)\}")
SENS_LEVEL = {"none": 0, "low": 1, "high": 2}


def slot_doc(doc, L, pool):
    from .slots import CELL_CATEGORY, SLOT_CATEGORY

    t = pick(SLOT_T[L.code][pool])
    r, people = L.r, {}
    western = not L.cjk
    born = rint(1500, 1925)
    years = {"DEAD.born": born, "DEAD.died": born + rint(38, 88)}
    others = {}
    orgs = {}

    def who(key):
        if key not in people:
            people[key] = L.person(pool)
        return people[key]

    last = 0
    for m in SLOT_RE.finditer(t["text"]):
        doc.add(t["text"][last : m.start()])
        last = m.end()
        ph = m.group(1)
        if ph.startswith("P") and ph[1:2].isdigit():
            n, _, attr = ph.partition(".")
            p = who(n)
            if attr == "":
                doc.person(L.mention(p, "full"))
            elif attr in ("s", "g"):
                doc.person(p.full if p.foreign else (p.surname if attr == "s" else p.given))
            elif attr == "i":
                doc.person(f"{p.given[0]}. {p.surname}" if western and not p.foreign else p.surname)
            elif attr == "rev":
                doc.person(f"{p.surname}, {p.given}" if western and not p.foreign else p.full)
            elif attr == "email":
                doc.span(email_variant(person_email(L, p)), "EMAIL")
            elif attr == "phone":
                doc.span(phone_variant(L.code, personal_phone(L.code)), "PHONE")
            elif attr == "govid":
                doc.span(*gov_id(L.code))
            elif attr == "passport":
                doc.span(passport(L.code), "PASSPORT_LICENCE")
            elif attr == "card":
                doc.span(card(L.code), "CREDIT_CARD")
            elif attr == "bank":
                doc.span(bank(L.code), "BANK_ACCOUNT")
            elif attr == "org":
                # One employer per person, so every mention of that person in the document matches.
                if n not in orgs:
                    orgs[n] = surname_org(L, pool) if chance(0.25) else pick(r["orgs"])
                doc.span(orgs[n], "EMPLOYMENT")
            elif attr == "dept":
                doc.span(pick(r["depts"]), "EMPLOYMENT")
            elif attr == "job":
                doc.span(pick(r["job_titles"]), "EMPLOYMENT")
            elif attr == "addr":
                doc.span(pick(r["addresses"]), "ADDRESS")
            elif attr == "dob":
                doc.span(date_str(r, True), "DATE_OF_BIRTH")
            elif attr == "handle":
                doc.span(handle(r), "HANDLE")
            elif attr == "ip":
                doc.span(ip(), "IP_PERSONAL")
            elif attr == "id":
                doc.span(personal_identifier(), "MEMBER_ID")
            doc.cats.add(SLOT_CATEGORY.get(attr, "person_name"))
        elif ph in ("PUB", "PUB.s"):
            p = others.setdefault("PUB", L.person(pool))
            doc.person(p.full if ph == "PUB" or p.foreign else p.surname)
        elif ph == "DEAD":
            doc.person(others.setdefault("DEAD", L.person(pool)).full, public=True)
        elif ph in years:
            doc.add(str(years[ph]))
        elif ph == "ORG.email":
            doc.span(f"{pick(r['generic_locals'])}@{pick(r['email_domains_company'])}", "EMAIL_GENERIC")
        elif ph == "ORG.phone":
            doc.span(corporate_phone(L.code), "PHONE_CORPORATE")
        elif ph == "ORG.addr":
            doc.span(pick(r["addresses"]), "ADDRESS_CORPORATE")
        elif ph == "CODE":
            doc.span(doc_code(), "ORDER_TRACKING")
        else:
            doc.add(
                {
                    "ORG": lambda: others.setdefault(
                        "ORG", surname_org(L, pool) if chance(0.25) else pick(r["orgs"])
                    ),
                    "DATE": lambda: date_str(r, False),
                    "PLACE": lambda: pick(r["places"]),
                    "AMOUNT": lambda: amount(L.code),
                    "GPS": gps,
                    "PW": password,
                    "PRODUCT": lambda: pick(r["products"]),
                    "ORG.bank": lambda: company_account(L.code),
                    "ORG.regno": company_registration,
                    "HASH": machine_id,
                }[ph]()
            )
            if ph == "PW":
                doc.cats.add("credentials")
    doc.add(t["text"][last:])
    if t["cell"] in CELL_CATEGORY:
        doc.cats.add(CELL_CATEGORY[t["cell"]])
    level = SENS_LEVEL[t["sensitivity"]]
    doc.personal = level >= 1
    doc.high = level == 2
    # A person picked out by an identifier alone ({P1.id}) is still an identified person.
    if not any(k == "PERSON" for *_, k in doc.spans) and not re.search(r"\{P\d\.id\}", t["text"]):
        doc.cats.discard("person_name")


def lookalike_doc(doc, L, pool):
    # Words the model took for names: letter closings, contract party labels ("甲", "Party A") and common
    # words that begin with a surname character ("유방암", "高血压"). Real names around them stay labelled.
    lk = LOOKALIKE[L.code]
    kind = pick(["contract", "closing", "closing", "words"])
    if kind == "contract":
        tpl, last = pick(lk["contract"]), 0
        for m in re.finditer(r"\{(ORG|P2?)\}", tpl):
            doc.add(tpl[last : m.start()])
            last = m.end()
            if m.group(1) == "ORG":
                doc.add(surname_org(L, pool) if chance(0.25) else pick(L.r["orgs"]))
            else:
                doc.person(L.mention(L.person(pool), "full"))
        doc.add(tpl[last:])
    elif kind == "closing":
        (negative if chance(0.5) else narrative)(doc, L, pool)
        doc.add("\n\n" + pick(lk["closings"]))
        if chance(0.5):
            doc.add("\n")
            doc.person(L.mention(L.person(pool), "full"))
    else:
        for i in range(rint(1, 3)):
            if i:
                join_next(doc, L)
            doc.add(pick(lk["words"]))
        if chance(0.5):
            join_next(doc, L)
            fill(doc, L, pool, "filler", L.person(pool))


def nameless_doc(doc, L, pool):
    # Personal information with no name: a user's device, cookie or IP address, or a recipient's address.
    # Server and service-account lines around it stay non-personal; the document is low.
    lines = rng.sample(NAMELESS[L.code], rint(1, 2))
    for i, line in enumerate(lines):
        if i or chance(0.5):
            if doc.text:
                doc.add("\n")
            if chance(0.4):
                tech_neg(doc)
            else:
                fill(doc, L, pool, "filler", L.person(pool))
            doc.add("\n")
        last = 0
        for m in re.finditer(r"\{([A-Z_]+)\}", line):
            doc.add(line[last : m.start()])
            last = m.end()
            ph = m.group(1)
            if ph == "IP":
                doc.span(ip(), "IP_PERSONAL")
            elif ph == "ADDR":
                doc.span(pick(L.r["addresses"]), "ADDRESS")
            elif ph == "ID":
                doc.add(machine_id())
        doc.add(line[last:])
        doc.cats.add("postal_address" if "{ADDR}" in line else "ip_address_of_a_person")
    doc.personal = True


def load_slot_templates(root, skip=frozenset()):
    from .slots import problems

    kept = Counter()
    paths = sorted(Path(root).glob("*.json"))
    if not paths:
        raise SystemExit(f"--templates: no *.json files in {root}")
    for path in paths:
        try:
            templates = json.loads(path.read_text())["templates"]
        except (json.JSONDecodeError, KeyError):
            print(f"!! skipping unreadable {path}")
            continue
        for i, t in enumerate(templates):
            if t["cell"] in skip:
                kept["skipped"] += 1
                continue
            if problems(t):
                kept["rejected"] += 1
                continue
            pool = "valid" if i % 7 == 3 else "train"
            SLOT_T.setdefault(t["lang"], {"train": [], "valid": []})[pool].append(t)
            kept[pool] += 1
    return kept


FORMATS = [
    (34, narrative),
    (20, negative),
    (14, email_doc),
    (16, form),
    (10, csv),
    (6, chat),
    (14, boundary),
    (10, sensitive_list),
    (10, transaction),
    (3, self_disclosure),
    (5, topic_article),
    (6, long_doc),
    (6, lookalike_doc),
    (5, nameless_doc),
    (10, minutes),
    (8, record),
]
BLOCKS = [narrative, negative, negative, email_doc, transaction, form, chat, topic_article, minutes, record]


def make_doc(L, pool):
    formats = FORMATS + ([(SLOT_WEIGHT, slot_doc)] if SLOT_T.get(L.code, {}).get(pool) else [])
    fn = rng.choices([f for _, f in formats], weights=[w for w, _ in formats])[0]
    doc = Doc()
    fn(doc, L, pool)
    if re.search(r"\{[A-Za-z_]+\}", doc.text):
        raise ValueError(f"leftover placeholder in {L.code}: {doc.text}")
    for s, e, _ in doc.spans:
        if not (e > s and doc.text[s:e].strip() == doc.text[s:e]):
            raise ValueError(f"span {s}-{e} is empty or has surrounding spaces in {L.code}: {doc.text!r}")
    return {
        "lang": L.code,
        "format": fn.__name__,
        "text": doc.text,
        "spans": doc.spans,
        "sensitivity": 2 if doc.high else 1 if doc.personal else 0,
        "categories": sorted(doc.cats),
    }


def packed_doc(L, pool, cap=None):
    """Several documents joined into one long one, so the model also learns from inputs longer than one
    window. The target length comes from the table's long or very_long bucket. Parts are joined until the text
    reaches the target, and a part that would take it more than 30% past the bucket's upper edge is left out. The
    labels are the union of the parts: the most sensitive part sets the level. With cap, every part is at most
    that sensitivity level, so the packed document can be a low or none one."""
    lo, hi = draw_long_bucket(L.code)
    target = rint(lo, hi - 1)
    text, spans, level, cats = "", [], 0, set()
    while len(text) < target:
        d = make_doc(L, pool)
        while cap is not None and d["sensitivity"] > cap:
            d = make_doc(L, pool)
        if text and len(text) + 2 + len(d["text"]) > hi * 1.3:
            break
        if text:
            text += "\n\n"
        spans += [[s + len(text), e + len(text), k] for s, e, k in d["spans"]]
        text += d["text"]
        level = max(level, d["sensitivity"])
        cats |= set(d["categories"])
    return {
        "lang": L.code,
        "format": "packed",
        "text": text,
        "spans": spans,
        "sensitivity": level,
        "categories": sorted(cats),
    }


# ---------------------------------------------------------------- coverage
# CJK text carries more meaning per character, so its length buckets use half the edges.
HALVED_LENGTH_LANGS = {"ja", "zh", "ko"}
LONG_BUCKETS = ("long", "very_long")


def check_coverage(cov):
    """Reject a coverage table whose shares do not sum to 1 or that leaves out a span kind or a category."""
    if not 0 <= cov["pool_packed_share"] <= 1:
        raise ValueError("coverage pool_packed_share must be between 0 and 1")
    if set(cov["sensitivity"]) != set(SENSITIVITY):
        raise ValueError(f"coverage sensitivity keys must be {SENSITIVITY}")
    shares = {
        "sensitivity": list(cov["sensitivity"].values()),
        "length": [bucket[2] for bucket in cov["length"].values()],
        "formats": list(cov["formats"].values()),
    }
    for dim, xs in shares.items():
        if abs(sum(xs) - 1) > 1e-6:
            raise ValueError(f"coverage {dim} shares sum to {sum(xs):.4f}, not 1")
    for key, names in (("spans_per_1000", SPAN_TYPES), ("categories_per_1000", CATEGORIES)):
        missing = [name for name in names if name not in cov[key]]
        if missing:
            raise ValueError(f"coverage {key} has no minimum for {', '.join(missing)}")


def length_buckets(lang, cov):
    scale = 0.5 if lang in HALVED_LENGTH_LANGS else 1
    return [(name, int(lo * scale), int(hi * scale)) for name, (lo, hi, _) in cov["length"].items()]


def length_bucket(lang, chars, cov):
    buckets = length_buckets(lang, cov)
    for name, lo, hi in buckets:
        if lo <= chars < hi:
            return name
    return buckets[-1][0]  # longer than the last edge counts as the last bucket


def draw_long_bucket(lang):
    """Edges (low, high) of a long or very_long bucket for lang, chosen with the table's shares as weights."""
    buckets = {name: (lo, hi) for name, lo, hi in length_buckets(lang, COVERAGE) if name in LONG_BUCKETS}
    name = rng.choices(list(buckets), weights=[COVERAGE["length"][k][2] for k in buckets])[0]
    return buckets[name]


def apportion(n, shares):
    """Whole counts that sum to n: each share rounded down, then the largest remainders take the rest."""
    raw = {k: n * s for k, s in shares.items()}
    counts = {k: int(v) for k, v in raw.items()}
    rest = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: max(rest, 0)]:
        counts[k] += 1
    return counts


def draw(xs):
    """Remove and return a random element, so that a candidate is chosen at most once."""
    return xs.pop(rng.randrange(len(xs)))


def cell_name(key):
    return f"{SENSITIVITY[key[0]]}/{key[1]}"


def candidate_pool(L, pool, size, cov):
    """Candidates for one language. The table's pool_packed_share is made of packed documents, split across the
    sensitivity levels in the table's proportions, because long documents are otherwise almost all high. The rest
    come from make_doc(). The "packed" format target is a reporting target only."""
    packed = round(cov["pool_packed_share"] * size)
    levels = apportion(packed, {level: cov["sensitivity"][name] for level, name in enumerate(SENSITIVITY)})
    docs = [packed_doc(L, pool, cap=level) for level, count in levels.items() for _ in range(count)]
    return docs + [make_doc(L, pool) for _ in range(size - packed)]


def select(docs, n, cov):
    """Choose n of one language's candidates so that each sensitivity x length cell gets its target count.

    Every cell first takes what its own candidates supply. A cell that falls short then takes the rest from the
    spare candidates of the cells with the same sensitivity and the nearest length, then from any cell. Lengths
    scale multiplicatively, so the distance between two buckets is the log ratio of their centres. Returns the
    chosen documents and the cells that needed the fallback.
    """
    centres = {name: math.log(math.sqrt(max(lo, 1) * hi)) for name, (lo, hi, _) in cov["length"].items()}
    cells = {(s, name): [] for s in range(len(SENSITIVITY)) for name in centres}
    for d in docs:
        cells[(d["sensitivity"], length_bucket(d["lang"], len(d["text"]), cov))].append(d)
    shares = {(s, name): cov["sensitivity"][SENSITIVITY[s]] * cov["length"][name][2] for s, name in cells}
    targets = apportion(n, shares)
    chosen, short = [], []
    for key in cells:
        own = min(targets[key], len(cells[key]))
        chosen += [draw(cells[key]) for _ in range(own)]
        if own < targets[key]:
            short.append((key, own))
    shortfalls = []
    for key, own in short:
        need, borrowed = targets[key] - own, {}
        neighbours = sorted(
            (k for k in cells if k != key),
            key=lambda k: (k[0] != key[0], abs(centres[k[1]] - centres[key[1]])),
        )
        for k in neighbours:
            take = min(need, len(cells[k]))
            chosen += [draw(cells[k]) for _ in range(take)]
            if take:
                borrowed[cell_name(k)] = take
            need -= take
            if need == 0:
                break
        shortfalls.append(
            {
                "cell": cell_name(key),
                "target": targets[key],
                "own": own,
                "borrowed": borrowed,
                "unfilled": need,
            }
        )
    return chosen, shortfalls


def measure(docs, cov):
    """Actual shares and rates of one pool, next to the table's targets and minimums."""
    total = len(docs) or 1
    sens = Counter(SENSITIVITY[d["sensitivity"]] for d in docs)
    lengths = Counter(length_bucket(d["lang"], len(d["text"]), cov) for d in docs)
    formats = Counter(d["format"] for d in docs)
    spans = Counter(s[2] for d in docs for s in d["spans"])
    cats = Counter(c for d in docs for c in d["categories"])
    format_names = list(cov["formats"]) + [f for f in formats if f not in cov["formats"]]

    def shares(counts, targets):
        return {k: {"target": t, "actual": round(counts[k] / total, 4)} for k, t in targets.items()}

    def rates(counts, minimums):
        return {k: {"min": m, "actual": round(counts[k] * 1000 / total, 1)} for k, m in minimums.items()}

    return {
        "docs": len(docs),
        "sensitivity": shares(sens, {k: cov["sensitivity"][k] for k in SENSITIVITY}),
        "length": shares(lengths, {k: v[2] for k, v in cov["length"].items()}),
        "formats": shares(formats, {f: cov["formats"].get(f, 0) for f in format_names}),
        "spans_per_1000": rates(spans, {k: cov["spans_per_1000"][k] for k in SPAN_TYPES}),
        "categories_per_1000": rates(cats, {k: cov["categories_per_1000"][k] for k in CATEGORIES}),
    }


def coverage_warnings(pool, m):
    out = []
    for dim in ("sensitivity", "length", "formats"):
        for k, v in m[dim].items():
            t, a = v["target"], v["actual"]
            if t == 0 and a == 0:
                continue
            if t == 0 or abs(a - t) > 0.2 * t:
                out.append(f"{pool} {dim} {k}: actual {a:.4f}, target {t:.4f}")
    for dim, label in (("spans_per_1000", "span"), ("categories_per_1000", "category")):
        for k, v in m[dim].items():
            if v["actual"] < v["min"]:
                out.append(
                    f"{pool} {label} {k}: {v['actual']} per 1000 documents, below the minimum {v['min']}"
                )
    return out


def shortfall_warning(s):
    text = f"{s['pool']} {s['lang']} {s['cell']}: {s['own']} of {s['target']} from its own cell"
    if s["borrowed"]:
        text += ", the rest from " + ", ".join(f"{k} ({v})" for k, v in s["borrowed"].items())
    if s["unfilled"]:
        text += f", {s['unfilled']} could not be filled"
    return text


def coverage_table(measured):
    lines = ["coverage, actual/target for shares and actual/minimum for rates per 1000 documents:"]
    for pool, m in measured.items():
        lines.append(f"{pool} ({m['docs']} documents)")
        for dim in ("sensitivity", "length", "formats"):
            cells = " ".join(
                f"{k} {v['actual']:.3f}/{v['target']:.2f}"
                for k, v in m[dim].items()
                if v["target"] or v["actual"]
            )
            lines.append(f"  {dim:<20}{cells}")
        for dim, label in (("spans_per_1000", "spans"), ("categories_per_1000", "categories")):
            cells = " ".join(f"{k} {v['actual']:g}/{v['min']:g}" for k, v in m[dim].items())
            lines.append(f"  {label:<20}{cells}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train-docs", type=int, default=1500)
    ap.add_argument("--valid-docs", type=int, default=150)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--templates", type=Path, help="directory of slot template files")
    ap.add_argument("--template-weight", type=float, default=100)
    ap.add_argument(
        "--packed-docs",
        type=int,
        default=0,
        help="long documents made of joined parts, per language, added after the coverage selection",
    )
    ap.add_argument("--skip-cells", default="", help="slot template cells to leave out, comma-separated")
    ap.add_argument(
        "--coverage", type=Path, default=COVERAGE_PATH, help="target distribution (coverage.json)"
    )
    ap.add_argument(
        "--pool-factor", type=float, default=2.0, help="candidates generated per kept document, per language"
    )
    a = ap.parse_args(argv)
    rng.seed(a.seed)
    # main() filters these tables and fills SLOT_T, so start each run from the files.
    TABLES.clear()
    TABLES.update(json.loads((HERE / "tables.json").read_text()))
    SLOT_T.clear()
    global SLOT_WEIGHT
    SLOT_WEIGHT = a.template_weight
    if a.templates:
        print(
            "slot templates:",
            dict(load_slot_templates(a.templates, set(filter(None, a.skip_cells.split(","))))),
        )
    names, texts = reserved_names([p for p in a.exclude.split(",") if p])
    excluded = []
    # Brands and public figures seen in evaluation text are left out of training.
    for key in ("BRANDS",):
        kept = [b for b in TABLES[key] if not any(b in t for t in texts)]
        excluded += [f"brands: {b}" for b in TABLES[key] if b not in kept]
        TABLES[key] = kept
    for lang, figs in TABLES["PUBLIC"].items():
        kept = [p for p in figs if not any(p in t for t in texts)]
        excluded += [f"public.{lang}: {p}" for p in figs if p not in kept]
        TABLES["PUBLIC"][lang] = kept
    langs = [
        # Names are also left out when they merely occur in evaluation text, so a missing label cannot leak them.
        Lang(json.loads((HERE / "resources" / f"{c}.json").read_text()), names, excluded, texts)
        for c in a.langs.split(",")
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    global COVERAGE
    COVERAGE = json.loads(a.coverage.read_text())
    cov = COVERAGE
    check_coverage(cov)
    report = {"excluded": excluded, "pools": {}}
    measured, shortfalls = {}, []
    for pool, n in (("train", a.train_docs), ("valid", a.valid_docs)):
        docs = []
        for L in langs:
            chosen, short = select(candidate_pool(L, pool, round(a.pool_factor * n), cov), n, cov)
            docs += chosen
            shortfalls += [{"pool": pool, "lang": L.code, **s} for s in short]
        packed = a.packed_docs if pool == "train" else a.packed_docs * a.valid_docs // max(1, a.train_docs)
        docs += [packed_doc(L, pool) for L in langs for _ in range(packed)]
        rng.shuffle(docs)
        with (a.out / f"{pool}.jsonl").open("w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        report["pools"][pool] = {
            "docs": len(docs),
            "sensitivity": Counter(d["sensitivity"] for d in docs),
            "formats": Counter(d["format"] for d in docs),
            "spans": Counter(s[2] for d in docs for s in d["spans"]),
            "categories": Counter(c for d in docs for c in d["categories"]),
            "max_chars": max(len(d["text"]) for d in docs),
        }
        measured[pool] = measure(docs, cov)
    unknown = {c for p in report["pools"].values() for c in p["categories"]} - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    warnings = [w for pool, m in measured.items() for w in coverage_warnings(pool, m)]
    warnings += [shortfall_warning(s) for s in shortfalls]
    report["coverage"] = {
        "source": str(a.coverage),
        "pool_factor": a.pool_factor,
        "pools": measured,
        "shortfalls": shortfalls,
        "warnings": warnings,
    }
    (a.out / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report["pools"], ensure_ascii=False, indent=1))
    print("excluded:", len(excluded))
    print(coverage_table(measured))
    print("coverage warnings:" if warnings else "coverage warnings: none")
    for w in warnings:
        print("  " + w)


if __name__ == "__main__":
    main()
