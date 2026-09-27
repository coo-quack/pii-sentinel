"""Synthetic multilingual training data with labels derived from the templates that built each document.

Every value planted in a document is recorded as a character span with its type, and the document gets a
sensitivity level and category set from the facts it contains. Nothing is labelled by a model.

    python -m pii_sentinel.gen.generator --out data [--train-docs 2000] [--valid-docs 150]
        [--langs ja,zh,ko,en,fr,it,de,es] [--seed 1] [--exclude eval/a.json,eval/b.json] [--templates templates]
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..labels import CATEGORIES

HERE = Path(__file__).parent
LANGS = ["ja", "zh", "ko", "en", "fr", "it", "de", "es"]
TABLES = json.loads((HERE / "tables.json").read_text())
SENS = json.loads((HERE / "sensitive.json").read_text())

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


def seen_in(name, texts):
    """Whether a name occurs in evaluation text (whole words for Latin script, any position for CJK)."""
    if len(name) < 2:
        return False
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]", name):
        return any(name in t for t in texts)
    pattern = re.compile(rf"(?<![^\W\d_]){re.escape(name)}(?![^\W\d_])")
    return any(pattern.search(t) for t in texts)


class Lang:
    def __init__(self, res, reserved, excluded_log, texts=()):
        self.r = res
        self.code = res["lang"]

        def split(xs, label):
            uniq = list(dict.fromkeys(xs))
            kept = [
                n
                for n in uniq
                if not any(r == n or (len(n) >= 2 and n in r) or r in n for r in reserved)
                and not seen_in(n, texts)
            ]
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
            if not self.cjk and not p.foreign and chance(0.04):
                name = f"{p.given[0]}. {p.surname}"
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


def email_local(L, who):
    g, f = ascii_name(who.given), ascii_name(who.surname)
    if L.cjk or who.foreign or not g or not f or chance(0.25):
        return pick(L.r["email_locals"])
    return pick([f"{g}.{f}", f"{g[0]}.{f}", f"{f}.{g}", f"{g}{f}{rint(1, 99)}", f"{g[0]}{f}"])


def nick():
    return f"{pick(TABLES['NICK_A'])}{pick(TABLES['NICK_B'])}{rint(1, 2026) if chance(0.6) else ''}"


TECH_NEG = [
    lambda: (
        "function validateEmail(email) {\n  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;\n  return pattern.test(email);\n}"
    ),
    lambda: (
        f"{rint(2024, 2026)}-0{rint(1, 9)}-1{rint(0, 9)} 1{rint(0, 9)}:{rint(10, 59)}:{rint(10, 59)} | IP: {ip()} | User: "
        f"{pick(['svc_backup', 'cron', 'deploy-bot', 'healthcheck', 'system'])} | Action: {pick(['connect', 'sync', 'rotate', 'ping'])} | Status: success"
    ),
    lambda: (
        f"server:\n  host: {ip()}\n  port: {rint(1024, 9999)}\n  admin_email: {pick(['ops', 'admin', 'noreply', 'alerts'])}@example.com\n  timeout: {rint(5, 60)}"
    ),
    lambda: (
        f"SELECT id, name, email FROM users WHERE created_at > '{rint(2024, 2026)}-01-01' LIMIT {rint(10, 500)};"
    ),
    lambda: "const phoneRegex = /^0\\d{1,4}-\\d{1,4}-\\d{4}$/; // matches numbers like 0X-XXXX-XXXX",
]


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
    "address_personal": 3,
    "dob": 3,
    "ip_personal": 2,
    "sns_personal": 2,
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
NEG_WEIGHTS = {"filler": 3, "place_distractor": 2, "brand_mention": 2, "general_sensitive": 2}


def weighted(keys, weights):
    return rng.choices(keys, weights=[weights.get(k, 1) for k in keys])[0]


def template_for(L, key):
    lang = L.code
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
            doc.span(email_variant(f"{email_local(L, who)}@{pick(r['email_domains_personal'])}"), "EMAIL")
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
            doc.add(nick())
            doc.cats.add("sns_handle")
            doc.personal = True
        else:
            doc.add(
                {
                    "ADDR": lambda: pick(r["addresses"]),
                    "DATE": lambda: date_str(r, key == "dob"),
                    "DIAG": lambda: pick(r["diagnoses"]),
                    "IP": ip,
                    "HANDLE": lambda: "@" + re.sub(r"[.-]", "_", pick(r["email_locals"])),
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
            )
    doc.add(tpl[last:])


def sep(L):
    return "" if L.cjk and L.code != "ko" else " "


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
        facts.add(weighted(PERSON_FACTS, PERSON_WEIGHTS))
    facts = list(facts) + [weighted(NEG_FACTS, NEG_WEIGHTS) for _ in range(rint(0, 2))]
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
            weighted([k for k in PERSON_FACTS if k != "employment"], PERSON_WEIGHTS),
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
        doc.add(
            "\n" + pick(L.r["orgs"]) + " " + (pick(L.r["depts"]) if chance(0.5) else pick(L.r["job_titles"]))
        )
        doc.cats.add("employment_info")
    if chance(0.6):
        doc.add(f"\n{lab(L, pick(['phone', 'mobile']))}{colon(L)}")
        doc.span(personal_phone(L.code), "PHONE")
        doc.cats.add("email_or_phone")
    if chance(0.5):
        doc.add(f"\n{lab(L, 'email')}{colon(L)}")
        doc.span(f"{email_local(L, who)}@{pick(L.r['email_domains_personal'])}", "EMAIL")
        doc.cats.add("email_or_phone")


def _field(doc, L, k, who):
    if k == "phone":
        doc.span(phone_variant(L.code, personal_phone(L.code)), "PHONE")
    elif k == "email":
        doc.span(email_variant(f"{email_local(L, who)}@{pick(L.r['email_domains_personal'])}"), "EMAIL")
    elif k == "gov_id":
        doc.span(*gov_id(L.code))
    elif k == "passport":
        doc.span(passport(L.code), "PASSPORT_LICENCE")
    elif k == "card":
        doc.span(card(L.code), "CREDIT_CARD")
    elif k == "bank":
        doc.span(bank(L.code), "BANK_ACCOUNT")
    else:
        doc.add(
            {
                "address": lambda: pick(L.r["addresses"]),
                "dob": lambda: date_str(L.r, True),
                "dept": lambda: pick(L.r["depts"]),
                "job": lambda: pick(L.r["job_titles"]),
                "diagnosis": lambda: pick(L.r["diagnoses"]),
                "ip": ip,
                "sns": lambda: "@" + re.sub(r"[.-]", "_", pick(L.r["email_locals"])),
            }[k]()
        )


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
    doc.person(L.mention(who, "full"))
    if chance(0.2):
        staff = f"{pick(['E-', 'M', '', 'No.'])}{digits(pick([5, 6, 7]))}"
        doc.add(f"{nl}{pick(SENS['STAFF_ID'][L.code])}{colon(L)}{staff}")
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
            rng.sample(["dept", "email", "phone"] + (["diagnosis"] if patients else []), rint(0, 2))
        )
    )
    s = pick([",", "\t", " | "])
    doc.add(s.join([lab(L, "name")] + [lab(L, c) for c in cols]))
    for _ in range(rint(2, 4)):
        doc.add("\n")
        who = L.person(pool)
        doc.person(L.mention(who, "full"))
        for c in cols:
            doc.add(s)
            _field(doc, L, c, who)
    for c in cols:
        doc.cats.add(FIELD_CAT[c])
    if patients or "diagnosis" in cols:
        doc.cats.add("health_info")
        doc.high = True


def chat(doc, L, pool):
    people = [L.person(pool), L.person(pool)]
    for i in range(rint(2, 4)):
        if i:
            doc.add("\n")
        who = people[i % 2]
        doc.person(L.mention(who, "surname" if L.cjk else pick(["given", "full"])))
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
        doc.add(pick(TECH_NEG)())
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
        doc.span(f"{email_local(L, who)}@{pick(L.r['email_domains_personal'])}", "EMAIL")
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
    extra = pick([None, None, "phone", "dept"])
    inline = chance(0.3)
    for i in range(rint(2, 5)):
        if i:
            doc.add(pick([", ", "、" if L.cjk else "; "]) if inline else "\n")
        who = L.person(pool)
        if not inline and chance(0.4):
            doc.add(f"{i + 1}. ")
        doc.person(L.mention(who, "full"))
        if extra == "phone":
            doc.add(" ")
            doc.span(personal_phone(L.code), "PHONE")
            doc.cats.add("email_or_phone")
        elif extra == "dept":
            doc.add(f" ({pick(L.r['depts'])})")
            doc.cats.add("employment_info")
    if not neutral:
        doc.cats.add(spec["cat"])
        doc.high = True


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
    # Several blocks in one document, so the deciding fact can sit anywhere in a long text.
    limit = 420 if L.cjk else 1300
    while len(doc.text) < limit * 0.6:
        if doc.text:
            doc.add("\n\n")
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
                doc.span(email_variant(f"{email_local(L, p)}@{pick(r['email_domains_personal'])}"), "EMAIL")
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
            else:
                doc.add(
                    {
                        "addr": lambda: pick(r["addresses"]),
                        "dob": lambda: date_str(r, True),
                        "job": lambda: pick(r["job_titles"]),
                        "handle": lambda: "@" + re.sub(r"[.-]", "_", pick(r["email_locals"])),
                        "ip": ip,
                    }[attr]()
                )
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
    doc.cats.discard("person_name") if not any(k == "PERSON" for *_, k in doc.spans) else None


def load_slot_templates(root):
    from .slots import problems

    kept = Counter()
    for path in sorted(Path(root).glob("*.json")):
        try:
            templates = json.loads(path.read_text())["templates"]
        except (json.JSONDecodeError, KeyError):
            print(f"!! skipping unreadable {path}")
            continue
        for i, t in enumerate(templates):
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
]
BLOCKS = [narrative, negative, negative, email_doc, transaction, form, chat, topic_article]


def make_doc(L, pool):
    formats = FORMATS + ([(SLOT_WEIGHT, slot_doc)] if SLOT_T.get(L.code, {}).get(pool) else [])
    fn = rng.choices([f for _, f in formats], weights=[w for w, _ in formats])[0]
    doc = Doc()
    fn(doc, L, pool)
    if re.search(r"\{[A-Za-z_]+\}", doc.text):
        raise ValueError(f"leftover placeholder in {L.code}: {doc.text}")
    for s, e, _ in doc.spans:
        assert doc.text[s:e].strip() == doc.text[s:e] and e > s, (doc.text, s, e)
    return {
        "lang": L.code,
        "format": fn.__name__,
        "text": doc.text,
        "spans": doc.spans,
        "sensitivity": 2 if doc.high else 1 if doc.personal else 0,
        "categories": sorted(doc.cats),
    }


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
    a = ap.parse_args(argv)
    rng.seed(a.seed)
    global SLOT_WEIGHT
    SLOT_WEIGHT = a.template_weight
    if a.templates:
        print("slot templates:", dict(load_slot_templates(a.templates)))
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
    report = {"excluded": excluded, "pools": {}}
    for pool, n in (("train", a.train_docs), ("valid", a.valid_docs)):
        docs = [make_doc(L, pool) for L in langs for _ in range(n)]
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
    assert set(CATEGORIES) >= {c for p in report["pools"].values() for c in p["categories"]}
    (a.out / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report["pools"], ensure_ascii=False, indent=1))
    print("excluded:", len(excluded))


if __name__ == "__main__":
    main()
