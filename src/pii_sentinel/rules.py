"""Rule engine for detecting sensitive information and PII in text.

Loads regex rules from canary_rules.json and applies them with validator functions.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import regex

# Test card numbers that should not be flagged (published test data)
_TEST_CARD_NUMBERS = {
    "4242424242424242",
    "4111111111111111",
    "4012888888881881",
    "4000056655665556",
    "5555555555554444",
    "5105105105105100",
    "5200828282828210",
    "378282246310005",
    "371449635398431",
    "6011111111111117",
    "6011000990139424",
    "3056930009020004",
    "3566002020360505",
}

# Spanish DNI/NIE control letter alphabet
_NIF_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

# Italian Codice Fiscale odd position values
_CF_ODD_VALUES = {
    "0": 1,
    "1": 0,
    "2": 5,
    "3": 7,
    "4": 9,
    "5": 13,
    "6": 15,
    "7": 17,
    "8": 19,
    "9": 21,
    "A": 1,
    "B": 0,
    "C": 5,
    "D": 7,
    "E": 9,
    "F": 13,
    "G": 15,
    "H": 17,
    "I": 19,
    "J": 21,
    "K": 2,
    "L": 4,
    "M": 18,
    "N": 20,
    "O": 11,
    "P": 3,
    "Q": 6,
    "R": 8,
    "S": 12,
    "T": 14,
    "U": 16,
    "V": 10,
    "W": 22,
    "X": 25,
    "Y": 24,
    "Z": 23,
}

# Placeholder detection patterns
_PLACEHOLDER_MARKERS = regex.compile(
    r"^(?:changeme|change|me|replace|insert|set|with|real|this|your|my|here|todo|tbd|fixme|dummy|placeholder|insecure|sample|example|test|fake|redacted|value|x{3,})$",
    regex.IGNORECASE,
)
_PLACEHOLDER_FILLER = regex.compile(
    r"^(?:api|key|keys|token|tokens|secret|secrets|password|passwd|pwd|pass|base|url|uri|host|hostname|name|user|username|id|access|refresh|client|auth|sk|pk|in|production|development|staging|local|dev|the|a|of|for|and|[0-9]+)$",
    regex.IGNORECASE,
)
_GENERIC_CREDENTIALS = regex.compile(
    r"\w{1,32}:\/\/(?:your[_-]?)?(?:user|username)(?:name)?:(?:your[_-]?)?(?:password|passwd|pwd)@",
    regex.IGNORECASE,
)
_GENERIC_HOST = regex.compile(
    r"^(?:localhost|127\.0\.0\.1|0\.0\.0\.0|host|hostname|db|database|example\.(?:com|org|net))\b",
    regex.IGNORECASE,
)
_DESCRIBES_A_SECRET = regex.compile(
    r"\b[A-Za-z0-9_]*_(?:PROJECT|NAME|PATH|FILE|DIR|URL|URI|ENDPOINT|HOST|PORT|ID|TYPE|HEADER|PREFIX|SUFFIX|FIELD|COLUMN|TABLE|ENV|REGION|BUCKET|ARN|VERSION|TTL|TIMEOUT|LENGTH|COUNT|ENABLED|ALGORITHM|ISSUER|AUDIENCE|SCOPE|PROVIDER|BACKEND|SOURCE)\b[ \t]*[:=]",
    regex.IGNORECASE,
)

_MIN_MEAN_WORD_LENGTH = 2.5
_SHORTEST_MEASURABLE_SEGMENT = 8


@dataclass
class Match:
    """A detected sensitive match in text."""

    rule: str
    description: str
    category: str
    start: int
    end: int
    value: str


def entropy(s: str) -> float:
    """Shannon entropy (bits per character; ≈0–8 for byte-sized alphabets)."""
    if len(s) == 0:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    h = 0.0
    n = len(s)
    for count in freq.values():
        p = count / n
        h -= p * math.log2(p)
    return h


def _reads_as_words(segment: str) -> bool:
    """Check if a segment reads as words rather than random characters."""
    if len(segment) < _SHORTEST_MEASURABLE_SEGMENT:
        return True
    # Match: capital runs, capital + lowercase runs, or digit runs
    words = regex.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", segment)
    if not words:
        return False
    return len(segment) / len(words) >= _MIN_MEAN_WORD_LENGTH


def is_not_secret_shaped(value: str) -> bool:
    """Whether a value's shape says it is not a credential."""
    v = value.strip()
    # A URL or URN
    if regex.match(r"^[a-z][a-z0-9+.-]*:\/\/", v, regex.IGNORECASE) and "?" not in v and "@" not in v:
        return True
    if regex.match(r"^urn:", v, regex.IGNORECASE):
        return True
    # A filesystem path
    if regex.match(r"^[~.]?\/[^\s]*$", v):
        return True
    # Variable name (SCREAMING_SNAKE_CASE)
    if regex.match(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$", v):
        return True
    # HTTP header name
    if regex.match(r"^[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)+$", v):
        return True
    # A number
    if regex.match(r"^\d+$", v):
        return True
    # Dotted lower-case identifier
    if regex.match(r"^[a-z][a-z0-9]*(?:\.[a-z0-9]+)+$", v):
        return True
    # Code reference like process.env.API_KEY, user.password_digest
    return bool(
        regex.match(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+$", v)
        and all(_reads_as_words(seg) for seg in v.split("."))
    )


def is_placeholder(value: str, following: str = "") -> bool:
    """Whether a value is written to be replaced, not a real credential."""
    v = value.strip()
    if not v:
        return False
    # Filler: xxxxxxxxx
    if regex.match(r"^[Xx]+$", v):
        return True
    # Slot: <token>, ${TOKEN}, {{ token }}
    if regex.match(r"^[<{[]", v) and regex.search(r"[>}\]]$", v):
        return True
    # Shell or template reference: $VAR, ${VAR}
    if regex.match(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$", v):
        return True
    # Unexpanded reference in connection string
    if regex.search(
        r":\/\/[^@\s]*(?:\$\{[A-Za-z_][^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|\{\})[^@\s]*@",
        v,
    ):
        return True
    # Same user and password naming the service
    same_pair = regex.search(r":\/\/([A-Za-z]{3,12}):([A-Za-z]{3,12})@", v)
    if same_pair:
        user = same_pair.group(1).lower()
        passwd = same_pair.group(2).lower()
        if user == passwd and regex.match(
            r"^(?:postgres|postgresql|mysql|mariadb|mongo|mongodb|redis|root|guest|admin|user|test|rabbitmq)$",
            user,
            regex.IGNORECASE,
        ):
            return True
    # Django insecure default
    if regex.match(r"^django-insecure-", v, regex.IGNORECASE):
        return True
    # Generic credentials with default host
    credentials = _GENERIC_CREDENTIALS.search(v)
    if credentials:
        after_at = v[credentials.end() :]
        if _GENERIC_HOST.match(after_at or following):
            return True
    # Split on separators and check if all parts are marker/filler words
    parts = [p for p in regex.split(r"[-_.\s]+", v) if p]
    if not parts:
        return False
    if not any(_PLACEHOLDER_MARKERS.match(p) for p in parts):
        return False
    return all(_PLACEHOLDER_MARKERS.match(p) or _PLACEHOLDER_FILLER.match(p) for p in parts)


def key_describes_rather_than_holds(match_text: str) -> bool:
    """Whether a match key says where a secret lives rather than what it is."""
    return bool(_DESCRIBES_A_SECRET.search(match_text))


def _luhn(s: str) -> bool:
    """Luhn algorithm checksum validation."""
    digits = regex.sub(r"\D", "", s)
    if not digits:
        return False
    total = 0
    double = False
    for i in range(len(digits) - 1, -1, -1):
        d = int(digits[i])
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return total % 10 == 0


def is_real_card_number(s: str) -> bool:
    """Check if a card number is real (not test data) and passes Luhn."""
    if regex.sub(r"\D", "", s) in _TEST_CARD_NUMBERS:
        return False
    return _luhn(s)


def is_real_aws_key(s: str) -> bool:
    """Check if AWS key is real (not example key)."""
    return not s.endswith("EXAMPLE")


def validate_my_number(s: str) -> bool:
    """Japanese Individual Number (My Number): 12 digits, weighted checksum."""
    cleaned = regex.sub(r"[-\s]", "", s)
    # All same digit is invalid (padding, zeroed records)
    if regex.match(r"^(\d)\1*$", cleaned):
        return False
    digits = regex.sub(r"\D", "", s)
    if len(digits) != 12:
        return False
    weights = [6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights[i] for i in range(11))
    remainder = total % 11
    check_digit = 0 if remainder <= 1 else 11 - remainder
    return check_digit == int(digits[11])


def validate_phone_jp(s: str) -> bool:
    """Japanese phone number: 10 or 11 digits, exclude freephone prefixes."""
    digits = regex.sub(r"\D", "", s)
    if not regex.match(r"^0\d{8,10}$", digits):
        return False
    if len(digits) not in (10, 11):
        return False
    return not regex.match(r"^0(?:120|800)", digits)


def validate_french_nir(s: str) -> bool:
    """French NIR / Social Security Number: 15 digits, 2-digit check key."""
    cleaned = regex.sub(r"\s", "", s)

    standard = regex.match(r"^([12]\d{12})(\d{2})$", cleaned)
    corse_a = regex.match(r"^([12]\d{4}2A\d{6})(\d{2})$", cleaned, regex.IGNORECASE)
    corse_b = regex.match(r"^([12]\d{4}2B\d{6})(\d{2})$", cleaned, regex.IGNORECASE)

    if standard:
        nir13 = standard.group(1)
        key_str = standard.group(2)
    elif corse_a:
        nir13 = regex.sub(r"2A", "19", corse_a.group(1), flags=regex.IGNORECASE)
        key_str = corse_a.group(2)
    elif corse_b:
        nir13 = regex.sub(r"2B", "18", corse_b.group(1), flags=regex.IGNORECASE)
        key_str = corse_b.group(2)
    else:
        return False

    num = int(nir13)
    computed_key = 97 - (num % 97)
    return computed_key == int(key_str)


def validate_codice_fiscale(s: str) -> bool:
    """Italian Codice Fiscale: 16 alphanumeric chars with control digit."""
    cf = regex.sub(r"\s", "", s.upper())
    if not regex.match(
        r"^[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]$",
        cf,
    ):
        return False

    total = 0
    for i in range(15):
        ch = cf[i]
        if i % 2 == 0:  # odd position (1-indexed)
            total += _CF_ODD_VALUES.get(ch, -1)
        elif ch.isdigit():
            total += int(ch)
        else:
            total += ord(ch) - 65

    expected = chr(65 + (total % 26))
    return expected == cf[15]


def validate_german_id_nr(s: str) -> bool:
    """German Steuer-Identifikationsnummer: 11 digits, ISO/IEC 7064 MOD 11,10."""
    cleaned = regex.sub(r"\s", "", s)
    if not regex.match(r"^[1-9]\d{10}$", cleaned):
        return False

    produkt = 10
    for i in range(10):
        summe = (int(cleaned[i]) + produkt) % 10
        if summe == 0:
            summe = 10
        produkt = (summe * 2) % 11

    check = 11 - produkt
    if check == 10:
        check = 0
    return check == int(cleaned[10])


def validate_spanish_nif(s: str) -> bool:
    """Spanish DNI (8 digits + letter) or NIE (X/Y/Z + 7 digits + letter)."""
    cleaned = regex.sub(r"[\s-]", "", s.upper())

    dni_match = regex.match(r"^(\d{8})([A-Z])$", cleaned)
    if dni_match:
        num = int(dni_match.group(1))
        return _NIF_LETTERS[num % 23] == dni_match.group(2)

    nie_match = regex.match(r"^([XYZ])(\d{7})([A-Z])$", cleaned)
    if nie_match:
        prefix = {"X": "0", "Y": "1", "Z": "2"}[nie_match.group(1)]
        num = int(prefix + nie_match.group(2))
        return _NIF_LETTERS[num % 23] == nie_match.group(3)

    return False


def validate_korean_rrn(s: str) -> bool:
    """Korean Resident Registration Number: 13 digits, weighted checksum."""
    cleaned = regex.sub(r"[-\s]", "", s)
    if not regex.match(r"^\d{13}$", cleaned):
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    total = sum(int(cleaned[i]) * weights[i] for i in range(12))
    check = (11 - (total % 11)) % 10
    return check == int(cleaned[12])


def validate_korean_brn(s: str) -> bool:
    """Korean Business Registration Number: 10 digits, NTS algorithm."""
    cleaned = regex.sub(r"[-\s]", "", s)
    if not regex.match(r"^\d{10}$", cleaned):
        return False
    weights = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    total = sum(int(cleaned[i]) * weights[i] for i in range(9))
    total += int(cleaned[8]) * 5 // 10
    check = (10 - (total % 10)) % 10
    return check == int(cleaned[9])


def validate_chinese_id(s: str) -> bool:
    """Chinese Resident Identity Card: 18 chars (17 digits + check)."""
    cleaned = regex.sub(r"[-\s]", "", s.upper())
    if not regex.match(r"^\d{17}[\dX]$", cleaned):
        return False
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    code = "10X98765432"
    total = sum(int(cleaned[i]) * weights[i] for i in range(17))
    return code[total % 11] == cleaned[17]


def is_reserved_ipv4(ip: str) -> bool:
    """Check if IPv4 is reserved/non-public."""
    octets = ip.split(".")
    if len(octets) != 4 or not all(regex.match(r"^\d{1,3}$", o) for o in octets):
        return True

    parts = [int(o) for o in octets]
    if any(p > 255 for p in parts):
        return True

    a, b, c = parts[0], parts[1], parts[2]

    if a == 0 or a == 10:
        return True
    if a == 100 and 64 <= b <= 127:  # CGN
        return True
    if a == 127:  # loopback
        return True
    if a == 169 and b == 254:  # link-local
        return True
    if a == 172 and 16 <= b <= 31:  # private
        return True
    if a == 192 and b == 0 and c == 0:  # IETF protocol
        return True
    if a == 192 and b == 0 and c == 2:  # TEST-NET-1
        return True
    if a == 192 and b == 88 and c == 99:  # 6to4 relay
        return True
    if a == 192 and b == 168:  # private
        return True
    if a == 198 and b in (18, 19):  # benchmark
        return True
    if a == 198 and b == 51 and c == 100:  # TEST-NET-2
        return True
    if a == 203 and b == 0 and c == 113:  # TEST-NET-3
        return True
    return a >= 224  # multicast + reserved


def is_reserved_ipv6(ip: str) -> bool:
    """Check if IPv6 is reserved/non-public."""
    lower = ip.lower()

    # Split by ::
    halves = lower.split("::")
    if len(halves) > 2:
        return True

    def is_hex_group(g: str) -> bool:
        return bool(regex.match(r"^[0-9a-f]{1,4}$", g))

    if len(halves) == 1:
        raw = lower.split(":")
        if not all(is_hex_group(g) for g in raw):
            return True
        groups = [int(g, 16) for g in raw]
    else:
        left_raw = halves[0].split(":") if halves[0] else []
        right_raw = halves[1].split(":") if halves[1] else []
        if not all(is_hex_group(g) for g in left_raw) or not all(is_hex_group(g) for g in right_raw):
            return True
        if len(left_raw) + len(right_raw) >= 8:
            return True
        left = [int(g, 16) for g in left_raw]
        right = [int(g, 16) for g in right_raw]
        zeros = [0] * (8 - len(left) - len(right))
        groups = left + zeros + right

    if len(groups) != 8:
        return True

    # Unspecified (::)
    if all(g == 0 for g in groups):
        return True
    # Loopback (::1)
    if all(g == 0 for g in groups[:7]) and groups[7] == 1:
        return True
    # Link-local fe80::/10
    if 0xFE80 <= groups[0] <= 0xFEBF:
        return True
    # Unique-local fc00::/7
    if (groups[0] & 0xFE00) == 0xFC00:
        return True
    # Multicast ff00::/8
    if (groups[0] & 0xFF00) == 0xFF00:
        return True
    # Documentation 2001:db8::/32
    return groups[0] == 0x2001 and groups[1] == 0x0DB8


# Validator registry
_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "luhn": is_real_card_number,
    "aws-key": is_real_aws_key,
    "mynumber-jp": validate_my_number,
    "phone-jp": validate_phone_jp,
    "nir-fr": validate_french_nir,
    "codice-fiscale-it": validate_codice_fiscale,
    "steuer-id-de": validate_german_id_nr,
    "dni-nie-es": validate_spanish_nif,
    "rrn-kr": validate_korean_rrn,
    "brn-kr": validate_korean_brn,
    "resident-id-cn": validate_chinese_id,
    "public-ipv4": lambda ip: not is_reserved_ipv4(ip),
    "public-ipv6": lambda ip: not is_reserved_ipv6(ip),
}


def _context_tokens(text: str) -> set[str]:
    """Extract context tokens from text, removing punctuation."""
    out = set()
    for raw in text.split():
        # Remove leading and trailing punctuation/symbols using regex module's Unicode support
        word = regex.sub(r"^[\p{P}\p{S}]+", "", raw, flags=regex.UNICODE)
        word = regex.sub(r"[\p{P}\p{S}]+$", "", word, flags=regex.UNICODE)
        word = word.lower()
        if word:
            out.add(word)
    return out


def _has_nearby_context_word(
    text: str,
    match_start: int,
    match_end: int,
    context_words: list[str],
    window_tokens: int,
) -> bool:
    """Check if context words appear near the match."""
    if not context_words:
        return True

    char_window = window_tokens * 8
    before = text[max(0, match_start - char_window) : match_start]
    after = text[match_end : match_end + char_window]
    window = f"{before} {after}"
    nearby = _context_tokens(window)
    lowered = window.lower()

    for raw in context_words:
        word = raw.lower()
        # Non-ASCII labels are matched as substrings
        if not all(ord(c) < 128 for c in word):
            if word in lowered:
                return True
        elif word in nearby:
            return True

    return False


@dataclass
class _CompiledRule:
    """Compiled regex rule with metadata."""

    id: str
    description: str
    category: str
    regex: "regex.Pattern[str]"
    secret_group: int | None = None
    entropy_threshold: float | None = None
    validate: Callable[[str], bool] | None = None
    context_words: list[str] | None = None
    require_context: bool = False
    exclude_context: list[str] | None = None
    context_window: int | None = None


# Global context window from config
_effective_context_window = 3


def _compile_rules() -> list[_CompiledRule]:
    """Load and compile rules from canary_rules.json."""
    global _effective_context_window
    config_path = Path(__file__).parent / "canary_rules.json"
    with open(config_path) as f:
        config = json.load(f)

    _effective_context_window = config.get("contextWindow", 3)

    rules: list[_CompiledRule] = []
    for rule_config in config.get("rules", []):
        rule_id = rule_config.get("id", "")

        # Compile regex with proper flags
        pattern_str = rule_config.get("regex", "")
        flags_str = rule_config.get("flags", "g")

        # Convert JS flags to Python regex flags
        py_flags = 0
        if "i" in flags_str:
            py_flags |= regex.IGNORECASE
        if "m" in flags_str:
            py_flags |= regex.MULTILINE
        if "s" in flags_str:
            py_flags |= regex.DOTALL
        if "u" in flags_str:
            py_flags |= regex.UNICODE

        try:
            compiled = regex.compile(pattern_str, py_flags)
        except regex.error as e:
            raise ValueError(f"Failed to compile regex for rule {rule_id}: {e}") from e

        # Get validator if specified
        validator = None
        validator_name = rule_config.get("validate")
        if validator_name:
            validator = _VALIDATORS.get(validator_name)
            if not validator:
                raise ValueError(f"Unknown validator '{validator_name}' in rule {rule_id}")

        rule = _CompiledRule(
            id=rule_id,
            description=rule_config.get("description", ""),
            category=rule_config.get("category", "secret"),
            regex=compiled,
            secret_group=rule_config.get("secretGroup"),
            entropy_threshold=rule_config.get("entropyThreshold"),
            validate=validator,
            context_words=rule_config.get("contextWords"),
            require_context=rule_config.get("requireContext", False),
            exclude_context=rule_config.get("excludeContext"),
            context_window=rule_config.get("contextWindow"),
        )
        rules.append(rule)

    return rules


# Compile rules at module load time
try:
    _COMPILED_RULES = _compile_rules()
except Exception as e:
    raise RuntimeError(f"Failed to load canary rules: {e}") from e


def scan(text: str) -> list[Match]:
    """Scan text for sensitive information and PII.

    Returns a list of Match objects sorted by start position.
    """
    matches: list[Match] = []

    for rule in _COMPILED_RULES:
        for match_obj in rule.regex.finditer(text):
            # Extract the value to validate/report
            if rule.secret_group is not None:
                secret_value = match_obj.group(rule.secret_group)
            else:
                secret_value = match_obj.group(0)

            if not secret_value:
                continue

            # Get match span (for the whole match)
            match_start = match_obj.start()
            match_end = match_obj.end()
            following = text[match_end : match_end + 64]

            # The shape test applies only where the rule captured a free-form value.
            captures_a_value = rule.secret_group is not None

            # Order of checks must match TS: shapes, then entropy, then validate, then context
            if rule.category == "secret":
                # Check placeholder first (always)
                if is_placeholder(secret_value, following):
                    continue

                # Then shape checks if secretGroup is set
                if captures_a_value and (
                    is_not_secret_shaped(secret_value)
                    or is_placeholder(match_obj.group(0), following)
                    or is_not_secret_shaped(match_obj.group(0))
                    or key_describes_rather_than_holds(match_obj.group(0))
                ):
                    continue

            # Check entropy threshold
            if rule.entropy_threshold is not None and entropy(secret_value) < rule.entropy_threshold:
                continue

            # Validate with custom validator if present
            if rule.validate and not rule.validate(secret_value):
                continue

            # Get context window: rule-specific or global default
            window = rule.context_window if rule.context_window is not None else _effective_context_window

            # Check context-based filtering
            has_context = not rule.context_words or _has_nearby_context_word(
                text, match_start, match_end, rule.context_words, window
            )

            # Require context rules are dropped if no context
            if rule.require_context and not has_context:
                continue

            # Exclude context rules are dropped if exclude words found
            if rule.exclude_context and _has_nearby_context_word(
                text, match_start, match_end, rule.exclude_context, window
            ):
                continue

            # Determine span: use secretGroup span if present, else whole match
            if rule.secret_group is not None:
                span_start = match_obj.start(rule.secret_group)
                span_end = match_obj.end(rule.secret_group)
            else:
                span_start = match_start
                span_end = match_end

            matches.append(
                Match(
                    rule=rule.id,
                    description=rule.description,
                    category=rule.category,
                    start=span_start,
                    end=span_end,
                    value=secret_value,
                )
            )

    # Sort by start position
    matches.sort(key=lambda m: m.start)
    return matches
