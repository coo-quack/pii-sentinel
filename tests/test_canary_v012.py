"""Cases ported from sensitive-canary v0.12.0 for the new and changed rules.

Source: sensitive-canary tag v0.12.0, src/lib/__tests__/rule-patterns.test.ts,
validators.test.ts and shapes.test.ts. Inputs and expected results are the
TypeScript ones; only the spelling of names changes.
"""

import pytest

from pii_sentinel.rules import (
    has_aws_secret_shape,
    is_basic_credentials,
    is_filler_body,
    is_iban,
    is_intl_phone,
    is_nino_uk,
    is_not_secret_shaped,
    is_password_like,
    is_personal_mailbox,
    is_placeholder,
    is_real_card_number,
    is_us_phone,
    scan,
)

# The fixtures from rule-fixtures.ts. Literals that look like live credentials
# are joined at run time so that GitHub push protection does not refuse them.
ALPHABET = "Xk9mP2qR7vL4nW1sYj3cBz8dEf5gHiKoNpQuTxMnA6bC0eFhJ"
HEX_ALPHABET = "9f3a7c1e5b0d8f2a"
AWS_KEY = "".join(["AKIA", "3QF7TZ9KLMN2", "PQRS"])
DOCKER_PAT = "_".join(["dckr", "pat", "xEFv_Zi-yLNKnOyppRuS6X3wEJB"])
STRIPE_DOCS_TEST_KEY = "_".join(["sk", "test", "4eC39HqLyjWDarjtT1zdp7dc"])


def _cycle(alphabet: str, n: int) -> str:
    """Return n characters cycling through the alphabet, so no run is longer than one."""
    return "".join(alphabet[i % len(alphabet)] for i in range(n))


def mixed(n: int) -> str:
    """Return a mixed-case alphanumeric value of length n."""
    return _cycle(ALPHABET, n)


def hex_run(n: int) -> str:
    """Return a lower-case hexadecimal value of length n."""
    return _cycle(HEX_ALPHABET, n)


def _rule_ids(text: str) -> list[str]:
    """Return the rule ids of every finding in text."""
    return [m.rule for m in scan(text)]


def _hit(text: str, rule_id: str) -> bool:
    """Return True when the rule reports at least one finding in text."""
    return rule_id in _rule_ids(text)


def _value(text: str, rule_id: str) -> str | None:
    """Return the matched value of the first finding of the rule, or None."""
    for m in scan(text):
        if m.rule == rule_id:
            return m.value
    return None


def _random(n: int) -> str:
    """Return a value of length n that is neither a run nor a word."""
    return ("A1b2C3d4E5f6G7h8I9j0" * 10)[:n]


def _words_of(size: int, count: int) -> str:
    """Return count camel-case words of the given size, joined without separators."""
    return "".join(("a" if i == 0 else "A") + "b" * (size - 1) for i in range(count))


# ── telegram-bot-token ────────────────────────────────────────────────────────


class TestTelegramBotToken:
    """telegram-bot-token: the hash after AA may be any length from 32 up."""

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param("110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw", id="32-after-AA"),
            pytest.param("8012345678:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsawXyz", id="35-after-AA"),
            pytest.param("123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw", id="six-digit-bot-id"),
        ],
    )
    def test_token_of_any_hash_length_is_found(self, token: str):
        """A bot id, a colon, AA and a hash of 32 or more characters is a token."""
        assert _hit(token, "telegram-bot-token")

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("1234567890:AA", id="no-hash"),
            pytest.param("12345:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw", id="too-few-digits"),
        ],
    )
    def test_short_id_or_no_hash_is_not_found(self, text: str):
        """A bot id with too few digits, or with no hash, is not a token."""
        assert not _hit(text, "telegram-bot-token")

    @pytest.mark.parametrize(
        "text",
        [
            "https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/getMe",
            "https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
            "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
        ],
        ids=["getMe-URL", "URL-ending-at-token", "token-alone"],
    )
    def test_token_after_bot_in_a_url_is_found(self, text: str):
        """A token written after bot in a URL, or on its own, is found."""
        assert _hit(text, "telegram-bot-token")

    @pytest.mark.parametrize(
        "text",
        [
            "https://api.telegram.org/bot123456789:BBHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/getMe",
            "robot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
        ],
        ids=["not-AA", "robot-is-not-bot"],
    )
    def test_colon_without_AA_or_without_bot_is_not_found(self, text: str):
        """The AA after the colon is required, and bot must not be a suffix."""
        assert not _hit(text, "telegram-bot-token")


# ── connection-string ─────────────────────────────────────────────────────────


class TestConnectionStringCredentials:
    """connection-string: an empty user name still carries a password."""

    @pytest.mark.parametrize(
        "text",
        [
            "redis://:4yBnLzQgJuCkCCJsJ7@cache-03.northgate-data.io:6379/0",
            "mongodb://:Xk2p9QmR7t@cluster0.example.net/db",
            "postgres://:Xk2p9QmR7t@db.internal/app",
        ],
        ids=["redis-no-user", "mongodb-no-user", "postgres-no-user"],
    )
    def test_empty_user_with_a_password_is_found(self, text: str):
        """A connection string with a password and no user name is a finding."""
        assert _hit(text, "connection-string")

    @pytest.mark.parametrize(
        "text",
        [
            "redis://:@cache-03.northgate-data.io:6379/0",
            "redis://cache-03.northgate-data.io:6379/0",
        ],
        ids=["no-user-no-password", "no-credentials"],
    )
    def test_no_password_is_not_found(self, text: str):
        """Without a password there is no credential to report."""
        assert not _hit(text, "connection-string")

    def test_srv_scheme_is_found(self):
        """The mongodb+srv scheme that MongoDB Atlas issues is covered."""
        assert _hit("mongodb+srv://svc:Xk2p9QmR7t@cluster0.abcd.mongodb.net/db", "connection-string")

    @pytest.mark.parametrize(
        "text",
        [
            "mongodb://user:pass@host/db",
            "postgres://u:p@h:5432/d",
            "postgresql://u:p@h/d",
            "mysql://root:secret@localhost",
            "redis://default:abc123@127.0.0.1:6379",
            "see mongodb://user:pass@host in the log",
        ],
    )
    def test_bounded_credentials_are_found(self, text: str):
        """The user and password forms the bounded pattern still reads."""
        assert _hit(text, "connection-string")

    @pytest.mark.parametrize(
        "text",
        [
            "mongodb://host/db",
            "https://user:pass@host",
            "mongodb://user@host",
        ],
        ids=["no-credentials", "scheme-not-covered", "no-password"],
    )
    def test_text_without_a_password_or_scheme_is_not_found(self, text: str):
        """No password, or a scheme the rule does not cover, is not a finding."""
        assert not _hit(text, "connection-string")

    def test_userinfo_longer_than_the_bound_is_not_found(self):
        """Userinfo past 1024 characters is not read; 1024 characters still is."""
        assert not _hit(f"mongodb://user:{mixed(1025)}@host", "connection-string")
        assert _hit(f"mongodb://user:{mixed(1024)}@host", "connection-string")


# ── pii-email ─────────────────────────────────────────────────────────────────

ROLE_ADDRESSES = [
    "noreply@acmecorp.io",
    "no-reply@acmecorp.io",
    "donotreply@acmecorp.io",
    "do-not-reply@acmecorp.io",
    "support@acmecorp.io",
    "sales@acmecorp.io",
    "info@acmecorp.io",
    "help@acmecorp.io",
    "contact@acmecorp.io",
    "press@acmecorp.io",
    "abuse@acmecorp.io",
    "postmaster@acmecorp.io",
    "webmaster@acmecorp.io",
    "hostmaster@acmecorp.io",
    "Support@acmecorp.io",
    "NOREPLY@acmecorp.io",
    "No-Reply@acmecorp.io",
]

NEAR_MISS_ADDRESSES = [
    "john.support@acmecorp.io",
    "support2@acmecorp.io",
    "supportteam@acmecorp.io",
    "info.jane@acmecorp.io",
    "sales-team@acmecorp.io",
    "x.noreply@acmecorp.io",
]

REMOTE_SHELL_ADDRESSES = [
    "rsync failed, notify alice@acmecorp.io",
    "# sftp creds, ask carol@acmecorp.io",
    "See ssh(1). Maintainer: bob@acmecorp.io",
    "the ssh key belongs to dave@acmecorp.io",
    "scp is slow; email erin@acmecorp.io",
]

REMOTE_SHELL_HOSTS = [
    "ssh deploy@bastion.acmecorp.io",
    "ssh -i ~/.ssh/id_ed25519 deploy@bastion.acmecorp.io",
    "ssh -o StrictHostKeyChecking=no deploy@bastion.acmecorp.io",
    "scp build.tar ops@files.acmecorp.io:/srv/",
    "rsync -a ops@files.acmecorp.io:/srv/ .",
    "sftp ops@files.acmecorp.io",
]

SPACED_REMOTE_SHELL_HOSTS = [
    "ssh deploy@prod.acme-corp.net",
    "ssh  deploy@prod.acme-corp.net",
    "ssh -p 22 deploy@prod.acme-corp.net",
    "scp -r build ops@files.acme-corp.net:/srv/",
    "rsync -avz ops@files.acme-corp.net:/srv/ .",
]

PACKAGE_SPECIFIERS = [
    "playwright-core@1.35.1.patch",
    "eslint@8.57.0",
    "@babel/core@7.24.0",
    "typescript@5.4.2",
    "golang.org/x/mobile@0.0.0",
]

PERSONAL_ADDRESSES = [
    "alice@analytical-engines.org",
    "ada.lovelace@analytical-engines.org",
    "user@123abc.com",
]


class TestEmailRoleAddresses:
    """pii-email: a role mailbox is not a person's address."""

    @pytest.mark.parametrize("address", ROLE_ADDRESSES)
    def test_role_address_is_not_found(self, address: str):
        """The role words, compared whole and case-insensitively, are exempt."""
        assert not _hit(f"write to {address}", "pii-email")

    @pytest.mark.parametrize("address", NEAR_MISS_ADDRESSES)
    def test_near_miss_of_a_role_address_is_found(self, address: str):
        """A name that contains a role word, or a role word with a suffix, is a person."""
        assert _hit(f"write to {address}", "pii-email")

    def test_role_address_in_a_url_query_is_not_found(self):
        """The percent-encoded role address in a query string is also exempt."""
        assert not _hit("?to=support%40acmecorp.io", "pii-email")


class TestEmailOperandOfRemoteShell:
    """pii-email: only the host operand of a remote-shell command is exempt."""

    @pytest.mark.parametrize("text", REMOTE_SHELL_ADDRESSES)
    def test_address_near_a_remote_shell_word_is_found(self, text: str):
        """A remote-shell word in the sentence does not exempt the address."""
        assert _hit(text, "pii-email")

    def test_address_after_an_exempt_host_is_still_found(self):
        """The exemption covers the host only, not the address that follows it."""
        assert _hit("ssh deploy@bastion.acmecorp.io admin@acmecorp.io", "pii-email")

    @pytest.mark.parametrize("text", REMOTE_SHELL_HOSTS)
    def test_host_operand_is_not_found(self, text: str):
        """user@host as the operand of ssh, scp, rsync or sftp is a host."""
        assert not _hit(text, "pii-email")

    @pytest.mark.parametrize("text", SPACED_REMOTE_SHELL_HOSTS)
    def test_host_operand_with_any_spacing_is_not_found(self, text: str):
        """The operand check does not depend on how many spaces separate the command."""
        assert not _hit(text, "pii-email")

    def test_address_further_from_the_command_is_found(self):
        """An address that is far from the command word is a person."""
        text = f"ssh into the box, then write to {mixed(45)} ada@analytical-engines.org"
        assert _hit(text, "pii-email")


class TestEmailLocalPartAndDomain:
    """pii-email: the local part is bounded, and the domain is taken whole."""

    @pytest.mark.parametrize(
        "text",
        [
            "ada@analytical-engines.org",
            "Alice.Smith@Example.COM",
            "user+tag@analytical-engines.org",
            "user.name@sub.analytical-engines.co.uk",
            "a@b.co",
            "user_name%foo-bar@analytical-engines.org",
            "see<ada@analytical-engines.org>now",
            f"{mixed(64)}@analytical-engines.org",
        ],
    )
    def test_plain_and_bounded_forms_are_found(self, text: str):
        """Common address forms, and a local part at the RFC 5321 limit, are found."""
        assert _hit(text, "pii-email")

    def test_local_part_past_the_limit_is_not_found(self):
        """65 characters with no dot before the at sign are not read."""
        assert not _hit(f"{mixed(65)}@analytical-engines.org", "pii-email")

    def test_long_local_part_broken_by_dots_is_found(self):
        """The limit applies to one unbroken run, so dots restart it."""
        local = ".".join(mixed(20) for _ in range(10))
        assert _hit(f"{local}@analytical-engines.org", "pii-email")

    @pytest.mark.parametrize("text", ["user@localhost", "user@example", "user@[192.168.0.1]"])
    def test_domain_without_a_dot_is_not_found(self, text: str):
        """A domain without a dot was never matched."""
        assert not _hit(text, "pii-email")

    @pytest.mark.parametrize(
        ("text", "address"),
        [
            ("contact dmitri.ferreira@fernhill.co.", "dmitri.ferreira@fernhill.co"),
            ("write to ada@analytical-engines.org.", "ada@analytical-engines.org"),
            ("the mailbox is a@b.co.uk.", "a@b.co.uk"),
            ("see (ada@analytical-engines.org).", "ada@analytical-engines.org"),
        ],
    )
    def test_full_stop_after_an_address_is_not_part_of_it(self, text: str, address: str):
        """A sentence full stop or closing parenthesis is not in the matched address."""
        assert _value(text, "pii-email") == address

    def test_multi_label_domain_is_taken_whole(self):
        """The whole domain is matched, not just its first label."""
        assert _value("a@b.co.uk", "pii-email") == "a@b.co.uk"

    def test_domain_followed_by_a_dot_and_a_word_is_not_found(self):
        """A dot followed by a word is part of the name, so nothing is cut out."""
        assert not _hit("a@b.co.uk.example", "pii-email")

    @pytest.mark.parametrize(
        ("text", "address"),
        [
            (
                "GET /api/users?email=tomas.reyes%40quillfield.com&expand=orders",
                "tomas.reyes%40quillfield.com",
            ),
            ("?to=ada%40analytical-engines.org", "ada%40analytical-engines.org"),
            ("mailto:ada%40analytical-engines.org", "ada%40analytical-engines.org"),
        ],
    )
    def test_percent_encoded_at_sign_is_found(self, text: str, address: str):
        """An address whose at sign is written %40 is found with that spelling."""
        assert _value(text, "pii-email") == address

    def test_git_percent_40_repository_address_is_not_found(self):
        """A repository address written with %40 is not a person."""
        assert not _hit("clone git%40github.com:org/repo.git", "pii-email")


class TestEmailPackageSpecifiers:
    """pii-email: name@version in a lockfile is a package, not an address."""

    @pytest.mark.parametrize("text", PACKAGE_SPECIFIERS)
    def test_package_specifier_is_not_found(self, text: str):
        """A package name followed by a version is not an address."""
        assert not _hit(text, "pii-email")

    @pytest.mark.parametrize("text", PERSONAL_ADDRESSES)
    def test_address_with_a_real_domain_is_found(self, text: str):
        """A domain label that mixes digits and letters is still a domain, so the address is found."""
        assert _hit(text, "pii-email")


# ── pii-phone-us ──────────────────────────────────────────────────────────────


class TestUsPhoneRule:
    """pii-phone-us: area code and exchange decide whether the number is real."""

    @pytest.mark.parametrize(
        "text",
        ["call (415) 867-5309", "(212) 555-0300", "(415) 867-5309"],
        ids=["real-looking", "555-outside-fiction-range", "ordinary"],
    )
    def test_ordinary_number_is_found(self, text: str):
        """A number with an allocated area code is found."""
        assert _hit(text, "pii-phone-us")

    @pytest.mark.parametrize(
        "text",
        [
            "(555) 867-5309",
            "(212) 555-0100",
            "(212) 555-0199",
            "(212) 555-1212",
            "1-800-555-1212",
            "(800) 123-4567",
            "(888) 123-4567",
            "(833) 123-4567",
        ],
        ids=[
            "area-code-555",
            "fiction-line-0100",
            "fiction-line-0199",
            "directory-assistance",
            "toll-free-in-dashes",
            "toll-free-800",
            "toll-free-888",
            "toll-free-833",
        ],
    )
    def test_fictional_or_toll_free_number_is_not_found(self, text: str):
        """Fiction-range lines, directory assistance and toll-free codes are not found."""
        assert not _hit(text, "pii-phone-us")

    def test_line_0200_is_outside_the_fiction_range(self):
        """The fiction range ends at 0199, so line 0200 is found."""
        assert _hit("(212) 555-0200", "pii-phone-us")


# ── pii-mynumber-jp ───────────────────────────────────────────────────────────


class TestMyNumberRule:
    """pii-mynumber-jp: placeholder digits and AWS account IDs are not My Numbers."""

    @pytest.mark.parametrize(
        "text",
        ["000000000000", "checksum 000000000000 ok", "111111111111"],
        ids=["twelve-zeros", "zeros-in-sentence", "twelve-ones"],
    )
    def test_repeated_digits_are_not_found(self, text: str):
        """A run of one repeated digit is a placeholder."""
        assert not _hit(text, "pii-mynumber-jp")

    def test_real_my_number_is_found(self):
        """A real My Number with a valid check digit is found."""
        assert _hit("My Number: 123456789018", "pii-mynumber-jp")

    @pytest.mark.parametrize(
        "text",
        [
            "arn:aws:iam::835997400938:role/ci-deployer",
            "arn:aws:sns:us-east-1:835997400938:topic",
            "arn:aws-cn:s3:cn-north-1:835997400938:bucket",
        ],
        ids=["iam-role", "sns-topic", "partition-with-dash"],
    )
    def test_account_id_in_an_arn_is_not_found(self, text: str):
        """A twelve-digit account ID inside an ARN is not a My Number."""
        assert not _hit(text, "pii-mynumber-jp")

    def test_same_digits_outside_an_arn_are_found(self):
        """The exemption is for the ARN position, not for the digits."""
        assert _hit("account 835997400938 is ours", "pii-mynumber-jp")


# ── each rule catches its example ─────────────────────────────────────────────


class TestRuleExamples:
    """Each rule that v0.12.0 added or changed finds its example text."""

    @pytest.mark.parametrize(
        ("rule_id", "example"),
        [
            ("telegram-bot-token", f"12345678:AA{mixed(33)}"),
            ("connection-string", "postgres://admin:hunter2xyz@db.internal/app"),
            ("pii-email", "contact alice@analytical-engines.org"),
            ("pii-phone-us", "call 415-867-5309"),
            ("pii-mynumber-jp", "My Number: 123456789018"),
            ("aws-secret-near-key-id", f"{AWS_KEY}\nrIXf+Y0WDSn5Ctrn2zsvegYvJGWJWRXzhwOLfPg6"),
            ("docker-hub-pat", f"docker login -u chatalab -p {DOCKER_PAT}"),
            ("http-basic-auth", "Authorization: Basic c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM="),
            ("http-bearer-token", "Authorization: Bearer sk-live-OqDqQa5ZD5sRIkeC8wLtO9BS"),
            ("password-cli-flag", "mysql -h 10.2.0.14 -u report -p'V1m*23zdFGB' analytics"),
            ("password-netrc", "machine git.example-corp.net\n  login deploy\n  password V9N7z2!pjcRs"),
            ("password-prose", "my password is CKj3Jg%V8Yr if you want"),
            ("password-json-key", '"db.user": "malvarez", "db.pass": "i!kyBV7DyKVcqNoixmQ"'),
            ("redis-auth", "127.0.0.1:6379> AUTH hMDvD-xED3A"),
            ("pii-iban", "DE89 3704 0044 0532 0130 00"),
            ("pii-phone-intl", "+49 151 85280948"),
            ("pii-nino-uk", "NI: RE 62 91 56 A"),
        ],
    )
    def test_rule_finds_its_example(self, rule_id: str, example: str):
        """The rule reports its own id for the example from the v0.12.0 table."""
        assert _hit(example, rule_id)


# ── aws-secret-near-key-id ────────────────────────────────────────────────────

SECRET = "wJ4rXUt0nFEM9K7MDENG2bPxRfiCYqZ8nMLVAa3D"
KEY_ID = "AKIA32M9QY4UEBOPC69U"


class TestAwsSecretNearKeyId:
    """aws-secret-near-key-id: a forty-character secret beside an Access Key ID."""

    @pytest.mark.parametrize(
        "text",
        [
            f"{KEY_ID}\nrIXf+Y0WDSn5Ctrn2zsvegYvJGWJWRXzhwOLfPg6",
            f"svc-report-1,{KEY_ID},/yvGmsLPhuxamXfuIt4mxDZncVBbdM4d4bs6SuZc",
            f"access key {KEY_ID} secret {SECRET}",
            f"アクセスキーは {KEY_ID}、シークレットは {SECRET}",
            f"{KEY_ID}, {SECRET}",
        ],
        ids=["next-line", "csv-row", "same-line", "japanese-labels", "comma-and-space"],
    )
    def test_secret_beside_key_id_is_found(self, text: str):
        """A secret on the same line or the next line, with any label, is found."""
        assert _hit(text, "aws-secret-near-key-id")

    def test_finding_is_the_forty_characters_only(self):
        """The key ID and the label beside the secret are not part of the value."""
        assert _value(f"{KEY_ID},{SECRET}", "aws-secret-near-key-id") == SECRET

    def test_gap_of_eighty_characters_on_same_line_is_the_limit(self):
        """A gap of 80 spaces is found; 81 is not."""
        assert _hit(f"{KEY_ID}{' ' * 80}{SECRET}", "aws-secret-near-key-id")
        assert not _hit(f"{KEY_ID}{' ' * 81}{SECRET}", "aws-secret-near-key-id")

    def test_gap_of_eighty_characters_on_next_line_is_the_limit(self):
        """The same bound applies when the secret is on the next line."""
        assert _hit(f"{KEY_ID}\n{' ' * 80}{SECRET}", "aws-secret-near-key-id")
        assert not _hit(f"{KEY_ID}\n{' ' * 81}{SECRET}", "aws-secret-near-key-id")

    def test_forty_one_base64_characters_are_not_a_secret(self):
        """The secret is exactly forty characters; a forty-first is a longer run."""
        assert not _hit(f"{KEY_ID},{SECRET}Q", "aws-secret-near-key-id")

    @pytest.mark.parametrize(
        "text",
        [
            f"{KEY_ID},{'0123456789abcdef' * 2}abcdef01",
            f"{KEY_ID},wJ4rXUt0nFEM9K7MDENG2bPxRfiCYqZ8nMLVAa3",
            f"{KEY_ID},wJ4rXUt0nFEM9K7MDENG2bPxRfiCYqZ8nMLVA",
            f"{KEY_ID},WJ4RXUT0NFEM9K7MDENG2BPXRFICYQZ8NMLVAA3",
            f"{KEY_ID}{' ' * 3}{'x' * 40}",
        ],
        ids=[
            "lower-case-hex",
            "thirty-nine-characters",
            "no-digit",
            "no-lower-case",
            "one-letter-repeated",
        ],
    )
    def test_value_failing_the_secret_shape_is_not_found(self, text: str):
        """A value that is hex, too short, or lacks a digit or lower-case letter is refused."""
        assert not _hit(text, "aws-secret-near-key-id")

    def test_documented_key_id_ending_in_example_is_not_found(self):
        """AWS documentation uses a key ending in EXAMPLE, and its secret is documentation too."""
        assert not _hit(f"AKIA{'IOSFODNN7'}EXAMPLE\n{SECRET}", "aws-secret-near-key-id")

    def test_key_id_without_a_secret_beside_it_is_not_found(self):
        """A key ID with no secret near it is not a finding."""
        assert not _hit(f"{KEY_ID} is the key", "aws-secret-near-key-id")


# ── docker-hub-pat ────────────────────────────────────────────────────────────


class TestDockerHubPat:
    """docker-hub-pat: dckr_pat_ followed by exactly 27 characters or more."""

    @pytest.mark.parametrize(
        "text",
        [
            f"docker login -u chatalab -p {DOCKER_PAT}",
            f"dckr_pat_{mixed(27)}",
            f"dckr_pat_{mixed(40)}",
            f'"token": "dckr_pat_{mixed(27)}_x"',
        ],
        ids=["docker-login", "exactly-27", "longer-token", "json-with-underscore"],
    )
    def test_token_is_found(self, text: str):
        """A personal access token of 27 or more characters after the prefix is found."""
        assert _hit(text, "docker-hub-pat")

    @pytest.mark.parametrize(
        "text",
        [f"dckr_pat_{mixed(26)}", "dckr_pat_", f"mydckr_pat_{mixed(27)}"],
        ids=["one-under", "prefix-alone", "prefix-not-at-word-boundary"],
    )
    def test_short_or_misplaced_prefix_is_not_found(self, text: str):
        """One character under the length, the prefix alone, or a longer word is not a token."""
        assert not _hit(text, "docker-hub-pat")


# ── http-basic-auth ───────────────────────────────────────────────────────────


class TestHttpBasicAuthHeader:
    """http-basic-auth: a well-formed user:password pair in the Authorization header."""

    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Basic c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM=",
            "authorization: basic c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM=",
            '"Authorization": "Basic c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM="',
        ],
        ids=["header", "lower-case", "json"],
    )
    def test_credential_is_found(self, text: str):
        """The header in any case, or inside JSON, is found."""
        assert _hit(text, "http-basic-auth")

    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Basic dXNlcg==",
            "Authorization: Basic OnBhc3M=",
            "Authorization: Basic dXNlcjo=",
            "Authorization: Basic c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM",
            "Authorization: Basic /w==",
            "Authorization: Basic dQE6cA==",
        ],
        ids=[
            "no-colon",
            "empty-user",
            "empty-password",
            "missing-padding",
            "not-utf-8",
            "control-character",
        ],
    )
    def test_base64_that_is_not_a_user_password_pair_is_not_found(self, text: str):
        """Each refusal on its own: no colon, empty half, bad padding, bytes, control character."""
        assert not _hit(text, "http-basic-auth")

    @pytest.mark.parametrize(
        "text",
        ["Authorization: Basic dXNlcjpwYXNz", "Authorization: Basic YWRtaW46cGFzc3dvcmQ="],
        ids=["user-pass", "admin-password"],
    )
    def test_tutorial_example_is_not_found(self, text: str):
        """A pair that every tutorial uses, with both halves on the list, is an example."""
        assert not _hit(text, "http-basic-auth")

    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Basic YWRtaW46WGs5bVAycVI3dg==",
            "Authorization: Basic cm9vdDpYazltUDJxUjd2",
        ],
        ids=["admin-real-password", "root-real-password"],
    )
    def test_real_password_with_a_tutorial_user_is_found(self, text: str):
        """A real password under a common user name is still a credential."""
        assert _hit(text, "http-basic-auth")


# ── http-bearer-token ─────────────────────────────────────────────────────────


class TestHttpBearerToken:
    """http-bearer-token: a Bearer value of at least twenty characters with entropy."""

    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer sk-live-OqDqQa5ZD5sRIkeC8wLtO9BS",
            "authorization: bearer sk-live-OqDqQa5ZD5sRIkeC8wLtO9BS",
            '"Authorization": "Bearer sk-live-OqDqQa5ZD5sRIkeC8wLtO9BS"',
            f"Authorization: Bearer {mixed(20)}",
        ],
        ids=["header", "lower-case", "json", "minimum-length-twenty"],
    )
    def test_token_is_found(self, text: str):
        """A bearer value of the minimum length, in any case or inside JSON, is found."""
        assert _hit(text, "http-bearer-token")

    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer $TOKEN",
            "Authorization: Bearer <token>",
            "Authorization: Bearer ${API_KEY}",
            "Authorization: Bearer abcdefghijklmnopqrs",
            "Authorization: Bearer abababababababababababab",
            "Authorization: Bearer aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ],
        ids=[
            "shell-reference",
            "slot",
            "template-reference",
            "nineteen-characters",
            "no-entropy",
            "run-of-one-character",
        ],
    )
    def test_placeholder_or_low_entropy_value_is_not_found(self, text: str):
        """A reference, a slot, a short value or a value with no entropy is not a token."""
        assert not _hit(text, "http-bearer-token")


# ── password-cli-flag ─────────────────────────────────────────────────────────


class TestPasswordCliFlag:
    """password-cli-flag: a password passed on a command line."""

    @pytest.mark.parametrize(
        "text",
        [
            "mysql -h 10.2.0.14 -u report -p'V1m*23zdFGB' analytics",
            'mysql -h 10.2.0.14 -u report -p"V1m*23zdFGB" analytics',
            "pg_dump --host db1 --username app --password 'S4Kv&DEu0GBmb' prod",
            'sync.exe --password "jQMhIKJNlyJl@u#Ulf" --user fwagner',
            "tool --password=S4Kv&DEu0GBmb",
            "tool --passwd hunter2!x",
            "tool --pass=Xk9mP2qR7v",
            "(mysql -p'V1m*23zdFGB')",
            '"cmd": "mysql -p\'V1m*23zdFGB\' db"',
        ],
        ids=[
            "mysql-single-quotes",
            "mysql-double-quotes",
            "password-quoted-with-space",
            "password-double-quotes",
            "password-equals-unquoted",
            "passwd-unquoted",
            "pass-equals",
            "inside-parentheses",
            "inside-json-string",
        ],
    )
    def test_password_on_the_command_line_is_found(self, text: str):
        """The value after the flag is a password in each quoting and grouping."""
        assert _hit(text, "password-cli-flag")

    @pytest.mark.parametrize(
        "text",
        [
            "pg_dump --password prod",
            "tool --password-stdin",
            "tool --password=abc1!",
            "ssh -p 22 host",
            "mysql -p analytics",
            "tool --password",
            "tool --password 'my pass1'",
        ],
        ids=[
            "word-not-password",
            "flag-names-source",
            "five-characters",
            "port",
            "mysql-unquoted",
            "flag-without-value",
            "value-with-space",
        ],
    )
    def test_flag_without_a_password_value_is_not_found(self, text: str):
        """A word, a port, a flag that names the source, or a short value is not a password."""
        assert not _hit(text, "password-cli-flag")

    def test_value_stops_before_a_trailing_full_stop(self):
        """The full stop that ends the sentence is not part of the password."""
        assert _value("run tool --password Xk9mP2qR7v.", "password-cli-flag") == "Xk9mP2qR7v"

    @pytest.mark.parametrize(
        "text",
        [
            '--password=os.environ.get("DB_PASSWORD")',
            "--password getpass()",
        ],
        ids=["call-on-value", "call-with-no-argument"],
    )
    def test_code_on_the_command_line_is_not_a_password(self, text: str):
        """A call in place of the value is code, not a password."""
        assert not _hit(text, "password-cli-flag")


# ── password-netrc ────────────────────────────────────────────────────────────


class TestPasswordNetrc:
    """password-netrc: a password line in a .netrc file or a vault listing."""

    @pytest.mark.parametrize(
        "text",
        [
            "machine git.example-corp.net\n  login deploy\n  password V9N7z2!pjcRs",
            "password V9N7z2!pjcRs",
            "username       meilin.zhou\npassword       Salt_lVi9UMB45",
            "\tpasswd\tSalt_lVi9UMB45",
        ],
        ids=["indented-netrc", "start-of-text", "vault-listing", "tab-indented-passwd"],
    )
    def test_password_line_is_found(self, text: str):
        """A password at the start of a line, indented or in a listing, is found."""
        assert _hit(text, "password-netrc")

    @pytest.mark.parametrize(
        "text",
        [
            "the password V9N7z2!pjcRs is old",
            "password required",
            "password_hint V9N7z2!pjcRs",
        ],
        ids=["word-mid-line", "no-digit-or-symbol", "name-not-the-word"],
    )
    def test_password_word_mid_line_or_without_a_value_is_not_found(self, text: str):
        """The word must start the line, and the value needs a digit or a symbol."""
        assert not _hit(text, "password-netrc")

    def test_password_line_that_is_a_call_is_not_a_password(self):
        """A netrc-style line whose value is a call is code."""
        assert not _hit('  password os.environ.get("DB_PASSWORD")', "password-netrc")


# ── password-prose ────────────────────────────────────────────────────────────


class TestPasswordProse:
    """password-prose: a password written after a label or in a sentence."""

    @pytest.mark.parametrize(
        "text",
        [
            "my password is CKj3Jg%V8Yr if you want",
            "Temporary password is Kessler?dlbRswI01j24 -- change it",
            "pw: hunter2!x",
            'password = "Xk9mP2qR7v"',
            "PASSWORD: Xk9mP2qR7v!",
            "パスワードは Xk9mP2qR7v です",
            "パスワード：Xk9mP2qR7v",
            "パスワード: Xk9mP2qR7v",
            "パスワードは Xk9mP2qR7v で届きます",
        ],
        ids=[
            "password-is",
            "password-is-capitalised",
            "pw-colon",
            "password-equals-quoted",
            "upper-case-keyword",
            "japanese-wa-desu",
            "japanese-full-width-colon",
            "japanese-colon",
            "japanese-stops-before-de",
        ],
    )
    def test_password_in_a_sentence_is_found(self, text: str):
        """A value written out after the label is a password, in any of these forms."""
        assert _hit(text, "password-prose")

    @pytest.mark.parametrize(
        "text",
        [
            "password is required",
            "password is incorrect.",
            "the password was reset",
            "passwords are stored elsewhere",
            "password: 123456",
            "password = None",
            "パスワードは設定済みです",
            "パスワードを忘れました",
        ],
        ids=[
            "required",
            "incorrect",
            "reset",
            "longer-word",
            "digits-only",
            "none",
            "japanese-no-latin-letters",
            "japanese-no-separator",
        ],
    )
    def test_word_without_a_digit_or_symbol_is_not_found(self, text: str):
        """A word with no digit or symbol, or no separator after the label, is not a password."""
        assert not _hit(text, "password-prose")

    @pytest.mark.parametrize(
        "text",
        [
            "password = hash_password(raw)",
            'password = getpass("Enter password: ")',
            'password=os.environ.get("DB_PASSWORD")',
            "pwd = os.getcwd()",
            "password: z.string().min(8),",
            "pw = self.pw_hash(salt+raw)",
            "password = new_password",
            "password: form.password2",
            "password: string | null;",
            "if password == confirm_password:",
        ],
        ids=[
            "call",
            "call-with-prompt",
            "attribute-call",
            "pwd-from-call",
            "schema-call",
            "method-call",
            "bare-identifier",
            "dotted-identifier",
            "type-annotation",
            "comparison",
        ],
    )
    def test_code_that_names_a_password_is_not_found(self, text: str):
        """A call, an identifier, or a type after the label is code, not a password value."""
        assert not _hit(text, "password-prose")

    @pytest.mark.parametrize(
        "text",
        [
            "password: :ContentChecker::MAX_LENGTH",
            "password = @user_data",
            "passwd = same_pair.group(2).lower()",
            "password = 4.times.map",
            "password: String!",
            "the `password is required` check",
        ],
        ids=[
            "ruby-symbol",
            "ruby-instance-variable",
            "method-call-on-name",
            "method-chain-from-number",
            "graphql-type",
            "backticked-phrase",
        ],
    )
    def test_code_shape_after_the_label_is_not_a_password(self, text: str):
        """A Ruby symbol, a method chain, a GraphQL type or a backticked phrase is not a value."""
        assert not _hit(text, "password-prose")

    @pytest.mark.parametrize(
        "text",
        [
            'password = "Summer2024!"',
            'password = "hunter2"',
            'password = "Summer(2024)"',
            "my password is Summer2024",
            "password: hunter2!x",
            "パスワードは Abc123!x です",
        ],
        ids=[
            "quoted-after-equals",
            "quoted-identifier-shaped",
            "quoted-with-parentheses",
            "is-form-identifier-shaped",
            "symbol-makes-a-value",
            "japanese-form",
        ],
    )
    def test_quoted_or_symbol_value_is_a_password(self, text: str):
        """A value in quotes, or with a symbol, is written out and so is found."""
        assert _hit(text, "password-prose")

    def test_unquoted_identifier_after_a_colon_is_not_found(self):
        """Without quotes or a symbol, an identifier after the colon reads as code."""
        assert not _hit("password: hunter2", "password-prose")

    def test_password_with_a_digit_before_its_bang_is_found(self):
        """A type name is letters only, so a digit and a bang keep the value."""
        assert _hit("password: Summer2024!", "password-prose")


# ── password-json-key ─────────────────────────────────────────────────────────


class TestPasswordJsonKey:
    """password-json-key: a value under a key that ends in pass, passwd, password or pwd."""

    @pytest.mark.parametrize(
        "text",
        [
            '"db.user": "malvarez", "db.pass": "i!kyBV7DyKVcqNoixmQ"',
            '{"user_password": "Xk9mP2qR7vL4"}',
            '{"userPassword": "Xk9mP2qR7vL4"}',
            '{"PASSWORD": "Xk9mP2qR7vL4"}',
            '{"dbPass": "Xk9mP2qR7vL4"}',
            '{"passwd": "Xk9mP2qR7vL4"}',
            '{"pwd": "Xk9mP2qR7vL4"}',
        ],
        ids=[
            "db-pass",
            "user-password",
            "camel-case",
            "upper-case",
            "camel-case-pass",
            "passwd",
            "pwd",
        ],
    )
    def test_value_under_a_password_key_is_found(self, text: str):
        """The last word of the key decides, and the four suffixes all count."""
        assert _hit(text, "password-json-key")

    @pytest.mark.parametrize(
        "text",
        [
            '{"bypass": "Xk9mP2qR7vL4"}',
            '{"password_hint": "Xk9mP2qR7vL4"}',
            '{"passwordless": "Xk9mP2qR7vL4"}',
            '{"db.pass": "12345"}',
        ],
        ids=["bypass-is-one-word", "key-ends-in-hint", "key-ends-in-less", "no-letter-in-value"],
    )
    def test_key_that_only_contains_the_suffix_is_not_found(self, text: str):
        """A key whose last word is not one of the four suffixes is not a password key."""
        assert not _hit(text, "password-json-key")


# ── redis-auth ────────────────────────────────────────────────────────────────


class TestRedisAuth:
    """redis-auth: the AUTH command and redis-cli -a."""

    @pytest.mark.parametrize(
        "text",
        [
            "127.0.0.1:6379> AUTH hMDvD-xED3A",
            '"AUTH" "Ember%0eWzV6g4lU59"',
            "AUTH hMDvD-xED3A",
            "redis-cli -h 10.0.0.5 -a hMDvD-xED3A ping",
            'AUTH "hMDvD-xED3A"',
        ],
        ids=["monitor-prompt", "quoted-monitor-line", "start-of-text", "redis-cli-a", "quoted-value"],
    )
    def test_auth_with_a_value_is_found(self, text: str):
        """The AUTH command in upper case, or redis-cli -a, with a value is found."""
        assert _hit(text, "redis-auth")

    @pytest.mark.parametrize(
        "text",
        [
            "auth hMDvD-xED3A",
            "AUTH required",
            "redis-cli -h 10.0.0.5 ping",
            "AUTHORIZED hMDvD-xED3A",
        ],
        ids=["lower-case", "word-not-password", "no-a-flag", "longer-word"],
    )
    def test_other_forms_are_not_found(self, text: str):
        """Lower case, a non-password word, no -a flag, or a longer word is not AUTH."""
        assert not _hit(text, "redis-auth")


# ── pii-iban ──────────────────────────────────────────────────────────────────


class TestIbanRule:
    """pii-iban: an IBAN with a valid check and the length for its country."""

    @pytest.mark.parametrize(
        "text",
        [
            "DE89 3704 0044 0532 0130 00",
            "GB82WEST12345698765432",
            "DE89370400440532013000",
            "IBAN: GB82 WEST 1234 5698 7654 32.",
            "(DE89 3704 0044 0532 0130 00)",
            '{"iban": "DE89370400440532013000"}',
            "IL17 0108 0000 0001 2612 345",
        ],
        ids=[
            "german-grouped",
            "british-unspaced",
            "german-unspaced",
            "after-label-before-full-stop",
            "inside-parentheses",
            "inside-json",
            "country-not-in-length-table",
        ],
    )
    def test_iban_is_found(self, text: str):
        """A valid IBAN is found in any grouping or surrounding punctuation."""
        assert _hit(text, "pii-iban")

    @pytest.mark.parametrize(
        "text",
        [
            "DE88 3704 0044 0532 0130 00",
            "DE89 3704 0044 0532 0130 0",
            "DE89 3704 0044 0532 0130 001",
            "AA12 3456 7890 1234 5678",
            "XX82WEST12345698765432",
        ],
        ids=["check-digit-changed", "one-short", "one-long", "no-check-passes", "not-a-country"],
    )
    def test_invalid_iban_is_not_found(self, text: str):
        """A changed check digit, a wrong length, or an unknown country is not an IBAN."""
        assert not _hit(text, "pii-iban")


# ── pii-phone-intl ────────────────────────────────────────────────────────────


class TestIntlPhoneRule:
    """pii-phone-intl: a plus, a country code and nine to fifteen digits."""

    @pytest.mark.parametrize(
        "text",
        [
            "+49 151 85280948",
            "+33 6 22 81 96 59",
            "+55 11 98206-4442",
            "+91 90848 57208",
            "+61 477 715 965",
            "+81 90-9384-1967",
            "+44 7801 997035",
            "+52 55 4268 5070",
            "+4915112345678",
            "call +49 151 85280948 now",
            "(+49 151 85280948)",
            '"phone": "+49.151.85280948"',
        ],
        ids=[
            "german",
            "french",
            "brazilian-hyphen",
            "indian",
            "australian",
            "japanese-hyphens",
            "british",
            "mexican",
            "unspaced",
            "inside-sentence",
            "inside-parentheses",
            "dotted-inside-json",
        ],
    )
    def test_international_number_is_found(self, text: str):
        """A plus, a country code and the digits are found in any spacing or context."""
        assert _hit(text, "pii-phone-intl")

    @pytest.mark.parametrize(
        "text",
        [
            "+0000",
            "+2026-10-09",
            "+12.5%",
            "version +3.2.1",
            "a+44 7801 997035",
            "1+44 7801 997035",
            "diff: + 123",
            "+11 111 111 111",
        ],
        ids=[
            "time-zone",
            "date-eight-digits",
            "percent-leading-one",
            "version",
            "letter-before-plus",
            "digit-before-plus",
            "space-after-plus",
            "every-digit-same",
        ],
    )
    def test_non_number_with_a_plus_is_not_found(self, text: str):
        """A time zone, a date, a version, or a repeated digit is not a phone number."""
        assert not _hit(text, "pii-phone-intl")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("+4 1234567", False),
            ("+4 12345678", True),
            ("+49 123456789012", True),
            ("+49 1234567890123", True),
            ("+49 12345678901234", False),
        ],
        ids=["eight-digits", "nine-digits", "fourteen-digits", "fifteen-digits", "sixteen-digits"],
    )
    def test_digit_count_boundary(self, text: str, expected: bool):
        """The digit count is nine to fifteen, counted with the country code."""
        assert _hit(text, "pii-phone-intl") is expected


# ── pii-nino-uk ───────────────────────────────────────────────────────────────


class TestNinoUkRule:
    """pii-nino-uk: the spaced National Insurance number with a suffix from A to D."""

    @pytest.mark.parametrize(
        "text",
        ["NI: RE 62 91 56 A", "CS 45 74 46 B", "AB 12 34 56 C", "(CE 12 34 56 D)"],
        ids=["RE-suffix-A", "CS-suffix-B", "AB-suffix-C", "suffix-D-in-parentheses"],
    )
    def test_spaced_number_is_found(self, text: str):
        """The spaced form with a suffix from A to D is found."""
        assert _hit(text, "pii-nino-uk")

    @pytest.mark.parametrize(
        "text",
        [
            "DA 12 34 56 A",
            "FA 12 34 56 A",
            "IA 12 34 56 A",
            "QA 12 34 56 A",
            "UA 12 34 56 A",
            "VA 12 34 56 A",
            "RD 12 34 56 A",
            "RO 12 34 56 A",
            "BG 12 34 56 A",
            "GB 12 34 56 A",
            "KN 12 34 56 A",
            "NK 12 34 56 A",
            "NT 12 34 56 A",
            "TN 12 34 56 A",
            "ZZ 12 34 56 A",
            "RE 62 91 56 E",
            "RE 62 91 56",
        ],
        ids=[
            "first-D",
            "first-F",
            "first-I",
            "first-Q",
            "first-U",
            "first-V",
            "second-D",
            "second-O",
            "prefix-BG",
            "prefix-GB",
            "prefix-KN",
            "prefix-NK",
            "prefix-NT",
            "prefix-TN",
            "prefix-ZZ",
            "suffix-E",
            "no-suffix",
        ],
    )
    def test_reserved_letter_prefix_or_bad_suffix_is_not_found(self, text: str):
        """The reserved letters, reserved pairs and a suffix outside A to D are refused."""
        assert not _hit(text, "pii-nino-uk")

    def test_unspaced_form_is_not_found(self):
        """Without spaces the letters and digits are too common to read as a number."""
        assert not _hit("RE629156A", "pii-nino-uk")


# ── credit card ───────────────────────────────────────────────────────────────


class TestCreditCardBranches:
    """pii-credit-card: every issuer branch is found, and 659 is not Discover."""

    @pytest.mark.parametrize(
        "number",
        [
            "36123456789013",
            "6011000991300009",
            "6450000000000002",
            "3530111333300000",
            "2221000000000009",
            "6221260000000000",
            "8171990000000008",
        ],
        ids=[
            "diners-14",
            "discover-6011",
            "discover-644-658",
            "jcb-3528-3589",
            "mastercard-2-series",
            "unionpay-62",
            "unionpay-81",
        ],
    )
    def test_card_of_each_branch_is_found(self, number: str):
        """Each card branch, Luhn-valid, is found."""
        assert _hit(f"card {number}", "pii-credit-card")

    def test_659_is_not_a_discover_range(self):
        """Discover's range stops at 658, so 659 is not a card here."""
        assert not _hit("card 6591111111111116", "pii-credit-card")


# ── published test card numbers ───────────────────────────────────────────────

PUBLISHED_TEST_CARDS = [
    "4242424242424242",
    "4000000000000002",
    "4000056655665556",
    "5555555555554444",
    "2223003122003222",
    "5200828282828210",
    "5105105105105100",
    "378282246310005",
    "371449635398431",
    "6011111111111117",
    "36227206271667",
    "6200000000000005",
    "4111111111111111",
    "4012888888881881",
]


class TestPublishedTestCards:
    """The card numbers Stripe publishes are test data, not real cards."""

    @pytest.mark.parametrize("number", PUBLISHED_TEST_CARDS)
    def test_published_test_card_is_not_found_by_the_rule(self, number: str):
        """A published test number, Luhn-valid, is not reported."""
        assert not _hit(f"card {number}", "pii-credit-card")

    def test_luhn_valid_number_that_is_not_published_is_found(self):
        """A card that is not on the list is still a card."""
        assert _hit("card 2221000000000009", "pii-credit-card")

    @pytest.mark.parametrize(
        "number",
        ["4242424242424242", "4000000000000002", "2223003122003222", "36227206271667", "6200000000000005"],
    )
    def test_validator_refuses_the_published_test_numbers(self, number: str):
        """The validator treats the four 2025 additions and the older list as test data."""
        assert is_real_card_number(number) is False

    def test_luhn_valid_number_outside_the_list_is_real(self):
        """A Luhn-valid number outside the list is a real card."""
        assert is_real_card_number("2221000000000009") is True

    def test_list_is_compared_on_digits_not_as_printed(self):
        """The printed spacing does not get a test card past the list."""
        assert is_real_card_number("4000 0000 0000 0002") is False


# ── every shape that is a placeholder ─────────────────────────────────────────


class TestPlaceholderLines:
    """A value written to be replaced is not a secret, and a real one is still one."""

    @pytest.mark.parametrize(
        "line",
        [
            "DB_PASSWORD=your-password-here",
            "API_TOKEN=REPLACE_ME_WITH_REAL",
            "GITHUB_TOKEN=your_token_here",
            "SECRET_KEY=django-insecure-CHANGE-THIS-IN-PRODUCTION",
            "DATABASE_URL=postgres://user:password@localhost:5432/db",
            "TOKEN=<your-token>",
            "API_KEY=${SOME_OTHER_VAR}",
            "CLIENT_SECRET=xxxxxxxxxxxxxxxx",
        ],
    )
    def test_placeholder_line_produces_no_finding(self, line: str):
        """A value written for replacement gives no finding."""
        assert scan(line) == []

    @pytest.mark.parametrize(
        "line",
        [
            "API_KEY=access-refresh-client-auth",
            "SECRET=production-staging-local",
            "TOKEN=api-key-token-secret",
        ],
    )
    def test_words_without_a_marker_are_not_a_placeholder(self, line: str):
        """Ordinary words around secrets, with no marker, are not a placeholder."""
        assert len(scan(line)) > 0

    @pytest.mark.parametrize(
        "line",
        [
            f"key={AWS_KEY}",
            "aws_secret_access_key = wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
            "POSTGRES_PASSWORD: Sup3rS3cretDbPassw0rd",
            "export GITHUB_TOKEN=ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
            "DB_PASSWORD=hunter2xyzabc",
            "postgres://admin:s3cr3tP4ss@db.corp.internal/app",
        ],
    )
    def test_documented_or_real_value_is_still_a_finding(self, line: str):
        """The list is short of example on purpose, so an AWS key ending in it is still a key."""
        assert len(scan(line)) > 0

    @pytest.mark.parametrize(
        "text",
        ["contact todo@analytical-engines.org", "write to placeholder@analytical-engines.org"],
    )
    def test_marker_word_in_an_address_is_still_an_address(self, text: str):
        """Only secret rules consult the marker list, so a mailbox is still an address."""
        assert _hit(text, "pii-email")


class TestConnectionStringPlaceholders:
    """A connection string with no real credential is a placeholder."""

    @pytest.mark.parametrize(
        "line",
        [
            "postgres://user:password@localhost:5432/appdb",
            "postgres://postgres:postgres@localhost:5432/appdb",
            "postgres://root:root@127.0.0.1:5432/test",
            "postgres://${PGUSER}:${PGPASSWORD}@${PGHOST}/${PGDATABASE}",
            "postgres://$DB_USER:$DB_PASS@$DB_HOST/app",
            '"postgres://{}:{}@{}:{}/{}"',
        ],
    )
    def test_unfilled_connection_string_is_not_a_finding(self, line: str):
        """A connection string with a placeholder credential is not reported."""
        assert not _hit(line, "connection-string")

    @pytest.mark.parametrize(
        "line",
        [
            "postgres://admin:s3cr3tP4ss@db.corp.internal/app",
            "mysql://root:secret@localhost",
            "postgres://user:password@prod.corp.internal/app",
        ],
    )
    def test_filled_connection_string_is_found(self, line: str):
        """A real host with a credential is found, even when the password is common."""
        assert _hit(line, "connection-string")


class TestSecretNamesAndValues:
    """Values that are names, paths or URLs for a secret are not the secret."""

    @pytest.mark.parametrize(
        "line",
        [
            "TOKEN_ENDPOINT=https://login.microsoftonline.com/common/oauth2/v2.0/token",
            '"token_url": "https://sts.googleapis.com/v1/token"',
            '"subject_token_type": "urn:ietf:params:oauth:token-type:jwt"',
            'secret_name = "BACKEND_SERVICE_API_KEY"',
            'secret_id   = "PROJECT_DB_PASSWORD"',
            "TOKEN_HEADER_NAME=X-Auth-Token",
            "VAULT_TOKEN_PATH=/var/run/secrets/vault/token",
            "AWS_WEB_IDENTITY_TOKEN_FILE=/var/run/secrets/token",
            "PASSWORD_MIN_LENGTH=12345678",
            "SECRET_MANAGER_PROJECT=my-company-production",
            'const TOKEN_STORAGE_KEY = "app.auth.token.v2";',
            "JWT_SECRET_ISSUER=https://auth.corp.example",
        ],
    )
    def test_name_path_or_url_is_not_a_secret(self, line: str):
        """A public endpoint, a URN, a name, a header, a path or an issuer gives no finding."""
        assert scan(line) == []

    @pytest.mark.parametrize(
        ("line", "rule_id"),
        [
            (
                "/".join(["https://hooks.slack.com", "services", "T00000000", "B00000000", "X" * 24]),
                "slack-webhook",
            ),
            (
                "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
                "env-assignment",
            ),
            ("GITHUB_TOKEN=ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "github-pat"),
            ("DB_PASSWORD=Sup3rS3cretDbPassw0rd", "env-assignment"),
        ],
    )
    def test_rule_with_a_fixed_prefix_is_still_found(self, line: str, rule_id: str):
        """A rule with a fixed prefix has already said what it found, so the shape test does not apply."""
        assert _hit(line, rule_id)


# ── isNotSecretShaped ─────────────────────────────────────────────────────────


class TestIsNotSecretShapedReferences:
    """A dotted name is a reference to a value, not a value."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "process.env.API_KEY",
            "user.password_digest",
            "response.data.accessToken",
            "this.options.tokenizer",
            "config.database.connectionString",
            "settings.oauth2ClientSecretIdentifier",
            "value.toLowerCase",
            "a.b",
        ],
    )
    def test_dotted_name_is_a_reference(self, identifier: str):
        """Words joined by dots, with long camel-case words, are a reference."""
        assert is_not_secret_shaped(identifier) is True

    @pytest.mark.parametrize(
        "value",
        [
            ".".join(["tok", _random(28), _random(28)]),
            ".".join(["k", _random(24)]),
            ".".join([_random(30), _random(20), _random(22), _random(40), _random(22)]),
        ],
        ids=["dotted-api-token", "two-part-secret", "five-parts"],
    )
    def test_dotted_credential_is_not_a_reference(self, value: str):
        """A token that changes case or slips in a digit every character or two is not a name."""
        assert is_not_secret_shaped(value) is False

    def test_mean_word_length_separates_a_name_from_a_token(self):
        """Words longer than the threshold read as a name; shorter ones read as a token."""
        assert is_not_secret_shaped(f"config.{_words_of(3, 4)}") is True
        assert is_not_secret_shaped(f"config.{_words_of(2, 6)}") is False


class TestIsNotSecretShapedPath:
    """A value that begins with a slash is a path only with a second slash or a home or working prefix."""

    @pytest.mark.parametrize(
        "value",
        ["/etc/ssl/key", "~/.ssh/id_ed25519", "./config/key"],
        ids=["absolute-with-two-separators", "from-home", "from-working-directory"],
    )
    def test_path_is_not_a_secret(self, value: str):
        """An absolute path with a second slash, or a home or working prefix, is a path."""
        assert is_not_secret_shaped(value) is True

    def test_forty_character_secret_beginning_with_slash_is_not_a_path(self):
        """A base64 value that begins with a slash is a secret."""
        assert is_not_secret_shaped("/yvGmsLPhuxamXfuIt4mxDZncVBbdM4d4bs6SuZc") is False


# ── isFillerBody and isPlaceholder ────────────────────────────────────────────


class TestFillerPlaceholders:
    """A value built from a run of one character after a short prefix is a placeholder."""

    @pytest.mark.parametrize(
        "value",
        [
            "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "glpat-XXXXXXXXXXXXXXXXXXXX",
            "_".join(["xoxb", "1" * 12, "2" * 12, "X" * 24]),
            "AC00000000000000000000000000000000",
        ],
        ids=["github-x-run", "gitlab-X-run", "slack-three-runs", "twilio-zeros"],
    )
    def test_filler_token_is_a_placeholder(self, value: str):
        """Every part is a run, and at least one run is eight or more characters."""
        assert is_placeholder(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "ghp_E64NXJeC2PsmvG1MkSioJyHDrW5vRGlMfaix",
            f"ghp_{mixed(10)}aaaaaa{mixed(20)}",
            f"ghp_kkkkkkk{mixed(29)}",
            f"glpat-{mixed(8)}1111{mixed(8)}",
            "_".join(["sk", "live", "a" * 26]),
        ],
        ids=[
            "random-github-token",
            "run-of-six-inside",
            "run-of-seven-at-start",
            "run-of-four-inside",
            "prefix-word-after-first-part",
        ],
    )
    def test_random_value_with_a_short_run_is_not_a_placeholder(self, value: str):
        """A run shorter than eight, or a random tail with a run inside it, is not filler."""
        assert is_placeholder(value) is False

    def test_run_of_seven_is_not_enough_on_its_own(self):
        """Seven is short of the eight-character run, and eight is enough."""
        assert is_filler_body("ghp_xxxxxxx") is False
        assert is_filler_body("ghp_xxxxxxxx") is True

    def test_letter_prefix_is_at_most_four_letters(self):
        """The letter prefix in the run part may be four letters, not six."""
        assert is_filler_body("ABCDxxxxxxxx") is True
        assert is_filler_body("ABCDEFxxxxxxxx") is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("abcdefgh-XXXXXXXXXXXX", True),
            ("abcdefghi-XXXXXXXXXXXX", False),
            ("token-value-XXXXXXXXXXXX", False),
        ],
        ids=["eight-letter-word", "nine-letter-word", "two-words"],
    )
    def test_first_part_may_be_a_bare_word_of_up_to_eight_letters(self, value: str, expected: bool):
        """Only the first part may be a bare word, and it is at most eight letters."""
        assert is_filler_body(value) is expected

    def test_empty_value_is_not_filler(self):
        """An empty value has no run, so it is not filler."""
        assert is_filler_body("") is False


class TestPlaceholderWords:
    """The marker and filler words that make a value a placeholder."""

    @pytest.mark.parametrize(
        "value",
        [
            "mock-token-value",
            "stub-secret",
            "tests-only-key",
            "fake-token-for-unit-tests",
            "use-only-fake-key",
        ],
        ids=["mock", "stub", "tests", "unit-with-tests", "use-and-only"],
    )
    def test_marker_and_filler_words_make_a_placeholder(self, value: str):
        """mock, stub and tests are markers; unit, only and use are filler."""
        assert is_placeholder(value) is True

    @pytest.mark.parametrize(
        "value",
        ["mock9F3x8KqLp2Vn7Ws", "Xk9mock2qR7vL4nW1s", "unit2Xk9mP2qR7vL4n"],
        ids=["mock-in-random-part", "mock-in-random-value", "unit-before-random-part"],
    )
    def test_marker_inside_a_random_part_is_not_a_marker(self, value: str):
        """The marker has to be a whole part, not a word that contains it."""
        assert is_placeholder(value) is False

    def test_filler_words_alone_are_not_a_marker(self):
        """Filler words with no marker do not make a placeholder."""
        assert is_placeholder("unit-only-use") is False


# ── documented example values ─────────────────────────────────────────────────

AWS_EXAMPLE = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
JWT_EXAMPLE = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


class TestDocumentedExamples:
    """Vendor documentation values are placeholders only when matched whole."""

    @pytest.mark.parametrize(
        "value",
        [
            AWS_EXAMPLE,
            f'"{AWS_EXAMPLE}"',
            f"'{AWS_EXAMPLE}'",
            STRIPE_DOCS_TEST_KEY,
            JWT_EXAMPLE,
        ],
        ids=["aws-secret", "aws-double-quotes", "aws-single-quotes", "stripe-test-key", "jwt-io-token"],
    )
    def test_documented_example_is_a_placeholder(self, value: str):
        """The whole example, with or without quotes, is a placeholder."""
        assert is_placeholder(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            f"{AWS_EXAMPLE}X",
            f"A{AWS_EXAMPLE}",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE",
            f"{STRIPE_DOCS_TEST_KEY}A",
            f"sk_live_{STRIPE_DOCS_TEST_KEY[8:]}",
            f"{JWT_EXAMPLE}.x",
        ],
        ids=[
            "one-past-aws",
            "one-before-aws",
            "aws-cut-short",
            "stripe-with-character-added",
            "stripe-body-under-live-prefix",
            "jwt-with-segment-added",
        ],
    )
    def test_changed_example_is_not_a_placeholder(self, value: str):
        """A real value that begins with an example, or the example cut short, is a value."""
        assert is_placeholder(value) is False

    def test_secret_with_aws_example_as_prefix_is_still_a_secret(self):
        """An assignment whose value extends the AWS example is still an env-assignment finding."""
        assert _hit(f"AWS_SECRET_ACCESS_KEY={AWS_EXAMPLE}X", "env-assignment")

    def test_aws_example_in_its_own_assignment_is_not_an_env_assignment_finding(self):
        """The AWS documentation line is not reported as an assignment."""
        assert not _hit(f"aws_secret_access_key = {AWS_EXAMPLE}", "env-assignment")


class TestFakeTokenConstants:
    """A constant that holds a fake token in a test is not flagged; a real value is."""

    def test_fake_token_constant_is_not_flagged(self):
        """The value is filler, so the constant name does not make it a secret."""
        assert not _hit("const password = 'fake-token-for-unit-tests'", "env-assignment")

    def test_real_value_under_the_same_name_is_flagged(self):
        """A real value under the same name is still an env-assignment finding."""
        assert _hit("const password = 'Xk9mP2qR7vL4nW1s'", "env-assignment")


class TestQuotedPlaceholders:
    """Quotes around a value do not change whether it is a placeholder."""

    @pytest.mark.parametrize(
        "value",
        [
            "'fake-token-for-unit-tests'",
            '"fake-token-for-unit-tests"',
            '"ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"',
        ],
        ids=["single-quotes", "double-quotes", "filler-run-in-quotes"],
    )
    def test_quoted_placeholder_is_a_placeholder(self, value: str):
        """The check reads the value inside the quotes."""
        assert is_placeholder(value) is True

    def test_quoted_real_value_is_not_a_placeholder(self):
        """A real value in quotes is not a placeholder."""
        assert is_placeholder('"Xk9mP2qR7vL4nW1s"') is False


# ── isPasswordLike ────────────────────────────────────────────────────────────


class TestIsPasswordLike:
    """A password is six to 128 characters with a letter and a digit or symbol."""

    @pytest.mark.parametrize(
        "value",
        ["hunter2!x", "V1m*23zdFGB", "S4Kv&DEu0GBmb", "Salt_lVi9UMB45"],
    )
    def test_password_is_password_like(self, value: str):
        """A mixed value with a letter and a digit or symbol is a password."""
        assert is_password_like(value) is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("abc1!", False),
            ("abc12!", True),
            ("a1" + "b" * 126, True),
            ("a1" + "b" * 127, False),
        ],
        ids=["five", "six", "128", "129"],
    )
    def test_length_is_six_to_128(self, value: str, expected: bool):
        """Six is the shortest and 128 the longest."""
        assert is_password_like(value) is expected

    @pytest.mark.parametrize("value", ["hunter 2!x", "hunter\t2!x"], ids=["space", "tab"])
    def test_whitespace_is_refused(self, value: str):
        """A password has no whitespace."""
        assert is_password_like(value) is False

    def test_needs_a_letter(self):
        """A value of digits and symbols alone is refused."""
        assert is_password_like("123456!!") is False

    @pytest.mark.parametrize("value", ["abcdefgh", "ABCDEFGH"], ids=["lower", "upper"])
    def test_needs_a_digit_or_a_symbol(self, value: str):
        """A letter run with neither a digit nor a symbol is refused."""
        assert is_password_like(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            "$HOME1x",
            "%VAR%ab1",
            "<token>1a",
            "{token}1a",
            "-password1",
            "/etc/key1a",
            "~/key1abc",
        ],
        ids=["shell-variable", "windows-variable", "slot", "brace", "flag", "path", "home-path"],
    )
    def test_reference_is_not_a_password(self, value: str):
        """A variable, a slot, a flag or a path is a reference, not a value."""
        assert is_password_like(value) is False

    def test_value_of_only_asterisks_is_a_mask(self):
        """A run of asterisks is a mask."""
        assert is_password_like("********") is False

    def test_asterisks_with_a_letter_and_a_digit_are_not_a_mask(self):
        """Asterisks alongside a letter and a digit are a password."""
        assert is_password_like("**a1xy") is True

    @pytest.mark.parametrize("value", ['"Summer2024!"', "'Summer2024!'"], ids=["double", "single"])
    def test_quoted_password_is_a_password(self, value: str):
        """The quotes are read past, so the value inside is judged."""
        assert is_password_like(value) is True

    @pytest.mark.parametrize(
        "value",
        ['"$MYSQL_ROOT_PASSWORD"', '"*****"'],
        ids=["quoted-shell-reference", "quoted-mask"],
    )
    def test_quoted_reference_or_mask_is_not_a_password(self, value: str):
        """A quoted reference or mask is still a reference or mask."""
        assert is_password_like(value) is False


# ── isBasicCredentials ────────────────────────────────────────────────────────


class TestIsBasicCredentials:
    """A Basic credential decodes to a user:password pair with printable UTF-8."""

    def test_decodes_to_user_and_password(self):
        """svc_billing:6MzX9nEW+T+LLc= is a credential."""
        assert is_basic_credentials("c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM=") is True

    @pytest.mark.parametrize(
        "value",
        ["dXNlcg==", "OnBhc3M=", "dXNlcjo=", "/w==", "dQE6cA=="],
        ids=["no-colon", "empty-user", "empty-password", "not-utf-8", "control-character"],
    )
    def test_each_refusal_on_its_own(self, value: str):
        """A missing colon, an empty half, non-UTF-8 bytes or a control character is refused."""
        assert is_basic_credentials(value) is False

    def test_length_must_be_a_multiple_of_four(self):
        """A lenient decoder would read the unpadded form, so it is refused here."""
        assert is_basic_credentials("c3ZjX2JpbGxpbmc6Nk16WDluRVcrVCtMTGM") is False

    @pytest.mark.parametrize("value", ["c3Zj X2Jp", "c3ZjX2!p"], ids=["space", "punctuation"])
    def test_characters_outside_the_base64_alphabet_are_refused(self, value: str):
        """Only the base64 alphabet is accepted."""
        assert is_basic_credentials(value) is False

    @pytest.mark.parametrize(
        "value",
        ["dXNlcjpwYXNz", "YWRtaW46cGFzc3dvcmQ=", "cm9vdDp0ZXN0"],
        ids=["user-pass", "admin-password", "root-test"],
    )
    def test_documented_pair_is_an_example(self, value: str):
        """Both halves on the tutorial list make the pair an example, not a credential."""
        assert is_basic_credentials(value) is False

    @pytest.mark.parametrize(
        "value",
        ["YWRtaW46WGs5bVAycVI3dg==", "cm9vdDpYazltUDJxUjd2"],
        ids=["admin-password-not-listed", "root-password-not-listed"],
    )
    def test_real_password_with_a_documented_user_is_a_credential(self, value: str):
        """Only the pair is an example; a real password under a tutorial user is a credential."""
        assert is_basic_credentials(value) is True


# ── isIban ────────────────────────────────────────────────────────────────────


class TestIsIban:
    """An IBAN is a valid country, a valid check and the length for its country."""

    @pytest.mark.parametrize(
        "value",
        [
            "DE89370400440532013000",
            "DE89 3704 0044 0532 0130 00",
            "GB82WEST12345698765432",
            "AE92070331234567890123456",
            "IL170108000000012612345",
        ],
        ids=["german", "german-spaced", "british", "country-not-in-table-25", "country-not-in-table-23"],
    )
    def test_valid_iban_is_an_iban(self, value: str):
        """A valid check and a length in range is an IBAN, spaced or not."""
        assert is_iban(value) is True

    def test_changed_check_digit_is_refused(self):
        """The two check digits must give a remainder of one."""
        assert is_iban("DE88370400440532013000") is False
        assert is_iban("DE99370400440532013000") is False

    def test_country_with_a_fixed_length_must_have_that_length(self):
        """Germany is 22 characters; one under and one over are both refused."""
        assert is_iban("DE8937040044053201300") is False
        assert is_iban("DE893704004405320130001") is False

    def test_country_not_in_the_table_is_held_to_fifteen_to_thirty_four(self):
        """A country outside the table is held only to the general range."""
        assert is_iban("IL00" + "1" * 10) is False
        assert is_iban("IL00" + "1" * 31) is False

    def test_lower_case_and_punctuation_are_refused(self):
        """The IBAN is upper case and has no punctuation."""
        assert is_iban("de89370400440532013000") is False
        assert is_iban("DE89-3704-0044-0532-0130-00") is False


# ── isIntlPhone ───────────────────────────────────────────────────────────────


class TestIsIntlPhone:
    """An international number has nine to fifteen digits and is not one repeated digit."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("+4 12345678", True),
            ("+49 1234567890123", True),
            ("+4 1234567", False),
            ("+49 12345678901234", False),
        ],
        ids=["nine", "fifteen", "eight", "sixteen"],
    )
    def test_digit_count_is_nine_to_fifteen(self, value: str, expected: bool):
        """The digits, counted with the country code, are nine to fifteen."""
        assert is_intl_phone(value) is expected

    @pytest.mark.parametrize("value", ["+11 111 111 111", "+00 0000 0000"], ids=["ones", "zeros"])
    def test_every_digit_the_same_is_a_placeholder(self, value: str):
        """A number of one repeated digit is a placeholder."""
        assert is_intl_phone(value) is False

    def test_only_the_digits_count(self):
        """Brackets, spaces and hyphens do not change the count."""
        assert is_intl_phone("+49 (151) 852-80948") is True


# ── isNinoUk ──────────────────────────────────────────────────────────────────


class TestIsNinoUk:
    """A National Insurance number has a valid prefix, six digits and a suffix A to D."""

    @pytest.mark.parametrize("value", ["RE 62 91 56 A", "CS 45 74 46 B", "AB 12 34 56 C"])
    def test_valid_number_is_a_national_insurance_number(self, value: str):
        """A valid prefix, six digits and a suffix from A to D is accepted."""
        assert is_nino_uk(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "DA 12 34 56 A",
            "FA 12 34 56 A",
            "IA 12 34 56 A",
            "QA 12 34 56 A",
            "UA 12 34 56 A",
            "VA 12 34 56 A",
            "RD 12 34 56 A",
            "RF 12 34 56 A",
            "RI 12 34 56 A",
            "RO 12 34 56 A",
            "RQ 12 34 56 A",
            "RU 12 34 56 A",
            "RV 12 34 56 A",
            "BG 12 34 56 A",
            "GB 12 34 56 A",
            "KN 12 34 56 A",
            "NK 12 34 56 A",
            "NT 12 34 56 A",
            "TN 12 34 56 A",
            "ZZ 12 34 56 A",
        ],
        ids=[
            "first-D",
            "first-F",
            "first-I",
            "first-Q",
            "first-U",
            "first-V",
            "second-D",
            "second-F",
            "second-I",
            "second-O",
            "second-Q",
            "second-U",
            "second-V",
            "prefix-BG",
            "prefix-GB",
            "prefix-KN",
            "prefix-NK",
            "prefix-NT",
            "prefix-TN",
            "prefix-ZZ",
        ],
    )
    def test_reserved_letter_or_prefix_is_refused(self, value: str):
        """Each reserved first letter, second letter or pair is refused on its own."""
        assert is_nino_uk(value) is False

    @pytest.mark.parametrize("value", ["BA 12 34 56 A", "GA 12 34 56 A"], ids=["BA", "GA"])
    def test_prefix_that_starts_with_a_reserved_letter_is_accepted(self, value: str):
        """The prefix is refused whole, so a pair beginning with a reserved letter is not refused for it."""
        assert is_nino_uk(value) is True

    @pytest.mark.parametrize("value", ["RE 62 91 5 A", "RE 62 91 567 A"], ids=["five-digits", "seven-digits"])
    def test_digit_count_is_six(self, value: str):
        """Six digits exactly."""
        assert is_nino_uk(value) is False


# ── isUsPhone ─────────────────────────────────────────────────────────────────


class TestIsUsPhone:
    """A US number is judged by its area code, its exchange and the fiction range."""

    @pytest.mark.parametrize("value", ["(415) 867-5309", "(212) 555-0300", "+1 212 555 0300"])
    def test_ordinary_number_is_a_phone_number(self, value: str):
        """An allocated area code with an ordinary line is a phone number."""
        assert is_us_phone(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "(555) 867-5309",
            "(212) 555-0100",
            "(212) 555-0199",
            "(212) 555-1212",
            "1-800-555-1212",
            "(888) 123-4567",
            "(877) 123-4567",
            "(866) 123-4567",
            "(855) 123-4567",
            "(844) 123-4567",
        ],
        ids=[
            "area-555",
            "fiction-0100",
            "fiction-0199",
            "directory-assistance",
            "toll-free-800-dashes",
            "toll-free-888",
            "toll-free-877",
            "toll-free-866",
            "toll-free-855",
            "toll-free-844",
        ],
    )
    def test_fiction_or_toll_free_number_is_refused(self, value: str):
        """Fiction, directory assistance and toll-free area codes are refused."""
        assert is_us_phone(value) is False

    def test_fiction_range_is_0100_to_0199_in_the_555_exchange_only(self):
        """Outside the fiction range, or outside the 555 exchange, the number is ordinary."""
        assert is_us_phone("(212) 555-0200") is True
        assert is_us_phone("(212) 556-0150") is True


# ── isPersonalMailbox ─────────────────────────────────────────────────────────


class TestIsPersonalMailbox:
    """A role mailbox is not personal; a name, or a role word with a suffix, is."""

    @pytest.mark.parametrize(
        ("address", "personal"),
        [
            ("support@acmecorp.io", False),
            ("Support@acmecorp.io", False),
            ("NO-REPLY@acmecorp.io", False),
            ("support%40acmecorp.io", False),
            ("john.support@acmecorp.io", True),
            ("support2@acmecorp.io", True),
            ("ada@analytical-engines.org", True),
        ],
    )
    def test_personal_mailbox_decision(self, address: str, personal: bool):
        """A role word, compared whole and case-insensitively, is not personal."""
        assert is_personal_mailbox(address) is personal


# ── hasAwsSecretShape ─────────────────────────────────────────────────────────


class TestHasAwsSecretShape:
    """An AWS secret access key has an upper case letter, a lower case letter and a digit."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("wJ4rXUt0nFEM9K7MDENG2bPxRfiCYqZ8nMLVAa3D", True),
            ("wj4rxut0nfem9k7mdeng2bpxrficyqz8nmlvaa3d", False),
            ("WJ4RXUT0NFEM9K7MDENG2BPXRFICYQZ8NMLVAA3D", False),
            ("wJrXUtnFEMIKMDENGbPxRfiCYqZnMLVAaD", False),
        ],
        ids=["mixed", "no-upper", "no-lower", "no-digit"],
    )
    def test_each_character_class_is_required(self, value: str, expected: bool):
        """The upper case letter, the lower case letter and the digit are each required."""
        assert has_aws_secret_shape(value) is expected

    def test_lower_case_hex_digest_is_refused(self):
        """A git commit ID is lower case hex with no upper case letter."""
        assert has_aws_secret_shape("0123456789abcdef0123456789abcdef01234567") is False

    def test_random_mixed_value_is_accepted(self):
        """A random mixed value of forty characters has the shape."""
        assert has_aws_secret_shape(mixed(40)) is True
