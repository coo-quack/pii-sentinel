"""Tests for the rule scan on whole texts."""

from pii_sentinel.rules import scan


class TestScanSecretFiltering:
    """Test that placeholder and shape filters work in scan."""

    def test_placeholder_secret_ignored(self):
        """Placeholder values should not be flagged even if they match patterns."""
        # Real AWS key pattern but placeholder text
        matches = scan("api_key = your-api-key-here")
        secret_matches = [m for m in matches if m.category == "secret"]
        # The placeholder should be filtered out
        assert len(secret_matches) == 0

    def test_variable_reference_ignored(self):
        """Code references like API_KEY_NAME should not be flagged."""
        matches = scan("export API_KEY_NAME = os.getenv('API_KEY')")
        # API_KEY_NAME is not a secret shape
        assert all(m.rule != "aws-access-key" for m in matches)

    def test_real_secret_not_ignored(self):
        """Real secrets should not be filtered by shape checks."""
        matches = scan("My key is AKIA3QF7TZ9KLMN2PQRS and secret is xyz")
        aws_matches = [m for m in matches if m.rule == "aws-access-key"]
        assert len(aws_matches) == 1
        assert aws_matches[0].value == "AKIA3QF7TZ9KLMN2PQRS"

    def test_valid_credit_card_not_ignored(self):
        """Valid credit cards should pass all filters."""
        matches = scan("payment card 4532015112830366 for processing")
        cc_matches = [m for m in matches if m.rule == "pii-credit-card"]
        assert len(cc_matches) == 1

    def test_test_card_filtered(self):
        """Test card numbers should be filtered."""
        matches = scan("test card 4242424242424242")
        cc_matches = [m for m in matches if m.rule == "pii-credit-card"]
        # Filtered by Luhn validator
        assert len(cc_matches) == 0


class TestScanMatchSpans:
    """Test that match spans are correct, especially with secretGroup."""

    def test_whole_match_span(self):
        """Without secretGroup, span should be whole match."""
        matches = scan("found: AKIA3QF7TZ9KLMN2PQRS here")
        aws_matches = [m for m in matches if m.rule == "aws-access-key"]
        assert len(aws_matches) == 1
        m = aws_matches[0]
        # Span should cover entire match
        assert text_at_span("found: AKIA3QF7TZ9KLMN2PQRS here", m) == m.value


class TestScanDescription:
    """Test that Match objects include description."""

    def test_match_has_description(self):
        """Match objects should include rule description."""
        matches = scan("AKIA3QF7TZ9KLMN2PQRS")
        assert len(matches) == 1
        m = matches[0]
        assert m.description != ""
        assert m.description == "AWS Access Key ID"

    def test_all_matches_have_description(self):
        """All matches should have descriptions from the rules."""
        matches = scan("card 4532015112830366 and key AKIA3QF7TZ9KLMN2PQRS")
        for m in matches:
            assert m.description
            assert isinstance(m.description, str)
            assert len(m.description) > 0


class TestScanContextWindow:
    """Test that contextWindow from config and rules is used."""

    def test_postal_code_requires_context(self):
        """Postal codes should require context words."""
        # pii-postal-code requires context
        matches = scan("version 1.2.345")
        postal_matches = [m for m in matches if m.rule == "pii-postal-code"]
        # Should NOT match without context word
        assert len(postal_matches) == 0

        matches = scan("zip code 12345")
        postal_matches = [m for m in matches if m.rule == "pii-postal-code"]
        # Should match WITH context word
        assert len(postal_matches) == 1


def text_at_span(text: str, match) -> str:
    """Extract text at match span."""
    return text[match.start : match.end]
