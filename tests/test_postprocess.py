"""Tests for the post-processing that lowers false positives in predict.py."""

import pytest

from pii_sentinel.labels import CATEGORIES
from pii_sentinel.predict import add_rule_findings, is_non_personal, settle_level


def probs(**over):
    """Category probabilities with every category at zero except the given ones."""
    out = dict.fromkeys(CATEGORIES, 0.0)
    out.update(over)
    return out


def finding(ftype, value="x", pii=True):
    return {"type": ftype, "value": value, "pii": pii}


class TestNotPersonalValues:
    """Values written as examples, tests or placeholders are judged not to be real identifiers."""

    @pytest.mark.parametrize(
        ("ftype", "number_type", "value"),
        [
            ("number", "credit_card", "4242 4242 4242 4242"),
            ("number", "credit_card", "3782 822463 10005"),
            (
                "number",
                "credit_card",
                "5555 5555 4444",
            ),  # a piece of a sandbox number, cut short by the model
            ("number", "national_id", "000-00-0000"),
            ("number", "national_id", "666-12-3456"),
            ("number", "national_id", "QQ 12 34 56 C"),
            ("number", "driver_licence_or_passport", "X0000000"),
        ],
    )
    def test_example_values_are_not_personal(self, ftype, number_type, value):
        assert is_non_personal(ftype, number_type, value)

    @pytest.mark.parametrize(
        ("ftype", "number_type", "value"),
        [
            ("number", "credit_card", "4532 0151 1283 0366"),  # a card number that is not published test data
            ("number", "credit_card", "4556 7375 8689 9855"),  # Luhn-valid and not a published test card
            ("number", "national_id", "123-45-6789"),  # an issued group and serial
            ("number", "national_id", "AB 12 34 56 C"),  # an NI prefix that HMRC issues
            ("number", "national_id", "0x8007000E"),  # not a number of the form the checks read
            ("number", "bank_account", "1234 5678 9012"),
        ],
    )
    def test_near_misses_are_still_personal(self, ftype, number_type, value):
        assert not is_non_personal(ftype, number_type, value)

    @pytest.mark.parametrize(
        ("ftype", "value"),
        [
            ("email", "alice@example.com"),
            ("email", "noreply@notify.test"),
            ("phone", "555-0100"),
            ("phone", "+1 415 555 0123"),
        ],
    )
    def test_contacts_are_never_judged_non_personal(self, ftype, value):
        # A contact is personal whatever its domain or number range (labelling policy).
        assert not is_non_personal(ftype, None, value)


class TestSettleLevel:
    """A level is lowered only when a judged-not-real value supports it and no tie to a person remains."""

    def test_a_high_on_a_judged_number_with_no_tie_falls_to_none(self):
        assert settle_level("high", probs(government_id=0.96), [], [], {"number"}) == "none"

    def test_a_high_on_a_number_without_a_judged_number_is_kept(self):
        # An Aadhaar-like number the model marks only as a category: nothing judged, so nothing lowers it.
        assert settle_level("high", probs(government_id=0.998), [], [], set()) == "high"

    def test_a_high_on_a_number_stays_when_a_number_finding_remains(self):
        findings = [finding("number", "123-45-6789")]
        assert settle_level("high", probs(government_id=0.96), findings, [], {"number"}) == "high"

    def test_a_high_stays_when_a_secret_remains(self):
        assert (
            settle_level("high", probs(financial_account=0.9), [], [{"type": "secret"}], {"number"}) == "high"
        )

    def test_a_high_with_a_name_stays_even_when_the_number_is_not_real(self):
        probs_name = probs(financial_account=0.9, person_name=0.9)
        assert settle_level("high", probs_name, [finding("person_name")], [], {"number"}) == "high"

    def test_a_high_with_a_weak_name_category_falls_when_no_name_is_found(self):
        probs_weak = probs(financial_account=0.9, person_name=0.2)
        assert settle_level("high", probs_weak, [], [], {"number"}) == "none"

    def test_a_judged_value_of_another_type_does_not_lower_a_number_high(self):
        # Only a judged number lowers a number high; another judged type does not.
        assert settle_level("high", probs(government_id=0.96), [], [], {"email"}) == "high"

    def test_a_high_on_health_is_left_alone(self):
        assert settle_level("high", probs(health_info=0.95), [], [], {"number"}) == "high"

    def test_a_high_on_health_and_a_number_is_left_alone(self):
        p = probs(health_info=0.95, government_id=0.9)
        assert settle_level("high", p, [], [], {"number"}) == "high"

    def test_a_high_with_no_reason_above_the_threshold_is_left_alone(self):
        assert settle_level("high", probs(health_info=0.3), [], [], {"number"}) == "high"

    def test_a_high_on_precise_location_is_left_alone(self):
        assert settle_level("high", probs(precise_location=0.99), [], [], {"number"}) == "high"

    def test_a_low_with_only_judged_not_personal_findings_falls_to_none(self):
        assert settle_level("low", probs(), [], [], {"number"}) == "none"

    def test_a_low_stays_when_an_address_is_in_the_document(self):
        # "Contact test@example.com. Address: 東京都港区芝公園4-2-8, 生年月日 1988年7月21日": the example address is
        # judged not personal, but the address and the date of birth still make the document low.
        p = probs(email_or_phone=0.99, postal_address=0.9)
        assert settle_level("low", p, [], [], {"number"}) == "low"

    def test_a_low_stays_when_a_date_of_birth_is_in_the_document(self):
        p = probs(email_or_phone=0.99, date_of_birth=0.9)
        assert settle_level("low", p, [], [], {"number"}) == "low"

    def test_a_low_stays_when_an_employer_is_in_the_document(self):
        p = probs(email_or_phone=0.99, employment_info=0.9)
        assert settle_level("low", p, [], [], {"number"}) == "low"

    def test_a_low_stays_when_a_handle_or_an_ip_address_is_in_the_document(self):
        assert settle_level("low", probs(sns_handle=0.9), [], [], {"number"}) == "low"
        assert settle_level("low", probs(ip_address_of_a_person=0.9), [], [], {"number"}) == "low"

    def test_a_low_on_an_unreal_value_alone_falls_when_the_attributes_are_low(self):
        p = probs(email_or_phone=0.3, postal_address=0.4, date_of_birth=0.2, employment_info=0.1)
        assert settle_level("low", p, [], [], {"number"}) == "none"

    def test_a_low_with_a_personal_finding_stays(self):
        findings = [finding("person_name")]
        assert settle_level("low", probs(person_name=0.9), findings, [], {"number"}) == "low"

    def test_a_low_without_an_excluded_finding_stays(self):
        assert settle_level("low", probs(email_or_phone=0.99), [], [], set()) == "low"

    def test_none_stays_none(self):
        assert settle_level("none", probs(), [], [], {"number"}) == "none"


class TestRuleMatchesJudgedNotPersonal:
    """A rule match judged not to be a real identifier does not raise the floor."""

    def test_a_fictional_looking_phone_is_still_personal(self):
        # A contact is personal whatever its number range (the rule set itself leaves out the NANP fiction block).
        text = "Call +1 415 782 0142 today."
        findings = []
        _, floor = add_rule_findings(text, findings)
        assert floor == "none"
        assert findings and findings[0]["pii"]

    def test_a_real_looking_phone_is_kept_as_personal(self):
        text = "Call +1 415 555 0223 today."
        findings = []
        add_rule_findings(text, findings)
        assert findings and findings[0]["pii"]

    def test_a_real_looking_ssn_still_raises_the_floor(self):
        _, floor = add_rule_findings("SSN 123-45-6789 on file", [])
        assert floor == "high"
