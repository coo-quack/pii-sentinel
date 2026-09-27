"""Label sets shared by the data generator, training, inference and evaluation."""

# Span types. Non-PII contacts and numbers are labelled too, so the detector can report them as
# pii=false instead of silently dropping them (a company hotline is still a phone number).
SPAN_TYPES = [
    "PERSON",
    "EMAIL",
    "EMAIL_GENERIC",
    "PHONE",
    "PHONE_CORPORATE",
    "MY_NUMBER",
    "NATIONAL_ID",
    "PASSPORT_LICENCE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "ORDER_TRACKING",
    "SERIAL",
]

# BIO tags over tokens: index 0 is O, then B-/I- per span type.
BIO = ["O"] + [f"{p}-{t}" for t in SPAN_TYPES for p in ("B", "I")]
BIO_INDEX = {tag: i for i, tag in enumerate(BIO)}

# Document-level categories (multi-label) and sensitivity (single label).
CATEGORIES = [
    "person_name",
    "email_or_phone",
    "postal_address",
    "date_of_birth",
    "government_id",
    "financial_account",
    "health_info",
    "biometric_or_genetic",
    "ip_address_of_a_person",
    "sns_handle",
    "employment_info",
    "race_or_religion",
    "political_or_union",
    "sex_life_or_orientation",
    "citizenship_or_immigration",
    "precise_location",
    "credentials",
    "private_communications",
    "hr_or_criminal_record",
]
SENSITIVITY = ["none", "low", "high"]

# How each span type is reported: (finding type, pii, number_type).
REPORT = {
    "PERSON": ("person_name", True, None),
    "EMAIL": ("email", True, None),
    "EMAIL_GENERIC": ("email", False, None),
    "PHONE": ("phone", True, None),
    "PHONE_CORPORATE": ("phone", False, None),
    "MY_NUMBER": ("number", True, "my_number"),
    "NATIONAL_ID": ("number", True, "national_id"),
    "PASSPORT_LICENCE": ("number", True, "driver_licence_or_passport"),
    "CREDIT_CARD": ("number", True, "credit_card"),
    "BANK_ACCOUNT": ("number", True, "bank_account"),
    "ORDER_TRACKING": ("number", False, "order_or_tracking_number"),
    "SERIAL": ("number", False, "product_serial"),
}
