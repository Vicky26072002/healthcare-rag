"""Healthcare guardrails: medical disclaimer and emergency red-flag detection."""

from __future__ import annotations

import re

DISCLAIMER = (
    "This information is for educational purposes only and is not a substitute for "
    "professional medical advice, diagnosis or treatment. Always consult a qualified "
    "healthcare provider."
)

EMERGENCY_MESSAGE = (
    "Your question mentions symptoms that may need urgent care. If you or someone else "
    "is in danger, call your local emergency number now (India: 112 / ambulance 108)."
)

_RED_FLAGS = [
    r"chest pain",
    r"can'?t breathe|cannot breathe|difficulty breathing|shortness of breath",
    r"unconscious|fainted|passed out",
    r"seizure",
    r"stroke|face drooping|slurred speech",
    r"severe bleeding|vomiting blood|coughing (up )?blood",
    r"overdose|poison(ed|ing)?",
    r"suicid\w*|kill (my|him|her)self|self[- ]harm",
    r"anaphyla\w*|throat (is )?swelling",
]
_RED_FLAG_RE = re.compile(r"\b(" + "|".join(_RED_FLAGS) + r")\b", re.IGNORECASE)


def detect_emergency(text: str) -> list[str]:
    """Return the red-flag phrases found in the text (empty list if none)."""
    return sorted({m.group(0).lower() for m in _RED_FLAG_RE.finditer(text)})
