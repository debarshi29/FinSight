from __future__ import annotations

import re

from core.models import SectionType

_AUDITED_PATTERNS = [
    r"independent auditor",
    r"audit report",
    r"statement of.*profit",
    r"profit.*loss",
    r"balance sheet",
    r"income statement",
    r"cash flow statement",
    r"financial statement",
    r"notes? to.*financial",
    r"consolidated statement",
]

_MDA_PATTERNS = [
    r"management.?s? discussion",
    r"md&a",
    r"management analysis",
    r"operating review",
    r"business review",
    r"performance review",
]

_NOTES_PATTERNS = [
    r"^note \d",
    r"notes? to accounts",
    r"accounting policies",
    r"significant accounting",
]

_LETTER_PATTERNS = [
    r"dear shareholder",
    r"chairman.?s? (letter|message|statement)",
    r"ceo.?s? (letter|message)",
    r"to our shareholders",
    r"managing director",
]

_FISCAL_YEAR_RE = re.compile(
    r"\b(fy|fiscal year|year ended)\s*(20\d{2}[-/]?\d{0,2})\b", re.IGNORECASE
)
_COMPANY_HINTS = {
    "infosys": "Infosys",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "wipro": "Wipro",
}


def detect_section_type(text: str, section_heading: str = "") -> SectionType:
    combined = (text + " " + section_heading).lower()

    for pat in _AUDITED_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            if "note" in combined:
                return SectionType.NOTES
            return SectionType.AUDITED_FINANCIALS

    for pat in _NOTES_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return SectionType.NOTES

    for pat in _MDA_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return SectionType.MDA

    for pat in _LETTER_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return SectionType.LETTER

    return SectionType.UNKNOWN


def detect_fiscal_year(text: str) -> str:
    match = _FISCAL_YEAR_RE.search(text)
    if match:
        raw = match.group(2).replace("/", "-").replace(" ", "")
        return raw
    for year in range(2024, 2019, -1):
        if str(year) in text:
            return str(year)
    return ""


def detect_company(source_filename: str, text: str = "") -> str:
    combined = (source_filename + " " + text[:200]).lower()
    for hint, name in _COMPANY_HINTS.items():
        if hint in combined:
            return name
    return ""


def section_type_confidence_weight(section_type: SectionType) -> float:
    weights = {
        SectionType.AUDITED_FINANCIALS: 1.0,
        SectionType.NOTES: 0.85,
        SectionType.MDA: 0.65,
        SectionType.LETTER: 0.40,
        SectionType.UNKNOWN: 0.50,
    }
    return weights[section_type]
