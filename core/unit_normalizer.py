"""
Deterministic financial unit normalizer.

Converts USD and multi-scale INR figures inside claim/KPI text to a common
₹ crore denomination so the Comparator LLM never has to do arithmetic.

Exchange rate: 1 USD = ₹84 (approximate FY2026 average).
Scale identities (INR): 1 crore = 10 million = 100 lakh = 0.01 billion.

Only values with an explicit currency marker (₹, Rs, $, USD) are touched.
Plain numbers without a currency symbol are left unchanged.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

# ── Constants ────────────────────────────────────────────────────────────────

USD_TO_INR = Decimal("84")  # approximate FY2026 average

# INR units → crore multiplier
_INR_SCALE: dict[str, Decimal] = {
    "crore": Decimal("1"),
    "crores": Decimal("1"),
    "cr": Decimal("1"),
    "lakh": Decimal("0.01"),
    "lakhs": Decimal("0.01"),
    "lac": Decimal("0.01"),
    "lacs": Decimal("0.01"),
    "million": Decimal("0.1"),
    "millions": Decimal("0.1"),
    "mn": Decimal("0.1"),
    "billion": Decimal("100"),
    "billions": Decimal("100"),
    "bn": Decimal("100"),
    "thousand": Decimal("0.001"),
    "thousands": Decimal("0.001"),
}

# USD units → USD-billion multiplier (then × 8400 → ₹ crore)
_USD_SCALE: dict[str, Decimal] = {
    "trillion": Decimal("1000"),
    "trillions": Decimal("1000"),
    "tn": Decimal("1000"),
    "billion": Decimal("1"),
    "billions": Decimal("1"),
    "bn": Decimal("1"),
    "million": Decimal("0.001"),
    "millions": Decimal("0.001"),
    "mn": Decimal("0.001"),
    "thousand": Decimal("0.000001"),
    "thousands": Decimal("0.000001"),
}

# $1 billion = ₹84 billion = ₹8,400 crore
_USD_BN_TO_CRORE = USD_TO_INR * Decimal("100")

_NUM = r"[0-9][0-9,]*(?:\.[0-9]*)?"  # e.g. 30 | 1,78,650 | 30.5

_INR_WORD_SCALES = r"crores?|cr|lakhs?|lacs?|millions?|mn|billions?|bn|thousands?"
_USD_WORD_SCALES = r"trillions?|tn|billions?|bn|millions?|mn|thousands?"

# ₹ / Rs. / INR followed by a number and optional scale word
_INR_RE = re.compile(
    rf"(?:₹\s*|Rs\.?\s*|INR\s*)({_NUM})\s*({_INR_WORD_SCALES})?",
    re.IGNORECASE,
)

# $ / USD followed by a number and optional scale word
_USD_RE = re.compile(
    rf"(?:USD\s*|\$\s*)({_NUM})\s*({_USD_WORD_SCALES})?",
    re.IGNORECASE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_num(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def _fmt(d: Decimal) -> str:
    """Format a Decimal with up to 2 decimal places, dropping trailing zeros."""
    rounded = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = f"{rounded:,}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ── Public API ───────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """
    Replace financial figures in *text* with ₹ crore equivalents, labeling
    conversions inline.  Figures already in ₹ crore are returned unchanged.
    Figures whose scale is unknown are left unchanged.

    Examples
    --------
    "$30 billion"          → "₹2,52,000 crore [converted from $30 billion at ₹84/USD]"
    "₹10,478 million"      → "₹1,047.8 crore [converted from ₹10,478 million]"
    "₹1,78,650 crore"      → "₹1,78,650 crore"  (no change)
    "growth of 15.2%"      → unchanged
    "270,000 employees"    → unchanged
    """
    text = _replace_inr(text)
    text = _replace_usd(text)
    return text


def normalize_subtask_results(subtask_results: list[dict]) -> list[dict]:
    """
    Walk subtask_results (as produced by query_stream) and normalize all
    string fields that may contain financial figures.

    Operates on a shallow copy — original dicts are not mutated.
    """
    out = []
    for sr in subtask_results:
        sr2 = dict(sr)
        sr2["claims"] = [_normalize_claim(c) for c in sr.get("claims", [])]
        sr2["kpis"] = [_normalize_kpi(k) for k in sr.get("kpis", [])]
        out.append(sr2)
    return out


# ── Internal ─────────────────────────────────────────────────────────────────


def _replace_inr(text: str) -> str:
    def _sub(m: re.Match) -> str:
        raw_num = m.group(1)
        scale_word = (m.group(2) or "crore").lower()
        multiplier = _INR_SCALE.get(scale_word)
        if multiplier is None:
            return m.group(0)  # unknown scale → leave unchanged

        try:
            amount = _parse_num(raw_num)
        except Exception:
            return m.group(0)

        crore_val = amount * multiplier

        if multiplier == Decimal("1"):
            # Already in crore — rebuild verbatim (normalise whitespace only)
            return f"₹{raw_num} crore"

        original_scale = m.group(2) or ""
        original = f"₹{raw_num}{' ' + original_scale if original_scale else ''}"
        return f"₹{_fmt(crore_val)} crore [converted from {original}]"

    return _INR_RE.sub(_sub, text)


def _replace_usd(text: str) -> str:
    def _sub(m: re.Match) -> str:
        raw_num = m.group(1)
        scale_word = (m.group(2) or "").lower()
        usd_bn_mult = _USD_SCALE.get(scale_word)
        if usd_bn_mult is None and scale_word:
            return m.group(0)  # unknown scale → leave unchanged
        if usd_bn_mult is None:
            # No scale word — treat as plain USD (e.g. "$500" = $0.000_000_5 B)
            # These are likely not revenue figures; leave unchanged
            return m.group(0)

        try:
            amount = _parse_num(raw_num)
        except Exception:
            return m.group(0)

        crore_val = amount * usd_bn_mult * _USD_BN_TO_CRORE

        original_scale = m.group(2) or ""
        original = f"${raw_num}{' ' + original_scale if original_scale else ''}"
        return f"₹{_fmt(crore_val)} crore [converted from {original} at ₹84/USD, approx]"

    return _USD_RE.sub(_sub, text)


def _normalize_claim(claim: dict) -> dict:
    c = dict(claim)
    for field in ("claim", "supporting_text"):
        if isinstance(c.get(field), str):
            c[field] = normalize_text(c[field])
    return c


def _normalize_kpi(kpi: dict) -> dict:
    k = dict(kpi)
    for field in ("value", "label", "description"):
        if isinstance(k.get(field), str):
            k[field] = normalize_text(k[field])
    return k
