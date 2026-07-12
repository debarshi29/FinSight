from __future__ import annotations

from decimal import Decimal

from core.unit_normalizer import (
    USD_TO_INR,
    normalize_subtask_results,
    normalize_text,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _crore(val: str) -> Decimal:
    """Parse a crore value string, stripping commas."""
    return Decimal(val.replace(",", ""))


# ── Currency constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_usd_inr_rate(self):
        assert USD_TO_INR == Decimal("84")

    def test_usd_bn_to_crore(self):
        # $1 billion = ₹84 billion = ₹8,400 crore
        # $30 billion → ₹252,000 crore
        from core.unit_normalizer import _USD_BN_TO_CRORE

        assert _USD_BN_TO_CRORE == Decimal("8400")


# ── USD conversions ───────────────────────────────────────────────────────────


class TestUSDNormalization:
    def test_usd_30_billion(self):
        out = normalize_text("TCS generated $30 billion in revenue")
        assert "252,000 crore" in out
        assert "[converted from $30 billion at" in out

    def test_usd_1_5_billion(self):
        out = normalize_text("operating profit of $1.5 billion")
        assert "12,600 crore" in out

    def test_usd_prefix_variant(self):
        out = normalize_text("USD 5 billion")
        assert "42,000 crore" in out

    def test_usd_million(self):
        # $1 million = ₹84 million = ₹8.4 crore
        out = normalize_text("profit of $500 million")
        assert "4,200 crore" in out

    def test_usd_thousand(self):
        # $1 thousand = ₹84 thousand = ₹0.0084 crore
        out = normalize_text("fee of $100 thousand")
        assert "0.84 crore" in out

    def test_usd_billion_abbreviated_bn(self):
        out = normalize_text("revenue USD 2 bn")
        assert "16,800 crore" in out

    def test_usd_no_scale_left_unchanged(self):
        # "$500" with no scale word should not be converted
        out = normalize_text("fee of $500")
        assert "$500" in out
        assert "crore" not in out

    def test_label_includes_at_84(self):
        out = normalize_text("$10 billion revenue")
        assert "at ₹84/USD" in out

    def test_original_value_preserved_in_label(self):
        out = normalize_text("$30 billion in revenue")
        assert "$30 billion" in out

    def test_multiple_usd_figures_in_one_string(self):
        out = normalize_text("Revenue $30 billion, profit $2 billion")
        assert "252,000 crore" in out
        assert "16,800 crore" in out

    def test_usd_trillion(self):
        out = normalize_text("GDP of $1 trillion")
        assert "8,40,00,000" in out or "8400000" in out.replace(",", "")


# ── INR scale conversions ─────────────────────────────────────────────────────


class TestINRNormalization:
    def test_inr_million_to_crore(self):
        # ₹10,478 million → ₹1,047.8 crore
        out = normalize_text("Wipro recorded Rs 10,478 million in revenue")
        assert "1,047.8 crore" in out
        assert "[converted from" in out

    def test_inr_large_million(self):
        # ₹926,240 million → ₹92,624 crore
        out = normalize_text("Rs 926,240 million")
        assert "92,624 crore" in out

    def test_inr_crore_unchanged(self):
        # Already in crore — no label needed
        out = normalize_text("Rs 1,78,650 crore total revenue")
        assert "1,78,650 crore" in out
        assert "[converted from" not in out

    def test_inr_lakh_to_crore(self):
        # ₹500 lakh = ₹5 crore
        out = normalize_text("Rs 500 lakh premium income")
        assert "5 crore" in out
        assert "[converted from" in out

    def test_inr_billion_to_crore(self):
        # INR 2 billion = 200 crore
        out = normalize_text("INR 2 billion")
        assert "200 crore" in out

    def test_rupee_symbol_prefix(self):
        out = normalize_text("revenue of ₹50,000 million")
        assert "5,000 crore" in out

    def test_rs_dot_prefix(self):
        out = normalize_text("Rs. 1,000 million profit")
        assert "100 crore" in out

    def test_inr_prefix(self):
        out = normalize_text("INR 100 lakh")
        assert "1 crore" in out

    def test_original_scale_preserved_in_label(self):
        out = normalize_text("Rs 10,478 million revenue")
        assert "₹10,478 million" in out

    def test_multiple_inr_figures(self):
        out = normalize_text("Revenue Rs 50,000 million, profit Rs 5,000 million")
        assert "5,000 crore" in out
        assert "500 crore" in out


# ── Non-financial strings left unchanged ─────────────────────────────────────


class TestNoChange:
    def test_percentages_unchanged(self):
        out = normalize_text("Revenue grew 15.2% year-on-year")
        assert out == "Revenue grew 15.2% year-on-year"

    def test_plain_employee_count(self):
        out = normalize_text("270,000 employees across 50 countries")
        assert out == "270,000 employees across 50 countries"

    def test_plain_number_unchanged(self):
        out = normalize_text("The company has 5 offices in 3 countries")
        assert out == "The company has 5 offices in 3 countries"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_unknown_inr_scale_left_alone(self):
        # "₹50" with no scale word — treated as crore (default), rebuild verbatim
        out = normalize_text("fee of ₹50")
        # No conversion label since it's treated as already in crore
        assert "₹50 crore" in out

    def test_already_normalized_label_not_double_converted(self):
        # If text already has a "[converted from ...]" label, the ₹ crore value
        # at the start should not get re-processed into another label
        already = "₹252,000 crore [converted from $30 billion at ₹84/USD, approx]"
        out = normalize_text(already)
        # The ₹252,000 crore part (already in crore) should be left verbatim
        assert "₹252,000 crore" in out


# ── Precision / rounding ──────────────────────────────────────────────────────


class TestPrecision:
    def test_no_trailing_zeros(self):
        out = normalize_text("Rs 10 million")
        assert "1 crore" in out
        assert "1.00 crore" not in out

    def test_two_decimal_places_when_needed(self):
        out = normalize_text("Rs 10,478 million")
        assert "1,047.8 crore" in out

    def test_large_number_formatted_with_commas(self):
        out = normalize_text("$30 billion revenue")
        assert "252,000 crore" in out


# ── Structured subtask normalization ─────────────────────────────────────────


class TestNormalizeSubtaskResults:
    def _make_result(self, claims: list[dict], kpis: list[dict] = None) -> dict:
        return {
            "subtask": "revenue comparison",
            "claims": claims,
            "kpis": kpis or [],
        }

    def test_normalizes_claim_field(self):
        sr = [
            self._make_result(
                claims=[
                    {
                        "claim": "TCS generated $30 billion revenue",
                        "supporting_text": "Revenue was $30 billion",
                        "confidence": 0.9,
                    }
                ]
            )
        ]
        out = normalize_subtask_results(sr)
        claim_text = out[0]["claims"][0]["claim"]
        assert "252,000 crore" in claim_text
        assert "$30 billion" in claim_text  # original preserved in label

    def test_normalizes_supporting_text(self):
        sr = [
            self._make_result(
                claims=[
                    {
                        "claim": "Revenue was Rs 50,000 million",
                        "supporting_text": "Total revenue stood at Rs 50,000 million",
                    }
                ]
            )
        ]
        out = normalize_subtask_results(sr)
        assert "5,000 crore" in out[0]["claims"][0]["supporting_text"]

    def test_normalizes_kpi_value_field(self):
        sr = [
            self._make_result(
                claims=[],
                kpis=[{"label": "Revenue", "value": "$30 billion", "description": "Total revenue"}],
            )
        ]
        out = normalize_subtask_results(sr)
        assert "252,000 crore" in out[0]["kpis"][0]["value"]

    def test_does_not_mutate_original(self):
        original_claim = {"claim": "$30 billion revenue", "confidence": 0.9}
        sr = [self._make_result(claims=[original_claim])]
        normalize_subtask_results(sr)
        assert original_claim["claim"] == "$30 billion revenue"

    def test_empty_list(self):
        assert normalize_subtask_results([]) == []

    def test_non_string_fields_untouched(self):
        sr = [self._make_result(claims=[{"claim": "profit", "confidence": 0.9, "page": 5}])]
        out = normalize_subtask_results(sr)
        assert out[0]["claims"][0]["confidence"] == 0.9
        assert out[0]["claims"][0]["page"] == 5

    def test_missing_claims_key_handled(self):
        sr = [{"subtask": "test", "kpis": []}]
        out = normalize_subtask_results(sr)
        assert out[0]["claims"] == []

    def test_mixed_currencies_in_one_result(self):
        sr = [
            self._make_result(
                claims=[
                    {"claim": "TCS: $30 billion, Wipro: Rs 926,240 million"},
                ]
            )
        ]
        out = normalize_subtask_results(sr)
        claim = out[0]["claims"][0]["claim"]
        assert "252,000 crore" in claim
        assert "92,624 crore" in claim


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_value(self):
        out = normalize_text("revenue of $0 billion")
        assert "0 crore" in out

    def test_decimal_value(self):
        out = normalize_text("profit of $1.234 billion")
        # 1.234 × 8400 = 10,365.6
        assert "10,365.6 crore" in out

    def test_value_with_spaces_around_currency(self):
        out = normalize_text("Rs  500  million")
        # Should still match despite extra spaces
        assert "50 crore" in out

    def test_indian_comma_format(self):
        # 1,78,650 is Indian lakh-comma grouping
        out = normalize_text("Rs 1,78,650 crore")
        assert "1,78,650 crore" in out
        assert "[converted from" not in out

    def test_very_small_lakh_value(self):
        out = normalize_text("Rs 1 lakh")
        assert "0.01 crore" in out

    def test_sentence_with_no_financial_figures(self):
        s = "The board meeting was held on 15 March 2024 at the registered office."
        assert normalize_text(s) == s

    def test_usd_and_inr_in_same_sentence(self):
        s = "TCS earned $30 billion while Wipro earned Rs 50,000 million"
        out = normalize_text(s)
        assert "252,000 crore" in out
        assert "5,000 crore" in out

    def test_currencies_are_idempotent_when_already_crore(self):
        # Running normalize_text twice on crore values should not keep adding labels
        s = "Rs 1,000 crore revenue"
        once = normalize_text(s)
        twice = normalize_text(once)
        assert once == twice

    def test_mn_abbreviation_inr(self):
        out = normalize_text("Rs 1,000 mn")
        assert "100 crore" in out

    def test_bn_abbreviation_inr(self):
        out = normalize_text("Rs 1 bn")
        assert "100 crore" in out
