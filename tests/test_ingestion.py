from __future__ import annotations

from core.models import SectionType
from ingestion.metadata import detect_company, detect_fiscal_year, detect_section_type


def test_detect_fiscal_year():
    assert "2024" in detect_fiscal_year("Revenue for FY2024 grew by 10%")
    assert "2023" in detect_fiscal_year("Year ended March 2023 results")
    assert detect_fiscal_year("some text without year") == ""


def test_detect_section_type_financials():
    st = detect_section_type("Profit and Loss Statement for the year ended March 2024")
    assert st == SectionType.AUDITED_FINANCIALS


def test_detect_section_type_mda():
    st = detect_section_type("Management Discussion and Analysis of Financial Condition")
    assert st == SectionType.MDA


def test_detect_section_type_letter():
    st = detect_section_type("Dear Shareholders, it is my pleasure to present")
    assert st == SectionType.LETTER


def test_detect_company():
    assert detect_company("Infosys_AR_2024.pdf") == "Infosys"
    assert detect_company("TCS_Annual_Report_2024.pdf") == "TCS"
    assert detect_company("Wipro_FY2024.pdf") == "Wipro"


def test_detect_company_tata_consultancy_alias():
    assert detect_company("tata_consultancy_services_2024.pdf") == "TCS"


def test_detect_company_unknown():
    assert detect_company("random_document.pdf") == ""


def test_detect_section_type_notes():
    st = detect_section_type("Note 5: Accounting Policies and Significant Judgements")
    assert st == SectionType.NOTES


def test_detect_section_type_unknown():
    st = detect_section_type("Page 1 of 200 — Cover Page")
    assert st == SectionType.UNKNOWN


def test_detect_fiscal_year_fy_prefix():
    assert "2023" in detect_fiscal_year("FY2023 annual results")


def test_detect_fiscal_year_no_match():
    assert detect_fiscal_year("No year in this string at all") == ""


def test_detect_section_type_heading_context():
    # The section heading should influence classification even if body text is neutral
    st = detect_section_type("Total revenue was 500bn", "Balance Sheet")
    assert st == SectionType.AUDITED_FINANCIALS
