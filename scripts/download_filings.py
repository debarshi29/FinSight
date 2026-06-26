"""
Download Infosys, TCS, and Wipro annual reports for FY2022-FY2024.

Usage:
    uv run python scripts/download_filings.py

The script scrapes each company's investor relations page, finds annual report
PDF links, and downloads them to data/filings/.

If a company's page structure changes and scraping fails, the script prints
the investor page URL so you can download manually and drop the PDF into
data/filings/ yourself.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

FILINGS_DIR = Path(__file__).parent.parent / "data" / "filings"
TARGET_YEARS = {"2022", "2023", "2024"}

# Realistic browser headers — investor pages often block bare Python requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Per-company configuration ─────────────────────────────────────────────────

COMPANIES = [
    {
        "name": "Infosys",
        "investor_page": "https://www.infosys.com/investors/reports-filings/annual-report/annual.html",
        "pdf_url_hints": [
            # Direct PDF paths used in recent Infosys annual reports
            "https://www.infosys.com/investors/reports-filings/annual-report/annual/Documents/infosys-ar-24.pdf",
            "https://www.infosys.com/investors/reports-filings/annual-report/annual/Documents/infosys-ar-23.pdf",
            "https://www.infosys.com/investors/reports-filings/annual-report/annual/Documents/infosys-ar-22.pdf",
        ],
        "filename_prefix": "Infosys",
    },
    {
        "name": "TCS",
        "investor_page": "https://www.tcs.com/investors/reports-and-filings/annual-reports",
        "pdf_url_hints": [
            # TCS hosts PDFs on their content delivery path; structure below is approximate
            "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2023-24/ar/tcs-annual-report-2023-2024.pdf",
            "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2022-23/ar/tcs-annual-report-2022-2023.pdf",
            "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2021-22/ar/tcs-annual-report-2021-2022.pdf",
        ],
        "filename_prefix": "TCS",
    },
    {
        "name": "Wipro",
        "investor_page": "https://www.wipro.com/investors/investors-reports/annual-report/",
        "pdf_url_hints": [
            # Wipro hosts on their investor reports page
            "https://www.wipro.com/content/dam/nexus/en/investors/annual-reports/2023-2024/wipro-annual-report-2023-24.pdf",
            "https://www.wipro.com/content/dam/nexus/en/investors/annual-reports/2022-2023/wipro-annual-report-2022-23.pdf",
            "https://www.wipro.com/content/dam/nexus/en/investors/annual-reports/2021-2022/wipro-annual-report-2021-22.pdf",
        ],
        "filename_prefix": "Wipro",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _year_from_text(text: str) -> str | None:
    """Extract the most recent 4-digit year that falls in TARGET_YEARS."""
    years = re.findall(r"\b(202[2-4])\b", text)
    return years[-1] if years else None


def _find_pdf_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """
    Return (url, year) pairs for PDF links found in HTML that look like
    annual reports for target years. Sorted newest-first.
    """
    # Find all hrefs ending in .pdf
    raw = re.findall(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, re.IGNORECASE)
    results: dict[str, str] = {}

    for href in raw:
        url = urljoin(base_url, href)
        year = _year_from_text(href) or _year_from_text(url)
        if year and year in TARGET_YEARS:
            # Prefer earlier entry to avoid overwriting with duplicates
            if year not in results:
                results[year] = url

    return sorted(results.items(), key=lambda x: x[0], reverse=True)


def _sanitise_filename(company: str, year: str) -> str:
    return f"{company}_Annual_Report_FY{year}.pdf"


def _download(client: httpx.Client, url: str, dest: Path) -> bool:
    """Stream-download url → dest. Returns True on success."""
    try:
        with client.stream("GET", url, follow_redirects=True, timeout=60) as resp:
            if resp.status_code != 200:
                return False
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                # Not a PDF response — skip silently
                return False
            total = 0
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    total += len(chunk)
            if total < 50_000:
                # Suspiciously small — likely an error page, not a real report
                dest.unlink(missing_ok=True)
                return False
            return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def _try_hints(client: httpx.Client, hints: list[str], prefix: str) -> list[str]:
    """Try hint URLs directly, return list of successfully saved filenames."""
    saved = []
    for url in hints:
        year = _year_from_text(url)
        if not year:
            continue
        dest = FILINGS_DIR / _sanitise_filename(prefix, year)
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            saved.append(dest.name)
            continue
        print(f"  [try]  {url}")
        if _download(client, url, dest):
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [ok]   {dest.name} ({size_mb:.1f} MB)")
            saved.append(dest.name)
        else:
            print(f"  [fail] {url}")
    return saved


def _try_scrape(client: httpx.Client, company: dict) -> list[str]:
    """Scrape investor page for PDF links, return list of saved filenames."""
    page_url = company["investor_page"]
    print(f"  [scrape] {page_url}")
    try:
        resp = client.get(page_url, follow_redirects=True, timeout=30)
        if resp.status_code != 200:
            print(f"  [scrape] HTTP {resp.status_code}")
            return []
    except Exception as exc:
        print(f"  [scrape] failed: {exc}")
        return []

    links = _find_pdf_links(resp.text, page_url)
    if not links:
        print("  [scrape] no annual-report PDF links found in page HTML")
        return []

    saved = []
    for year, url in links:
        dest = FILINGS_DIR / _sanitise_filename(company["filename_prefix"], year)
        if dest.exists():
            print(f"  [skip]  {dest.name} already exists")
            saved.append(dest.name)
            continue
        print(f"  [dl]    {url}")
        if _download(client, url, dest):
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [ok]    {dest.name} ({size_mb:.1f} MB)")
            saved.append(dest.name)
        else:
            print(f"  [fail]  {url}")
        time.sleep(1)  # polite crawl delay

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {FILINGS_DIR.resolve()}\n")

    total_saved: list[str] = []
    manual: list[dict] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for company in COMPANIES:
            name = company["name"]
            print(f"── {name} ───────────────────────────────────────")

            # 1. Try hint URLs first (faster, no scraping needed)
            saved = _try_hints(client, company["pdf_url_hints"], company["filename_prefix"])

            # 2. If hints didn't cover all three years, try scraping
            missing_years = TARGET_YEARS - {_year_from_text(f) or "" for f in saved}
            if missing_years:
                print(
                    f"  [scrape] hints missed years {sorted(missing_years)}, trying investor page…"
                )
                scraped = _try_scrape(client, company)
                saved.extend(scraped)

            # 3. Report what's still missing
            final_years = {_year_from_text(f) for f in saved} - {None}
            still_missing = TARGET_YEARS - final_years
            if still_missing:
                manual.append(
                    {
                        "company": name,
                        "years": sorted(still_missing),
                        "investor_page": company["investor_page"],
                    }
                )

            total_saved.extend(saved)
            print()
            time.sleep(2)  # polite delay between companies

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Downloaded / already present: {len(total_saved)} file(s)")

    if total_saved:
        for f in sorted(total_saved):
            path = FILINGS_DIR / f
            size_mb = path.stat().st_size / 1_048_576 if path.exists() else 0
            print(f"  {f}  ({size_mb:.1f} MB)")

    if manual:
        print("\n── Manual downloads needed ──────────────────────────")
        print("Some PDFs could not be downloaded automatically.")
        print("Download them from the links below and place in:")
        print(f"  {FILINGS_DIR.resolve()}\n")
        for m in manual:
            print(f"  {m['company']} — FY{', FY'.join(m['years'])}")
            print(f"    {m['investor_page']}")
            expected = [_sanitise_filename(m["company"], y) for y in m["years"]]
            print(f"    Save as: {', '.join(expected)}\n")

    if not total_saved and not manual:
        print("\nNothing downloaded. Check your internet connection.")
        sys.exit(1)

    needed = len(COMPANIES) * len(TARGET_YEARS)
    if len(total_saved) < needed:
        print(f"\n{needed - len(total_saved)} file(s) still needed for full corpus.")
    else:
        print("\nFull corpus ready. Run ingestion with:")
        print(
            "  python -c \"import asyncio; from ingestion.pipeline import ingest_directory; asyncio.run(ingest_directory('data/filings'))\""
        )


if __name__ == "__main__":
    main()
