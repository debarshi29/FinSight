"""
Download Infosys, TCS, and Wipro annual reports for FY2022-FY2024.

Usage:
    uv run python scripts/download_filings.py

Strategy (tried in order per company):
  1. BSE India filing API  — exchange-hosted, most reliably accessible
  2. Direct PDF hint URLs  — known paths from company sites
  3. Browser fallback      — opens the investor page in your browser
                             and waits for you to manually save the file

Files are saved to data/filings/ with names like Infosys_Annual_Report_FY2024.pdf
"""

from __future__ import annotations

import re
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urljoin

import httpx

FILINGS_DIR = Path(__file__).parent.parent / "data" / "filings"
TARGET_YEARS = {"2022", "2023", "2024"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# BSE scrip codes for each company
BSE_SCRIP = {
    "Infosys": "500209",
    "TCS": "532540",
    "Wipro": "507685",
}

# Known direct PDF URLs — tried if BSE fails
DIRECT_HINTS: dict[str, list[str]] = {
    "Infosys": [
        "https://www.infosys.com/investors/reports-filings/annual-report/annual/Documents/infosys-ar-24.pdf",
        "https://www.infosys.com/investors/reports-filings/annual-report/annual/Documents/infosys-ar-23.pdf",
        "https://www.infosys.com/investors/reports-filings/annual-report/annual/Documents/infosys-ar-22.pdf",
    ],
    "TCS": [
        "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2023-24/ar/tcs-annual-report-2023-2024.pdf",
        "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2022-23/ar/tcs-annual-report-2022-2023.pdf",
        "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2021-22/ar/tcs-annual-report-2021-2022.pdf",
    ],
    "Wipro": [
        "https://www.wipro.com/content/dam/nexus/en/investors/annual-reports/2023-2024/wipro-annual-report-2023-24.pdf",
        "https://www.wipro.com/content/dam/nexus/en/investors/annual-reports/2022-2023/wipro-annual-report-2022-23.pdf",
        "https://www.wipro.com/content/dam/nexus/en/investors/annual-reports/2021-2022/wipro-annual-report-2021-22.pdf",
    ],
}

INVESTOR_PAGES: dict[str, str] = {
    "Infosys": "https://www.infosys.com/investors/reports-filings/annual-report/annual.html",
    "TCS": "https://www.tcs.com/investors/reports-and-filings/annual-reports",
    "Wipro": "https://www.wipro.com/investors/investors-reports/annual-report/",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _year_from_text(text: str) -> str | None:
    years = re.findall(r"\b(202[2-4])\b", text)
    return years[-1] if years else None


def _sanitise_filename(company: str, year: str) -> str:
    return f"{company}_Annual_Report_FY{year}.pdf"


def _download(client: httpx.Client, url: str, dest: Path, label: str = "") -> bool:
    """Stream-download url to dest. Returns True on success."""
    try:
        with client.stream("GET", url, follow_redirects=True, timeout=90) as resp:
            if resp.status_code != 200:
                print(f"  [fail] {label or url}  HTTP {resp.status_code}")
                return False
            ct = resp.headers.get("content-type", "")
            if "pdf" not in ct and not url.lower().split("?")[0].endswith(".pdf"):
                print(f"  [fail] {label or url}  not a PDF (content-type: {ct[:40]})")
                return False
            total = 0
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    total += len(chunk)
            if total < 50_000:
                dest.unlink(missing_ok=True)
                print(f"  [fail] {label or url}  too small ({total} bytes) — likely an error page")
                return False
            return True
    except Exception as exc:
        dest.unlink(missing_ok=True)
        print(f"  [fail] {label or url}  {exc}")
        return False


# ── Source 1: BSE India filing API ───────────────────────────────────────────


def _bse_annual_report_urls(client: httpx.Client, scrip_code: str) -> dict[str, str]:
    """
    Query BSE's public annual-report API and return {year: pdf_url}.
    BSE hosts exchange-submitted filings and the API is more accessible
    than company-direct sites that sit behind Cloudflare.
    """
    url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w?scripcode={scrip_code}&type=EQ"
    headers = {
        **HEADERS,
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }
    try:
        resp = client.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"  [bse] HTTP {resp.status_code}")
            return {}
        data = resp.json()
    except Exception as exc:
        print(f"  [bse] {exc}")
        return {}

    results: dict[str, str] = {}
    # Response is a list of objects; each has PDFLINKTOOPEN and YEAR_OF_REPORT (or similar)
    for item in data if isinstance(data, list) else data.get("Table", []):
        raw_year = str(item.get("YEAR_OF_REPORT") or item.get("Year") or "")
        pdf_link = str(item.get("PDFLINKTOOPEN") or item.get("PDFLink") or "")
        year = _year_from_text(raw_year) or _year_from_text(pdf_link)
        if year and year in TARGET_YEARS and pdf_link:
            if not pdf_link.startswith("http"):
                pdf_link = "https://www.bseindia.com" + pdf_link
            if year not in results:
                results[year] = pdf_link
    return results


def _try_bse(client: httpx.Client, company: str) -> list[str]:
    scrip = BSE_SCRIP.get(company)
    if not scrip:
        return []
    print(f"  [bse] querying exchange filings for scrip {scrip}...")
    urls = _bse_annual_report_urls(client, scrip)
    if not urls:
        print("  [bse] no PDF links returned from BSE API")
        return []

    saved = []
    for year, url in sorted(urls.items(), reverse=True):
        dest = FILINGS_DIR / _sanitise_filename(company, year)
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            saved.append(dest.name)
            continue
        print(f"  [bse] FY{year}  {url[:80]}...")
        if _download(client, url, dest, label=f"BSE FY{year}"):
            mb = dest.stat().st_size / 1_048_576
            print(f"  [ok]   {dest.name}  ({mb:.1f} MB)")
            saved.append(dest.name)
        time.sleep(1)
    return saved


# ── Source 2: Direct PDF hints ────────────────────────────────────────────────


def _try_hints(client: httpx.Client, company: str) -> list[str]:
    saved = []
    for url in DIRECT_HINTS.get(company, []):
        year = _year_from_text(url)
        if not year:
            continue
        dest = FILINGS_DIR / _sanitise_filename(company, year)
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            saved.append(dest.name)
            continue
        print(f"  [hint] FY{year}  {url[:80]}")
        if _download(client, url, dest, label=f"hint FY{year}"):
            mb = dest.stat().st_size / 1_048_576
            print(f"  [ok]   {dest.name}  ({mb:.1f} MB)")
            saved.append(dest.name)
        time.sleep(1)
    return saved


# ── Source 3: Investor page scrape ────────────────────────────────────────────


def _try_scrape(client: httpx.Client, company: str) -> list[str]:
    page_url = INVESTOR_PAGES[company]
    print(f"  [scrape] {page_url}")
    try:
        resp = client.get(page_url, follow_redirects=True, timeout=30)
        if resp.status_code != 200:
            print(f"  [scrape] HTTP {resp.status_code}")
            return []
    except Exception as exc:
        print(f"  [scrape] {exc}")
        return []

    raw = re.findall(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', resp.text, re.IGNORECASE)
    found: dict[str, str] = {}
    for href in raw:
        url = urljoin(page_url, href)
        year = _year_from_text(href) or _year_from_text(url)
        if year and year in TARGET_YEARS and year not in found:
            found[year] = url

    if not found:
        print("  [scrape] no PDF links found in page HTML")
        return []

    saved = []
    for year, url in sorted(found.items(), reverse=True):
        dest = FILINGS_DIR / _sanitise_filename(company, year)
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            saved.append(dest.name)
            continue
        print(f"  [scrape] FY{year}  {url[:80]}")
        if _download(client, url, dest, label=f"scrape FY{year}"):
            mb = dest.stat().st_size / 1_048_576
            print(f"  [ok]   {dest.name}  ({mb:.1f} MB)")
            saved.append(dest.name)
        time.sleep(1)
    return saved


# ── Source 4: Browser fallback ────────────────────────────────────────────────


def _browser_fallback(company: str, missing_years: list[str]) -> None:
    page = INVESTOR_PAGES[company]
    filenames = [_sanitise_filename(company, y) for y in missing_years]
    print()
    print(f"  [browser] Opening {page}")
    print(f"  Save the FY{', FY'.join(missing_years)} annual report PDF(s) as:")
    for fn in filenames:
        print(f"    {FILINGS_DIR / fn}")
    print("  Then press Enter to continue...")
    webbrowser.open(page)
    input()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {FILINGS_DIR.resolve()}\n")

    all_saved: list[str] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for company in ("Infosys", "TCS", "Wipro"):
            print(f"-- {company} " + "-" * (40 - len(company)))

            saved: list[str] = []

            def present_years() -> set[str | None]:
                return {_year_from_text(f) for f in saved} - {None}

            # Check what's already on disk
            for year in TARGET_YEARS:
                dest = FILINGS_DIR / _sanitise_filename(company, year)
                if dest.exists():
                    print(f"  [skip] {dest.name} already exists")
                    saved.append(dest.name)

            missing = TARGET_YEARS - present_years()

            # Stage 1: BSE API
            if missing:
                saved.extend(_try_bse(client, company))
                missing = TARGET_YEARS - present_years()

            # Stage 2: direct hints
            if missing:
                print(f"  [hint] trying direct PDF URLs for years {sorted(missing)}...")
                saved.extend(_try_hints(client, company))
                missing = TARGET_YEARS - present_years()

            # Stage 3: scrape investor page
            if missing:
                print(f"  [scrape] trying investor page for years {sorted(missing)}...")
                saved.extend(_try_scrape(client, company))
                missing = TARGET_YEARS - present_years()

            # Stage 4: open browser, ask user to save manually
            if missing:
                print(f"  [manual] could not auto-download FY{', FY'.join(sorted(missing))}")
                _browser_fallback(company, sorted(missing))
                # Check again after user interaction
                for year in list(missing):
                    dest = FILINGS_DIR / _sanitise_filename(company, year)
                    if dest.exists():
                        saved.append(dest.name)

            all_saved.extend(saved)
            print()
            time.sleep(2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("=" * 55)
    present = [f for f in all_saved if (FILINGS_DIR / f).exists()]
    print(f"Files in data/filings/: {len(present)}")
    for f in sorted(set(present)):
        mb = (FILINGS_DIR / f).stat().st_size / 1_048_576
        print(f"  {f}  ({mb:.1f} MB)")

    needed = len(("Infosys", "TCS", "Wipro")) * len(TARGET_YEARS)
    if len(present) >= needed:
        print("\nFull corpus ready. Next step — ingest:")
        print(
            '  uv run python -c "import asyncio; '
            "from ingestion.pipeline import ingest_directory; "
            "asyncio.run(ingest_directory('data/filings'))\""
        )
    else:
        still = needed - len(present)
        print(f"\n{still} file(s) still needed.")
        print(f"Place them in:  {FILINGS_DIR.resolve()}")
        print("Then re-run this script or ingest directly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
