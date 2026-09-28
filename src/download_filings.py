import os
import time
import requests


# ============================================================
# CONFIGURATION
# ============================================================

USER_AGENT = "SEC-RAG-Project larixo6289@kingdais.com"

OUTPUT_DIR = "data/raw"

# Companies we want to collect
COMPANIES = {
    "unitedhealth": "UNH",
    "eli-lilly": "LLY",
    "abbvie": "ABBV",
    "merck": "MRK",
    "pfizer": "PFE",
    "bristol-myers-squibb": "BMY",
    "amgen": "AMGN",

    "home-depot": "HD",
    "lowes": "LOW",
    "target": "TGT",
    "starbucks": "SBUX",
    "mcdonalds": "MCD",
    "nike": "NKE",
    "booking-holdings": "BKNG",
    "airbnb": "ABNB",

    "american-express": "AXP",
    "goldman-sachs": "GS",
    "morgan-stanley": "MS",
    "citigroup": "C",
    "wells-fargo": "WFC",
    "blackrock": "BLK",
    "charles-schwab": "SCHW",
    "paypal": "PYPL",

    "uber": "UBER",
    "lyft": "LYFT",
    "netflix": "NFLX",
    "disney": "DIS",
    "comcast": "CMCSA",

    "accenture": "ACN",
    "ibm": "IBM",
    "cisco": "CSCO",
    "salesforce": "CRM",
    "servicenow": "NOW",
    "palantir": "PLTR",

    "general-motors": "GM",
    "ford": "F",
    "caterpillar": "CAT",
    "deere": "DE",
    "lockheed-martin": "LMT",
}


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
})


# ============================================================
# GET TICKER -> CIK MAPPING
# ============================================================

print("Getting SEC ticker/CIK mapping...")

response = session.get(
    "https://www.sec.gov/files/company_tickers.json",
    timeout=30
)

response.raise_for_status()

ticker_data = response.json()

ticker_to_cik = {}

for item in ticker_data.values():

    ticker = item["ticker"].upper()

    cik = str(item["cik_str"]).zfill(10)

    ticker_to_cik[ticker] = cik


print("Ticker mapping loaded.")
print()


# ============================================================
# PROCESS EACH COMPANY
# ============================================================

for company_name, ticker in COMPANIES.items():

    print("=" * 60)
    print(f"Processing {company_name.upper()} ({ticker})")
    print("=" * 60)

    ticker = ticker.upper()

    # --------------------------------------------------------
    # Find CIK
    # --------------------------------------------------------

    if ticker not in ticker_to_cik:

        print(f"ERROR: Could not find CIK for {ticker}")
        continue

    cik = ticker_to_cik[ticker]

    print(f"CIK: {cik}")

    # --------------------------------------------------------
    # Create company directory
    # --------------------------------------------------------

    company_dir = os.path.join(
        OUTPUT_DIR,
        company_name
    )

    os.makedirs(company_dir, exist_ok=True)

    # --------------------------------------------------------
    # Get filing history
    # --------------------------------------------------------

    submissions_url = (
        f"https://data.sec.gov/submissions/CIK{cik}.json"
    )

    print("Getting filing history...")

    response = session.get(
        submissions_url,
        timeout=30
    )

    response.raise_for_status()

    company_data = response.json()

    print(
        f"Company: {company_data['name']}"
    )

    # --------------------------------------------------------
    # Extract 10-K and 10-Q filings
    # --------------------------------------------------------

    recent = company_data["filings"]["recent"]

    filings = []

    for i, form in enumerate(recent["form"]):

        if form not in ["10-K", "10-Q"]:
            continue

        filing = {
            "form": form,
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "accession_number": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
        }

        filings.append(filing)

    # --------------------------------------------------------
    # Select filings
    # --------------------------------------------------------

    ten_k = [
        filing
        for filing in filings
        if filing["form"] == "10-K"
    ][:3]

    ten_q = [
        filing
        for filing in filings
        if filing["form"] == "10-Q"
    ][:4]

    selected_filings = ten_k + ten_q

    print()
    print("Selected filings:")

    for filing in selected_filings:

        print(
            f"  {filing['form']} | "
            f"{filing['filing_date']} | "
            f"{filing['primary_document']}"
        )

    # --------------------------------------------------------
    # Download documents
    # --------------------------------------------------------

    for filing in selected_filings:

        accession = (
            filing["accession_number"]
            .replace("-", "")
        )

        primary_document = filing["primary_document"]

        # SEC archive URL
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/"
            f"{accession}/"
            f"{primary_document}"
        )

        filename = (
            f"{filing['filing_date']}_"
            f"{filing['form']}_"
            f"{primary_document}"
        )

        filepath = os.path.join(
            company_dir,
            filename
        )

        # ----------------------------------------------------
        # Don't download if already exists
        # ----------------------------------------------------

        if os.path.exists(filepath):

            print(
                f"Already exists: {filename}"
            )

            continue

        print()
        print(
            f"Downloading {filing['form']} "
            f"{filing['filing_date']}..."
        )

        try:

            response = session.get(
                filing_url,
                timeout=60
            )

            response.raise_for_status()

            with open(
                filepath,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(response.text)

            print(
                f"Saved: {filepath}"
            )

        except requests.RequestException as error:

            print(
                f"ERROR downloading {filing_url}"
            )

            print(error)

        # ----------------------------------------------------
        # Be polite to SEC servers
        # ----------------------------------------------------

        time.sleep(1)


print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print()
print(
    f"Your filings are inside: {OUTPUT_DIR}"
)
