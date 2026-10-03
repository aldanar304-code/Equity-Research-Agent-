"""Download a company's SEC filings, keep them on disk, and make them searchable.

Fetched up front, in code, before the agent starts (no tokens spent):
  - last 3 annual reports (10-K), last 4 quarterly reports (10-Q)
  - last 12 current reports (8-K) with their press-release exhibits (EX-99)
  - the latest proxy statement (DEF 14A): executive pay, board, governance
  - insider transactions (Form 4) from the past 12 months
plus 10 years of standardized financials from the SEC's XBRL "company facts" API.

The agent then *searches* the filings instead of reading them end to end, which keeps
a full due-diligence run affordable.
"""

from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from .data import _cik, _filing_text, _sec_get

FILINGS_DIR = Path("filings")

# form type -> how many of the most recent to download
WANTED = {"10-K": 3, "10-Q": 4, "8-K": 12, "DEF 14A": 1}
FORM4_LOOKBACK_DAYS = 365
FORM4_MAX = 40


@dataclass
class Filing:
    doc_id: str          # short id the agent uses, e.g. "10-K-2026"
    form: str
    filed: str
    period: str
    description: str
    url: str
    chars: int = 0


def _polite_get(url: str):
    time.sleep(0.12)  # SEC fair-access limit is 10 requests/second
    return _sec_get(url)


def _archive_url(cik: int, accession: str, name: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{name}"


def _exhibit_urls(cik: int, accession: str, primary_doc: str) -> list[tuple[str, str]]:
    """Exhibits attached to an 8-K - usually the earnings press release and CFO commentary.

    Most filers name these *ex99*.htm, but some (e.g. NVIDIA: q2fy27pr.htm) don't, so any
    other .htm document in the filing that isn't SEC boilerplate counts too."""
    try:
        idx = _polite_get(_archive_url(cik, accession, "index.json")).json()
    except Exception:
        return []
    out = []
    for item in idx.get("directory", {}).get("item", []):
        name = item.get("name", "")
        lower = name.lower()
        if not lower.endswith((".htm", ".html")) or name == primary_doc:
            continue
        if re.match(r"r\d+\.htm$", lower) or "index" in lower:  # XBRL viewer pages, index pages
            continue
        out.append((name, _archive_url(cik, accession, name)))
    out.sort(key=lambda nu: not re.search(r"ex-?99|pr\.htm|release", nu[0], re.IGNORECASE))
    return out[:2]


def download_filings(ticker: str, emit=lambda kind, payload: None) -> list[Filing]:
    """Download the filing set for a ticker (cached on disk) and return a manifest."""
    ticker = ticker.upper()
    cik = _cik(ticker)
    folder = FILINGS_DIR / ticker
    folder.mkdir(parents=True, exist_ok=True)

    recent = _polite_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    rows = list(zip(recent["form"], recent["filingDate"], recent["reportDate"],
                    recent["accessionNumber"], recent["primaryDocument"], recent["primaryDocDescription"]))

    manifest: list[Filing] = []
    counts: Counter = Counter()
    used_ids: Counter = Counter()

    def add(form, filed, period, desc, url):
        base = f"{form.replace(' ', '')}-{(period or filed)[:7]}"
        used_ids[base] += 1
        doc_id = base if used_ids[base] == 1 else f"{base}-{used_ids[base]}"
        emit("download", {"doc_id": doc_id, "form": form, "filed": filed})
        text = _filing_text(url)
        (folder / f"{doc_id}.txt").write_text(text)
        manifest.append(Filing(doc_id, form, filed, period, desc, url, len(text)))

    for form, filed, period, accession, doc, desc in rows:
        if form not in WANTED or counts[form] >= WANTED[form]:
            continue
        counts[form] += 1
        try:
            add(form, filed, period, desc or form, _archive_url(cik, accession, doc))
            if form == "8-K":
                for name, url in _exhibit_urls(cik, accession, doc):
                    add("8-K-EX", filed, period, f"8-K exhibit {name}", url)
        except Exception as exc:  # one broken filing shouldn't stop the rest
            emit("download_error", {"form": form, "filed": filed, "error": str(exc)[:120]})

    insiders = _download_form4(cik, rows, emit)
    (folder / "insider_transactions.json").write_text(json.dumps(insiders, indent=1))
    (folder / "manifest.json").write_text(json.dumps([asdict(f) for f in manifest], indent=1))
    return manifest


# ---------------------------------------------------------------------------
# Insider transactions (Form 4)
# ---------------------------------------------------------------------------

def _xml_text(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def _download_form4(cik: int, rows, emit) -> list[dict]:
    cutoff = (date.today() - timedelta(days=FORM4_LOOKBACK_DAYS)).isoformat()
    txns, n = [], 0
    for form, filed, _period, accession, doc, _desc in rows:
        if form != "4" or filed < cutoff:
            continue
        if n >= FORM4_MAX:
            break
        n += 1
        # primaryDocument points at the XSL-rendered view; the raw XML sits next to the folder.
        raw = doc.split("/")[-1]
        try:
            root = ET.fromstring(_polite_get(_archive_url(cik, accession, raw)).content)
        except Exception:
            continue
        owner = _xml_text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
        title = _xml_text(root, "reportingOwner/reportingOwnerRelationship/officerTitle")
        if not title and _xml_text(root, "reportingOwner/reportingOwnerRelationship/isDirector") in ("1", "true"):
            title = "Director"
        for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
            shares = _xml_text(t, "transactionAmounts/transactionShares/value")
            price = _xml_text(t, "transactionAmounts/transactionPricePerShare/value")
            txns.append({
                "date": _xml_text(t, "transactionDate/value"),
                "insider": owner, "title": title,
                "code": _xml_text(t, "transactionCoding/transactionCode"),
                "shares": float(shares) if shares else 0.0,
                "price": float(price) if price else None,
            })
    emit("download", {"doc_id": f"{n} insider filings (Form 4)", "form": "4", "filed": ""})
    return txns


_CODES = {"P": "open-market buy", "S": "open-market sale", "A": "grant/award", "M": "option exercise",
          "F": "tax withholding", "G": "gift"}


def insider_summary(ticker: str) -> str:
    path = FILINGS_DIR / ticker.upper() / "insider_transactions.json"
    if not path.exists():
        return "No insider data downloaded."
    txns = json.loads(path.read_text())
    if not txns:
        return "No Form 4 insider transactions in the past 12 months."

    by_code = defaultdict(lambda: [0.0, 0.0, 0])
    by_person = defaultdict(lambda: {"bought_usd": 0.0, "sold_usd": 0.0})
    for t in txns:
        value = t["shares"] * (t["price"] or 0)
        agg = by_code[t["code"]]
        agg[0] += t["shares"]; agg[1] += value; agg[2] += 1
        key = f"{t['insider']} ({t['title'] or 'insider'})"
        if t["code"] == "P":
            by_person[key]["bought_usd"] += value
        elif t["code"] == "S":
            by_person[key]["sold_usd"] += value

    lines = [f"Insider transactions, past {FORM4_LOOKBACK_DAYS} days ({len(txns)} transactions):"]
    for code, (shares, value, count) in sorted(by_code.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {code} {_CODES.get(code, 'other')}: {count} txns, {shares:,.0f} shares, ${value / 1e6:,.1f}M")
    lines.append("Open-market activity by insider (codes P/S; grants and tax withholding excluded):")
    for person, v in sorted(by_person.items(), key=lambda kv: -(kv[1]["bought_usd"] + kv[1]["sold_usd"]))[:10]:
        lines.append(f"  {person}: bought ${v['bought_usd'] / 1e6:,.1f}M, sold ${v['sold_usd'] / 1e6:,.1f}M")
    lines.append("Note: many insider sales are pre-scheduled (10b5-1 plans); open-market buys are the stronger signal.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search across downloaded filings
# ---------------------------------------------------------------------------

_INDEX: dict[str, tuple[list[dict], Counter, float]] = {}
_WORD = re.compile(r"[a-z0-9][a-z0-9\-&]*")


def _manifest(ticker: str) -> list[Filing]:
    path = FILINGS_DIR / ticker.upper() / "manifest.json"
    return [Filing(**f) for f in json.loads(path.read_text())] if path.exists() else []


def _build_index(ticker: str):
    chunks, df = [], Counter()
    for f in _manifest(ticker):
        text = (FILINGS_DIR / ticker.upper() / f"{f.doc_id}.txt").read_text()
        for start in range(0, len(text), 1300):
            body = text[start: start + 1500]
            terms = Counter(_WORD.findall(body.lower()))
            chunks.append({"doc": f, "start": start, "body": body, "terms": terms, "len": sum(terms.values())})
            df.update(terms.keys())
    avg_len = sum(c["len"] for c in chunks) / max(1, len(chunks))
    _INDEX[ticker.upper()] = (chunks, df, avg_len)


def search_filings(ticker: str, query: str, form_types: list[str] | None = None, top_k: int = 6) -> str:
    """BM25 keyword search over every downloaded filing; returns the best-matching passages."""
    ticker = ticker.upper()
    if ticker not in _INDEX:
        _build_index(ticker)
    chunks, df, avg_len = _INDEX[ticker]
    if not chunks:
        return "No filings downloaded for this company."
    q_terms = [t for t in _WORD.findall(query.lower()) if len(t) > 1]
    wanted = {f.upper() for f in form_types} if form_types else None
    n = len(chunks)
    scored = []
    for c in chunks:
        if wanted and c["doc"].form.upper() not in wanted:
            continue
        score = 0.0
        for t in q_terms:
            tf = c["terms"].get(t, 0)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * c["len"] / avg_len))
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda sc: -sc[0])

    results, seen = [], set()
    for score, c in scored:
        key = (c["doc"].doc_id, c["start"] // 3000)  # avoid near-duplicate neighbouring chunks
        if key in seen:
            continue
        seen.add(key)
        d = c["doc"]
        results.append(f"--- [{d.doc_id}] {d.form} filed {d.filed} (offset {c['start']:,})\n{c['body'].strip()}")
        if len(results) >= top_k:
            break
    return "\n\n".join(results) if results else f"No passages matched '{query}'."


def read_filing(ticker: str, doc_id: str, section: str = "full_text", offset: int = 0) -> str:
    from .data import read_sec_filing
    for f in _manifest(ticker):
        if f.doc_id == doc_id:
            if f.form not in ("10-K", "10-Q"):
                section = "full_text"
            return f"[{f.doc_id}: {f.form} filed {f.filed}]\n" + read_sec_filing(f.url, section, offset)
    raise ValueError(f"Unknown doc_id '{doc_id}'. Use an id from the filing manifest.")


def manifest_summary(manifest: list[Filing]) -> str:
    if not manifest:
        return "No SEC filings could be downloaded (the company may not be an SEC registrant)."
    lines = [f"  {f.doc_id:<16} {f.form:<8} filed {f.filed}  {f.chars / 1000:,.0f}k chars  {f.description[:60]}"
             for f in manifest]
    return "Downloaded filings (use these doc_ids):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# 10-year standardized financials (XBRL company facts)
# ---------------------------------------------------------------------------

_XBRL = {
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "Gross profit": ["GrossProfit"],
    "Operating income": ["OperatingIncomeLoss"],
    "Net income": ["NetIncomeLoss"],
    "Diluted EPS": ["EarningsPerShareDiluted"],
    "Operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"],
    "R&D": ["ResearchAndDevelopmentExpense"],
    "Buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "Dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "Diluted shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}


def financial_history_data(ticker: str, years: int = 10) -> dict[str, dict[str, float]]:
    """{metric: {fiscal-year-end 'YYYY-MM': value}} from 10-K XBRL data, as reported to the SEC."""
    cik = _cik(ticker)
    facts = _polite_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    gaap = facts.get("facts", {}).get("us-gaap", {})

    table: dict[str, dict[str, float]] = {}
    for label, tags in _XBRL.items():
        # Companies switch tags over time (e.g. NVIDIA moved from RevenueFromContract... to Revenues),
        # so merge them: earlier tags in the list win for a given year, later tags fill the gaps.
        merged: dict[str, float] = {}
        for tag in tags:
            best: dict[str, tuple[str, float]] = {}
            units = gaap.get(tag, {}).get("units", {})
            for unit_rows in units.values():
                for r in unit_rows:
                    if r.get("form") != "10-K" or r.get("fp") != "FY" or "start" not in r:
                        continue
                    span = (datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).days
                    if not 340 <= span <= 380:
                        continue
                    year = r["end"][:7]  # fiscal year-end month, e.g. 2026-06
                    # keep the most recently filed value (restatements win)
                    if year not in best or r["filed"] > best[year][0]:
                        best[year] = (r["filed"], r["val"])
            for y, (_, v) in best.items():
                merged.setdefault(y, v)
        table[label] = merged
    if table["Operating cash flow"] and table["Capex"]:
        # only where capex is known - otherwise FCF would silently equal operating cash flow
        table["Free cash flow"] = {y: ocf - table["Capex"][y]
                                   for y, ocf in table["Operating cash flow"].items() if y in table["Capex"]}
    keep = sorted({y for col in table.values() for y in col})[-years:]
    return {label: {y: v for y, v in col.items() if y in keep} for label, col in table.items()}


def financial_history(ticker: str, years: int = 10) -> str:
    """Annual figures from 10-K XBRL data, as reported to the SEC."""
    table = financial_history_data(ticker, years)
    all_years = sorted({y for col in table.values() for y in col})[-years:]
    if not all_years:
        return "No XBRL annual data available."

    def fmt(label, v):
        if v is None:
            return "n/a"
        if label == "Diluted EPS":
            return f"{v:.2f}"
        return f"{v / 1e9:,.2f}B" if abs(v) >= 1e8 else f"{v / 1e6:,.1f}M"

    header = f"{'FY ending':<22}" + "".join(f"{y:>11}" for y in all_years)
    lines = [f"{ticker.upper()} annual financials from SEC XBRL (10-K, as reported; EPS and share counts "
             f"are NOT adjusted for later stock splits)", header]
    for label, col in table.items():
        if col:
            lines.append(f"{label:<22}" + "".join(f"{fmt(label, col.get(y)):>11}" for y in all_years))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------

def dcf_value(base_fcf_billions: float, growth_rates: list[float], terminal_growth: float,
              discount_rate: float, net_cash_billions: float, shares_billions: float):
    """Returns (value per share, yearly rows, terminal value, PV of TV, PV of FCF, equity value)."""
    fcf, pv, rows = base_fcf_billions, 0.0, []
    for year, g in enumerate(growth_rates, start=1):
        fcf *= 1 + g
        disc = fcf / (1 + discount_rate) ** year
        pv += disc
        rows.append((year, g, fcf, disc))
    tv = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** len(growth_rates)
    equity = pv + pv_tv + net_cash_billions
    return equity / shares_billions, rows, tv, pv_tv, pv, equity


def run_dcf(base_fcf_billions: float, growth_rates: list[float], terminal_growth: float,
            discount_rate: float, net_cash_billions: float, shares_billions: float,
            current_price: float | None = None) -> str:
    """Discounted cash flow valuation with a sensitivity table. Rates are decimals (0.09 = 9%)."""
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must exceed terminal_growth.")
    if not growth_rates or shares_billions <= 0:
        raise ValueError("Need at least one growth rate and a positive share count.")

    def value(dr, tg):
        return dcf_value(base_fcf_billions, growth_rates, tg, dr, net_cash_billions, shares_billions)

    per_share, rows, tv, pv_tv, pv_fcf, equity = value(discount_rate, terminal_growth)
    out = ["Year  Growth   FCF ($B)   PV ($B)"]
    out += [f"{y:>4}  {g:>6.1%}  {f:>9.1f}  {d:>8.1f}" for y, g, f, d in rows]
    out += [
        f"PV of forecast FCF:     ${pv_fcf:,.1f}B",
        f"Terminal value:         ${tv:,.1f}B  (PV ${pv_tv:,.1f}B = {pv_tv / (pv_fcf + pv_tv):.0%} of EV)",
        f"Net cash (debt):        ${net_cash_billions:,.1f}B",
        f"Equity value:           ${equity:,.1f}B",
        f"Value per share:        ${per_share:,.2f}"
        + (f"  ({per_share / current_price - 1:+.1%} vs ${current_price:,.2f})" if current_price else ""),
        "",
        "Sensitivity (value per share): rows = discount rate, columns = terminal growth",
    ]
    tgs = [terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005]
    out.append("        " + "".join(f"{tg:>10.1%}" for tg in tgs))
    for dr in (discount_rate - 0.01, discount_rate, discount_rate + 0.01):
        cells = "".join(f"{value(dr, tg)[0]:>10,.0f}" if dr > tg else f"{'n/a':>10}" for tg in tgs)
        out.append(f"{dr:>7.1%} {cells}")
    return "\n".join(out)
