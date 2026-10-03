"""Free market and filing data: Yahoo Finance (via yfinance) and SEC EDGAR.

Every public function returns a plain-text/JSON string sized for an LLM context
window. Outputs are deliberately compact - every character here is billed as
input tokens on each later turn of the agent loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

CACHE_DIR = Path(os.getenv("EQUITY_AGENT_CACHE", ".cache"))

# The SEC asks every client to identify itself with a contact email.
# https://www.sec.gov/os/accessing-edgar-data
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "equity-research-agent contact@example.com")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_num(value) -> str | None:
    """Format large numbers as 1.23B / 456.7M; pass through ratios."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(v) >= div:
                return f"{v / div:.2f}{suffix}"
        return f"{v:.4g}"
    return str(value)


def _pct(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return f"{value * 100:.1f}%"


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _sec_get(url: str) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def get_company_profile(ticker: str) -> str:
    """Snapshot: description, price, valuation multiples, margins, analyst view."""
    info = yf.Ticker(ticker).info or {}
    if not info.get("longName") and not info.get("shortName"):
        raise ValueError(f"No Yahoo Finance data found for ticker '{ticker}'.")

    summary = info.get("longBusinessSummary") or ""
    profile = _clean({
        "name": info.get("longName") or info.get("shortName"),
        "ticker": ticker.upper(),
        "exchange": info.get("exchange"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "employees": info.get("fullTimeEmployees"),
        "currency": info.get("currency"),
        "description": summary[:1200] + ("..." if len(summary) > 1200 else ""),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "market_cap": _fmt_num(info.get("marketCap")),
        "enterprise_value": _fmt_num(info.get("enterpriseValue")),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("trailingPegRatio"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "price_to_book": info.get("priceToBook"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "ev_to_revenue": info.get("enterpriseToRevenue"),
        "revenue_ttm": _fmt_num(info.get("totalRevenue")),
        "revenue_growth_yoy": _pct(info.get("revenueGrowth")),
        "earnings_growth_yoy": _pct(info.get("earningsGrowth")),
        "gross_margin": _pct(info.get("grossMargins")),
        "operating_margin": _pct(info.get("operatingMargins")),
        "profit_margin": _pct(info.get("profitMargins")),
        "return_on_equity": _pct(info.get("returnOnEquity")),
        "free_cash_flow_ttm": _fmt_num(info.get("freeCashflow")),
        "total_cash": _fmt_num(info.get("totalCash")),
        "total_debt": _fmt_num(info.get("totalDebt")),
        "dividend_yield": f"{info['dividendYield']:.2f}%" if info.get("dividendYield") else None,
        "beta": info.get("beta"),
        "analyst_recommendation": info.get("recommendationKey"),
        "analyst_count": info.get("numberOfAnalystOpinions"),
        "analyst_target_mean": info.get("targetMeanPrice"),
        "analyst_target_low": info.get("targetLowPrice"),
        "analyst_target_high": info.get("targetHighPrice"),
    })
    return json.dumps(profile, indent=1, default=str)


# Rows worth showing per statement - the full yfinance frames have 40-80 rows.
_STATEMENT_ROWS = {
    "income": [
        "Total Revenue", "Cost Of Revenue", "Gross Profit", "Research And Development",
        "Selling General And Administration", "Operating Income", "EBITDA",
        "Interest Expense", "Pretax Income", "Tax Provision", "Net Income",
        "Diluted EPS", "Diluted Average Shares",
    ],
    "balance": [
        "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
        "Accounts Receivable", "Inventory", "Current Assets", "Total Assets",
        "Current Liabilities", "Total Debt", "Long Term Debt",
        "Total Liabilities Net Minority Interest", "Stockholders Equity",
        "Retained Earnings", "Ordinary Shares Number",
    ],
    "cashflow": [
        "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
        "Stock Based Compensation", "Depreciation And Amortization",
        "Repurchase Of Capital Stock", "Cash Dividends Paid",
        "Acquisitions Net", "Issuance Of Debt", "Repayment Of Debt",
    ],
}


def get_financial_statements(ticker: str, statement: str, period: str = "annual") -> str:
    """Key rows of the income statement, balance sheet or cash flow statement."""
    if statement not in _STATEMENT_ROWS:
        raise ValueError(f"statement must be one of {list(_STATEMENT_ROWS)}")
    t = yf.Ticker(ticker)
    quarterly = period == "quarterly"
    frame: pd.DataFrame = {
        "income": t.quarterly_income_stmt if quarterly else t.income_stmt,
        "balance": t.quarterly_balance_sheet if quarterly else t.balance_sheet,
        "cashflow": t.quarterly_cashflow if quarterly else t.cashflow,
    }[statement]
    if frame is None or frame.empty:
        raise ValueError(f"No {period} {statement} statement available for {ticker}.")

    rows = [r for r in _STATEMENT_ROWS[statement] if r in frame.index]
    frame = frame.loc[rows].iloc[:, : (5 if quarterly else 4)]
    frame.columns = [c.strftime("%Y-%m-%d") for c in frame.columns]
    out = frame.map(lambda v: _fmt_num(v) or "n/a")
    return f"{ticker.upper()} {period} {statement} statement (most recent first)\n{out.to_string()}"


def get_price_performance(ticker: str) -> str:
    """Trailing returns vs the S&P 500, volatility and drawdown over 2 years."""
    # Sequential Ticker.history calls: yf.download's threads can lock yfinance's sqlite cache.
    closes = {tk: yf.Ticker(tk).history(period="2y", auto_adjust=True)["Close"] for tk in (ticker, "SPY")}
    hist = pd.DataFrame({tk: s.tz_localize(None) for tk, s in closes.items()}).dropna()
    if hist.empty:
        raise ValueError(f"No price history for {ticker}.")
    px, spy = hist[ticker], hist["SPY"]

    def ret(series, days):
        return _pct(series.iloc[-1] / series.iloc[-min(days, len(series))] - 1)

    windows = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "2y": len(px)}
    daily = px.pct_change().dropna()
    last_year = px.iloc[-252:]
    drawdown = (last_year / last_year.cummax() - 1).min()
    result = {
        "last_close": round(float(px.iloc[-1]), 2),
        "as_of": px.index[-1].strftime("%Y-%m-%d"),
        "returns": {w: ret(px, d) for w, d in windows.items()},
        "spy_returns": {w: ret(spy, d) for w, d in windows.items()},
        "annualized_volatility_1y": _pct(daily.iloc[-252:].std() * math.sqrt(252)),
        "max_drawdown_1y": _pct(drawdown),
        "200d_moving_avg": round(float(px.iloc[-200:].mean()), 2),
        "50d_moving_avg": round(float(px.iloc[-50:].mean()), 2),
    }
    return json.dumps(result, indent=1)


def compare_peers(tickers: list[str]) -> str:
    """Side-by-side valuation and profitability table for a list of tickers."""
    rows = {}
    for tk in tickers[:8]:
        try:
            info = yf.Ticker(tk).info or {}
        except Exception as exc:  # one bad ticker shouldn't sink the table
            rows[tk.upper()] = {"error": str(exc)[:80]}
            continue
        rows[tk.upper()] = {
            "mkt_cap": _fmt_num(info.get("marketCap")),
            "fwd_pe": _round(info.get("forwardPE")),
            "ttm_pe": _round(info.get("trailingPE")),
            "ev_ebitda": _round(info.get("enterpriseToEbitda")),
            "p_sales": _round(info.get("priceToSalesTrailing12Months")),
            "rev_growth": _pct(info.get("revenueGrowth")),
            "gross_mgn": _pct(info.get("grossMargins")),
            "op_mgn": _pct(info.get("operatingMargins")),
            "roe": _pct(info.get("returnOnEquity")),
        }
    return pd.DataFrame(rows).T.fillna("n/a").to_string()


def _round(v, nd=1):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ticker_to_cik_map() -> dict[str, int]:
    data = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def _cik(ticker: str) -> int:
    cik = _ticker_to_cik_map().get(ticker.upper().replace(".", "-"))
    if cik is None:
        raise ValueError(
            f"{ticker} not found in SEC EDGAR. Non-US companies that file 20-F/40-F "
            "or are not SEC registrants have no filings here."
        )
    return cik


def list_sec_filings(ticker: str, form_types: list[str] | None = None, limit: int = 8) -> str:
    """Recent filings (10-K, 10-Q, 8-K, ...) with URLs usable by read_sec_filing."""
    cik = _cik(ticker)
    recent = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    wanted = {f.upper() for f in (form_types or ["10-K", "10-Q", "8-K"])}
    out = []
    for form, filed, period, accession, doc, desc in zip(
        recent["form"], recent["filingDate"], recent["reportDate"],
        recent["accessionNumber"], recent["primaryDocument"], recent["primaryDocDescription"],
    ):
        if form.upper() not in wanted:
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}"
        out.append({"form": form, "filed": filed, "period": period, "description": desc, "url": url})
        if len(out) >= limit:
            break
    return json.dumps(out, indent=1)


def _heading(item: str, title: str) -> str:
    """Regex for e.g. 'Item 1A. Risk Factors'. Filings often split words across
    styled <span>s, which shows up as stray spaces ('RIS K FACTORS'), so allow an
    optional space between letters."""
    def word(w: str) -> str:
        return r"\s?".join(re.escape(ch) if ch != "'" else ".{0,3}" for ch in w)
    # The look-arounds skip quoted cross-references such as: see "Item 1A. Risk Factors," below.
    return (r"(?<![“\"])item\s*" + word(item) + r"\.?\s*"
            + r"\s*".join(word(w) for w in title.split()) + r"(?!\s*[,”\"’])(?!(?-i:\s+(?!and\b|of\b)[a-z]))")


def _any(*headings: tuple[str, str]) -> str:
    return "|".join(_heading(i, t) for i, t in headings)


# Section boundaries. Each start pattern also appears in the table of contents,
# so _extract_section picks the longest start->end span (the real section body).
_SECTIONS = {
    "10-K": {
        "business": (_heading("1", "business"), _heading("1a", "risk factors")),
        "risk_factors": (_heading("1a", "risk factors"),
                         _any(("1b", "unresolved"), ("1c", "cybersecurity"), ("2", "properties"))),
        "mdna": (_heading("7", "management's discussion"),
                 _any(("7a", "quantitative"), ("8", "financial statements"))),
    },
    "10-Q": {
        "mdna": (_heading("2", "management's discussion"),
                 _any(("3", "quantitative"), ("4", "controls"))),
        "risk_factors": (_heading("1a", "risk factors"),
                         _any(("2", "unregistered"), ("3", "defaults"), ("5", "other"), ("6", "exhibits"))),
    },
}


def _filing_text(url: str) -> str:
    """Download a filing and convert it to plain text, cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if path.exists():
        return path.read_text()
    if not url.startswith("https://www.sec.gov/"):
        raise ValueError("Only https://www.sec.gov/ filing URLs are supported.")
    soup = BeautifulSoup(_sec_get(url).content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline XBRL hides a large metadata header; drop it.
    for tag in soup.find_all(["ix:header"]):
        tag.decompose()
    text = soup.get_text(" ").replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    path.write_text(text)
    return text


def _extract_section(text: str, start_pat: str, end_pat: str) -> str | None:
    best = None
    for m in re.finditer(start_pat, text, flags=re.IGNORECASE):
        end = re.compile(end_pat, flags=re.IGNORECASE).search(text, m.end())
        stop = end.start() if end else min(len(text), m.end() + 200_000)
        if best is None or stop - m.start() > best[1] - best[0]:
            best = (m.start(), stop)  # leading newline is trimmed by .strip() below
    return text[best[0]:best[1]].strip() if best else None


def read_sec_filing(url: str, section: str = "mdna", offset: int = 0, max_chars: int = 12000) -> str:
    """Read one section of a 10-K/10-Q (or the start of any other filing), paginated."""
    text = _filing_text(url)
    form = "10-Q" if re.search(r"form\s*10-q", text[:5000], re.IGNORECASE) else "10-K"
    body = None
    if section != "full_text":
        patterns = _SECTIONS[form].get(section)
        if patterns is None:
            raise ValueError(f"Section '{section}' is not available for a {form}. "
                             f"Options: {list(_SECTIONS[form])} or 'full_text'.")
        body = _extract_section(text, *patterns)
    note = ""
    if body is None:
        if section != "full_text":
            note = "[section heading not found - showing the full filing text]\n"
        body, section, form = text, "full_text", "filing"
    elif len(body) < 2000:
        note = ("[this section is very short - the company probably incorporates it by reference "
                "(e.g. a 10-Q pointing back to the 10-K, or MD&A filed as an exhibit). "
                "Try another filing or section='full_text'.]\n")

    chunk = body[offset: offset + max_chars]
    remaining = max(0, len(body) - offset - max_chars)
    footer = (f"\n\n[{remaining:,} more characters - call again with offset={offset + max_chars} to continue]"
              if remaining else "\n\n[end of section]")
    return f"{note}[{form} - section: {section} - chars {offset:,}-{offset + len(chunk):,} of {len(body):,}]\n{chunk}{footer}"
