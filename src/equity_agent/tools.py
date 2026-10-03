"""Tool schemas exposed to Claude and the dispatcher that runs them."""

from __future__ import annotations

from . import data, filings

TICKER = {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}

CUSTOM_TOOLS = [
    {
        "name": "get_company_profile",
        "description": (
            "Company snapshot from Yahoo Finance: business description, sector, current price, "
            "52-week range, market cap, valuation multiples (P/E, EV/EBITDA, P/S), margins, "
            "growth, balance sheet totals and analyst consensus. Call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": TICKER},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_financial_statements",
        "description": (
            "Key line items from the income statement, balance sheet or cash flow statement. "
            "Annual returns the last 4 fiscal years; quarterly returns the last 5 quarters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": TICKER,
                "statement": {"type": "string", "enum": ["income", "balance", "cashflow"]},
                "period": {"type": "string", "enum": ["annual", "quarterly"]},
            },
            "required": ["ticker", "statement", "period"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_price_performance",
        "description": (
            "Trailing 1m/3m/6m/1y/2y total returns versus the S&P 500 (SPY), 1-year volatility, "
            "max drawdown and 50/200-day moving averages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": TICKER},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_peers",
        "description": (
            "Side-by-side table of market cap, forward/trailing P/E, EV/EBITDA, P/S, revenue growth, "
            "margins and ROE. Pass the target company first, then 3-6 close competitors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "2-8 tickers"},
            },
            "required": ["tickers"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_analyst_estimates",
        "description": (
            "Wall Street consensus: EPS and revenue estimates for the current/next quarter and fiscal "
            "year, how the EPS consensus moved over 90 days, up/down revisions, price targets and "
            "buy/hold/sell counts. Use it to state what the market expects and where you differ."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": TICKER},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_financial_history",
        "description": (
            "Up to 10 years of annual financials as reported to the SEC (XBRL): revenue, gross profit, "
            "operating income, net income, diluted EPS, operating cash flow, capex, free cash flow, R&D, "
            "buybacks, dividends and diluted shares. Use for long-term trends and DCF inputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": TICKER},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_filings",
        "description": (
            "Keyword search across every downloaded SEC filing for this company (10-Ks, 10-Qs, 8-Ks, "
            "earnings press releases, proxy statement). Returns the best-matching passages with their "
            "doc_id. Use specific terms, e.g. 'customer concentration percent of revenue', "
            "'share repurchase authorization', 'CEO total compensation'. Optionally restrict to form "
            "types such as ['10-K'] or ['DEF 14A']."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": TICKER,
                "query": {"type": "string"},
                "form_types": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ticker", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_filing",
        "description": (
            "Read a downloaded filing by doc_id, 12,000 characters at a time. For 10-K/10-Q pick a "
            "section: 'business' (10-K), 'risk_factors', 'mdna' or 'full_text'. Other forms (press "
            "releases, proxy) are read as full text. The response says how to fetch the next page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": TICKER,
                "doc_id": {"type": "string", "description": "doc_id from the filing manifest"},
                "section": {"type": "string", "enum": ["business", "risk_factors", "mdna", "full_text"]},
                "offset": {"type": "integer", "description": "Character offset (default 0)"},
            },
            "required": ["ticker", "doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_insider_activity",
        "description": "Summary of insider (Form 4) buying, selling and awards over the past 12 months.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": TICKER},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_dcf",
        "description": (
            "Discounted cash flow valuation computed in code, with a sensitivity table. Rates are "
            "decimals (0.09 = 9%). Base FCF should be a normalized free cash flow in $ billions; "
            "growth_rates gives one rate per forecast year (5-10 years). net_cash is cash minus debt "
            "in $ billions (negative for net debt)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string", "enum": ["bear", "base", "bull"]},
                "base_fcf_billions": {"type": "number"},
                "growth_rates": {"type": "array", "items": {"type": "number"}},
                "terminal_growth": {"type": "number"},
                "discount_rate": {"type": "number"},
                "net_cash_billions": {"type": "number"},
                "shares_billions": {"type": "number"},
                "current_price": {"type": "number"},
            },
            "required": ["scenario", "base_fcf_billions", "growth_rates", "terminal_growth", "discount_rate",
                         "net_cash_billions", "shares_billions", "current_price"],
            "additionalProperties": False,
        },
    },
]

_DISPATCH = {
    "get_company_profile": lambda a: data.get_company_profile(a["ticker"]),
    "get_financial_statements": lambda a: data.get_financial_statements(
        a["ticker"], a["statement"], a.get("period", "annual")),
    "get_price_performance": lambda a: data.get_price_performance(a["ticker"]),
    "compare_peers": lambda a: data.compare_peers(a["tickers"]),
    "get_analyst_estimates": lambda a: data.get_analyst_estimates(a["ticker"]),
    "get_financial_history": lambda a: filings.financial_history(a["ticker"]),
    "search_filings": lambda a: filings.search_filings(a["ticker"], a["query"], a.get("form_types")),
    "read_filing": lambda a: filings.read_filing(
        a["ticker"], a["doc_id"], a.get("section", "full_text"), a.get("offset", 0)),
    "get_insider_activity": lambda a: filings.insider_summary(a["ticker"]),
    "run_dcf": lambda a: filings.run_dcf(
        a["base_fcf_billions"], a["growth_rates"], a["terminal_growth"], a["discount_rate"],
        a["net_cash_billions"], a["shares_billions"], a.get("current_price")),
}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool call. Returns (output, is_error)."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name}", True
    try:
        return fn(args), False
    except Exception as exc:  # surface the failure to Claude so it can adapt
        return f"{type(exc).__name__}: {exc}", True
