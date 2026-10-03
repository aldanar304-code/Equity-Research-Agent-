"""Tool schemas exposed to Claude and the dispatcher that runs them."""

from __future__ import annotations

from . import data

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
        "name": "list_sec_filings",
        "description": (
            "List a US-listed company's recent SEC EDGAR filings with their URLs. "
            "Use the URL with read_sec_filing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": TICKER,
                "form_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Form types to include, e.g. ['10-K', '10-Q', '8-K']",
                },
                "limit": {"type": "integer", "description": "Max filings to return (default 8)"},
            },
            "required": ["ticker", "form_types"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_sec_filing",
        "description": (
            "Read one section of a 10-K or 10-Q from SEC EDGAR, 12,000 characters at a time. "
            "Sections: 'business' (10-K only), 'risk_factors', 'mdna' (Management's Discussion & "
            "Analysis), or 'full_text'. The response says how to fetch the next page via offset. "
            "Filings are long - read only what the report needs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Filing URL from list_sec_filings"},
                "section": {"type": "string", "enum": ["business", "risk_factors", "mdna", "full_text"]},
                "offset": {"type": "integer", "description": "Character offset to start from (default 0)"},
            },
            "required": ["url", "section"],
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
    "list_sec_filings": lambda a: data.list_sec_filings(
        a["ticker"], a.get("form_types"), a.get("limit", 8)),
    "read_sec_filing": lambda a: data.read_sec_filing(
        a["url"], a.get("section", "mdna"), a.get("offset", 0)),
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
