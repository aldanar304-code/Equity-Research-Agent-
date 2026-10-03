"""Turn a Markdown memo into a polished, self-contained HTML report (and PDF) with charts.

Charts are drawn from the same free data the agent used (SEC XBRL, Yahoo Finance) plus the
DCF inputs the agent chose, so the visuals always match the analysis.
"""

from __future__ import annotations

import io
import math
import re
import shutil
import subprocess
from datetime import date
from html import escape
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from . import filings  # noqa: E402

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, LIGHT_BLUE, GREEN, RED, AMBER = "#1d4ed8", "#93c5fd", "#15803d", "#b91c1c", "#b45309"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def _figure(svg: str, caption: str) -> str:
    return f'<figure class="chart">{svg}<figcaption>{escape(caption)}</figcaption></figure>'


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_revenue_margin(hist: dict) -> str | None:
    rev, op = hist.get("Revenue", {}), hist.get("Operating income", {})
    years = sorted(y for y in rev if y in op)
    if len(years) < 3:
        return None
    labels = [f"FY{y[2:4]}" if y[5:7] != "12" else y[:4] for y in years]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar(labels, [rev[y] / 1e9 for y in years], color=LIGHT_BLUE, label="Revenue ($B)")
    ax.set_ylabel("Revenue ($B)")
    ax2 = ax.twinx()
    ax2.plot(labels, [op[y] / rev[y] * 100 for y in years], color=BLUE, marker="o", lw=2, label="Operating margin")
    ax2.set_ylabel("Operating margin (%)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Revenue and operating margin")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)
    return _figure(_svg(fig), "Source: SEC XBRL (10-K as reported).")


def chart_cash_vs_earnings(hist: dict) -> str | None:
    ni, fcf = hist.get("Net income", {}), hist.get("Free cash flow", {})
    years = sorted(y for y in ni if y in fcf)
    if len(years) < 3:
        return None
    labels = [f"FY{y[2:4]}" if y[5:7] != "12" else y[:4] for y in years]
    x = range(len(years))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar([i - 0.2 for i in x], [ni[y] / 1e9 for y in years], 0.4, color=MUTED, label="Net income")
    ax.bar([i + 0.2 for i in x], [fcf[y] / 1e9 for y in years], 0.4, color=GREEN, label="Free cash flow")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("$B")
    ax.set_title("Cash conversion: free cash flow vs net income")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return _figure(_svg(fig), "Free cash flow = operating cash flow - capital expenditure. Source: SEC XBRL.")


def chart_price_vs_market(ticker: str) -> str | None:
    try:
        closes = {tk: yf.Ticker(tk).history(period="2y", auto_adjust=True)["Close"] for tk in (ticker, "SPY")}
        df = pd.DataFrame({tk: s.tz_localize(None) for tk, s in closes.items()}).dropna()
    except Exception:
        return None
    if df.empty:
        return None
    rebased = df / df.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(rebased.index, rebased[ticker], color=BLUE, lw=1.8, label=ticker)
    ax.plot(rebased.index, rebased["SPY"], color=MUTED, lw=1.4, label="S&P 500 (SPY)")
    ax.axhline(100, color=GRID, lw=1)
    ax.set_ylabel("Indexed (start = 100)")
    ax.set_title("Two-year total return vs the S&P 500")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return _figure(_svg(fig), "Source: Yahoo Finance, dividend-adjusted.")


def chart_peers(tickers: list[str], focus: str) -> str | None:
    rows = []
    for tk in tickers[:8]:
        try:
            info = yf.Ticker(tk).info or {}
        except Exception:
            continue
        pe, om = info.get("forwardPE"), info.get("operatingMargins")
        if pe and om is not None and 0 < pe < 150:
            rows.append((tk.upper(), pe, om * 100))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 0.45 * len(rows) + 1.0), sharey=True)
    colors = [BLUE if r[0] == focus.upper() else LIGHT_BLUE for r in rows]
    a1.barh([r[0] for r in rows], [r[1] for r in rows], color=colors)
    a1.set_title("Forward P/E (x)")
    a2.barh([r[0] for r in rows], [r[2] for r in rows], color=colors)
    a2.set_title("Operating margin (%)")
    for ax, idx, fmt in ((a1, 1, "{:.1f}"), (a2, 2, "{:.0f}%")):
        ax.grid(axis="y", visible=False)
        for i, r in enumerate(rows):
            ax.text(r[idx], i, " " + fmt.format(r[idx]), va="center", fontsize=8, color=INK)
    fig.tight_layout()
    return _figure(_svg(fig), "Peers chosen by the agent. Source: Yahoo Finance.")


def chart_scenarios(dcf_calls: list[dict], price: float | None) -> str | None:
    vals = []
    for c in dcf_calls:
        try:
            v = filings.dcf_value(c["base_fcf_billions"], c["growth_rates"], c["terminal_growth"],
                                  c["discount_rate"], c["net_cash_billions"], c["shares_billions"])[0]
        except Exception:
            continue
        vals.append((c.get("scenario", "dcf").title(), v))
    order = {"Bear": 0, "Base": 1, "Bull": 2}
    vals.sort(key=lambda kv: order.get(kv[0], 3))
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    colors = [{"Bear": RED, "Base": BLUE, "Bull": GREEN}.get(k, MUTED) for k, _ in vals]
    ax.bar([k for k, _ in vals], [v for _, v in vals], color=colors, width=0.55)
    for i, (_, v) in enumerate(vals):
        label = f"${v:,.0f}" + (f"\n({v / price - 1:+.0%})" if price else "")
        ax.text(i, v, label, ha="center", va="bottom", fontsize=8, color=INK)
    if price:
        ax.axhline(price, color=AMBER, lw=1.5, ls="--")
        ax.text(len(vals) - 0.5, price, f"  price ${price:,.0f}", color=AMBER, va="bottom", fontsize=8)
    ax.set_ylabel("Value per share ($)")
    ax.set_title("DCF value per share by scenario")
    ax.margins(y=0.25)
    return _figure(_svg(fig), "Values recomputed from the agent's DCF inputs.")


def chart_sensitivity(base: dict, price: float | None) -> str | None:
    try:
        drs = [base["discount_rate"] + d for d in (-0.02, -0.01, 0, 0.01, 0.02)]
        tgs = [base["terminal_growth"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        grid = [[filings.dcf_value(base["base_fcf_billions"], base["growth_rates"], tg, dr,
                                   base["net_cash_billions"], base["shares_billions"])[0] if dr > tg else math.nan
                 for tg in tgs] for dr in drs]
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    if price:
        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vcenter=price, vmin=min(min(r) for r in grid) - 1, vmax=max(max(r) for r in grid) + 1)
        im = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto")
    else:
        im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tgs)), [f"{t:.1%}" for t in tgs])
    ax.set_yticks(range(len(drs)), [f"{d:.1%}" for d in drs])
    ax.set_xlabel("Terminal growth")
    ax.set_ylabel("Discount rate")
    ax.grid(False)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, "n/a" if math.isnan(v) else f"${v:,.0f}", ha="center", va="center", fontsize=8,
                    color=INK, fontweight="bold" if (i, j) == (2, 2) else "normal")
    ax.set_title("Base-case DCF sensitivity (value per share)")
    del im
    return _figure(_svg(fig), "Green = above the current price, red = below. Centre cell is the base case.")


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
:root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --accent:#1d4ed8; --bg:#ffffff; --soft:#f8fafc; }
* { box-sizing: border-box; }
body { margin:0; background:var(--soft); color:var(--ink);
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif; }
.page { max-width: 860px; margin: 32px auto; background: var(--bg); padding: 48px 56px;
        border: 1px solid var(--line); border-radius: 10px; }
.brand { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
h1 { font-size: 28px; line-height: 1.2; margin: 6px 0 18px; }
h2 { font-size: 18px; margin: 34px 0 10px; padding-top: 14px; border-top: 2px solid var(--ink); }
h3 { font-size: 15px; margin: 22px 0 6px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 0 0 22px; }
.kpi { background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.kpi .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.kpi .value { font-size: 17px; font-weight: 700; }
.kpi.rec .value { color: var(--accent); }
table { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 13px; }
th { text-align: left; background: var(--soft); font-weight: 600; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: top; }
td:not(:first-child), th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
figure.chart { margin: 18px 0 22px; }
figure.chart svg { width: 100%; height: auto; }
figcaption { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
code { background: var(--soft); padding: 1px 4px; border-radius: 4px; font-size: 13px; }
hr { border: 0; border-top: 1px solid var(--line); margin: 30px 0 12px; }
em { color: var(--muted); }
.footer { font-size: 11.5px; color: var(--muted); margin-top: 24px; }
@media (max-width: 700px) { .page { margin: 0; padding: 24px 16px; border-radius: 0; } }
@media print {
  body { background: #fff; font-size: 11pt; }
  .page { margin: 0; padding: 0; border: 0; max-width: none; }
  h2 { break-after: avoid; } figure, table, .kpis { break-inside: avoid; }
  @page { size: A4; margin: 16mm 14mm; }
}
"""


def _kpis(header_line: str) -> tuple[str, str]:
    """Turn the '**Date:** x | **Price:** y | ...' line into KPI cards; returns (cards_html, remainder)."""
    pairs = re.findall(r"\*\*([^*]+?):\*\*\s*([^|]+)", header_line)
    if not pairs:
        return "", header_line
    cards = []
    for k, v in pairs:
        cls = "kpi rec" if k.lower().startswith("recommendation") or k.lower() == "stance" else "kpi"
        cards.append(f'<div class="{cls}"><div class="label">{escape(k)}</div>'
                     f'<div class="value">{escape(v.strip())}</div></div>')
    return f'<div class="kpis">{"".join(cards)}</div>', ""


def _insert(sections: dict[str, list[str]], md: str) -> str:
    """Append chart HTML to the end of numbered '## N.' sections."""
    out, current = [], None
    lines = md.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(\d+)\.", line)
        if line.startswith("## ") and current and sections.get(current):
            out += ["", *sections.pop(current), ""]
        if line.startswith("## "):
            current = m.group(1) if m else None
        out.append(line)
    if current and sections.get(current):
        out += ["", *sections.pop(current), ""]
    return "\n".join(out)


def _fix_lists(md: str) -> str:
    """Python-Markdown needs a blank line before a list; LLM output often omits it."""
    out = []
    for line in md.splitlines():
        is_item = re.match(r"^\s*(?:[-*+]|\d+\.)\s+", line)
        if is_item and out and out[-1].strip() and not re.match(r"^\s*(?:[-*+]|\d+\.)\s+", out[-1]) \
                and not out[-1].lstrip().startswith("|"):
            out.append("")
        out.append(line)
    return "\n".join(out)


def render_html(memo_md: str, ticker: str, tool_calls: list[dict], meta: str = "") -> str:
    ticker = ticker.upper()
    lines = memo_md.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else f"{ticker} memo"
    kpi_html, body_start = "", 1
    if len(lines) > 1 and lines[1].startswith("**"):
        kpi_html, _ = _kpis(lines[1])
        body_start = 2
    body_md = "\n".join(lines[body_start:])

    price = None
    m = re.search(r"\*\*Price:\*\*\s*\$([\d,\.]+)", memo_md)
    if m:
        price = float(m.group(1).replace(",", ""))

    try:
        hist = filings.financial_history_data(ticker)
    except Exception:
        hist = {}
    peers = next((c["input"]["tickers"] for c in reversed(tool_calls) if c["tool"] == "compare_peers"), [])
    dcfs = [c["input"] for c in tool_calls if c["tool"] == "run_dcf"]
    latest = {}
    for c in dcfs:  # the last run per scenario is the one the memo settled on
        latest[c.get("scenario", f"dcf{len(latest)}")] = c
    base = latest.get("base") or (dcfs[-1] if dcfs else None)

    charts = {
        "1": [chart_price_vs_market(ticker)],
        "3": [chart_peers(peers, ticker) if peers else None],
        "4": [chart_revenue_margin(hist), chart_cash_vs_earnings(hist)],
        "6": [chart_scenarios(list(latest.values()), price), chart_sensitivity(base, price) if base else None],
    }
    charts = {k: [c for c in v if c] for k, v in charts.items()}
    body_md = _insert(charts, _fix_lists(body_md))
    body_html = markdown.markdown(body_md, extensions=["tables", "sane_lists"])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head>
<body><main class="page">
<div class="brand">AI Equity Research Agent &middot; Investment memo</div>
<h1>{escape(title)}</h1>
{kpi_html}
{body_html}
<div class="footer">Generated {date.today():%B %d, %Y}. {escape(meta)}</div>
</main></body></html>"""


def find_chrome() -> str | None:
    for path in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                 "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"):
        if Path(path).exists():
            return path
    return shutil.which("chromium") or shutil.which("google-chrome")


def html_to_pdf(html_path: Path) -> Path | None:
    """Print the HTML to PDF with a locally installed Chromium browser, if there is one."""
    chrome = find_chrome()
    if not chrome:
        return None
    pdf = html_path.with_suffix(".pdf")
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                   check=False, capture_output=True, timeout=120)
    return pdf if pdf.exists() else None
