"""Streamlit web app. Run locally with: uv run streamlit run app.py

Two tabs:
- Sample memos: pre-generated memos, free to browse.
- Write a new memo: runs the agent for any visitor.
    * Free tries are paid by the app owner's key (ANTHROPIC_API_KEY in Streamlit secrets), protected by
      a daily budget, a per-visitor daily limit, a per-memo cap and reuse of recent memos.
    * Visitors can also paste their own Anthropic API key (kept only in their browser session).

Optional Streamlit secrets:
    ANTHROPIC_API_KEY        owner key for free tries (leave unset to require visitors' own keys)
    DAILY_BUDGET_USD         total free-try spend per day (default 3.00)
    FREE_MEMOS_PER_VISITOR   free memos per visitor per day (default 1)
    SEC_USER_AGENT           "app-name your-email" contact string the SEC asks API users to send
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))  # so it runs without installing the package (Streamlit Cloud)

load_dotenv()
load_dotenv(Path.home() / ".config" / "equity-research-agent" / ".env")  # key saved outside the repo


def secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets.toml
        value = None
    return value if value not in (None, "") else os.getenv(name, default)


if secret("SEC_USER_AGENT"):
    os.environ["SEC_USER_AGENT"] = secret("SEC_USER_AGENT")

from equity_agent import ResearchError, Settings, research  # noqa: E402
from equity_agent import data as data_mod  # noqa: E402
from equity_agent import filings as filings_mod  # noqa: E402

SAMPLES = Path(__file__).parent / "sample_reports"
WORK = Path(tempfile.gettempdir()) / "equity_agent"
MEMO_CACHE = WORK / "memos"
USAGE_FILE = WORK / "usage.json"
for d in (WORK, MEMO_CACHE):
    d.mkdir(parents=True, exist_ok=True)
filings_mod.FILINGS_DIR = WORK / "filings"   # writable on Streamlit Cloud
data_mod.CACHE_DIR = WORK / "cache"

OWNER_KEY = secret("ANTHROPIC_API_KEY")
DAILY_BUDGET = float(secret("DAILY_BUDGET_USD", "3.00"))
FREE_PER_VISITOR = int(secret("FREE_MEMOS_PER_VISITOR", "1"))
MAX_COST_PER_MEMO = 1.00
EST_COST = 0.55          # typical memo, used to decide whether a free try still fits today's budget
EST_COST_SPANISH = 0.15
REUSE_DAYS = 7

USAGE_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()  # the renderer switches language globally; one memo at a time


# ---------------------------------------------------------------------------
# Free-try accounting (owner key)
# ---------------------------------------------------------------------------

def _load_usage() -> dict:
    today = date.today().isoformat()
    try:
        usage = json.loads(USAGE_FILE.read_text())
    except Exception:
        usage = {}
    if usage.get("date") != today:
        usage = {"date": today, "spent": 0.0, "visitors": {}}
    return usage


def _save_usage(usage: dict) -> None:
    USAGE_FILE.write_text(json.dumps(usage))


def visitor_id() -> str:
    """Anonymous per-visitor id (hashed IP address) used only to enforce the daily free limit."""
    ip = ""
    try:
        # Behind a hosting proxy the visitor's own address is in X-Forwarded-For; the socket address
        # may be the proxy's, which would make every visitor look like the same person.
        headers = st.context.headers
        ip = ((headers.get("X-Forwarded-For") or "").split(",")[0].strip()
              or (headers.get("X-Real-Ip") or "").strip()
              or getattr(st.context, "ip_address", None) or "")
    except Exception:
        pass
    raw = ip or st.session_state.setdefault("_anon", os.urandom(8).hex())
    return hashlib.sha256(f"{date.today()}:{raw}".encode()).hexdigest()[:16]


def free_try_status() -> tuple[bool, str]:
    """(allowed, reason) for a free memo right now."""
    if not OWNER_KEY:
        return False, "Free tries aren't enabled on this site. Use your own API key below."
    with USAGE_LOCK:
        usage = _load_usage()
    left = DAILY_BUDGET - usage["spent"]
    if left < EST_COST:
        return False, "Today's free budget has been used up. Try again tomorrow, or use your own API key."
    if usage["visitors"].get(visitor_id(), 0) >= FREE_PER_VISITOR:
        return False, (f"You've used your {FREE_PER_VISITOR} free memo(s) for today. "
                       "Come back tomorrow, or use your own API key.")
    return True, f"Free memos left today on this site: about {int(left // EST_COST)}."


def record_free_spend(cost: float) -> None:
    with USAGE_LOCK:
        usage = _load_usage()
        usage["spent"] = round(usage["spent"] + cost, 4)
        vid = visitor_id()
        usage["visitors"][vid] = usage["visitors"].get(vid, 0) + 1
        _save_usage(usage)


# ---------------------------------------------------------------------------
# Memo storage and reuse
# ---------------------------------------------------------------------------

def recent_memo(ticker: str) -> Path | None:
    """Most recent English memo for this ticker from the last REUSE_DAYS days (samples or generated)."""
    cutoff = date.today() - timedelta(days=REUSE_DAYS)
    best = None
    for folder in (MEMO_CACHE, SAMPLES):
        for md in folder.glob(f"{ticker}_memo_*.md"):
            if md.stem.endswith("_es"):
                continue
            m = re.search(r"_memo_(\d{4}-\d{2}-\d{2})", md.name)
            if m and date.fromisoformat(m.group(1)) >= cutoff and (best is None or md.name > best.name):
                best = md
    return best


def memo_bundle(md_path: Path) -> dict:
    """Load a saved memo (and its Spanish version, if any) for display."""
    bundle = {"ticker": md_path.name.split("_")[0], "date": md_path.name.split("_memo_")[1][:10],
              "en_md": md_path.read_text().split("\n\n<!--")[0], "en_html": None, "es_md": None, "es_html": None}
    if md_path.with_suffix(".html").exists():
        bundle["en_html"] = md_path.with_suffix(".html").read_text()
    es = md_path.with_name(md_path.stem + "_es.md")
    if es.exists():
        bundle["es_md"] = es.read_text().split("\n\n<!--")[0]
        if es.with_suffix(".html").exists():
            bundle["es_html"] = es.with_suffix(".html").read_text()
    return bundle


def valid_ticker(ticker: str) -> bool:
    try:
        filings_mod._cik(ticker)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Equity Research Agent", page_icon="📈", layout="wide")
st.title("📈 AI Equity Research Agent")
st.caption(
    "An AI agent that downloads a company's SEC filings, researches it like an analyst - 10 years of financials, "
    "filing search, DCF valuation, peers, insider activity, analyst consensus and news - fact-checks its own draft, "
    "and writes an investment memo. Educational project, not investment advice."
)

with st.sidebar:
    st.markdown(
        "**How it works**\n"
        "1. Downloads ~30 SEC filings for the company\n"
        "2. Claude researches it with ~20 tool calls\n"
        "3. Writes the memo, then fact-checks it\n"
        "4. Delivers a report with charts (+ Spanish)\n\n"
        "[Source code on GitHub](https://github.com/aldanar304-code/Equity-Research-Agent-)"
    )


def show_bundle(bundle: dict, key: str) -> None:
    lang = "English"
    if bundle.get("es_md"):
        lang = st.radio("Language / Idioma", ["English", "Español"], horizontal=True, key=f"lang_{key}")
    es = lang == "Español"
    md, html = (bundle["es_md"], bundle["es_html"]) if es else (bundle["en_md"], bundle["en_html"])
    base = f"{bundle['ticker']}_memo_{bundle['date']}{'_es' if es else ''}"
    c1, c2 = st.columns(2)
    if html:
        c1.download_button("⬇️ Download report (.html)", html, file_name=f"{base}.html", mime="text/html",
                           help="Open it in your browser; use Print → Save as PDF for a PDF.", key=f"h_{key}_{lang}")
    c2.download_button("⬇️ Download text (.md)", md, file_name=f"{base}.md", key=f"m_{key}_{lang}")
    if html:
        st.iframe(html, height=1400)
    else:
        st.markdown(md)


tab_samples, tab_new = st.tabs(["📚 Sample memos", "✍️ Write a new memo"])

# ---- Sample memos ------------------------------------------------------------
with tab_samples:
    reports = sorted((p for p in SAMPLES.glob("*_memo_*.md") if not p.stem.endswith("_es")), reverse=True)
    if not reports:
        st.info("No sample memos yet.")
    else:
        choice = st.selectbox("Memo", reports, format_func=lambda p: f"{p.name.split('_')[0]} - {p.name.split('_memo_')[1][:10]}")
        meta = re.search(r"<!-- generated by .*?model: (\S+) \| cost: \$(\S+) .*?tool calls: (\d+)", choice.read_text())
        if meta:
            c1, c2, c3 = st.columns(3)
            c1.metric("Model", meta.group(1).replace("claude-", "Claude ").replace("-", " ").replace(" 5 5", " 5.5").title())
            c2.metric("Cost to generate", f"${float(meta.group(2)):.2f}")
            c3.metric("Tool calls", meta.group(3))
        bundle = memo_bundle(choice)
        pdf = choice.with_suffix(".pdf")
        if pdf.exists():
            st.download_button("⬇️ Download PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")
        show_bundle(bundle, "sample")

# ---- Write a new memo ---------------------------------------------------------
with tab_new:
    st.markdown("Enter the ticker of any **US-listed company** (for example `AAPL`, `TSLA`, `KO`, `JPM`). "
                "A memo takes about **4–5 minutes**.")

    c1, c2 = st.columns([2, 1])
    ticker = c1.text_input("Ticker", placeholder="e.g. TSLA", max_chars=10).strip().upper()
    spanish = c2.checkbox("Also in Spanish / También en español", value=False)

    options = ["Free try (paid by this site)", "Use my own Anthropic API key"] if OWNER_KEY else ["Use my own Anthropic API key"]
    how = st.radio("How to run it", options, horizontal=True)
    own_key = None
    if how.startswith("Use my own"):
        own_key = st.text_input("Your Anthropic API key", type="password",
                                help="Used only for this request and never stored. It starts with sk-ant-.")
        st.caption("Don't have one? Create a key at [console.anthropic.com](https://console.anthropic.com) "
                   "(add a little credit first). A memo costs about US\\$0.45 of your credit, US\\$0.60 with Spanish.")
        allowed, note = bool(own_key), ""
    else:
        allowed, note = free_try_status()
        (st.caption if allowed else st.warning)(note)

    run = st.button("Write investment memo", type="primary", disabled=not (ticker and allowed))

    if run:
        if not re.fullmatch(r"[A-Z][A-Z.\-]{0,9}", ticker) or not valid_ticker(ticker):
            st.error(f"'{ticker}' wasn't found among SEC-registered companies. Check the ticker - "
                     "this tool covers US-listed companies only.")
            st.stop()

        reuse = recent_memo(ticker)
        if reuse is not None and (not spanish or reuse.with_name(reuse.stem + "_es.md").exists()):
            st.session_state["result"] = memo_bundle(reuse)
            st.session_state["result_note"] = (f"✅ A memo on {ticker} from {st.session_state['result']['date']} "
                                               "already exists, so here it is - no cost, no wait.")
        else:
            free = not how.startswith("Use my own")
            key = OWNER_KEY if free else own_key
            client = anthropic.Anthropic(api_key=key)
            settings = Settings(max_cost_usd=MAX_COST_PER_MEMO)
            spent = 0.0
            cost_box = st.empty()
            with st.status(f"Researching {ticker}…", expanded=True) as status:
                def on_event(kind: str, p: dict) -> None:
                    if kind == "tool_call":
                        args = ", ".join(f"{k}={v}" for k, v in p["input"].items() if k not in ("url",))
                        st.write(f"🔧 `{p['name']}` {args[:120]}")
                    elif kind == "download":
                        st.write(f"📄 downloaded `{p['doc_id']}`")
                    elif kind == "web_search" and p.get("query"):
                        st.write(f"🔎 web search: *{p['query']}*")
                    elif kind == "status":
                        status.update(label=f"{ticker}: {p['message']}")
                    elif kind == "cost":
                        cost_box.caption(f"Cost so far: ${p['cost_usd']:.3f}")

                try:
                    result = research(ticker, settings, on_event, client=client)
                except ResearchError as exc:
                    if free:
                        record_free_spend(0.30)  # a failed run still spent some credit
                    status.update(label="Failed", state="error")
                    msg = str(exc)
                    if "401" in msg or "authentication" in msg.lower():
                        msg = "That API key wasn't accepted. Check that you copied the whole key (it starts with sk-ant-)."
                    elif "credit balance" in msg.lower():
                        msg = "The API key has no credit left. Add credit at console.anthropic.com → Billing."
                    st.error(msg)
                    st.stop()
                spent = result.cost_usd

                es_md = es_cost = None
                if spanish:
                    status.update(label=f"{ticker}: translating to Spanish")
                    try:
                        from equity_agent.translate import translate_memo
                        es_md, es_cost, _ = translate_memo(result.report, client)
                        spent += es_cost
                    except Exception as exc:
                        st.warning(f"The Spanish translation failed ({exc}); the English memo is ready.")
                status.update(label=f"{ticker}: done - {len(result.tool_calls)} tool calls, ${spent:.2f}",
                              state="complete", expanded=False)

            if free:
                record_free_spend(spent)

            from equity_agent.render import previous_recommendations, render_html
            today = date.today().isoformat()
            stem = f"{ticker}_memo_{today}"
            prev = previous_recommendations(ticker, SAMPLES)
            with RENDER_LOCK:
                en_html = render_html(result.report, ticker, result.tool_calls,
                                      f"Model: {result.model} | cost ${result.cost_usd:.2f} | {len(result.tool_calls)} tool calls.",
                                      generated_at=datetime.now(), previous=prev)
                es_html = render_html(es_md, ticker, result.tool_calls,
                                      f"Modelo: {result.model} | coste ${spent:.2f} | {len(result.tool_calls)} llamadas a herramientas.",
                                      generated_at=datetime.now(), previous=prev, lang="es") if es_md else None
            # keep a copy so other visitors asking for the same company get it for free
            (MEMO_CACHE / f"{stem}.md").write_text(result.report)
            (MEMO_CACHE / f"{stem}.html").write_text(en_html)
            if es_md:
                (MEMO_CACHE / f"{stem}_es.md").write_text(es_md)
                (MEMO_CACHE / f"{stem}_es.html").write_text(es_html)
            st.session_state["result"] = {"ticker": ticker, "date": today, "en_md": result.report, "en_html": en_html,
                                          "es_md": es_md, "es_html": es_html}
            st.session_state["result_note"] = f"✅ Done in {len(result.tool_calls)} research steps, cost ${spent:.2f}."

    # Shown on every rerun, so clicking a download button doesn't make the memo disappear.
    if st.session_state.get("result"):
        st.success(st.session_state.get("result_note", ""))
        show_bundle(st.session_state["result"], "new")
