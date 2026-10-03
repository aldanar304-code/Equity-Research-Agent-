"""Offline tests for the agent loop, using a scripted fake Claude client (no API key, no cost)."""

from types import SimpleNamespace as NS

import pytest

from equity_agent import agent
from equity_agent.agent import ResearchError, Settings, research


def usage(inp=1000, out=200, cw=0, cr=0, searches=0):
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=cw,
              cache_read_input_tokens=cr, server_tool_use=NS(web_search_requests=searches))


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def text(t):
    return NS(type="text", text=t)


def no_downloads():
    return Settings(download_filings=False, fact_check=False)


def response(content, stop_reason, u=None):
    return NS(content=content, stop_reason=stop_reason, usage=u or usage(), model="claude-opus-5-5")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setattr(agent, "run_tool", lambda name, args: (f"{name} ok", False))


def test_runs_tools_then_returns_report():
    client = FakeClient([
        response([tool_use("t1", "get_company_profile", {"ticker": "ACME"}),
                  tool_use("t2", "get_price_performance", {"ticker": "ACME"})], "tool_use"),
        response([text("# ACME report")], "end_turn"),
    ])
    events = []
    result = research("acme", no_downloads(), lambda k, p: events.append(k), client=client)

    assert result.report == "# ACME report"
    assert result.ticker == "ACME"
    assert [c["tool"] for c in result.tool_calls] == ["get_company_profile", "get_price_performance"]
    # Both tool results go back in a single user message.
    results_msg = client.calls[1]["messages"][-1]
    assert results_msg["role"] == "user"
    assert [r["tool_use_id"] for r in results_msg["content"]] == ["t1", "t2"]
    assert "tool_call" in events and "cost" in events


def test_budget_forces_wrap_up():
    expensive = usage(inp=200_000, out=10_000)  # ~$1.00 on Opus 5.5 pricing
    client = FakeClient([
        response([tool_use("t1", "get_company_profile", {"ticker": "ACME"})], "tool_use", expensive),
        response([text("# Report from partial data")], "end_turn"),
    ])
    result = research("ACME", Settings(model="claude-opus-5-5", max_cost_usd=1.0, download_filings=False, fact_check=False), client=client)

    final_call = client.calls[1]
    assert final_call["tool_choice"] == {"type": "none"}
    assert "Do not call any more tools" in final_call["messages"][-1]["content"][-1]["text"]
    assert result.report.startswith("# Report")


def test_pause_turn_resumes_without_new_user_message():
    client = FakeClient([
        response([NS(type="server_tool_use", name="web_search", input={"query": "ACME news"})], "pause_turn"),
        response([text("# done")], "end_turn"),
    ])
    result = research("ACME", no_downloads(), client=client)
    assert client.calls[1]["messages"][-1]["role"] == "assistant"
    assert result.tool_calls == [{"tool": "web_search", "input": {"query": "ACME news"}}]


def test_refusal_raises():
    client = FakeClient([response([], "refusal")])
    with pytest.raises(ResearchError):
        research("ACME", no_downloads(), client=client)


def test_cost_math():
    u = agent.Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000, web_searches=3)
    assert u.cost("claude-opus-5-5") == pytest.approx(4.00 + 2.00 + 0.20 + 0.03)


def test_haiku_skips_current_gen_only_params():
    kw = agent._request_kwargs(Settings(model="claude-haiku-4-5"))
    assert "output_config" not in kw and "fallbacks" not in kw
    assert kw["tools"][-1]["type"] == "web_search_20250305"


def test_dcf_math():
    from equity_agent.filings import run_dcf
    # 1 year at 0% growth, FCF 10, r=10%, g=0: PV(FCF)=9.09, TV=100 -> PV 90.91, equity 100 + net cash 0
    out = run_dcf(10, [0.0], 0.0, 0.10, 0.0, 1.0, 50.0)
    assert "Value per share:        $100.00  (+100.0% vs $50.00)" in out


def test_dcf_rejects_bad_rates():
    from equity_agent.filings import run_dcf
    with pytest.raises(ValueError):
        run_dcf(10, [0.05], 0.10, 0.08, 0, 1)


def test_fact_check_pass_replaces_draft():
    client = FakeClient([
        response([text("# Draft with a wrong number")], "end_turn"),
        response([tool_use("t1", "search_filings", {"ticker": "ACME", "query": "revenue"})], "tool_use"),
        response([text("# Corrected memo\n### Fact-check notes\n- fixed revenue")], "end_turn"),
    ])
    result = research("ACME", Settings(download_filings=False), client=client)
    assert result.report.startswith("# Corrected memo")
    assert "Fact-check pass" in client.calls[1]["messages"][-1]["content"]
    assert [c["tool"] for c in result.tool_calls] == ["search_filings"]


def test_render_escapes_model_html(monkeypatch):
    from equity_agent import render
    monkeypatch.setattr(render.filings, "financial_history_data", lambda t: {})
    monkeypatch.setattr(render, "chart_price_vs_market", lambda t: None)
    html = render.render_html("# T\n\n## 1. Thesis\nhi <script>alert(1)</script> & more\n> quote", "ACME", [])
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "<blockquote>" in html


def test_disclosures_state_holdings(monkeypatch):
    from datetime import datetime
    from equity_agent import render
    monkeypatch.setattr(render.filings, "financial_history_data", lambda t: {})
    monkeypatch.setattr(render, "chart_price_vs_market", lambda t: None)
    memo = "# T\n\n**Date:** x | **Price:** $10 | **Recommendation:** Hold | **12-month target:** $11 (+10%)\n\n## 1. A\ntext"
    cfg = {"author": "Ana", "holdings": ["ACME"], "short_positions": []}
    html = render.render_html(memo, "ACME", [], generated_at=datetime(2026, 1, 2, 9, 30), disclosure_cfg=cfg)
    assert "Ana holds a long position in ACME" in html and "the author holds ACME shares" in html
    assert "02 January 2026, 09:30" in html and "MiFID II" in html
    other = render.render_html(memo, "OTHR", [], disclosure_cfg=cfg)
    assert "holds no position in OTHR" in other and "the author holds" not in other


def test_number_check_flags_changed_figures():
    from equity_agent.translate import number_mismatches
    en = "Revenue was $96.2B, up 106% on October 3, 2026."
    assert number_mismatches(en, "Los ingresos fueron $96.2B, un 106% más, el 3 de octubre de 2026.") == {}
    assert number_mismatches(en, "Los ingresos fueron $96,2B, un 106% más, el 3 de octubre de 2026.") == {"96.2": 1, "96,2": -1}
    assert number_mismatches("Q2 revenue grew 106%", "Los ingresos del segundo trimestre crecieron un 106%") == {}


def test_spanish_render_uses_spanish_labels(monkeypatch):
    from equity_agent import render
    monkeypatch.setattr(render.filings, "financial_history_data", lambda t: {})
    monkeypatch.setattr(render, "chart_price_vs_market", lambda t: None)
    memo = ("# Acme (ACME) - Informe de inversión\n\n**Fecha:** x | **Precio:** $10 | **Recomendación:** Mantener | "
            "**Precio objetivo a 12 meses:** $11 (+10%)\n\n## 1. Recomendación y tesis\ntexto")
    html = render.render_html(memo, "ACME", [], lang="es", disclosure_cfg={"author": "Ana", "holdings": ["ACME"]})
    assert 'lang="es"' in html and "Información legal" in html and "Mantener, precio objetivo a 12 meses $11" in html
    assert "Ana mantiene una posición larga en acciones de ACME" in html and "traducido automáticamente" in html
    english = render.render_html(memo.replace("Informe", "Memo"), "ACME", [], disclosure_cfg={"author": "Ana", "holdings": []})
    assert "Disclosures" in english and 'lang="en"' in english


def test_all_charts_render_in_spanish(monkeypatch):
    from equity_agent import render
    years = [f"20{y}-01" for y in range(18, 26)]
    hist = {"Revenue": {y: 100e9 + i * 10e9 for i, y in enumerate(years)},
            "Operating income": {y: 30e9 for y in years},
            "Net income": {y: 20e9 for y in years}, "Free cash flow": {y: 25e9 for y in years}}
    base = dict(scenario="base", base_fcf_billions=10, growth_rates=[0.05] * 5, terminal_growth=0.02,
                discount_rate=0.09, net_cash_billions=0, shares_billions=1)
    for lang in ("en", "es"):
        render._LANG = lang
        assert render.chart_revenue_margin(hist) and render.chart_cash_vs_earnings(hist)
        assert render.chart_scenarios([base], 150.0) and render.chart_sensitivity(base, 150.0)
    render._LANG = "en"
