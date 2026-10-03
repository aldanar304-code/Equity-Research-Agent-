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
    result = research("acme", Settings(), lambda k, p: events.append(k), client=client)

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
    result = research("ACME", Settings(model="claude-opus-5-5", max_cost_usd=1.0), client=client)

    final_call = client.calls[1]
    assert final_call["tool_choice"] == {"type": "none"}
    assert "Do not call any more tools" in final_call["messages"][-1]["content"][-1]["text"]
    assert result.report.startswith("# Report")


def test_pause_turn_resumes_without_new_user_message():
    client = FakeClient([
        response([NS(type="server_tool_use", name="web_search", input={"query": "ACME news"})], "pause_turn"),
        response([text("# done")], "end_turn"),
    ])
    result = research("ACME", Settings(), client=client)
    assert client.calls[1]["messages"][-1]["role"] == "assistant"
    assert result.tool_calls == [{"tool": "web_search", "input": {"query": "ACME news"}}]


def test_refusal_raises():
    client = FakeClient([response([], "refusal")])
    with pytest.raises(ResearchError):
        research("ACME", Settings(), client=client)


def test_cost_math():
    u = agent.Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000, web_searches=3)
    assert u.cost("claude-opus-5-5") == pytest.approx(4.00 + 2.00 + 0.20 + 0.03)


def test_haiku_skips_current_gen_only_params():
    kw = agent._request_kwargs(Settings(model="claude-haiku-4-5"))
    assert "output_config" not in kw and "fallbacks" not in kw
    assert kw["tools"][-1]["type"] == "web_search_20250305"
