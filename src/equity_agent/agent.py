"""The research agent: a tool-use loop over Claude with cost tracking and a hard budget."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import anthropic

from .prompts import SYSTEM_PROMPT
from .tools import CUSTOM_TOOLS, run_tool

# USD per million tokens: (input, output, cache write, cache read). Web search is $10 per 1,000.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}
WEB_SEARCH_USD = 0.01

# Models that support the newer web search tool, effort control and server-side refusal fallbacks.
CURRENT_GEN = {"claude-opus-5-5", "claude-sonnet-5-5"}

EventHandler = Callable[[str, dict], None]


def _env(name: str, default: str, cast=str):
    return field(default_factory=lambda: cast(os.getenv(name, default)))


@dataclass
class Settings:
    model: str = _env("EQUITY_AGENT_MODEL", "claude-opus-5-5")
    effort: str = _env("EQUITY_AGENT_EFFORT", "medium")  # low | medium | high
    max_cost_usd: float = _env("EQUITY_AGENT_MAX_COST_USD", "1.00", float)
    max_turns: int = _env("EQUITY_AGENT_MAX_TURNS", "16", int)
    web_searches: int = _env("EQUITY_AGENT_WEB_SEARCHES", "5", int)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    web_searches: int = 0

    def add(self, u) -> None:
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_write_tokens += u.cache_creation_input_tokens or 0
        self.cache_read_tokens += u.cache_read_input_tokens or 0
        if getattr(u, "server_tool_use", None):
            self.web_searches += u.server_tool_use.web_search_requests or 0

    def cost(self, model: str) -> float:
        inp, out, cw, cr = PRICING.get(model, PRICING["claude-opus-5-5"])
        return (self.input_tokens * inp + self.output_tokens * out
                + self.cache_write_tokens * cw + self.cache_read_tokens * cr) / 1e6 \
            + self.web_searches * WEB_SEARCH_USD


@dataclass
class ResearchResult:
    ticker: str
    report: str
    model: str
    usage: Usage
    cost_usd: float
    tool_calls: list[dict] = field(default_factory=list)


class ResearchError(RuntimeError):
    pass


def _request_kwargs(settings: Settings) -> dict:
    current = settings.model in CURRENT_GEN
    web_search = {
        "type": "web_search_20260209" if current else "web_search_20250305",
        "name": "web_search",
        "max_uses": settings.web_searches,
    }
    kwargs = {
        "model": settings.model,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "tools": [*CUSTOM_TOOLS, web_search],
        # Automatic prompt caching: each turn re-reads the growing conversation from cache
        # at ~5% of the normal input price instead of paying full price again.
        "cache_control": {"type": "ephemeral"},
    }
    if current:
        kwargs["output_config"] = {"effort": settings.effort}
        # If a safety classifier declines a request, rerun it on Anthropic's recommended fallback model.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return kwargs


def research(
    ticker: str,
    settings: Settings | None = None,
    on_event: EventHandler | None = None,
    client: anthropic.Anthropic | None = None,
) -> ResearchResult:
    """Run the agent on one ticker and return the finished Markdown report."""
    settings = settings or Settings()
    client = client or anthropic.Anthropic()
    emit = on_event or (lambda kind, payload: None)
    ticker = ticker.strip().upper()

    kwargs = _request_kwargs(settings)
    messages: list[dict] = [{
        "role": "user",
        "content": f"Write an equity research report on {ticker}. Today's date is {date.today():%B %d, %Y}.",
    }]
    usage = Usage()
    tool_log: list[dict] = []
    wrapping_up = False

    for turn in range(1, settings.max_turns + 1):
        emit("status", {"message": "Writing report" if wrapping_up else f"Researching (step {turn})"})
        try:
            response = client.beta.messages.create(messages=messages, **kwargs)
        except anthropic.APIStatusError as exc:
            raise ResearchError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ResearchError("Could not reach the Claude API - check your network.") from exc

        usage.add(response.usage)
        cost = usage.cost(settings.model)
        emit("cost", {"cost_usd": cost, "usage": usage})

        if response.stop_reason == "refusal":
            raise ResearchError("Claude declined this request.")

        for block in response.content:
            if block.type == "server_tool_use" and block.name == "web_search":
                emit("web_search", {"query": block.input.get("query", "")})
                tool_log.append({"tool": "web_search", "input": block.input})

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "pause_turn":
            continue  # server-side web search loop paused; resend to let it resume

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            report = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "max_tokens":
                report += "\n\n*[Report truncated: hit the output token limit.]*"
            if not report:
                raise ResearchError(f"Claude finished without a report (stop_reason={response.stop_reason}).")
            return ResearchResult(ticker, report, response.model, usage, cost, tool_log)

        results = []
        for tu in tool_uses:
            emit("tool_call", {"name": tu.name, "input": tu.input})
            tool_log.append({"tool": tu.name, "input": tu.input})
            output, is_error = run_tool(tu.name, tu.input)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output, "is_error": is_error})

        # Out of budget or turns: hand back the results, then require the report with no further tool use.
        over_budget = cost >= settings.max_cost_usd * 0.8
        if not wrapping_up and (over_budget or turn >= settings.max_turns - 1):
            wrapping_up = True
            reason = "research budget" if over_budget else "step limit"
            results.append({"type": "text", "text": (
                f"You have reached the {reason}. Do not call any more tools. "
                "Write the full report now using what you have gathered, marking gaps as n/a.")})
            kwargs["tool_choice"] = {"type": "none"}
        messages.append({"role": "user", "content": results})

    raise ResearchError(f"Stopped after {settings.max_turns} steps without a finished report.")
