"""The research agent: a tool-use loop over Claude with cost tracking and a hard budget."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import anthropic

from . import filings
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
    model: str = _env("EQUITY_AGENT_MODEL", "claude-sonnet-5-5")
    effort: str = _env("EQUITY_AGENT_EFFORT", "medium")  # low | medium | high
    max_cost_usd: float = _env("EQUITY_AGENT_MAX_COST_USD", "1.50", float)
    max_turns: int = _env("EQUITY_AGENT_MAX_TURNS", "24", int)
    web_searches: int = _env("EQUITY_AGENT_WEB_SEARCHES", "6", int)
    download_filings: bool = True
    fact_check: bool = True


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


FACT_CHECK_PROMPT = """\
Fact-check pass. Re-read your memo line by line. For every number, date and factual claim, confirm \
that it appears in a tool result or cited web source in this conversation, or is clearly labelled \
as your own estimate with the arithmetic shown. Check that figures agree with each other across \
sections (target, scenario values, DCF outputs, peer table). Where you are unsure, verify with \
search_filings, read_filing or the data tools (at most 6 tool calls). Fix anything wrong or \
unsupported - correct it, label it as an estimate, or remove it. Do not add new analysis.

Return the complete corrected memo in the same format, starting with the title line. At the very \
end of the Appendix add a "### Fact-check notes" list describing each correction you made, or \
"No corrections needed." if there were none."""


class _Session:
    """One conversation with Claude: runs tool-use turns until Claude returns a final text answer."""

    def __init__(self, client, settings: Settings, emit: EventHandler, kwargs: dict, messages: list[dict]):
        self.client, self.settings, self.emit = client, settings, emit
        self.kwargs, self.messages = kwargs, messages
        self.usage, self.tool_log = Usage(), []
        self.model = settings.model

    @property
    def cost(self) -> float:
        return self.usage.cost(self.settings.model)

    def run(self, max_turns: int, budget_fraction: float, label: str) -> str:
        """Loop until a final answer. Past budget_fraction of the cap (or the turn limit), tools
        are switched off and Claude must answer with what it has."""
        wrapping_up = False
        for turn in range(1, max_turns + 1):
            self.emit("status", {"message": f"{label} - writing" if wrapping_up else f"{label} (step {turn})"})
            try:
                response = self.client.beta.messages.create(messages=self.messages, **self.kwargs)
            except anthropic.APIStatusError as exc:
                raise ResearchError(f"Claude API error {exc.status_code}: {exc.message}") from exc
            except anthropic.APIConnectionError as exc:
                raise ResearchError("Could not reach the Claude API - check your network.") from exc

            self.usage.add(response.usage)
            self.model = response.model
            self.emit("cost", {"cost_usd": self.cost, "usage": self.usage})
            if response.stop_reason == "refusal":
                raise ResearchError("Claude declined this request.")

            for block in response.content:
                if block.type == "server_tool_use" and block.name == "web_search":
                    self.emit("web_search", {"query": block.input.get("query", "")})
                    self.tool_log.append({"tool": "web_search", "input": block.input})

            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue  # server-side web search loop paused; resend to let it resume

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n*[Truncated: hit the output token limit.]*"
                if not text:
                    raise ResearchError(f"Claude finished without an answer (stop_reason={response.stop_reason}).")
                return text

            results = []
            for tu in tool_uses:
                self.emit("tool_call", {"name": tu.name, "input": tu.input})
                self.tool_log.append({"tool": tu.name, "input": tu.input})
                output, is_error = run_tool(tu.name, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output, "is_error": is_error})

            over_budget = self.cost >= self.settings.max_cost_usd * budget_fraction
            if not wrapping_up and (over_budget or turn >= max_turns - 1):
                wrapping_up = True
                reason = "research budget" if over_budget else "step limit"
                results.append({"type": "text", "text": (
                    f"You have reached the {reason}. Do not call any more tools. "
                    "Write the full answer now using what you have gathered, marking gaps as n/a.")})
                self.kwargs["tool_choice"] = {"type": "none"}
            self.messages.append({"role": "user", "content": results})

        raise ResearchError(f"Stopped after {max_turns} steps without a finished answer.")


def research(
    ticker: str,
    settings: Settings | None = None,
    on_event: EventHandler | None = None,
    client: anthropic.Anthropic | None = None,
) -> ResearchResult:
    """Download the filings, research one ticker, fact-check the draft and return the Markdown memo."""
    settings = settings or Settings()
    client = client or anthropic.Anthropic()
    emit = on_event or (lambda kind, payload: None)
    ticker = ticker.strip().upper()

    manifest = []
    if settings.download_filings:
        emit("status", {"message": "Downloading SEC filings"})
        try:
            manifest = filings.download_filings(ticker, emit)
        except Exception as exc:  # e.g. non-US company not in EDGAR
            emit("status", {"message": f"Could not download SEC filings: {exc}"})

    session = _Session(client, settings, emit, _request_kwargs(settings), [{
        "role": "user",
        "content": (f"Write a full investment memo on {ticker}. Today's date is {date.today():%B %d, %Y}.\n\n"
                    + filings.manifest_summary(manifest)),
    }])
    # Leave ~30% of the budget for the fact-check pass.
    memo = session.run(settings.max_turns, 0.7 if settings.fact_check else 0.8, "Researching")

    if settings.fact_check:
        if session.cost < settings.max_cost_usd * 0.85:
            session.kwargs.pop("tool_choice", None)
            session.messages.append({"role": "user", "content": FACT_CHECK_PROMPT})
            memo = session.run(8, 0.95, "Fact-checking")
        else:
            memo += "\n\n*Fact-check pass skipped: spending cap reached.*"

    return ResearchResult(ticker, memo, session.model, session.usage, session.cost, session.tool_log)
