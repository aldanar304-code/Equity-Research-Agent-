"""AI equity research agent built on Claude and free market data."""

from .agent import ResearchError, ResearchResult, Settings, research

__all__ = ["research", "Settings", "ResearchResult", "ResearchError"]
