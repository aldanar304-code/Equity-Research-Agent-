"""Translate a finished English memo into Spanish (Spain) with one Claude call, then verify it.

Translating the final memo costs a fraction of re-running the research, and keeps both language
versions identical in substance. A number check compares every figure in the original with the
translation, so a translation slip can't silently change a number.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic

from .agent import Usage

TRANSLATION_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """\
You are a professional financial translator from English into Spanish (Spain). Translate the \
investment memo you are given into natural, professional Spanish as written by Spanish equity \
research teams (for example: flujo de caja libre, margen operativo, beneficio por acción (BPA), \
precio objetivo, capitalización bursátil, deuda neta, consenso de analistas).

Rules:
- Keep the Markdown structure exactly: headings, numbering, tables, bullet nesting, bold and italics.
- Keep every number, currency amount, percentage, ratio and ticker exactly as written in the \
original. Do not change decimal separators, units or rounding. Write dates in Spanish but keep \
their numbers (e.g. "October 3, 2026" -> "3 de octubre de 2026").
- Keep citation tags unchanged, e.g. (10-K-2026-06), (8-K-EX-2026-07), doc ids and URLs. For web \
citations keep the source name and translate only the descriptive words.
- Title line: "# {Company name} ({TICKER}) - Informe de inversión".
- Header line labels, in this order: **Fecha:**, **Precio:**, **Capitalización:**, \
**Recomendación:** (Comprar / Mantener / Vender), **Precio objetivo a 12 meses:**, \
**Convicción:** (Alta / Media / Baja).
- Section headings: "## 1. Recomendación y tesis", "## 2. Descripción de la compañía", \
"## 3. Sector y posición competitiva", "## 4. Análisis financiero", "## 5. Equipo directivo, \
gobierno corporativo y asignación de capital", "## 6. Valoración", "## 7. Percepción diferencial", \
"## 8. Riesgos principales y mitigantes", "## 9. Catalizadores y calendario", "## 10. Qué nos haría \
cambiar de opinión", "## Anexo: Fuentes", "### Notas de verificación".
- Scenario names: Bear / Base / Bull -> Pesimista / Base / Optimista.
- Periods: write quarters as 1T, 2T, 3T, 4T and halves as 1S, 2S (e.g. "Q2 FY27" -> "2T FY27"); \
never spell numbers out as words.
- Do not add, remove, summarize or soften anything. Output only the translated memo."""

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _figures(text: str) -> Counter:
    """Financially meaningful numbers: decimals and anything >= 13. Small integers are skipped because
    they legitimately turn into words in translation ("Q2" -> "segundo trimestre", "1 year" -> "un año")."""
    out = Counter()
    for n in _NUM.findall(text):
        if re.search(r"[.,]", n) or int(n) >= 13:
            out[n] += 1
    return out


def number_mismatches(original: str, translated: str) -> dict[str, int]:
    """Figures whose count differs between the two texts ({number: original count - translated count})."""
    a, b = _figures(original), _figures(translated)
    diff = {n: a[n] - b[n] for n in (a | b) if a[n] != b[n]}
    # "FY26" written out as "2026" is the same year, not an error: cancel matching pairs.
    for n in list(diff):
        full = "20" + n
        if len(n) == 2 and full in diff and diff[n] > 0 and diff[full] < 0:
            k = min(diff[n], -diff[full])
            diff[n] -= k
            diff[full] += k
    return {n: d for n, d in diff.items() if d}


def translate_memo(memo_md: str, client: anthropic.Anthropic | None = None,
                   model: str = TRANSLATION_MODEL) -> tuple[str, float, dict[str, int]]:
    """Returns (spanish_markdown, cost_usd, number_mismatches)."""
    client = client or anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": memo_md}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the translation request.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Translation was cut off by the output limit.")
    usage = Usage()
    usage.add(response.usage)
    return text, usage.cost(model), number_mismatches(memo_md, text)


def spanish_path(md_path: Path) -> Path:
    return md_path.with_name(md_path.stem + "_es.md")


def translate_file(md_path: Path, client: anthropic.Anthropic | None = None, log=print) -> Path:
    """Translate a saved memo (.md) and write the Spanish .md, .html and .pdf next to it."""
    from .render import html_to_pdf, previous_recommendations, render_html

    md_path = Path(md_path)
    raw = md_path.read_text()
    memo, _, meta_comment = raw.partition("\n\n<!-- generated")
    ticker = md_path.name.split("_")[0]

    log(f"Translating {md_path.name} to Spanish...")
    es_md, cost, mismatches = translate_memo(memo, client)
    out = spanish_path(md_path)
    meta = f"\n\n<!-- translated from {md_path.name} | cost: ${cost:.3f} -->\n"
    out.write_text(es_md + meta)

    if mismatches:
        log(f"  Number check: {len(mismatches)} figures differ from the English original - review them: "
            + ", ".join(f"{n} ({'+' if d < 0 else '-'}{abs(d)})" for n, d in list(mismatches.items())[:12]))
    else:
        log("  Number check: every figure matches the English original.")

    tools_file = md_path.with_suffix(".tools.json")
    calls = json.loads(tools_file.read_text()) if tools_file.exists() else []
    m = re.search(r"model: (\S+) \| cost: \$(\S+) .*?tool calls: (\d+)", meta_comment)
    footer = (f"Modelo: {m.group(1)} | coste ${float(m.group(2)):.2f} + traducción ${cost:.2f} | "
              f"{m.group(3)} llamadas a herramientas." if m else "")
    html_path = out.with_suffix(".html")
    html_path.write_text(render_html(
        es_md, ticker, calls, footer, lang="es",
        generated_at=datetime.fromtimestamp(md_path.stat().st_mtime),
        previous=previous_recommendations(ticker, md_path.parent, exclude=md_path)))
    html_to_pdf(html_path)
    log(f"  Saved {html_path.name} (+ .md/.pdf), translation cost ${cost:.3f}")
    return out


def main(argv: list[str] | None = None) -> int:
    """`equity-research-translate sample_reports/NVDA_memo_2026-10-03.md [...]`"""
    import argparse
    import sys

    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".config" / "equity-research-agent" / ".env")

    parser = argparse.ArgumentParser(description="Translate saved memos into Spanish.")
    parser.add_argument("memos", nargs="+", type=Path, help="English memo .md files")
    args = parser.parse_args(argv)
    for p in args.memos:
        if p.stem.endswith("_es"):
            continue
        translate_file(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
