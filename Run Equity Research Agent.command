#!/bin/zsh
# Double-click this file in Finder to write an investment memo.
cd "$(dirname "$0")"

if [ ! -x .venv/bin/equity-research ]; then
  echo "Setting up for the first time (a minute or two)..."
  UV="$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")"
  "$UV" sync || { echo "Setup failed. Install uv: https://docs.astral.sh/uv/"; read -k1; exit 1; }
fi

echo ""
echo "=== AI Equity Research Agent ==="
echo "Each memo costs about \$0.45 of API credit (capped at \$1.50)."
echo ""
printf "Ticker (e.g. AAPL): "
read TICKER
TICKER="${TICKER:u}"
[ -z "$TICKER" ] && { echo "No ticker entered."; read -k1; exit 0; }

printf "Also write a Spanish version? (about \$0.10 extra) (y/n): "
read SPANISH
EXTRA=()
[ "$SPANISH" = "y" ] && EXTRA=(--spanish)

printf "Publish it to GitHub and the live demo when done? (y/n): "
read PUBLISH

OUT="reports"
[ "$PUBLISH" = "y" ] && OUT="sample_reports"

echo ""
.venv/bin/equity-research "$TICKER" --out "$OUT" "${EXTRA[@]}" || { echo ""; echo "Something went wrong (see above)."; read -k1; exit 1; }

HTML="$(ls -t "$OUT"/"${TICKER}"_memo_*.html 2>/dev/null | grep -v '_es.html' | head -1)"
[ -n "$HTML" ] && open "$HTML"
HTML_ES="$(ls -t "$OUT"/"${TICKER}"_memo_*_es.html 2>/dev/null | head -1)"
[ "$SPANISH" = "y" ] && [ -n "$HTML_ES" ] && open "$HTML_ES"

if [ "$PUBLISH" = "y" ]; then
  echo ""
  echo "Publishing..."
  git add "$OUT" && git commit -q -m "Add $TICKER memo" && git pull -q --rebase && git push -q \
    && echo "Published. The live demo updates in a minute or two." \
    || echo "Publishing failed - the memo is still saved in $OUT/."
fi

echo ""
echo "Done. Press any key to close."
read -k1
