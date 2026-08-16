#!/bin/bash
# hpo_raporu.html -> PDF
# HTML'i elle duzenledikten sonra SADECE bunu calistir.
# DIKKAT: build_hpo_report.py HTML'i sifirdan uretir, elle yapilan degisiklikleri siler.

set -e

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML="$DIR/hpo_raporu.html"
PDF="$DIR/hpo_sonuc_raporu.pdf"

if [ ! -f "$HTML" ]; then
    echo "HTML bulunamadi: $HTML" >&2
    exit 1
fi
[ -f "$PDF" ] && rm -f "$PDF"

URL="file://$HTML"

"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
          --print-to-pdf="$PDF" --virtual-time-budget=30000 \
          --user-data-dir="$(mktemp -d)" "$URL" 2>/dev/null

sleep 3

if [ -f "$PDF" ]; then
    MB=$(echo "scale=2; $(stat -f%z "$PDF") / 1048576" | bc)
    echo "PDF olusturuldu: $PDF  (${MB} MB)"
else
    echo "PDF olusturulamadi." >&2
    exit 1
fi
