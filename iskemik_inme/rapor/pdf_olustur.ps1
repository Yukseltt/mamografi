# isles_raporu.html -> PDF
# HTML'i elle duzenledikten sonra SADECE bunu calistir.
# DIKKAT: build_report.py HTML'i sifirdan uretir ve elle yapilan degisiklikleri siler.

$ErrorActionPreference = "Stop"

$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chrome)) {
    $chrome = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
}
if (-not (Test-Path $chrome)) { throw "Chrome/Edge bulunamadi" }

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $dir "isles_raporu.html"
$pdf  = Join-Path $dir "isles_sonuc_raporu.pdf"

if (-not (Test-Path $html)) { throw "HTML bulunamadi: $html" }
if (Test-Path $pdf) { Remove-Item $pdf -Force }

$url = "file:///" + ($html -replace '\\', '/')

& $chrome --headless --disable-gpu --no-pdf-header-footer `
          --print-to-pdf="$pdf" --virtual-time-budget=60000 `
          --user-data-dir="$env:TEMP\chrome_pdf_isles" $url 2>$null

Start-Sleep -Seconds 3

if (Test-Path $pdf) {
    $mb = [math]::Round((Get-Item $pdf).Length / 1MB, 2)
    Write-Output "PDF olusturuldu: $pdf  ($mb MB)"
} else {
    throw "PDF olusturulamadi."
}
