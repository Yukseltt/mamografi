# literatur_raporu.html -> PDF
# HTML'i elle duzenledikten sonra SADECE bunu calistir.
# DIKKAT: build_report.py HTML'i sifirdan uretir ve elle yapilan degisiklikleri siler.

$ErrorActionPreference = "Stop"

$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chrome)) {
    $chrome = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
}
if (-not (Test-Path $chrome)) { throw "Chrome/Edge bulunamadi" }

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $dir "literatur_raporu.html"
$pdf  = Join-Path (Split-Path -Parent $dir) "Mamografi_Meme_Kanseri_Tahmini_Literatur_Raporu.pdf"

if (-not (Test-Path $html)) { throw "HTML bulunamadi: $html" }
if (Test-Path $pdf) { Remove-Item $pdf -Force }

$url = "file:///" + ($html -replace '\\', '/')

& $chrome --headless --disable-gpu --no-pdf-header-footer `
          --print-to-pdf="$pdf" --virtual-time-budget=60000 `
          --user-data-dir="$env:TEMP\chrome_pdf_literatur" $url 2>$null

Start-Sleep -Seconds 3

if (Test-Path $pdf) {
    $kb = [math]::Round((Get-Item $pdf).Length / 1KB, 1)
    Write-Output "PDF olusturuldu: $pdf  ($kb kB)"
} else {
    throw "PDF olusturulamadi."
}
