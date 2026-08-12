# RTMDet-Ins egitimi + otomatik degerlendirme
#
# Girdi boyutu / augmentasyon seviyesi / tohum rtmdet_ins_busi.py icindeki
# GIRDI ve AUG_SEVIYE degiskenlerinden okunur (komut satirindan ezilmez:
# ikisi hem pipeline'da hem work_dir adinda kullaniliyor, ayrisirlarsa
# sessizce yanlis klasore yazilir).
#
# Kullanim:
#   powershell -ExecutionPolicy Bypass -File real_time_segmentasyon\rtmdet\kos.ps1
#   ... -Resume            # kesilen kosuya devam
#   ... -Epoch 40          # kisa deneme

param(
    [switch]$Resume,
    [int]$Epoch = 0,
    [int]$Workers = 0
)

$ErrorActionPreference = 'Stop'

$PY   = "d:\mamografi\.venv_mmdet\Scripts\python.exe"
$KOK  = "d:\mamografi\real_time_segmentasyon\rtmdet"
$CFG  = "$KOK\rtmdet_ins_busi.py"

# config'ten GIRDI / AUG_SEVIYE / TOHUM oku, work_dir'i ayni kuralla kur
$icerik = Get-Content $CFG -Raw
$girdi = [regex]::Match($icerik, 'GIRDI\s*=\s*(\d+)').Groups[1].Value
$aug   = [regex]::Match($icerik, "AUG_SEVIYE\s*=\s*'(\w+)'").Groups[1].Value
$tohum = [regex]::Match($icerik, 'TOHUM\s*=\s*(\d+)').Groups[1].Value
$work  = "$KOK\calisma\rtmdet_ins_tiny_${girdi}_${aug}_s${tohum}"

Write-Host "girdi=$girdi  aug=$aug  tohum=$tohum"
Write-Host "work_dir=$work"
Write-Host ("=" * 60)

# ---- 1) egitim ----
$argv = @('-u', "$KOK\egit.py", '--workers', $Workers)
if ($Resume) { $argv += '--resume' }
if ($Epoch -gt 0) { $argv += @('--epoch', $Epoch) }

$t0 = Get-Date
& $PY @argv
if ($LASTEXITCODE -ne 0) { throw "Egitim hata ile bitti (exit $LASTEXITCODE)" }
Write-Host ("Egitim suresi: {0:hh\:mm\:ss}" -f ((Get-Date) - $t0))

# ---- 2) en iyi checkpoint ----
# Checkpoint secimi artik Dice uzerinden (busi/dice_tum), segm_mAP uzerinden degil
$ckpt = Get-ChildItem $work -Filter 'best_busi_dice*.pth' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $ckpt) {
    Write-Host "UYARI: best_busi_dice* bulunamadi, son epoch checkpointi kullanilacak"
    $ckpt = Get-ChildItem $work -Filter '*.pth' | Sort-Object LastWriteTime | Select-Object -Last 1
}
Write-Host ("=" * 60)
Write-Host "degerlendirilen checkpoint: $($ckpt.Name)"

# ---- 3) orijinal cozunurlukte Dice (val) ----
& $PY -u "$KOK\dice_degerlendirme.py" `
    --checkpoint $ckpt.FullName `
    --split val `
    --cikti "$work\dice_val_${girdi}_${aug}_s${tohum}.xlsx"
