# RTMDet-Ins yerel egitim ortami - tekrarlanabilir kurulum
#
# Neden yerel: Colab'in torch 2.11 / cu128 / py3.12 ortamina uyan mmcv wheel'i yok.
# Resmi wheel'ler torch 2.1'de, topluluk wheel'leri 2.6'da bitiyor. Yerelde surumu
# kendimiz sabitleyip bir kez kuruyoruz.
#
# Kullanim:  powershell -ExecutionPolicy Bypass -File kurulum.ps1

$ErrorActionPreference = 'Stop'

$PY310 = "C:\Users\yt100\AppData\Local\Programs\Python\Python310\python.exe"
$VENV  = "d:\mamografi\.venv_mmdet"
$PY    = "$VENV\Scripts\python.exe"
$YAMA  = "d:\mamografi\real_time_segmentasyon\yerel_ortam\mmdet_yama.py"

# Python 3.10 zorunlu: mmcv wheel'i cp310/cp311/cp312 icin var, cp313 icin yok
Write-Host "1/6  venv olusturuluyor (Python 3.10)"
& $PY310 -m venv $VENV

# Makinede TLS araya girme var (antivirus/proxy) - pip sertifika dogrulamasinda takiliyor
Write-Host "2/6  pip trusted-host yapilandirmasi"
@"
[global]
trusted-host = pypi.org
               files.pythonhosted.org
               download.pytorch.org
               miropsota.github.io
               github.com
               objects.githubusercontent.com
"@ | Out-File -FilePath "$VENV\pip.ini" -Encoding utf8

& $PY -m pip install --upgrade pip --quiet

# mmcv wheel'inin derlendigi tam surum - baska bir torch surumu ile calismaz
Write-Host "3/6  torch 2.6.0 + torchvision 0.21.0 (cu124)  [~2.5 GB]"
& $PY -m pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124

# Resmi mmcv wheel'i yok, topluluk build'i kullaniliyor (derleme gerektirmez)
Write-Host "4/6  mmcv 2.2.0 (topluluk prebuilt wheel)"
& $PY -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "mmcv==2.2.0+a8073c7pt2.6.0cu124"

Write-Host "5/6  mmdet 3.3.0"
& $PY -m pip install "mmdet==3.3.0"

# mmdet 3.3.0 `mmcv<2.2.0` istiyor, elimizdeki tam 2.2.0 -> assert'i gevset
Write-Host "6/6  mmdet surum kontrolu yamasi"
& $PY $YAMA

Write-Host ""
Write-Host "Dogrulama:"
& $PY "d:\mamografi\real_time_segmentasyon\yerel_ortam\ortam_dogrulama.py"
