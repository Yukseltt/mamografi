"""mmdet 3.3.0'in mmcv ust surum kontrolunu gevset.

mmdet 3.3.0 `mmcv>=2.0.0rc4, <2.2.0` istiyor. Elimizdeki tek secenek olan topluluk
prebuilt wheel'i tam 2.2.0 (torch 2.6 icin resmi wheel yok). Sinir disi birakan tek
sey bu assert; 2.2.0 ile mmdet 3.3.0 arasinda bilinen API kirilmasi yok.

Idempotent: birden fazla kez calistirilabilir. mmdet yeniden kurulursa tekrar
calistirilmali.
"""
import importlib.util
import sys
from pathlib import Path

ESKI = "mmcv_maximum_version = '2.2.0'"
YENI = "mmcv_maximum_version = '2.3.0'  # yama: bkz. yerel_ortam/mmdet_yama.py"

# find_spec modulu calistirmadan yolunu verir - `import mmdet` zaten assert'e takiliyor
spec = importlib.util.find_spec('mmdet')
if spec is None or not spec.origin:
    print('HATA: mmdet bulunamadi')
    sys.exit(1)
hedef = Path(spec.origin)
metin = hedef.read_text(encoding='utf-8')

if YENI in metin:
    print('zaten yamali:', hedef)
    sys.exit(0)

if ESKI not in metin:
    print('HATA: beklenen satir bulunamadi, mmdet surumu farkli olabilir')
    print('  hedef:', hedef)
    sys.exit(1)

yedek = hedef.with_suffix('.py.yedek')
if not yedek.exists():
    yedek.write_text(metin, encoding='utf-8')

hedef.write_text(metin.replace(ESKI, YENI), encoding='utf-8')
print('yamalandi:', hedef)
print('yedek    :', yedek)
