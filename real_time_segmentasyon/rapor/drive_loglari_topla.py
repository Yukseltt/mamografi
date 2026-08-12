"""Colab hucresi: dokuz kosunun loglarini Drive'da tek klasore + zip'e toplar.

Bu dosyayi Colab'da bir hucreye yapistirip calistir. Drive'in bagli olmasi disinda
on kosulu yok; notebook'un CONFIG hucrelerine ihtiyac duymuyor.

Cikti:
    /content/drive/MyDrive/real_time_segmentasyon/rapor_loglari/        (6 dosya)
    /content/drive/MyDrive/real_time_segmentasyon/rapor_loglari.zip

Zip'i Drive arayuzunden indirip icerigini yerelde
    real_time_segmentasyon/rapor/loglar_faz2/ham/
klasorune ac.
"""
import shutil
import zipfile
from pathlib import Path

PROJE = Path('/content/drive/MyDrive/real_time_segmentasyon')
CIKTI = PROJE / 'rapor_loglari'
TOHUMLAR = (42, 43, 44)

if not PROJE.exists():
    raise SystemExit(
        f'{PROJE} yok. Drive bagli mi?\n'
        "  from google.colab import drive; drive.mount('/content/drive')")

CIKTI.mkdir(exist_ok=True)

# (kaynak, hedef ad) ciftleri. Hedef adlar loglari_topla_faz2.py'nin bekledigi adlar.
istenen = []
for t in TOHUMLAR:
    k = PROJE / f'segformer_b0_256_orta_s{t}'
    istenen.append((k / f'segformer_b0_256_orta_s{t}_log.xlsx',
                    f'segformer_b0_256_orta_s{t}_log.xlsx'))
    k = PROJE / f'yolo11n_seg_256_orta_s{t}'
    istenen.append((k / 'results.csv', f'yolo11n_seg_256_orta_s{t}_results.csv'))

bulunan, eksik = [], []
for kaynak, hedef in istenen:
    if kaynak.exists():
        shutil.copy2(kaynak, CIKTI / hedef)
        bulunan.append(hedef)
        print(f'  ok      {hedef}  ({kaynak.stat().st_size / 1024:.0f} KB)')
    else:
        eksik.append(kaynak)
        print(f'  EKSIK   {kaynak}')

if eksik:
    # Dosya adi tahminimiz tutmadiysa klasorde ne oldugunu goster: sessizce
    # eksik birakmak, yerelde figuru yarim uretmekten daha kotu.
    print('\nEksik olanlarin klasorlerinde ne var:')
    for p in eksik:
        d = p.parent
        print(f'  {d.name}/:',
              sorted(x.name for x in d.iterdir()) if d.exists() else 'KLASOR YOK')

zip_yolu = PROJE / 'rapor_loglari.zip'
with zipfile.ZipFile(zip_yolu, 'w', zipfile.ZIP_DEFLATED) as z:
    for ad in bulunan:
        z.write(CIKTI / ad, ad)

print(f'\n{len(bulunan)}/{len(istenen)} dosya toplandi')
print('zip:', zip_yolu, f'({zip_yolu.stat().st_size / 1024:.0f} KB)')
print('\nDrive arayuzunden rapor_loglari.zip dosyasini indir, icerigini'
      ' rapor/loglar_faz2/ham/ icine ac.')
