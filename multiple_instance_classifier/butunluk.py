"""hazir/ klasorunun butunlugunu dogrular.

  python butunluk.py uret     -> yerelde hazir/butunluk.json referansini yazar
  python butunluk.py kontrol  -> bulundugu ortamdaki hazir/ klasorunu referansa karsi dogrular

Colab'da: !python butunluk.py kontrol
Drive'a yuklenirken kesilen/bozulan dosyalari yakalamak icin md5 + boyut karsilastirir; ayrica
manifest/split/eval_cases tutarliligini ve her PNG'nin gercekten acilabildigini kontrol eder.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ADAY = [Path(r'D:\mamografi\multiple_instance_classifier'),
        Path('/content/drive/MyDrive/multiple_instance_classifier')]
KOK = next((k for k in ADAY if k.exists()), Path.cwd())
HAZIR = KOK / 'hazir'
REF = HAZIR / 'butunluk.json'
META = ['manifest.csv', 'split.json', 'eval_cases.json', 'config.json', 'onisleme.py']


def md5(p, blok=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for parca in iter(lambda: f.read(blok), b''):
            h.update(parca)
    return h.hexdigest()


def dosyalar():
    return sorted([p for p in HAZIR.iterdir()
                   if p.is_file() and p.name != REF.name and p.suffix != '.pyc'],
                  key=lambda p: p.name)


def uret():
    kayit = {}
    ds = dosyalar()
    for i, p in enumerate(ds, 1):
        kayit[p.name] = [p.stat().st_size, md5(p)]
        if i % 500 == 0 or i == len(ds):
            print(f'  {i}/{len(ds)}')
    toplam = sum(v[0] for v in kayit.values())
    REF.write_text(json.dumps({'kok': str(HAZIR), 'n': len(kayit),
                               'toplam_bayt': toplam, 'dosyalar': kayit}), encoding='utf-8')
    print(f'referans yazildi: {REF}')
    print(f'{len(kayit)} dosya, {toplam / 1e9:.2f} GB')
    return 0


def kontrol(hizli=False):
    print(f'kok: {KOK}')
    if not HAZIR.exists():
        print(f'PATLADI: {HAZIR} yok')
        return 1
    hata = []

    # 1) referansa karsi md5 + boyut
    if REF.exists():
        ref = json.loads(REF.read_text(encoding='utf-8'))
        var = {p.name: p for p in dosyalar()}
        eksik = sorted(set(ref['dosyalar']) - set(var))
        fazla = sorted(set(var) - set(ref['dosyalar']))
        print(f"referans: {ref['n']} dosya | ortamda: {len(var)} dosya")
        if eksik:
            hata.append(f'{len(eksik)} dosya eksik')
            print('  eksik ilk 10:', eksik[:10])
        if fazla:
            print(f'  referansta olmayan {len(fazla)} dosya:', fazla[:10])

        boyut_hata, md5_hata = [], []
        ortak = [a for a in ref['dosyalar'] if a in var]
        for i, ad in enumerate(ortak, 1):
            bek_boyut, bek_md5 = ref['dosyalar'][ad]
            if var[ad].stat().st_size != bek_boyut:
                boyut_hata.append(ad)
            elif not hizli and md5(var[ad]) != bek_md5:
                md5_hata.append(ad)
            if not hizli and (i % 500 == 0 or i == len(ortak)):
                print(f'  hash {i}/{len(ortak)}')
        if hizli:
            print(f'  hizli mod: {len(ortak)} dosyada yalnizca boyut karsilastirildi')
        if boyut_hata:
            hata.append(f'{len(boyut_hata)} dosyada boyut farki (yukleme yarim kalmis)')
            print('  boyut farki ilk 10:', boyut_hata[:10])
        if md5_hata:
            hata.append(f'{len(md5_hata)} dosyada md5 farki (icerik bozulmus)')
            print('  md5 farki ilk 10:', md5_hata[:10])
        if not (eksik or boyut_hata or md5_hata):
            print(f'  {len(ortak)} dosya {"boyut olarak" if hizli else "bit bazinda"} ayni')
    else:
        print(f'referans yok ({REF.name}) -- md5 karsilastirmasi atlandi, yapisal kontrol yapiliyor')

    # 2) meta dosyalari
    for ad in META:
        if not (HAZIR / ad).exists():
            hata.append(f'{ad} yok')
    if any(s.endswith(' yok') for s in hata):
        print('meta dosya eksik, yapisal kontrol yapilamiyor')
        return _bitir(hata)

    man = pd.read_csv(HAZIR / 'manifest.csv')
    split = json.loads((HAZIR / 'split.json').read_text(encoding='utf-8'))
    cfg = json.loads((HAZIR / 'config.json').read_text(encoding='utf-8'))
    print(f'manifest: {len(man)} satir | {man.hasta.nunique()} hasta | {man.meme.nunique()} meme')

    # 3) manifestteki her PNG var mi, fazlalik var mi
    png = {p.name for p in HAZIR.glob('*.png')}
    m_eksik = sorted(set(man.dosya) - png)
    m_fazla = sorted(png - set(man.dosya))
    if m_eksik:
        hata.append(f'manifestteki {len(m_eksik)} PNG diskte yok')
        print('  ilk 10:', m_eksik[:10])
    if m_fazla:
        hata.append(f'manifestte olmayan {len(m_fazla)} PNG diskte var')
        print('  ilk 10:', m_fazla[:10])

    # 4) split hasta seviyesinde: manifesti tam kaplasin, kumeler ortusmesin
    kume = {k: set(v) for k, v in split.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        ortak = kume[a] & kume[b]
        if ortak:
            hata.append(f'hasta sizmasi {a}/{b}: {len(ortak)} hasta')
    hepsi = set().union(*kume.values())
    if hepsi != set(man.hasta):
        hata.append(f'split manifesti kaplamiyor: split {len(hepsi)} hasta, '
                    f'manifest {man.hasta.nunique()} hasta')
    # manifest.split kolonu split.json ile ayni seyi soylemeli
    for k, v in kume.items():
        kolon = set(man[man.hasta.isin(v)].split)
        if kolon != {k}:
            hata.append(f'manifest.split kolonu split.json ile celisiyor ({k}): {kolon}')
    print('split (hasta):', {k: len(v) for k, v in kume.items()},
          '| goruntu:', man.split.value_counts().to_dict())

    # 5) etiket tutarliligi: bir memenin iki view'i ayni etikete sahip olmali
    tutarsiz = man.groupby('meme').y.nunique()
    if (tutarsiz > 1).any():
        hata.append(f'{int((tutarsiz > 1).sum())} memede view etiketleri celisiyor')

    # 6) PNG'ler gercekten acilabiliyor mu (yarim inen dosya burada patlar)
    import cv2
    bozuk, sekil_hata = [], []
    # hizli modda md5 zaten onceki tam kosuda gecmis kabul edilir, sadece ornek acilir
    if hizli:
        man_png = man.sample(min(50, len(man)), random_state=0)
        print(f'  hizli mod: {len(man_png)} ornek PNG aciliyor')
    else:
        man_png = man
    n = len(man_png)
    for i, r in enumerate(man_png.itertuples(), 1):
        im = cv2.imread(str(HAZIR / r.dosya), cv2.IMREAD_UNCHANGED)
        if im is None or im.size == 0:
            bozuk.append(r.dosya)
        elif im.ndim != 2 or im.shape != (r.h, r.w) or im.dtype != np.uint8:
            # uint8 bekleniyor: DICOM'larin 5200/5202'si 8-bit, 16-bit olan tek hasta
            # Parca 2'de max-olcekleme ile ayni araliga indirildi
            sekil_hata.append(f'{r.dosya} {im.shape} {im.dtype} != ({r.h},{r.w}) uint8')
        if i % 500 == 0 or i == n:
            print(f'  png {i}/{n}')
    if bozuk:
        hata.append(f'{len(bozuk)} PNG acilamadi')
        print('  ilk 10:', bozuk[:10])
    if sekil_hata:
        hata.append(f'{len(sekil_hata)} PNG manifest ile uyusmayan sekil/tip')
        print('  ilk 5:', sekil_hata[:5])

    # 7) eval_cases ve config
    # eval_cases.json: saliency map gorselleri icin secilmis ornek memelerin listesi
    ev = json.loads((HAZIR / 'eval_cases.json').read_text(encoding='utf-8'))
    memeler = set(man.meme)
    yok = [d['meme'] for d in ev if d['meme'] not in memeler]
    if yok:
        hata.append(f'eval_cases icinde manifeste olmayan {len(yok)} meme: {yok[:5]}')
    if tuple(cfg['GLOBAL_SIZE']) != (2304, 1280) or cfg['K'] != 6 or cfg['ESIK'] != 10:
        hata.append(f'config beklenenden farkli: {cfg}')
    print(f"eval_cases: {len(ev)} ornek meme | config GLOBAL_SIZE "
          f"{cfg['GLOBAL_SIZE']} K {cfg['K']} ESIK {cfg['ESIK']}")
    return _bitir(hata)


def _bitir(hata):
    print()
    if hata:
        print(f'BUTUNLUK BOZUK -- {len(hata)} sorun:')
        for s in hata:
            print('  -', s)
        return 1
    print('BUTUNLUK TAM -- hazir/ egitime uygun.')
    return 0


if __name__ == '__main__':
    arg = sys.argv[1:]
    if arg and arg[0] == 'uret':
        sys.exit(uret())
    sys.exit(kontrol(hizli='hizli' in arg))
