"""Servis motorunun uctan uca dogrulanmasi - dort arka uc, ayni protokol.

Iki sey olculuyor:
  1. **Gecikme, tek kosuda.** Parca 8'de "75 ms" bilesenlerden derlenmisti (on isleme bir
     olcumden, ag baska bir olcumden). Burada tek hatta, tek kosuda dogrulaniyor.
  2. **Kalite kilidi.** Her arka ucun urettigi maske, notebook'un kaydettigi vaka basina
     referans Dice ile karsilastiriliyor. Sapma toleransi asilirsa arka uc ELENIYOR -
     hizli ama farkli bir sey ureten bir yol dagitima giremez.

Kullanim:
    python servis/dogrula.py            # 4 arka uc, 20 vaka
    python servis/dogrula.py --n 50     # tum test seti
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'servis'))

import nibabel as nib      # noqa: E402
import hat                 # noqa: E402
from motor import Motor    # noqa: E402

ISINMA = 3
DICE_TOLERANS = 2e-3
ARKA_UCLAR = ['torch-fp32', 'torch-fp16', 'onnx-cuda', 'trt-fp16']


def veri_yollari():
    vaka_re = re.compile(r'(sub-[A-Za-z0-9]+)')
    d = {}
    for q in glob.glob(str(KOK / 'ISLES-2022' / '**' / '*.nii.gz'), recursive=True):
        v = next(iter(vaka_re.findall(q)), None)
        if v is None:
            continue
        ad = os.path.basename(q).lower().replace('.nii.gz', '')
        for s in ('dwi', 'adc', 'msk'):
            if ad.endswith('_' + s):
                d.setdefault(v, {})[s] = q
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--model', default='unet3d')
    a = ap.parse_args()

    ref = json.load(open(KOK / 'dagitim' / 'olcum_referansi.json', encoding='utf-8'))
    yollar = veri_yollari()
    test = [v for v in ref['split_test'] if 'dwi' in yollar.get(v, {})][:a.n + ISINMA]
    referans = ref['referans_dice'][a.model]
    print('model:', a.model, '| vaka:', len(test),
          f'({ISINMA} isinma + {len(test)-ISINMA} olcum)')
    print()

    satir = []
    for arka in ARKA_UCLAR:
        try:
            m = Motor(model_key=a.model, arka_uc=arka)
        except Exception as e:
            print(f'{arka:11s} KURULAMADI: {str(e)[:90]}')
            satir.append(dict(arka_uc=arka, durum=str(e)[:120]))
            continue

        kayit = []
        for i, v in enumerate(test):
            r = m.calistir(yollar[v]['dwi'], yollar[v]['adc'])
            if i < ISINMA:
                continue
            gt = hat.gt_hazirla(nib.load(yollar[v]['msk']), m.cfg, r['kirpma'])
            d = hat.dice(r['maske'], gt)
            kayit.append(dict(vaka=v, dice=d, referans=referans.get(v),
                              sapma=abs(d - referans[v]) if v in referans else np.nan,
                              **r['sure'], **{k: r['klinik'][k] for k in
                                              ('hacim_ml', 'lezyon_sayisi', 'hemisfer')}))
        df = pd.DataFrame(kayit)
        sap = df.sapma.max()
        gecti = not (np.isfinite(sap) and sap > DICE_TOLERANS)
        print(f'{arka:11s} {df.toplam_sn.mean()*1000:7.1f} ms/vaka  '
              f'(on {df.on_isleme_sn.mean()*1000:6.1f} | ag {df.ag_sn.mean()*1000:6.1f} | '
              f'son {df.son_isleme_sn.mean()*1000:5.1f})  Dice {df.dice.mean():.4f}  '
              f'sapma {sap:.5f}  {"GECTI" if gecti else "KALDI"}')
        satir.append(dict(arka_uc=arka, vaka=len(df),
                          toplam_ms=df.toplam_sn.mean() * 1000,
                          toplam_std_ms=df.toplam_sn.std() * 1000,
                          on_isleme_ms=df.on_isleme_sn.mean() * 1000,
                          ag_ms=df.ag_sn.mean() * 1000,
                          son_isleme_ms=df.son_isleme_sn.mean() * 1000,
                          okuma_ms=df.okuma_sn.mean() * 1000,
                          dice=df.dice.mean(), max_sapma=sap,
                          kalite_kilidi='GECTI' if gecti else 'KALDI',
                          hedef_1sn=bool(df.toplam_sn.mean() < 1.0),
                          durum='OK'))
        df.to_excel(KOK / 'servis' / f'vaka_bazinda_{arka}.xlsx', index=False)
        del m

    ozet = pd.DataFrame(satir)
    print()
    print('=' * 78)
    print(ozet.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    ozet.to_excel(KOK / 'servis' / 'servis_dogrulama.xlsx', index=False)
    print()
    print('yazildi: servis/servis_dogrulama.xlsx')

    iyi = ozet[(ozet.durum == 'OK') & (ozet.kalite_kilidi == 'GECTI')]
    if len(iyi):
        en = iyi.loc[iyi.toplam_ms.idxmin()]
        print(f"\nEN IYI: {en.arka_uc} - {en.toplam_ms:.1f} ms/vaka, Dice {en.dice:.4f}")


if __name__ == '__main__':
    main()
