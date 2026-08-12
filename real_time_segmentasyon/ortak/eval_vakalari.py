"""Uc modelde ortak kullanilacak sabit degerlendirme vakalarini secer.

Gorsel karsilastirmanin adil olmasi icin ayni vakalar kullanilmali; her model
kendi iyi ornegini secerse rapor hicbir sey soylemez. Secim deterministik
(seed 42) ve sinif x lezyon boyutu uzerinde katmanli.

Cikti: ortak/eval_cases.json

Kullanim:
    python ortak/eval_vakalari.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import degerlendirme as dg

TOHUM = 42
BOYUT_SINIR = [0, 80, 140, 220, np.inf]
BOYUT_AD = ['kucuk', 'orta', 'buyuk', 'cok_buyuk']
BASINA = 1          # sinif x boyut kovasi basina vaka
NORMAL_ADET = 3     # bos maskeli vaka (false-positive davranisi burada gorunuyor)


def sec(split='val'):
    df = dg.manifest_oku(split)
    rng = np.random.default_rng(TOHUM)
    secili = []

    lez = df[df.lesion_px > 0].copy()
    lez['kova'] = pd.cut(lez.eq_diam, BOYUT_SINIR, labels=BOYUT_AD)
    for (cls, kova), g in lez.groupby(['cls', 'kova'], observed=True):
        g = g.sort_values('case_id')            # groupby sirasindan bagimsiz olsun
        for i in rng.permutation(len(g))[:BASINA]:
            r = g.iloc[int(i)]
            secili.append(dict(case_id=r.case_id, cls=r.cls, kova=str(kova),
                               eq_diam=round(float(r.eq_diam), 1),
                               lesion_px=int(r.lesion_px), n_components=int(r.n_components)))

    bos = df[df.lesion_px == 0].sort_values('case_id')
    for i in rng.permutation(len(bos))[:NORMAL_ADET]:
        r = bos.iloc[int(i)]
        secili.append(dict(case_id=r.case_id, cls=r.cls, kova='bos',
                           eq_diam=0.0, lesion_px=0, n_components=0))

    return sorted(secili, key=lambda d: (d['kova'], d['case_id']))


def main():
    vakalar = sec('val')
    cikti = Path(__file__).parent / 'eval_cases.json'
    cikti.write_text(json.dumps(
        {'split': 'val', 'tohum': TOHUM, 'n': len(vakalar), 'vakalar': vakalar},
        indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'{len(vakalar)} vaka -> {cikti}\n')
    print(pd.DataFrame(vakalar).to_string(index=False))


if __name__ == '__main__':
    main()
