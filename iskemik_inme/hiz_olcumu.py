"""Gercek zamanli gecikme olcumu - yerel RTX 2060, PyTorch merdiveni (basamak 0-1).

PLAN.md Bolum 3 protokolu:
  - Olculen sey UCTAN UCA: on isleme + ileri gecis + son isleme.
  - Diskten .nii.gz okuma butcenin DISINDA (PACS'tan gelen hacim modelin maliyeti degil),
    ama ayri kolonda raporlaniyor.
  - Isinma kosulari atilir, her vaka tek tek senkronlanir.
  - Bilesenler (on isleme / ag / son isleme) ayri tutulur.

Ve BUSI projesindeki disiplin: hizi olculen kodun Dice'i, kalitesi olculen kodla ayni
olmali. Her vakada uretilen maske notebook'un kaydettigi referans Dice ile
karsilastiriliyor; sapma toleransi asilirsa script DURUR. Aksi halde "su model su kadar
hizli" cumlesi iki farkli seyi olcer.

Kullanim:
    python hiz_olcumu.py              # GPU, 50 test vakasi
    python hiz_olcumu.py --n 15       # daha az vaka (hizli deneme)
    python hiz_olcumu.py --cpu        # CPU olcumunu de ekle
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / 'ortak'))
import hat                                   # noqa: E402
import hat_gpu                               # noqa: E402
import modeller as md                        # noqa: E402

DAGITIM = KOK / 'dagitim'
VERI = KOK / 'ISLES-2022'
ISINMA = 3
DICE_TOLERANS = 2e-3
D_HEDEF = 96                                 # Parca 2: max D 90 -> 8'in kati 96


def veri_yollari():
    vaka_re = re.compile(r'(sub-[A-Za-z0-9]+)')
    d = {}
    for q in glob.glob(str(VERI / '**' / '*.nii.gz'), recursive=True):
        v = next(iter(vaka_re.findall(q)), None)
        if v is None:
            continue
        ad = os.path.basename(q).lower().replace('.nii.gz', '')
        for s in ('dwi', 'adc', 'msk'):
            if ad.endswith('_' + s):
                d.setdefault(v, {})[s] = q
    return d


def olc(model, konfig, secim, cfg, vakalar, yollar, referans, cihaz, etiket,
        onisleme='cpu', hassasiyet='fp32'):
    satir = []
    for i, v in enumerate(vakalar):
        ham, okuma_sn = hat.oku(yollar[v]['dwi'], yollar[v]['adc'])
        msk_im = nib.load(yollar[v]['msk'])

        if cihaz == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if onisleme == 'gpu':
            xt, dilim = hat_gpu.on_isle_gpu(ham, cfg, cihaz)
            x = xt.cpu().numpy()
        else:
            x, dilim = hat.on_isle(ham, cfg)
        girdi, sekil = hat.girdi_tensoru(x, konfig, cfg, D_HEDEF)
        if cihaz == 'cuda':
            torch.cuda.synchronize()
        t_on = time.perf_counter() - t0

        t = torch.from_numpy(girdi).to(cihaz)
        if hassasiyet == 'fp16':
            t = t.half()
        if cihaz == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            if konfig['mimari'] == 'unet3d':
                p = torch.sigmoid(model(t))[0, 0]
            else:
                parca = [torch.sigmoid(model(t[j:j + konfig['batch']]))[:, 0]
                         for j in range(0, len(t), konfig['batch'])]
                p = torch.cat(parca)
        if cihaz == 'cuda':
            torch.cuda.synchronize()
        t_ag = time.perf_counter() - t1

        t2 = time.perf_counter()
        olasilik = p.float().cpu().numpy()
        maske = hat.son_isle(olasilik, sekil, konfig, cfg, D_HEDEF,
                             secim['esik'], secim['min_voxel'])
        t_son = time.perf_counter() - t2

        if i < ISINMA:
            continue

        # GT'yi tahminle ayni uzaya sokmak zamanlamanin DISINDA - cikarimda GT yok
        gt = hat.gt_hazirla(msk_im, cfg, dilim)
        d = hat.dice(maske, gt)
        ref = referans.get(v)
        satir.append(dict(vaka=v, okuma_sn=okuma_sn, on_isleme_sn=t_on, ag_sn=t_ag,
                          son_isleme_sn=t_son, toplam_sn=t_on + t_ag + t_son,
                          dice=d, referans_dice=ref,
                          dice_sapma=abs(d - ref) if ref is not None else np.nan))
    df = pd.DataFrame(satir)
    df.insert(0, 'ortam', etiket)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--onisleme', default='her-ikisi', choices=['cpu', 'gpu', 'her-ikisi'],
                    help='on isleme nerede kossun (merdiven basamagi)')
    ap.add_argument('--hassasiyet', default='fp32', choices=['fp32', 'fp16', 'her-ikisi'],
                    help='ag hassasiyeti (merdiven basamagi)')
    a = ap.parse_args()

    if not DAGITIM.exists():
        sys.exit(str(DAGITIM) + ' yok. Colab notebookundaki "Parca 8 hazirligi" hucresini '
                 'calistirip dagitim/ klasorunu buraya indir.')

    ref_json = json.load(open(DAGITIM / 'olcum_referansi.json', encoding='utf-8'))
    cfg = ref_json['cfg']
    test = ref_json['split_test'][:a.n + ISINMA]
    yollar = veri_yollari()
    # Yereldeki ISLES-2022 kopyasi eksik olabilir (acilma yarim kalmis olabilir).
    # Gecikme olcumu vaka altkumesiyle de gecerli - ama KAC vaka uzerinde olculdugu
    # raporlanmali, sessizce daralan bir orneklem uzerinde konusulmamali.
    tam = [v for v in test if 'dwi' in yollar.get(v, {}) and 'adc' in yollar.get(v, {})]
    eksik = [v for v in test if v not in tam]
    if eksik:
        print('UYARI: {}/{} test vakasinin ham DWI/ADC dosyasi yerelde YOK.'
              .format(len(eksik), len(test)))
        print('       Olcum mevcut {} vaka uzerinde yapilacak.'.format(len(tam)))
        print('       Tam olcum icin ISLES-2022.zip Drive"dan yeniden acilmali.')
        print()
    if len(tam) < 10:
        sys.exit('yerelde yalniz {} vaka var - gecikme olcumu icin yetersiz.'.format(len(tam)))
    test = tam

    print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print('gpu:', pr.name, round(pr.total_memory / 1e9, 1), 'GB')
    else:
        print('!! CUDA yok - yalniz CPU olculecek.')
        print('   RTX 2060 sayilari icin CUDA derlemeli torch gerekli (bkz. PLAN Bolum 6).')


    print('olculecek vaka:', len(test), '(' + str(ISINMA) + ' isinma + ' +
          str(len(test) - ISINMA) + ' olcum)')
    print()

    ortamlar = [('cuda', 'GPU fp32')] if torch.cuda.is_available() else []
    if a.cpu or not torch.cuda.is_available():
        ortamlar.append(('cpu', 'CPU fp32'))

    hepsi = []
    for pt in sorted(DAGITIM.glob('*_cikarim.pt')):
        mk = pt.stem.replace('_cikarim', '')
        for cihaz, etiket in ortamlar:
            model, konfig, secim = md.model_yukle(pt, cihaz)
            onislemeler = (['cpu', 'gpu'] if a.onisleme == 'her-ikisi' else [a.onisleme])
            if cihaz == 'cpu':
                onislemeler = ['cpu']            # GPU on isleme CPU olcumunde anlamsiz
            hassasiyetler = (['fp32', 'fp16'] if a.hassasiyet == 'her-ikisi' else [a.hassasiyet])
            if cihaz == 'cpu':
                hassasiyetler = ['fp32']         # fp16 CPU'da hizlandirmiyor
          
            for oi in onislemeler:
              for hs in hassasiyetler:
                # model.half() YIKICI: agirliklar yerinde fp16'ya cevriliyor ve .float()
                # onlari geri getirmiyor, yalnizca yukari cast ediyor. Ilk surumde ayni
                # nesne fp16'ya cevrilip sonra fp32 diye tekrar kullaniliyordu; fp32
                # olcumleri sessizce fp16-kuantize agirliklarla yapiliyordu. Dice kilidi
                # bunu 0.0047 sapmayla yakaladi. Her hassasiyet icin taze yukleme:
                model_h, konfig, secim = md.model_yukle(pt, cihaz)
                if hs == 'fp16':
                    model_h = model_h.half()
                df = olc(model_h, konfig, secim, cfg, test, yollar,
                         ref_json['referans_dice'][mk], cihaz, etiket,
                         onisleme=oi, hassasiyet=hs)
                df.insert(0, 'hassasiyet', hs)
                df.insert(0, 'onisleme', oi)
                df.insert(0, 'model', mk)
                df['parametre'] = md.parametre_sayisi(model)
                if cihaz == 'cuda':
                    df['vram_gb'] = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                hepsi.append(df)
                del model_h
                if cihaz == 'cuda':
                    torch.cuda.empty_cache()

                sap = df.dice_sapma.max()
                print('{:13s} {:4s} on-isl:{:3s} {:7.1f} ms/vaka  '
                      '(on {:6.1f} | ag {:6.1f} | son {:5.1f})  Dice {:.4f}  sapma {:.5f}'
                      .format(mk, hs, oi, df.toplam_sn.mean() * 1000,
                              df.on_isleme_sn.mean() * 1000, df.ag_sn.mean() * 1000,
                              df.son_isleme_sn.mean() * 1000, df.dice.mean(), sap))
                if np.isfinite(sap) and sap > DICE_TOLERANS:
                    sys.exit('DURDU: ' + mk + '/' + oi + '/' + hs +
                             ' yerel Dice referanstan ' +
                             str(round(sap, 4)) + ' sapiyor (tolerans ' + str(DICE_TOLERANS) +
                             '). Hizi olculen kod kalitesi olculen koddan farkli.')
            del model
            if cihaz == 'cuda':
                torch.cuda.empty_cache()

    ham = pd.concat(hepsi, ignore_index=True)
    ozet = (ham.groupby(['model', 'ortam', 'onisleme', 'hassasiyet'])
            .agg(vaka=('vaka', 'count'),
                 parametre=('parametre', 'first'),
                 okuma_ms=('okuma_sn', lambda s: s.mean() * 1000),
                 on_isleme_ms=('on_isleme_sn', lambda s: s.mean() * 1000),
                 ag_ms=('ag_sn', lambda s: s.mean() * 1000),
                 son_isleme_ms=('son_isleme_sn', lambda s: s.mean() * 1000),
                 toplam_ms=('toplam_sn', lambda s: s.mean() * 1000),
                 toplam_std_ms=('toplam_sn', lambda s: s.std() * 1000),
                 dice=('dice', 'mean'))
            .reset_index())
    ozet['hedef_2sn'] = ozet.toplam_ms < 2000
    ozet['hedef_1sn'] = ozet.toplam_ms < 1000
    ozet['ag_payi_yuzde'] = 100 * ozet.ag_ms / ozet.toplam_ms

    print()
    print('=' * 78)
    print(ozet.to_string(index=False, float_format=lambda v: '{:.2f}'.format(v)))
    with pd.ExcelWriter(KOK / 'hiz_olcumu.xlsx') as w:
        ozet.to_excel(w, sheet_name='ozet', index=False)
        ham.to_excel(w, sheet_name='vaka_basina', index=False)
    print()
    print('yazildi: hiz_olcumu.xlsx')


if __name__ == '__main__':
    main()
