"""Uc mimarinin goruntu basina eslestirilmis karsilastirmasi.

Neden gerekli: test'te SegFormer bir sonrakini +0.026 geciyor, ama RTMDet 256<->512
icin olctugumuz eslestirilmis gurultu bandi +-0.03. Ortalamalara bakarak siralama
yapmak bu farkta anlamsiz; ayni goruntuler uzerinde eslestirilmis test gerekiyor.

Betik once **Colab sayilarini yerelde yeniden uretiyor**. Tutmuyorsa cikarim yolu
ayrismis demektir (farkli on isleme, farkli esik, yanlis checkpoint) ve devam
etmenin anlami yok -- bu yuzden sapma esigi assa hata veriyor.

Kullanim:
    python uc_model_karsilastirma.py                 # dogrulama + test karsilastirmasi
    python uc_model_karsilastirma.py --split val
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'rtmdet'))

import torch_uyum        # noqa: F401  torch 2.6 checkpoint uyumu
import ozel_transformlar  # noqa: F401  mmdet transform/metrik kaydi
import degerlendirme as dg
import modeller as md

# Her modelin val'de secilmis esigi. Test'e sabit uygulanir (PLAN.md protokolu).
MODELLER = {
    'SegFormer-B0': dict(tur='segformer', ckpt=KOK / 'segformer_best.pt', esik=0.4,
                         val_beklenen=0.8195, test_beklenen=0.7365),
    'RTMDet-Ins 256': dict(tur='rtmdet', esik=0.4, kosu='rtmdet_ins_tiny_256_orta',
                           val_beklenen=0.8084, test_beklenen=0.6998),
    'RTMDet-Ins 512': dict(tur='rtmdet', esik=0.2, kosu='rtmdet_ins_tiny_512_orta',
                           val_beklenen=0.8109, test_beklenen=0.7101),
    'YOLOv11n-Seg': dict(tur='yolo', ckpt=KOK / 'yolo11n_seg_best.pt', esik=0.2,
                         val_beklenen=0.7697, test_beklenen=0.6936),
}
SAPMA_ESIGI = 0.005      # yeniden uretimde kabul edilen fark


def harita_fn_kur(cfg, cihaz):
    if cfg['tur'] == 'segformer':
        return md.segformer_kur(cfg['ckpt'], cihaz=cihaz)[1]
    if cfg['tur'] == 'yolo':
        return md.yolo_kur(cfg['ckpt'], cihaz=0 if cihaz.startswith('cuda') else 'cpu')[1]
    from metrik_ayrismasi import kosu_bul
    config, ckpt = kosu_bul(cfg['kosu'])
    return md.rtmdet_kur(config, ckpt, cihaz=cihaz)[1]


def eslestirilmis(a, b, ad_a, ad_b, tohum=42, n=10000):
    """Ayni goruntulerde fark: bootstrap GA + Wilcoxon + isaret sayimi."""
    from scipy.stats import wilcoxon
    fark = a - b
    rng = np.random.default_rng(tohum)
    boot = fark[rng.integers(0, len(fark), size=(n, len(fark)))].mean(axis=1)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return dict(
        a=ad_a, b=ad_b, ort_a=round(float(a.mean()), 4), ort_b=round(float(b.mean()), 4),
        fark=round(float(fark.mean()), 4),
        ga_alt=round(float(np.percentile(boot, 2.5)), 4),
        ga_ust=round(float(np.percentile(boot, 97.5)), 4),
        wilcoxon_p=round(p, 4),
        a_iyi=int((fark > 1e-9).sum()), b_iyi=int((fark < -1e-9).sum()),
        anlamli='EVET' if (p < 0.05 and np.sign(np.percentile(boot, 2.5))
                           == np.sign(np.percentile(boot, 97.5))) else 'hayir')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--cihaz', default='cuda:0')
    ap.add_argument('--cikti', default=str(KOK / 'uc_model_karsilastirma.xlsx'))
    ap.add_argument('--dogrulamayi-atla', action='store_true')
    args = ap.parse_args()

    beklenen_anahtar = f'{args.split}_beklenen'
    goruntu, ozet = {}, []

    for ad, cfg in MODELLER.items():
        print(f'\n[{ad}] esik {cfg["esik"]}', flush=True)
        harita_fn = harita_fn_kur(cfg, args.cihaz)
        gb = dg.goruntu_basina(harita_fn, args.split, esikler=(cfg['esik'],))
        goruntu[ad] = gb

        lez = gb[~gb.bos_gt]
        bos = gb[gb.bos_gt]
        olculen = float(lez.dice.mean())
        beklenen = cfg[beklenen_anahtar]
        sapma = abs(olculen - beklenen)
        print(f'  lezyonlu Dice {olculen:.4f} (beklenen {beklenen:.4f}, '
              f'sapma {sapma:.4f})', flush=True)
        if not args.dogrulamayi_atla and sapma > SAPMA_ESIGI:
            raise SystemExit(
                f'{ad}: yerel olcum Colab sonucundan {sapma:.4f} sapiyor '
                f'(esik {SAPMA_ESIGI}). Cikarim yolu ayrismis olabilir -- on isleme,\n'
                'esik veya checkpoint kontrol edilmeli. Karsilastirmaya devam etmek\n'
                'yanlis sonuc uretir.')

        ozet.append(dict(model=ad, esik=cfg['esik'], n_lezyonlu=len(lez),
                         dice_lezyonlu=round(olculen, 4),
                         dice_tum=round(float(gb.dice.mean()), 4),
                         iou_lezyonlu=round(float(lez.iou.mean()), 4),
                         precision=round(float(lez.precision.mean()), 4),
                         recall=round(float(lez.recall.mean()), 4),
                         bos_dogru=int((bos.dice == 1.0).sum()), bos_n=len(bos),
                         beklenen=beklenen, sapma=round(sapma, 4)))

    ozet = pd.DataFrame(ozet).sort_values('dice_lezyonlu', ascending=False)

    # eslestirilmis testler: yalnizca lezyonlu goruntuler, ayni sirada
    adlar = list(goruntu)
    ref = goruntu[adlar[0]]
    lez_maske = ~ref.bos_gt.values
    testler = []
    for x, y in itertools.combinations(adlar, 2):
        gx, gy = goruntu[x], goruntu[y]
        assert (gx.case_id.values == gy.case_id.values).all(), 'goruntu sirasi ayrisiyor'
        testler.append(eslestirilmis(gx.loc[lez_maske, 'dice'].values,
                                     gy.loc[lez_maske, 'dice'].values, x, y))
    testler = pd.DataFrame(testler)

    print(f'\n=== {args.split} | ozet ===')
    print(ozet.to_string(index=False))
    print(f'\n=== {args.split} | eslestirilmis fark (lezyonlu, n={int(lez_maske.sum())}) ===')
    print(testler.to_string(index=False))
    anlamli = testler[testler.anlamli == 'EVET']
    print(f'\nistatistiksel olarak ayrisan cift: {len(anlamli)}/{len(testler)}')

    with pd.ExcelWriter(args.cikti) as w:
        ozet.to_excel(w, sheet_name='ozet', index=False)
        testler.to_excel(w, sheet_name='eslestirilmis', index=False)
        for ad, gb in goruntu.items():
            gb.to_excel(w, sheet_name=ad.replace('-', '_')[:31], index=False)
    print('kaydedildi:', args.cikti)


if __name__ == '__main__':
    main()
