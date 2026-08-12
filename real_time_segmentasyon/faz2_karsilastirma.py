"""Faz 2: uc mimari x uc tohum -- tohum varyansini ayirarak model karsilastirmasi.

Faz 1'deki uc_model_karsilastirma.py tek kosu varsayiyordu. Uc tohumla ogrendik ki
mimariler arasi farklarin bir kismi tohum gurultusu (RTMDet'in kendi std'si 0.0127).
Bu betik ikisini ayiriyor.

Esikler burada hardcode edilmiyor: her kosu icin val'de bastan taranip seciliyor
(olcut dice_tum, dg.en_iyi_esik sinirda optimumda hata veriyor), sonra test'e sabit
uygulaniyor. Boylece dokuz kosunun hepsi ayni protokolden geciyor.

Kullanim:
    python faz2_karsilastirma.py
    python faz2_karsilastirma.py --dogrulamayi-atla
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'rtmdet'))

import torch_uyum         # noqa: F401  torch 2.6 checkpoint uyumu
import ozel_transformlar  # noqa: F401  mmdet transform/metrik kaydi
import ozel_metrikler     # noqa: F401  BusiDiceMetric / ValLossHook kaydi
import degerlendirme as dg
import modeller as md

TOHUMLAR = (42, 43, 44)
CALISMA = KOK / 'rtmdet' / 'calisma'
CKPT = KOK / 'faz2_ckpt'

# Notebook/log'larda kaydedilen val dice_tum. Yerel olcum bunlardan saparsa
# cikarim yolu ayrismis demektir (yanlis checkpoint, farkli on isleme).
BEKLENEN_VAL = {
    ('RTMDet-Ins', 42): 0.8290, ('RTMDet-Ins', 43): 0.8538, ('RTMDet-Ins', 44): 0.8366,
    ('SegFormer-B0', 42): 0.8275, ('SegFormer-B0', 43): 0.8361, ('SegFormer-B0', 44): 0.8235,
    ('YOLOv11n-Seg', 42): 0.7985, ('YOLOv11n-Seg', 43): 0.8269, ('YOLOv11n-Seg', 44): 0.8136,
}
SAPMA_ESIGI = 0.005

MIMARILER = ('RTMDet-Ins', 'SegFormer-B0', 'YOLOv11n-Seg')


def rtmdet_kosu(tohum):
    """Faz 2 work_dir'i: checkpoint artik busi/dice_tum ile seciliyor."""
    work = CALISMA / f'rtmdet_ins_tiny_256_orta_s{tohum}'
    adaylar = list(work.glob('best_busi_dice_tum_epoch_*.pth'))
    if not adaylar:
        raise FileNotFoundError(f'best_busi_dice_tum checkpointi yok: {work}')
    ckpt = max(adaylar, key=lambda p: int(re.search(r'epoch_(\d+)', p.name).group(1)))
    return work / 'rtmdet_ins_busi.py', ckpt


def harita_fn_kur(mimari, tohum, cihaz):
    if mimari == 'SegFormer-B0':
        return md.segformer_kur(CKPT / f'segformer_s{tohum}.pt', cihaz=cihaz)[1]
    if mimari == 'YOLOv11n-Seg':
        return md.yolo_kur(CKPT / f'yolo_s{tohum}.pt',
                           cihaz=0 if cihaz.startswith('cuda') else 'cpu')[1]
    return md.rtmdet_kur(*rtmdet_kosu(tohum), cihaz=cihaz)[1]


def eslestirilmis(a, b, ad_a, ad_b, tohum=42, n=10000):
    """Ayni goruntulerde fark: bootstrap GA + Wilcoxon + isaret sayimi."""
    from scipy.stats import wilcoxon
    fark = a - b
    rng = np.random.default_rng(tohum)
    boot = fark[rng.integers(0, len(fark), size=(n, len(fark)))].mean(axis=1)
    alt, ust = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return dict(
        a=ad_a, b=ad_b, ort_a=round(float(a.mean()), 4), ort_b=round(float(b.mean()), 4),
        fark=round(float(fark.mean()), 4),
        ga_alt=round(float(alt), 4), ga_ust=round(float(ust), 4),
        wilcoxon_p=round(p, 4),
        a_iyi=int((fark > 1e-9).sum()), b_iyi=int((fark < -1e-9).sum()),
        anlamli='EVET' if (p < 0.05 and np.sign(alt) == np.sign(ust)) else 'hayir')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cihaz', default='cuda:0')
    ap.add_argument('--cikti', default=str(KOK / 'faz2_karsilastirma.xlsx'))
    ap.add_argument('--dogrulamayi-atla', action='store_true')
    args = ap.parse_args()

    test_gb, val_ozet, test_ozet = {}, [], []

    for mimari in MIMARILER:
        for tohum in TOHUMLAR:
            etiket = f'{mimari} s{tohum}'
            print(f'\n[{etiket}]', flush=True)
            harita_fn = harita_fn_kur(mimari, tohum, args.cihaz)

            # --- esik val'de seciliyor ---
            gb_val = dg.goruntu_basina(harita_fn, 'val')
            tablo = dg.esik_tablosu(gb_val)
            esik = dg.en_iyi_esik(tablo, olcut='dice_tum')
            satir = tablo.loc[tablo.esik == esik].iloc[0]
            olculen = float(satir.dice_tum)
            beklenen = BEKLENEN_VAL[(mimari, tohum)]
            sapma = abs(olculen - beklenen)
            print(f'  val esik {esik}  dice_tum {olculen:.4f} '
                  f'(beklenen {beklenen:.4f}, sapma {sapma:.4f})', flush=True)
            if not args.dogrulamayi_atla and sapma > SAPMA_ESIGI:
                raise SystemExit(
                    f'{etiket}: yerel val olcumu kayitli degerden {sapma:.4f} sapiyor '
                    f'(esik {SAPMA_ESIGI}).\nYanlis checkpoint veya ayrismis on isleme '
                    'olabilir; karsilastirmaya devam etmek yanlis sonuc uretir.')
            val_ozet.append(dict(mimari=mimari, tohum=tohum, esik=esik,
                                 dice_tum=round(olculen, 4),
                                 dice_lezyonlu=round(float(satir.dice_lezyonlu), 4),
                                 beklenen=beklenen, sapma=round(sapma, 4)))

            # --- esik sabit, test ---
            gb = dg.goruntu_basina(harita_fn, 'test', esikler=(esik,))
            test_gb[(mimari, tohum)] = gb
            lez, bos = gb[~gb.bos_gt], gb[gb.bos_gt]
            test_ozet.append(dict(
                mimari=mimari, tohum=tohum, esik=esik,
                dice_tum=round(float(gb.dice.mean()), 4),
                dice_lezyonlu=round(float(lez.dice.mean()), 4),
                iou_lezyonlu=round(float(lez.iou.mean()), 4),
                precision=round(float(lez.precision.mean()), 4),
                recall=round(float(lez.recall.mean()), 4),
                bos_dogru=int((bos.dice == 1.0).sum()), bos_n=len(bos)))
            print(f'  test dice_tum {test_ozet[-1]["dice_tum"]:.4f}  '
                  f'lezyonlu {test_ozet[-1]["dice_lezyonlu"]:.4f}', flush=True)

    val_ozet = pd.DataFrame(val_ozet)
    test_ozet = pd.DataFrame(test_ozet)

    # --- tohumlar arasi varyans ---
    varyans = (test_ozet.groupby('mimari')[['dice_tum', 'dice_lezyonlu']]
               .agg(['mean', 'std']).round(4))
    varyans.columns = ['_'.join(c) for c in varyans.columns]
    varyans = varyans.reindex(MIMARILER).reset_index()

    # --- eslestirilmis testler ---
    ref = test_gb[(MIMARILER[0], TOHUMLAR[0])]
    lez_maske = ~ref.bos_gt.values
    for gb in test_gb.values():
        assert (gb.case_id.values == ref.case_id.values).all(), 'goruntu sirasi ayrisiyor'

    def dice_vek(mimari, tohum):
        return test_gb[(mimari, tohum)].loc[lez_maske, 'dice'].values

    # (a) tohum-ortalamasi: her goruntude uc tohumun ortalamasi alinip mimariler
    #     karsilastiriliyor. Tohum gurultusunu bastiriyor, mimari sinyalini birakiyor.
    ortalama = {m: np.mean([dice_vek(m, t) for t in TOHUMLAR], axis=0) for m in MIMARILER}
    tohum_ort = pd.DataFrame([eslestirilmis(ortalama[x], ortalama[y], x, y)
                              for x, y in itertools.combinations(MIMARILER, 2)])

    # (b) dokuz capraz tohum cifti: sonuc tohum secimine bagli mi?
    capraz = []
    for x, y in itertools.combinations(MIMARILER, 2):
        satirlar = [eslestirilmis(dice_vek(x, tx), dice_vek(y, ty),
                                  f'{x} s{tx}', f'{y} s{ty}')
                    for tx in TOHUMLAR for ty in TOHUMLAR]
        d = pd.DataFrame(satirlar)
        capraz.append(dict(
            a=x, b=y, n_cift=len(d),
            fark_min=round(d.fark.min(), 4), fark_ort=round(d.fark.mean(), 4),
            fark_max=round(d.fark.max(), 4),
            anlamli=int((d.anlamli == 'EVET').sum()),
            a_hep_ustte='EVET' if (d.fark > 0).all() else 'hayir'))
    capraz = pd.DataFrame(capraz)

    # --- boyut kirilimi: tek kosuda gorulen 140-220 cukuru dokuz kosuda duruyor mu ---
    kirilim = []
    for (mimari, tohum), gb in test_gb.items():
        bk = dg.boyut_kirilimi(gb, gb.esik.iloc[0])
        for r in bk.itertuples():
            kirilim.append(dict(mimari=mimari, tohum=tohum, kova=r.kova,
                                n=r.n, dice=r.dice, iou=r.iou))
    kirilim = pd.DataFrame(kirilim)
    kirilim_ozet = (kirilim.pivot_table(index='kova', columns='mimari', values='dice',
                                        aggfunc=['mean', 'std'], observed=True)
                    .round(4))
    kirilim_ozet.columns = ['_'.join(c) for c in kirilim_ozet.columns]

    print('\n=== val | esik secimi ve dogrulama ===')
    print(val_ozet.to_string(index=False))
    print('\n=== test | kosu basina ===')
    print(test_ozet.to_string(index=False))
    print('\n=== test | tohumlar arasi ===')
    print(varyans.to_string(index=False))
    print(f'\n=== test | tohum-ortalamasi eslestirilmis '
          f'(lezyonlu, n={int(lez_maske.sum())}) ===')
    print(tohum_ort.to_string(index=False))
    print('\n=== test | dokuz capraz tohum cifti ===')
    print(capraz.to_string(index=False))
    print('\n=== test | lezyon boyutu kirilimi (3 tohum ort/std) ===')
    print(kirilim_ozet.to_string())

    with pd.ExcelWriter(args.cikti) as w:
        val_ozet.to_excel(w, sheet_name='val_esik', index=False)
        test_ozet.to_excel(w, sheet_name='test_kosu', index=False)
        varyans.to_excel(w, sheet_name='tohum_varyansi', index=False)
        tohum_ort.to_excel(w, sheet_name='eslestirilmis_ort', index=False)
        capraz.to_excel(w, sheet_name='capraz_tohum', index=False)
        kirilim.to_excel(w, sheet_name='boyut_kirilimi', index=False)
        kirilim_ozet.to_excel(w, sheet_name='boyut_ozet')
        # goruntu basina Dice: rapor figurleri ve sonraki analizler icin
        for (mimari, tohum), gb in test_gb.items():
            gb.to_excel(w, index=False,
                        sheet_name=f'{mimari.split("-")[0]}_s{tohum}'[:31])
    print('\nkaydedildi:', args.cikti)


if __name__ == '__main__':
    main()
