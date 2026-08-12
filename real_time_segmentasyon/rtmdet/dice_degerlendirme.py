"""Orijinal cozunurlukte Dice degerlendirmesi - uc modelde ortak metrik.

Neden ayri bir modul:
  1. Uc mimariyi (SegFormer / YOLO / RTMDet) ayni metrikle karsilastirmak icin
     ortak bir tanima ihtiyacimiz var; SegFormer semantic segmentation yaptigi
     icin segm_mAP onun icin tanimli bile degil. PLAN.md karari: degerlendirme
     **orijinal cozunurlukte**, orijinal ikili maskeye karsi.
  2. segm_mAP bu kurulumda kirilgan oldugunu gosterdi: RTMDet-Ins maskeyi
     ori_shape'ten 1 px kisa uretebiliyor ve COCOeval o goruntuyu sifir sayiyor
     (bkz. ozel_transformlar.CocoMetricHizali). Ayni hata daha once
     `keep_ratio=False` doneminde kare maske olarak cok daha buyuk olcekte
     yasanmisti. Bu modul maskeyi her durumda orijinal boyuta getirdigi icin
     ikisinden de etkilenmiyor.

Maske boyutu duzeltmesi: model kendi girdi uzayinda maske uretip buyutuyor,
cikan maske ori_shape'ten farkli olabiliyor. Dogrudan (ori_w, ori_h)'ye nearest
resize etmek dogru esleme veriyor.

Bos maskeli goruntuler (131 `normal` vaka): GT bos ve tahmin de bossa Dice 1
sayilir (0/0 tanimsiz). Rapor bu vakalari ayri kirilimda verir -- tek bir
ortalama, 131 bos goruntunun etkisiyle yaniltici olurdu.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI))

import torch_uyum        # noqa: F401  torch 2.6 checkpoint uyumu
import ozel_transformlar  # noqa: F401  transform kaydi

sys.path.insert(0, str(BURASI.parent / 'ortak'))
import degerlendirme as ortak_dg   # esik izgarasi ve secim olcutu tek yerden

ESIKLER = ortak_dg.ESIKLER
ORTAK_OLCUT = 'dice_tum'

KOK = BURASI.parent
DATA_ROOT = KOK / 'Dataset_BUSI_with_GT'
MANIFEST = KOK / 'veri' / 'veri_manifest.csv'
COLAB_ROOT = '/content/drive/MyDrive/real_time_segmentasyon/Dataset_BUSI_with_GT'


read_gray = ortak_dg.gri_oku   # tek tanim; kanal duzeni varsayimi orada ele aliniyor


def yerel_yol(p):
    return Path(str(p).replace(COLAB_ROOT, str(DATA_ROOT)))


def gt_maskesi(mask_paths, shape):
    """EDA / augmentasyon / COCO donusumu ile ayni birlesim tanimi."""
    out = np.zeros(shape, np.uint8)
    for p in str(mask_paths).split(';'):
        m = read_gray(yerel_yol(p))
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, (m > 127).astype(np.uint8))
    return out


def metrikler(tahmin, gt):
    """Goruntu basina TP/FP/FN uzerinden. Binary'de f1 = dice (ozdes)."""
    tp = float(np.logical_and(tahmin, gt).sum())
    fp = float(np.logical_and(tahmin, 1 - gt).sum())
    fn = float(np.logical_and(1 - tahmin, gt).sum())
    if tp + fp + fn == 0:                      # GT bos ve tahmin de bos
        return dict(dice=1.0, iou=1.0, precision=1.0, recall=1.0)
    return dict(
        dice=2 * tp / (2 * tp + fp + fn),
        iou=tp / (tp + fp + fn),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
    )


def tahmin_maskesi(pred, ori_shape, esik):
    """Skoru esigi gecen instance maskelerini birlestir, orijinal boyuta olcekle."""
    h, w = ori_shape
    out = np.zeros((h, w), np.uint8)
    if len(pred) == 0:
        return out
    skor = pred.scores.cpu().numpy()
    sec = skor >= esik
    if not sec.any():
        return out
    m = pred.masks.cpu().numpy()[sec]
    for tek in m:
        # kare maske duzeltmesi: model uzayindan orijinale dogrudan olcekleme
        if tek.shape != (h, w):
            tek = cv2.resize(tek.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, tek.astype(np.uint8))
    return out


def degerlendir(config, checkpoint, split='val', esikler=ESIKLER, cihaz='cuda:0'):
    from mmdet.apis import init_detector, inference_detector

    df = pd.read_csv(MANIFEST)
    alt = df[df['split'] == split].reset_index(drop=True)
    model = init_detector(str(config), str(checkpoint), device=cihaz)

    # her goruntude bir kez cikarim, esikler ayni tahmin uzerinden taranir
    tahminler, gtler = [], []
    for r in alt.itertuples():
        yol = yerel_yol(r.image_path)
        img = read_gray(yol)
        gtler.append(gt_maskesi(r.mask_paths, img.shape))
        tahminler.append(inference_detector(model, str(yol)).pred_instances)

    satirlar = []
    for esik in esikler:
        kayit = []
        for r, pred, gt in zip(alt.itertuples(), tahminler, gtler):
            m = metrikler(tahmin_maskesi(pred, gt.shape, esik), gt)
            m['bos_gt'] = r.lesion_px == 0
            kayit.append(m)
        k = pd.DataFrame(kayit)
        lezyonlu, bos = k[~k['bos_gt']], k[k['bos_gt']]
        satirlar.append(dict(
            esik=esik,
            dice_tum=round(float(k['dice'].mean()), 4),
            dice_lezyonlu=round(float(lezyonlu['dice'].mean()), 4),
            iou_lezyonlu=round(float(lezyonlu['iou'].mean()), 4),
            precision_lezyonlu=round(float(lezyonlu['precision'].mean()), 4),
            recall_lezyonlu=round(float(lezyonlu['recall'].mean()), 4),
            bos_dogru=int((bos['dice'] == 1.0).sum()),
            bos_toplam=len(bos),
        ))
    return pd.DataFrame(satirlar)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(BURASI / 'rtmdet_ins_busi.py'))
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--esik', type=float, default=None,
                    help='verilirse yalnizca bu esikte olculur. Test icin sart: '
                         'esik val`de secilir, test`e sabit uygulanir')
    ap.add_argument('--cikti', default=None)
    args = ap.parse_args()

    if args.split == 'test' and args.esik is None:
        raise SystemExit(
            'test icin --esik zorunlu. Esik val`de secilmeli ve test`e sabit\n'
            'uygulanmali; test uzerinde tarama yapip en iyisini raporlamak\n'
            'sizintidir (PLAN.md degerlendirme protokolu).')

    esikler = (args.esik,) if args.esik else ortak_dg.ESIKLER
    tablo = degerlendir(args.config, args.checkpoint, args.split, esikler)
    print(f'\n=== {args.split} | orijinal cozunurlukte Dice ===', flush=True)
    print(tablo.to_string(index=False), flush=True)

    if args.esik is None:
        # Esik olcutu uc modelde ortak olmali; tanim ortak/degerlendirme.py'de
        esik = ortak_dg.en_iyi_esik(tablo)
        satir = tablo.loc[tablo['esik'] == esik].iloc[0]
        print(f"\nEn iyi esik ({ORTAK_OLCUT}): {esik} -> "
              f"lezyonlu Dice {satir['dice_lezyonlu']}, tum {satir['dice_tum']}, "
              f"bos dogru {satir['bos_dogru']}/{satir['bos_toplam']}", flush=True)

    if args.cikti:
        tablo.to_excel(args.cikti, index=False)
        print('kaydedildi:', args.cikti, flush=True)


if __name__ == '__main__':
    main()
