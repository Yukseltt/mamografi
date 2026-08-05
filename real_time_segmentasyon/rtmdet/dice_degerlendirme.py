"""Orijinal cozunurlukte Dice degerlendirmesi - uc modelde ortak metrik.

Neden ayri bir modul:
  1. mmdet'in segm_mAP'i bu kurulumda kullanilamiyor. `Resize(keep_ratio=False)`
     ile x/y olcek katsayilari farkli, RTMDet-Ins'in maske geri-olceklemesi tek
     katsayi uygulayip kare maske uretiyor (ori_shape (473,323) -> maske (323,323)).
     Maske goruntuyle hizalanmadigi icin IoU sifir cikiyor.
  2. Uc mimariyi (SegFormer / YOLO / RTMDet) ayni metrikle karsilastirmak icin
     ortak bir tanima ihtiyacimiz var. PLAN.md karari: degerlendirme **orijinal
     cozunurlukte**, orijinal ikili maskeye karsi.

Kare maske duzeltmesi: model 256x256 uzayinda maske uretiyor, mmdet bunu tek
katsayiyla buyutuyor -- yani cikan kare maske model uzayinin duzgun bir
olceklemesi. Dogrudan (ori_w, ori_h)'ye resize etmek dogru esleme veriyor.

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

KOK = BURASI.parent
DATA_ROOT = KOK / 'Dataset_BUSI_with_GT'
MANIFEST = KOK / 'veri' / 'veri_manifest.csv'
COLAB_ROOT = '/content/drive/MyDrive/real_time_segmentasyon/Dataset_BUSI_with_GT'


def read_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im


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


def degerlendir(config, checkpoint, split='val', esikler=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                cihaz='cuda:0'):
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
    ap.add_argument('--cikti', default=None)
    args = ap.parse_args()

    tablo = degerlendir(args.config, args.checkpoint, args.split)
    print(f'\n=== {args.split} | orijinal cozunurlukte Dice ===', flush=True)
    print(tablo.to_string(index=False), flush=True)

    en_iyi = tablo.loc[tablo['dice_lezyonlu'].idxmax()]
    print(f"\nEn iyi esik: {en_iyi['esik']} -> lezyonlu Dice {en_iyi['dice_lezyonlu']}", flush=True)

    if args.cikti:
        tablo.to_excel(args.cikti, index=False)
        print('kaydedildi:', args.cikti, flush=True)


if __name__ == '__main__':
    main()
