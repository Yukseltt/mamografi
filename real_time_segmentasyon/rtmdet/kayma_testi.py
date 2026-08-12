"""Egitilmis modelin tahmini GT'ye gore sistematik kaymis mi?

Yarim piksel yamasi (ResizeHassas) su zinciri varsayiyordu: maske nearest ile
kucultulurken +0.5 px kayiyor -> model kaymis hedefi ogreniyor -> tahmin orijinal
cozunurlukte ~1.1 px kaymis cikiyor -> Dice dusuyor.

Yamali kosu Dice'ta daha kotu ciktigi icin bu zincirin gercekten isleyip
islemedigi dogrudan olculuyor: eslesen instance ciftlerinde tahmin merkezi ile GT
merkezi arasindaki fark. Zincir gecerliyse yamasiz kosuda medyan kayma ~+1 px
(her iki eksende, ayni isaretli), yamalida ~0 olmali.

Kullanim:
    python kayma_testi.py --kosular <work_dir_adi> <work_dir_adi> ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI))

import torch_uyum        # noqa: F401
import ozel_transformlar  # noqa: F401
from dice_degerlendirme import MANIFEST, gt_maskesi, read_gray, yerel_yol
from metrik_ayrismasi import kosu_bul

ESIK = 0.2


def merkez(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def olc(config, ckpt, alt, cihaz='cuda:0'):
    from mmdet.apis import inference_detector, init_detector
    import cv2

    model = init_detector(str(config), str(ckpt), device=cihaz)
    farklar = []
    for r in alt.itertuples():
        if r.lesion_px == 0 or r.n_components != 1:
            continue                      # tek lezyonlu vakalar: esleme belirsizligi yok
        yol = yerel_yol(r.image_path)
        gt = gt_maskesi(r.mask_paths, read_gray(yol).shape)
        pred = inference_detector(model, str(yol)).pred_instances
        skor = pred.scores.cpu().numpy()
        if not (skor >= ESIK).any():
            continue
        m = pred.masks.cpu().numpy()[int(np.argmax(skor))].astype(np.uint8)
        if m.shape != gt.shape:
            m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        if m.sum() == 0:
            continue
        # yalnizca iyi ortusen ciftler: kacirilmis tespit kaymayi olcmez
        if np.logical_and(m, gt).sum() / np.logical_or(m, gt).sum() < 0.5:
            continue
        farklar.append(merkez(m) - merkez(gt))
    del model
    return np.array(farklar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kosular', nargs='+', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--cihaz', default='cuda:0')
    args = ap.parse_args()

    alt = pd.read_csv(MANIFEST).query('split == @args.split').reset_index(drop=True)
    satir = []
    for ad in args.kosular:
        config, ckpt = kosu_bul(ad)
        d = olc(config, ckpt, alt, args.cihaz)
        satir.append(dict(
            kosu=ad, n=len(d),
            dx_medyan=round(float(np.median(d[:, 0])), 3),
            dy_medyan=round(float(np.median(d[:, 1])), 3),
            dx_ort=round(float(d[:, 0].mean()), 3),
            dy_ort=round(float(d[:, 1].mean()), 3),
            norm_medyan=round(float(np.median(np.hypot(*d.T))), 3),
        ))
        print(f'{ad}: {len(d)} cift olculdu', flush=True)

    tablo = pd.DataFrame(satir)
    print('\n=== tahmin merkezi - GT merkezi (orijinal cozunurlukte px) ===')
    print(tablo.to_string(index=False))
    print('\nYamanin dayandigi zincir gecerliyse yamasiz kosuda dx/dy medyani')
    print('~+1 px ve ayni isaretli, yamalida ~0 olmali.')


if __name__ == '__main__':
    main()
