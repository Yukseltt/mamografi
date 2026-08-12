"""RTMDet-Ins icin sabit degerlendirme vakasi gorselleri + hizalama kontrolu.

Vakalar `ortak/eval_cases.json`'dan geliyor, yani SegFormer ve YOLO ile birebir
ayni goruntuler. Her model kendi iyi ornegini secerse gorsel karsilastirma
hicbir sey soylemez.

Hizalama kontrolu de burada calisiyor: diger iki model Colab notebook'unda
`dg.kaydirma_taramasi` ile olculdu, RTMDet ayni fonksiyondan gecsin ki uc sonuc
ayni koddan gelsin.

Kullanim:
    python vaka_gorselleri.py --kosu rtmdet_ins_tiny_256_orta --esik 0.4
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent / 'ortak'))

import torch_uyum        # noqa: F401  torch 2.6 checkpoint uyumu
import ozel_transformlar  # noqa: F401  transform kaydi
import degerlendirme as dg
from metrik_ayrismasi import kosu_bul

MIN_SKOR = 0.005


def harita_fn_kur(config, ckpt, cihaz):
    """Model -> ortak protokolun bekledigi skor haritasi fonksiyonu."""
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(str(config), str(ckpt), device=cihaz)

    def harita_fn(satir):
        yol = str(dg.yol_coz(satir.image_path))
        pred = inference_detector(model, yol).pred_instances
        skor = pred.scores.cpu().numpy()
        sec = skor >= MIN_SKOR
        sekil = (int(satir.height), int(satir.width))
        if not sec.any():
            return np.zeros(sekil, np.float32)
        return dg.instans_haritasi(pred.masks.cpu().numpy()[sec], skor[sec], sekil)

    return harita_fn


def cizim(vakalar, lut, harita_fn, esik, cikti):
    # Suptitle yok: SegFormer ve YOLO notebook'larindaki figurlerde de yok, uc
    # gorsel rapora yan yana konulacak. Model adi dosya adinda zaten var.
    fig, ax = plt.subplots(len(vakalar), 3, figsize=(11, 3.4 * len(vakalar)))
    for i, v in enumerate(vakalar):
        r = lut[v['case_id']]
        img = dg.gri_oku(dg.yol_coz(r.image_path))
        gt = dg.gt_maskesi(r)
        pred = harita_fn(r) >= esik
        d = dg.metrikler(pred, gt)['dice']

        for j, (katman, alt) in enumerate([(img, 'orijinal'), (gt, 'ground truth'),
                                           (pred, f'tahmin  Dice {d:.3f}')]):
            ax[i, j].imshow(img, cmap='gray')
            if j:
                ax[i, j].imshow(np.ma.masked_where(katman == 0, katman),
                                cmap='autumn', alpha=0.45)
            ax[i, j].set_title(f"{v['case_id']}  [{v['kova']}]\n{alt}", fontsize=9)
            ax[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(cikti, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kosu', default='rtmdet_ins_tiny_256_orta')
    ap.add_argument('--esik', type=float, required=True,
                    help='val`de secilen esik (256 -> 0.4, 512 -> 0.2)')
    ap.add_argument('--cihaz', default='cuda:0')
    args = ap.parse_args()

    config, ckpt = kosu_bul(args.kosu)
    print(f'{args.kosu}: {ckpt.name} | esik {args.esik}', flush=True)
    harita_fn = harita_fn_kur(config, ckpt, args.cihaz)

    vakalar = json.loads((BURASI.parent / 'ortak' / 'eval_cases.json')
                         .read_text(encoding='utf-8'))['vakalar']
    lut = {r.case_id: r for r in dg.manifest_oku().itertuples()}

    work = BURASI / 'calisma' / args.kosu
    png = work / f'{args.kosu}_vakalar.png'
    cizim(vakalar, lut, harita_fn, args.esik, png)
    print('gorseller:', png, flush=True)

    # Hizalama: diger iki modelle ayni fonksiyon, ayni yorum
    tablo = dg.kaydirma_taramasi(harita_fn, 'val', esik=args.esik)
    print('\n=== hizalama taramasi (val, lezyonlu) ===')
    print(tablo.to_string())
    for k, v in dg.hizalama_ozeti(tablo).items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
