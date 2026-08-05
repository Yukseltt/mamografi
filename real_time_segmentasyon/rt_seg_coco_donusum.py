"""veri_manifest.csv -> COCO instance segmentation JSON (train / val / test).

RTMDet-Ins ve YOLOv11-Seg ikili maske kabul etmiyor, poligon anotasyonu bekliyor.
Bu betik tek kaynak: COCO JSON uretir, YOLO formati daha sonra bundan turetilir ki
iki instance modeli birebir ayni anotasyonu gorsun.

Parametreler PLAN.md'deki olcumlerle secildi:
  - epsilon 1.0 px : Dice 0.9968 (medyan), 44 nokta/goruntu. epsilon=0 kayipsiz ama
                     241 nokta/goruntu; epsilon=2.0'da Dice 0.9936'ya iniyor.
  - min alan 10 px : bunun altindaki bilesenler poligona cevrilemiyor. Tum veri
                     setinde toplam 40 piksel, en kotu goruntude Dice etkisi 0.99977.

Bos maskeli `normal` goruntuler sifir anotasyonlu goruntu olarak dahil edilir --
negatif ornek olarak kalmalari icin sart.

Kullanim:
    .venv_mmdet\\Scripts\\python.exe rt_seg_coco_donusum.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

KOK = Path(r'd:\mamografi\real_time_segmentasyon')
DATA_ROOT = KOK / 'Dataset_BUSI_with_GT'
MANIFEST = KOK / 'veri' / 'veri_manifest.csv'
CIKTI = KOK / 'veri' / 'coco'

COLAB_ROOT = '/content/drive/MyDrive/real_time_segmentasyon/Dataset_BUSI_with_GT'
SPLITS = ['train', 'val', 'test']
EPSILON = 1.0
MIN_ALAN = 10
KATEGORI = [{'id': 1, 'name': 'lezyon', 'supercategory': 'lezyon'}]


def read_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.ndim == 3:
        im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    return im


def yerel_yol(colab_yolu):
    return Path(colab_yolu.replace(COLAB_ROOT, str(DATA_ROOT)))


def maske_birlesimi(mask_paths, shape):
    # EDA ve augmentasyon notebook'lariyla ayni tanim
    out = np.zeros(shape, np.uint8)
    for p in mask_paths.split(';'):
        m = read_gray(yerel_yol(p))
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, (m > 127).astype(np.uint8))
    return out


def bilesen_poligonlari(mask):
    """Her bagli bilesen ayri bir instance. COCO ve YOLO ikisi de boyle bekler."""
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    instanslar = []
    atilan_px = 0
    for k in range(1, n):
        parca = (lab == k).astype(np.uint8)
        alan = int(parca.sum())
        if alan < MIN_ALAN:
            atilan_px += alan
            continue
        cnts, _ = cv2.findContours(parca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            c = cv2.approxPolyDP(c, EPSILON, True)
            if len(c) >= 3:
                polys.append(c.reshape(-1, 2).astype(np.float32))
        if not polys:
            atilan_px += alan
            continue
        instanslar.append((polys, parca))
    return instanslar, atilan_px


def coco_uret(df, split):
    images, annotations = [], []
    ann_id = 1
    istatistik = defaultdict(int)

    for img_id, r in enumerate(df.itertuples(), start=1):
        yol = yerel_yol(r.image_path)
        img = read_gray(yol)
        h, w = img.shape
        if (h, w) != (r.height, r.width):
            raise ValueError(f'{r.case_id}: manifest {r.height}x{r.width}, dosya {h}x{w}')

        images.append({
            'id': img_id,
            'file_name': f'{r.cls}/{yol.name}',   # data_root = Dataset_BUSI_with_GT
            'height': h,
            'width': w,
            'case_id': r.case_id,                 # izlenebilirlik icin, COCO disi alan
        })

        mask = maske_birlesimi(r.mask_paths, img.shape)
        if mask.sum() == 0:
            istatistik['bos_goruntu'] += 1
            continue

        instanslar, atilan = bilesen_poligonlari(mask)
        istatistik['atilan_px'] += atilan
        if not instanslar:
            istatistik['tamamen_kaybolan'] += 1
            continue

        for polys, parca in instanslar:
            ys, xs = np.nonzero(parca)
            x0, y0 = float(xs.min()), float(ys.min())
            bw, bh = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'segmentation': [p.reshape(-1).tolist() for p in polys],
                'area': float(parca.sum()),
                'bbox': [x0, y0, bw, bh],
                'iscrowd': 0,
            })
            ann_id += 1
            istatistik['instance'] += 1

    coco = {
        'info': {'description': f'BUSI lezyon segmentasyonu - {split}',
                 'epsilon_px': EPSILON, 'min_alan_px': MIN_ALAN},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': KATEGORI,
    }
    return coco, istatistik


def dogrula(coco_yolu, df):
    """Uretilen COCO'yu geri rasterize edip orijinal maskelerle karsilastir."""
    from pycocotools.coco import COCO

    coco = COCO(str(coco_yolu))
    lut = {r.case_id: r for r in df.itertuples()}

    dices, bos_dogru, sorunlu = [], 0, []
    for img_id in coco.getImgIds():
        bilgi = coco.loadImgs(img_id)[0]
        r = lut[bilgi['case_id']]
        orijinal = maske_birlesimi(r.mask_paths, (bilgi['height'], bilgi['width']))

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        geri = np.zeros((bilgi['height'], bilgi['width']), np.uint8)
        for a in anns:
            geri = np.maximum(geri, coco.annToMask(a))

        t = orijinal.sum() + geri.sum()
        if t == 0:
            bos_dogru += 1
            continue
        d = 2 * np.logical_and(orijinal, geri).sum() / t
        dices.append(d)
        if d < 0.95:
            sorunlu.append((bilgi['case_id'], round(float(d), 4)))

    return np.array(dices), bos_dogru, sorunlu


def main():
    if not MANIFEST.exists():
        sys.exit(f'manifest bulunamadi: {MANIFEST}')
    CIKTI.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)
    print(f'manifest: {len(df)} satir')

    ozet = []
    for split in SPLITS:
        alt = df[df['split'] == split].reset_index(drop=True)
        coco, ist = coco_uret(alt, split)

        yol = CIKTI / f'busi_{split}.json'
        yol.write_text(json.dumps(coco), encoding='utf-8')

        d, bos_dogru, sorunlu = dogrula(yol, alt)
        ozet.append(dict(
            split=split,
            goruntu=len(coco['images']),
            instance=len(coco['annotations']),
            bos_goruntu=ist['bos_goruntu'],
            coklu_instance=sum(1 for i in coco['images']
                               if sum(1 for a in coco['annotations'] if a['image_id'] == i['id']) > 1),
            dice_medyan=round(float(np.median(d)), 5) if len(d) else None,
            dice_min=round(float(d.min()), 5) if len(d) else None,
            atilan_px=ist['atilan_px'],
        ))
        print(f'\n{split}: {yol.name}  ({yol.stat().st_size / 1e6:.1f} MB)')
        print(f'  goruntu {len(coco["images"])}, instance {len(coco["annotations"])}, '
              f'bos {ist["bos_goruntu"]}, dogrulanan bos {bos_dogru}')
        print(f'  geri-rasterize Dice: medyan {np.median(d):.5f}  min {d.min():.5f}')
        if sorunlu:
            print(f'  DIKKAT Dice<0.95 olan {len(sorunlu)}:', sorunlu[:5])

    tablo = pd.DataFrame(ozet)
    print('\n=== OZET ===')
    print(tablo.to_string(index=False))
    tablo.to_excel(CIKTI / 'coco_donusum_ozeti.xlsx', index=False)

    # sert kontroller
    assert tablo['goruntu'].sum() == len(df), 'goruntu sayisi manifest ile uyusmuyor'
    assert tablo['bos_goruntu'].sum() == int((df['lesion_px'] == 0).sum()), 'bos goruntu sayisi tutmuyor'

    # Esik 0.985: olculen deger ~0.990. Poligonun kendisi daha sadik (cv2 ile
    # rasterize edildiginde 0.9968), aradaki fark pycocotools'un tarama
    # konvansiyonundan geliyor -- poligonu yarim piksel daraltiyor. Egitimde mmdet
    # de ayni konvansiyonu kullandigi icin dogrulamayi bilerek pycocotools ile
    # yapiyoruz: modelin gercekte gorecegi GT bu. Sabit offset ile telafi denendi,
    # kapatmadi (+0.5 offset 0.9883 -> 0.9903).
    assert (tablo['dice_medyan'] > 0.985).all(), 'geri-rasterize Dice beklenenin altinda'
    assert (tablo['dice_min'] > 0.90).all(), 'tek bir goruntude kabul edilemez kayip'
    print('\nTum kontroller gecti.')


if __name__ == '__main__':
    main()
