"""COCO instance segmentation JSON -> Ultralytics YOLO-seg formati.

Kaynak bilerek COCO JSON, orijinal maskeler degil: RTMDet-Ins ile YOLOv11-Seg
**birebir ayni anotasyonu** gormeli, yoksa karsilastirma mimariyi degil poligon
cikarma farkini olcer. Poligon parametreleri (epsilon 1.0 px, min alan 10 px)
rt_seg_coco_donusum.py'de secildi ve orada dogrulandi.

Uretilen yapi (Ultralytics etiket yolunu `/images/` -> `/labels/` ile buluyor):

    veri/yolo/
      data.yaml
      labels/{train,val,test}/<stem>.txt
      dosya_haritasi.csv          # stem <-> case_id <-> kaynak goruntu
      yolo_donusum_ozeti.xlsx

Goruntuler bilerek kopyalanmiyor: Colab'da veri seti zaten /content'e cekiliyor,
`images/` agacini oradan kurmak hem hizli hem Drive'da 769 dosyayi ikilemiyor.
Egitim notebook'u `dosya_haritasi.csv`'yi okuyup agaci kuruyor.

Bos maskeli 131 `normal` goruntu **bos .txt** ile temsil ediliyor -- Ultralytics
bunlari arkaplan ornegi sayar, negatif ornek islevleri boyle korunur.

Kullanim:
    .venv_mmdet\\Scripts\\python.exe rt_seg_yolo_donusum.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(r'd:\mamografi\real_time_segmentasyon')
COCO_DIZIN = KOK / 'veri' / 'coco'
CIKTI = KOK / 'veri' / 'yolo'
SPLITS = ['train', 'val', 'test']
SINIFLAR = {1: 0}                      # COCO category_id -> YOLO sinif indeksi
ONDALIK = 6


def stem_uret(case_id):
    """'benign/benign (100)' -> 'benign_100'. Bosluk/parantez barindirmayan, tekil."""
    cls, ad = case_id.split('/', 1)
    n = re.search(r'(\d+)', ad)
    if not n:
        raise ValueError(f'case_id icinde numara yok: {case_id}')
    return f'{cls}_{n.group(1)}'


def poligon_satirlari(anns, w, h):
    """COCO poligonlarini normalize YOLO satirlarina cevirir.

    YOLO-seg satir basina tek poligon tasiyor. COCO anotasyonu birden fazla
    poligon icerdiginde (tek bilesenin birden cok dis konturu) her biri ayri
    satira yaziliyor; bu bilgi kaybi degil, ayni pikselleri temsil ediyor.
    """
    satirlar, cok_poligon = [], 0
    for a in anns:
        segm = a['segmentation']
        if not isinstance(segm, list):
            raise TypeError(f'poligon bekleniyordu, RLE geldi: ann {a["id"]}')
        if len(segm) > 1:
            cok_poligon += 1
        for poly in segm:
            p = np.asarray(poly, np.float64).reshape(-1, 2)
            if len(p) < 3:
                continue
            p[:, 0] = np.clip(p[:, 0] / w, 0.0, 1.0)
            p[:, 1] = np.clip(p[:, 1] / h, 0.0, 1.0)
            koord = ' '.join(f'{v:.{ONDALIK}f}' for v in p.reshape(-1))
            satirlar.append(f'{SINIFLAR[a["category_id"]]} {koord}')
    return satirlar, cok_poligon


def _cv2_rasterize(poligonlar, h, w):
    import cv2
    m = np.zeros((h, w), np.uint8)
    for p in poligonlar:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def dogrula(coco, yol_haritasi, split):
    """YOLO etiketlerini geri rasterize edip COCO poligonlariyla karsilastir.

    Bu adimin tek isi **normalize etme + yuvarlamanin** bir sey bozmadigini
    gostermek, o yuzden iki taraf da ayni yontemle (cv2.fillPoly) rasterize
    ediliyor. COCO tarafini pycocotools ile rasterize etmek hizalama testindeki
    tuzagin aynisi olurdu: cikan ~0.990, normalize kaybi degil pycocotools'un
    poligonu yarim piksel daraltan tarama konvansiyonu (PLAN.md, poligon
    parametreleri). Poligonun orijinal maskeye sadakati zaten COCO adiminda
    olculdu. Ultralytics de maskeyi cv2 poligon doldurmayla uretiyor, yani
    modelin gorecegi GT bu taraf.
    """
    dices, bos_dogru = [], 0
    for img_id in coco.getImgIds():
        bilgi = coco.loadImgs(img_id)[0]
        h, w = bilgi['height'], bilgi['width']

        kaynak = [np.asarray(poly, np.float64).reshape(-1, 2)
                  for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                  for poly in a['segmentation']]
        geri = [np.column_stack([v[:, 0] * w, v[:, 1] * h]) for v in
                (np.fromstring(s, sep=' ')[1:].reshape(-1, 2)
                 for s in yol_haritasi[bilgi['case_id']])]

        coco_m = _cv2_rasterize(kaynak, h, w)
        yolo_m = _cv2_rasterize(geri, h, w)

        t = int(coco_m.sum()) + int(yolo_m.sum())
        if t == 0:
            bos_dogru += 1
            continue
        dices.append(2 * float(np.logical_and(coco_m, yolo_m).sum()) / t)
    return np.array(dices), bos_dogru


def main():
    from pycocotools.coco import COCO

    if not COCO_DIZIN.exists():
        sys.exit(f'COCO dizini yok, once rt_seg_coco_donusum.py: {COCO_DIZIN}')

    ozet, harita_satirlari = [], []
    for split in SPLITS:
        coco = COCO(str(COCO_DIZIN / f'busi_{split}.json'))
        etiket_dizin = CIKTI / 'labels' / split
        etiket_dizin.mkdir(parents=True, exist_ok=True)
        for eski in etiket_dizin.glob('*.txt'):      # kalinti etiket birakma
            eski.unlink()

        yol_haritasi, bos, cok_poligon, instans = {}, 0, 0, 0
        for img_id in coco.getImgIds():
            bilgi = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            satirlar, cp = poligon_satirlari(anns, bilgi['width'], bilgi['height'])
            cok_poligon += cp
            instans += len(anns)

            stem = stem_uret(bilgi['case_id'])
            (etiket_dizin / f'{stem}.txt').write_text(
                '\n'.join(satirlar) + ('\n' if satirlar else ''), encoding='utf-8')
            yol_haritasi[bilgi['case_id']] = satirlar
            bos += not satirlar

            harita_satirlari.append(dict(split=split, case_id=bilgi['case_id'],
                                         stem=stem, kaynak=bilgi['file_name'],
                                         height=bilgi['height'], width=bilgi['width'],
                                         instans=len(anns)))

        d, bos_dogru = dogrula(coco, yol_haritasi, split)
        ozet.append(dict(split=split, goruntu=len(yol_haritasi), instans=instans,
                         satir=sum(len(v) for v in yol_haritasi.values()),
                         bos_etiket=bos, bos_dogrulanan=bos_dogru,
                         coklu_poligon_ann=cok_poligon,
                         dice_medyan=round(float(np.median(d)), 5),
                         dice_min=round(float(d.min()), 5)))
        print(f'{split}: {len(yol_haritasi)} etiket -> {etiket_dizin}')

    harita = pd.DataFrame(harita_satirlari)
    harita.to_csv(CIKTI / 'dosya_haritasi.csv', index=False)

    (CIKTI / 'data.yaml').write_text(
        '# rt_seg_yolo_donusum.py tarafindan uretildi -- elle duzenlenmez.\n'
        '# `path` egitim notebook\'unda images/ agaci kurulduktan sonra oraya isaret eder.\n'
        'path: /content/yolo_veri\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'names:\n'
        '  0: lezyon\n', encoding='utf-8')

    tablo = pd.DataFrame(ozet)
    tablo.to_excel(CIKTI / 'yolo_donusum_ozeti.xlsx', index=False)
    print('\n=== OZET ===')
    print(tablo.to_string(index=False))

    manifest = pd.read_csv(KOK / 'veri' / 'veri_manifest.csv')
    assert len(harita) == len(manifest), 'etiket sayisi manifest ile uyusmuyor'
    assert harita['stem'].is_unique, 'stem cakismasi var'
    assert tablo['bos_etiket'].sum() == int((manifest['lesion_px'] == 0).sum()), \
        'bos etiket sayisi bos maskeli goruntu sayisiyla uyusmuyor'
    # Ayni rasterize yontemiyle olculdugu icin beklenen deger 1.0'a cok yakin;
    # kalan fark yalnizca normalize etme + 6 hane yuvarlamadan.
    assert (tablo['dice_medyan'] > 0.999).all(), 'YOLO geri-rasterize Dice dusuk'
    assert (tablo['dice_min'] > 0.99).all(), 'tek bir goruntude kabul edilemez kayip'
    print('\nTum kontroller gecti.')


if __name__ == '__main__':
    main()
