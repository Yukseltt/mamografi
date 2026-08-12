"""Dort kosunun epoch logunu tek yerde toplar.

Kaynaklar farkli: RTMDet mmengine `scalars.json`'dan uretilmis Excel'de, SegFormer'in
logu Drive'da ama notebook ciktisinda epoch satirlari duruyor, YOLO'nunki Ultralytics
`results.csv`'sinde. Rapor figurlerinin hepsi tek koddan ciksin diye burada
normalize ediliyor.

Val loss notu: SegFormer ve RTMDet'te validation adimi kayip degil kalite metrigi
uretiyor (Dice / mAP), dolayisiyla val loss hic hesaplanmadi. Ultralytics
hesapliyor ama **rapora alinmiyor**: tek modelde gosterip digerlerinde
gostermemek uc egriyi karsilastirilamaz hale getirir. Ayrica uc mimarinin kayip
fonksiyonu farkli oldugu icin val loss zaten modeller arasi okunamaz.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).parent.parent
CIKTI = Path(__file__).parent / 'loglar'

SF_SATIR = re.compile(
    r'ep\s+(\d+)/\d+\s+kayip\s+([\d.]+)\s+dice_tum\s+([\d.]+)\s+\(esik\s+([\d.]+)\)\s+'
    r'lezyonlu\s+([\d.]+)\s+bos\s+(\d+)/(\d+)')


def segformer_logu(nb_yolu):
    """Notebook ciktisindaki epoch satirlarindan tabloyu geri kazanir."""
    nb = json.loads(Path(nb_yolu).read_text(encoding='utf-8'))
    metin = ''
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        if 'for epoch in range' not in ''.join(c['source']):
            continue
        metin = ''.join(''.join(o.get('text', [])) for o in c.get('outputs', []))
    satir = [dict(epoch=int(m[0]), train_loss=float(m[1]), dice_tum=float(m[2]),
                  esik=float(m[3]), dice=float(m[4]), bos_dogru=int(m[5]), bos_n=int(m[6]))
             for m in SF_SATIR.findall(metin)]
    if not satir:
        raise SystemExit('SegFormer epoch satirlari bulunamadi -- notebook ciktilariyla '
                         'kaydedilmis olmali')
    return pd.DataFrame(satir)


def rtmdet_logu(kosu):
    return pd.read_excel(KOK / 'rtmdet' / 'calisma' / kosu / f'{kosu}_log.xlsx')


def yolo_logu(csv_yolu):
    """Ultralytics results.csv -> ortak sema (egitim kaybi + val mAP)."""
    ham = pd.read_csv(csv_yolu)
    ham.columns = [c.strip() for c in ham.columns]

    def kol(*adaylar):
        for a in adaylar:
            if a in ham.columns:
                return ham[a]
        return pd.Series([None] * len(ham))

    d = pd.DataFrame({
        'epoch': ham['epoch'],
        'train_box': kol('train/box_loss'), 'train_seg': kol('train/seg_loss'),
        'train_cls': kol('train/cls_loss'), 'train_dfl': kol('train/dfl_loss'),
        'mask_mAP': kol('metrics/mAP50-95(M)'), 'mask_mAP_50': kol('metrics/mAP50(M)'),
        'box_mAP': kol('metrics/mAP50-95(B)'), 'lr': kol('lr/pg0'),
    })
    d['train_loss'] = d[['train_box', 'train_seg', 'train_cls', 'train_dfl']].sum(axis=1)
    return d


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)

    sf = segformer_logu(KOK / 'rt_seg_egitim_segformer.ipynb')
    sf.to_excel(CIKTI / 'segformer_b0.xlsx', index=False)
    print(f'SegFormer-B0   : {len(sf)} epoch  ')

    for kosu, ad in [('rtmdet_ins_tiny_256_orta', 'rtmdet_256'),
                     ('rtmdet_ins_tiny_512_orta', 'rtmdet_512')]:
        d = rtmdet_logu(kosu)
        d.to_excel(CIKTI / f'{ad}.xlsx', index=False)
        print(f'{ad:15s}: {len(d)} epoch  ')

    csv = KOK / 'yolo_results.csv'
    if csv.exists():
        y = yolo_logu(csv)
        y.to_excel(CIKTI / 'yolo11n_seg.xlsx', index=False)
        print(f'YOLOv11n-Seg   : {len(y)} epoch  (egitim egrisi icin)')
    else:
        print(f'YOLOv11n-Seg   : ATLANDI -- {csv} yok.\n'
              '  Ultralytics results.csv Drive`da: '
              'yolo11n_seg_256_orta kosu klasorunde.\n'
              '  Val loss egrisi yalnizca bu dosyada mevcut.')


if __name__ == '__main__':
    main()
