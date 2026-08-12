"""Dokuz Faz 2 kosusunun epoch logunu tek semaya toplar.

RTMDet logu yerelde (work_dir icinde xlsx). SegFormer ve YOLO Colab'da egitildi,
loglari Drive'da -- indirilip `loglar_faz2/` icine ham adiyla konuluyor, bu betik
normalize ediyor.

Faz 1'den fark: val loss artik uc modelde de var (`val_kayip`), cunku Faz 2'de
RTMDet'e ValLossHook, SegFormer'a val dongusunde kayip hesabi eklendi. Faz 1
raporunda egrilerde yalnizca train loss vardi; gerekce "iki modelde val loss yok"
idi ve o gerekce artik gecersiz.

Beklenen girdi adlari (loglar_faz2/ham/ icine):
    segformer_b0_256_orta_s{42,43,44}_log.xlsx
    yolo11n_seg_256_orta_s{42,43,44}_results.csv   (Ultralytics results.csv)
"""
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).parent.parent
CIKTI = Path(__file__).parent / 'loglar_faz2'
HAM = CIKTI / 'ham'
TOHUMLAR = (42, 43, 44)


def rtmdet_logu(tohum):
    kosu = f'rtmdet_ins_tiny_256_orta_s{tohum}'
    return pd.read_excel(KOK / 'rtmdet' / 'calisma' / kosu / f'{kosu}_log.xlsx')


def yolo_logu(csv_yolu):
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
        'val_box': kol('val/box_loss'), 'val_seg': kol('val/seg_loss'),
        'val_cls': kol('val/cls_loss'), 'val_dfl': kol('val/dfl_loss'),
        'mask_mAP': kol('metrics/mAP50-95(M)'), 'box_mAP': kol('metrics/mAP50-95(B)'),
        'lr': kol('lr/pg0'),
    })
    d['train_loss'] = d[['train_box', 'train_seg', 'train_cls', 'train_dfl']].sum(axis=1)
    d['val_kayip'] = d[['val_box', 'val_seg', 'val_cls', 'val_dfl']].sum(axis=1)
    return d


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    HAM.mkdir(parents=True, exist_ok=True)
    eksik = []

    for t in TOHUMLAR:
        d = rtmdet_logu(t)
        d.to_excel(CIKTI / f'rtmdet_s{t}.xlsx', index=False)
        vk = 'var' if 'val_kayip' in d.columns else 'YOK'
        print(f'rtmdet_s{t}    : {len(d)} epoch, val_kayip {vk}')

    for t in TOHUMLAR:
        p = HAM / f'segformer_b0_256_orta_s{t}_log.xlsx'
        if not p.exists():
            eksik.append(p.name)
            continue
        d = pd.read_excel(p)
        d.to_excel(CIKTI / f'segformer_s{t}.xlsx', index=False)
        vk = 'var' if 'val_kayip' in d.columns else 'YOK'
        print(f'segformer_s{t} : {len(d)} epoch, val_kayip {vk}')

    for t in TOHUMLAR:
        p = HAM / f'yolo11n_seg_256_orta_s{t}_results.csv'
        if not p.exists():
            eksik.append(p.name)
            continue
        d = yolo_logu(p)
        d.to_excel(CIKTI / f'yolo_s{t}.xlsx', index=False)
        print(f'yolo_s{t}      : {len(d)} epoch, val_kayip var')

    if eksik:
        print(f'\nEKSIK ({len(eksik)}) -- {HAM} icine konulmali:')
        for e in eksik:
            print('  ', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
