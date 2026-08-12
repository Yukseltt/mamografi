"""mmengine `vis_data/scalars.json` -> Excel epoch logu + egitim egrileri.

Egitim sirasinda ek bir hook calistirmiyoruz: mmengine zaten her seyi
scalars.json'a yaziyor, biz koşu bitince donusturuyoruz. Boylece egitime yeni
bir cokme riski eklemiyoruz ve biten kosular icin geriye donuk da uretebiliyoruz.

Excel semasi mass_lezyon_projesi / mendeley_data_density ile ayni mantikta:
epoch basina tek satir, resume'da tekrar satir olusmaz (epoch'a gore tekillestirilir).

Kullanim:
    python egitim_grafikleri.py --work_dir ...\\calisma\\rtmdet_ins_tiny_256_orta
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEMA = ['epoch', 'train_loss', 'loss_cls', 'loss_bbox', 'loss_mask',
          'val_kayip', 'val_kayip_cls', 'val_kayip_bbox', 'val_kayip_mask',
          'lr', 'grad_norm', 'dice_tum', 'dice_lezyonlu', 'bos_dogru',
          'bbox_mAP', 'bbox_mAP_50', 'segm_mAP', 'segm_mAP_50']


def scalars_oku(work_dir):
    """En son kosunun scalars.json'ini oku. Train ve val satirlari ayri semada."""
    dosyalar = sorted(Path(work_dir).glob('*/vis_data/scalars.json'),
                      key=lambda p: p.stat().st_mtime)
    if not dosyalar:
        raise FileNotFoundError(f'scalars.json bulunamadi: {work_dir}')
    son = dosyalar[-1]

    train, val = [], []
    for satir in son.read_text(encoding='utf-8').splitlines():
        satir = satir.strip()
        if not satir:
            continue
        d = json.loads(satir)
        # Yalnizca 'coco/' aramak yetmiyor: model henuz esigi gecen tahmin
        # uretmediginde CocoMetric bos donuyor ve o epoch'un val satiri train
        # gibi siniflanip kayboluyordu. busi/ ve val_kayip her epoch yaziliyor.
        val_satiri = any(k.startswith(('coco/', 'busi/', 'val_kayip')) for k in d)
        (val if val_satiri else train).append(d)
    return son, pd.DataFrame(train), pd.DataFrame(val)


def epoch_tablosu(train, val):
    # train satirlari epoch basina ortalanir
    t = train.groupby('epoch').agg(
        train_loss=('loss', 'mean'), loss_cls=('loss_cls', 'mean'),
        loss_bbox=('loss_bbox', 'mean'), loss_mask=('loss_mask', 'mean'),
        lr=('lr', 'last'), grad_norm=('grad_norm', 'mean'),
    ).reset_index()

    # val satirlarinda epoch alani yok. Sirayla eslestirmek hatali: bazi epoch'larin
    # validation'i bos sonuc verip coco/ anahtari hic yazmayabiliyor (ornegin epoch 1,
    # model henuz esigi gecen tahmin uretmiyorken) -- o zaman tum eslesme bir kayiyor.
    # Bunun yerine val satirinin 'step' degeri, epoch'larin son train adimiyla eslestiriliyor.
    if len(val):
        v = pd.DataFrame({
            'epoch': val['step'].to_numpy(),
            'bbox_mAP': val.get('coco/bbox_mAP'),
            'bbox_mAP_50': val.get('coco/bbox_mAP_50'),
            'segm_mAP': val.get('coco/segm_mAP'),
            'segm_mAP_50': val.get('coco/segm_mAP_50'),
            'dice_tum': val.get('busi/dice_tum'),
            'dice_lezyonlu': val.get('busi/dice_lezyonlu'),
            'bos_dogru': val.get('busi/bos_dogru'),
            'val_kayip': val.get('val_kayip'),
            'val_kayip_cls': val.get('val_kayip_cls'),
            'val_kayip_bbox': val.get('val_kayip_bbox'),
            'val_kayip_mask': val.get('val_kayip_mask'),
        })
        t = t.merge(v, on='epoch', how='left')

    for k in SCHEMA:
        if k not in t.columns:
            t[k] = np.nan
    # resume'da ayni epoch iki kez loglanmis olabilir: sonuncusu gecerli
    return t[SCHEMA].drop_duplicates('epoch', keep='last').sort_values('epoch')


def egrileri_ciz(df, cikti, baslik):
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))

    a = ax[0, 0]
    a.plot(df.epoch, df.train_loss, lw=2, color='tab:blue', label='train')
    if 'val_kayip' in df and df.val_kayip.notna().any():
        a.plot(df.epoch, df.val_kayip, lw=2, ls='--', color='tab:blue', label='val')
    for k, r in [('loss_cls', 'tab:orange'), ('loss_bbox', 'tab:green'), ('loss_mask', 'tab:red')]:
        a.plot(df.epoch, df[k], lw=1, alpha=0.6, color=r, label=k)
    a.set_title('Kayip (train / val)'); a.set_xlabel('epoch'); a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(df.epoch, df.segm_mAP, marker='o', ms=3, color='tab:red', label='segm mAP')
    a.plot(df.epoch, df.bbox_mAP, marker='o', ms=3, color='tab:blue', label='bbox mAP')
    if df.segm_mAP.notna().any():
        i = int(df.segm_mAP.idxmax())
        a.axvline(df.epoch[i], ls='--', lw=1, color='k')
        # etiket egrinin ustune binmesin: asagi kaydirilip kutuya alindi
        a.annotate(f'zirve ep{int(df.epoch[i])}: {df.segm_mAP[i]:.3f}',
                   (df.epoch[i], df.segm_mAP[i]), textcoords='offset points',
                   xytext=(6, -46), fontsize=8.5,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb',
                             alpha=0.9))
    a.set_title('Validation mAP'); a.set_xlabel('epoch'); a.legend(); a.grid(alpha=0.3)

    a = ax[0, 2]
    a.plot(df.epoch, df.segm_mAP_50, marker='o', ms=3, color='tab:red', label='segm mAP@50')
    a.plot(df.epoch, df.bbox_mAP_50, marker='o', ms=3, color='tab:blue', label='bbox mAP@50')
    a.set_title('mAP@50'); a.set_xlabel('epoch'); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(df.epoch, df.lr, color='tab:brown')
    a.set_yscale('log'); a.set_title('Learning rate'); a.set_xlabel('epoch')
    a.grid(alpha=0.3, which='both')

    a = ax[1, 1]
    a.plot(df.epoch, df.grad_norm, color='tab:purple')
    a.axhline(1.0, ls='--', lw=1, color='k')
    a.set_yscale('log')
    a.set_title('Gradyan normu (kesikli cizgi: clip esigi 1.0)')
    a.set_xlabel('epoch'); a.grid(alpha=0.3, which='both')

    a = ax[1, 2]
    if 'dice_tum' in df and df.dice_tum.notna().any():
        a.plot(df.epoch, df.dice_tum, color='tab:green', lw=1.8, label='Dice (tum)')
        a.plot(df.epoch, df.dice_lezyonlu, color='tab:green', lw=1, alpha=0.6,
               label='Dice (lezyonlu)')
        a.set_ylim(0, 1); a.legend(fontsize=8)
        a.set_title('Val Dice (checkpoint secim olcutu)')
    else:
        a.plot(df.epoch, df.loss_mask, color='tab:red')
        a.set_title('loss_mask (yakin plan)')
    a.set_xlabel('epoch'); a.grid(alpha=0.3)

    plt.suptitle(baslik, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(cikti, dpi=140, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', required=True)
    args = ap.parse_args()

    work = Path(args.work_dir)
    kaynak, train, val = scalars_oku(work)
    df = epoch_tablosu(train, val)

    ad = work.name
    xlsx = work / f'{ad}_log.xlsx'
    png = work / f'{ad}_curves.png'
    df.to_excel(xlsx, index=False)
    egrileri_ciz(df, png, f'{ad}  ({len(df)} epoch)')

    print('kaynak :', kaynak)
    print('epoch  :', len(df))
    print('excel  :', xlsx)
    print('grafik :', png)
    if df.segm_mAP.notna().any():
        i = int(df.segm_mAP.idxmax())
        print(f'segm_mAP zirvesi: epoch {int(df.epoch[i])} -> {df.segm_mAP[i]:.4f}')
        print(f'son epoch       : {df.segm_mAP.iloc[-1]:.4f}')
    print()
    print(df.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
