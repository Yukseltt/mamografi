"""Rapor figurlerini uretir. Uc modelin gorseli de ayni koddan cikar.

Vaka gorselleri Colab'da uretilmisti ama yeniden burada uretiliyor: ayni cizim
kodu, ayni vakalar, ayni esikler -- rapora yan yana konulacaklari icin gorsel
farkin modelden gelmesi lazim, cizim ayarindan degil.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BURASI = Path(__file__).parent
KOK = BURASI.parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'rtmdet'))

import torch_uyum        # noqa: F401
import ozel_transformlar  # noqa: F401
import degerlendirme as dg
import modeller as md

FIG = BURASI / 'figurler'
LOG = BURASI / 'loglar'

MODELLER = {
    'segformer_b0': dict(ad='SegFormer-B0', esik=0.4, log='segformer_b0.xlsx',
                         tur='segformer', ckpt=KOK / 'segformer_best.pt', girdi=256),
    'rtmdet_256': dict(ad='RTMDet-Ins 256', esik=0.4, log='rtmdet_256.xlsx',
                       tur='rtmdet', kosu='rtmdet_ins_tiny_256_orta'),
    'rtmdet_512': dict(ad='RTMDet-Ins 512', esik=0.2, log='rtmdet_512.xlsx',
                       tur='rtmdet', kosu='rtmdet_ins_tiny_512_orta'),
    'yolo11n_seg': dict(ad='YOLOv11n-Seg', esik=0.2, log='yolo11n_seg.xlsx',
                        tur='yolo', ckpt=KOK / 'yolo11n_seg_best.pt', girdi=256),
}
RENK = {'segformer_b0': '#2c5aa0', 'rtmdet_256': '#c0392b',
        'rtmdet_512': '#e67e22', 'yolo11n_seg': '#27ae60'}


def kur(cfg):
    if cfg['tur'] == 'segformer':
        return md.segformer_kur(cfg['ckpt'], cfg['girdi'], 'cuda:0')[1]
    if cfg['tur'] == 'yolo':
        return md.yolo_kur(cfg['ckpt'], cfg['girdi'], 0)[1]
    from metrik_ayrismasi import kosu_bul
    return md.rtmdet_kur(*kosu_bul(cfg['kosu']), cihaz='cuda:0')[1]


# ---- 1. egitim egrileri ----

def egri(anahtar, cfg):
    yol = LOG / cfg['log']
    if not yol.exists():
        print(f'  ATLANDI (log yok): {yol.name}')
        return
    d = pd.read_excel(yol)
    # SegFormer logu notebook ciktisindan geri kazanildi, LR sutunu icermiyor
    lr_var = 'lr' in d.columns and d['lr'].notna().any()
    n = 3 if lr_var else 2
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4))

    # Yalnizca egitim kaybi ciziliyor. Val loss SegFormer ve RTMDet'te hic
    # hesaplanmadi (validation adimi kayip degil kalite metrigi uretiyor);
    # Ultralytics hesapliyor ama tek modelde gosterip digerlerinde gostermemek
    # uc figuru karsilastirilamaz hale getirirdi. Gerekce raporda bir kez yaziyor.
    a = ax[0]
    a.plot(d.epoch, d.train_loss, color=RENK[anahtar], lw=1.6)
    a.set_title('Egitim kaybi'); a.set_xlabel('epoch'); a.grid(alpha=0.3)

    a = ax[1]
    if 'dice_tum' in d:
        a.plot(d.epoch, d.dice_tum, color=RENK[anahtar], lw=1.6, label='Dice (tum)')
        a.plot(d.epoch, d.dice, color=RENK[anahtar], lw=1, alpha=0.55, label='Dice (lezyonlu)')
        i = int(d.dice_tum.idxmax()); en = d.dice_tum[i]
    else:
        a.plot(d.epoch, d.segm_mAP if 'segm_mAP' in d else d.mask_mAP,
               color=RENK[anahtar], lw=1.6, label='maske mAP')
        a.plot(d.epoch, d.bbox_mAP if 'bbox_mAP' in d else d.box_mAP,
               color=RENK[anahtar], lw=1, alpha=0.55, label='kutu mAP')
        s = d.segm_mAP if 'segm_mAP' in d else d.mask_mAP
        i = int(s.idxmax()); en = s[i]
    a.axvline(d.epoch[i], ls=':', lw=1.2, color='k')
    a.annotate(f'zirve ep{int(d.epoch[i])}\n{en:.3f}', (d.epoch[i], en),
               textcoords='offset points', xytext=(8, -22), fontsize=8)
    a.set_ylim(0, 1); a.set_title('Validation kalitesi'); a.set_xlabel('epoch')
    a.legend(fontsize=8); a.grid(alpha=0.3)

    if lr_var:
        a = ax[2]
        a.plot(d.epoch, d['lr'], color='#7f5539')
        a.set_yscale('log'); a.set_title('Learning rate'); a.set_xlabel('epoch')
        a.grid(alpha=0.3, which='both')

    plt.suptitle(f"{cfg['ad']} — {len(d)} epoch", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / f'egri_{anahtar}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  egri_{anahtar}.png')


# ---- 2. sabit vaka gorselleri ----

def vakalar(anahtar, cfg, vakalar_json, lut):
    harita_fn = kur(cfg)
    v = vakalar_json
    fig, ax = plt.subplots(len(v), 3, figsize=(10, 3.1 * len(v)))
    for i, vk in enumerate(v):
        r = lut[vk['case_id']]
        img = dg.gri_oku(dg.yol_coz(r.image_path))
        gt = dg.gt_maskesi(r)
        pred = harita_fn(r) >= cfg['esik']
        d = dg.metrikler(pred, gt)['dice']
        for j, (katman, alt) in enumerate([(img, 'orijinal'), (gt, 'ground truth'),
                                           (pred, f'tahmin · Dice {d:.3f}')]):
            ax[i, j].imshow(img, cmap='gray')
            if j:
                ax[i, j].imshow(np.ma.masked_where(katman == 0, katman),
                                cmap='autumn', alpha=0.45)
            ax[i, j].set_title(f"{vk['case_id']} [{vk['kova']}]\n{alt}", fontsize=8)
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(FIG / f'vakalar_{anahtar}.png', dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  vakalar_{anahtar}.png')


# ---- 3. karsilastirma figurleri ----

def kalite_hiz():
    h = pd.read_excel(KOK / 'hiz_olcumu.xlsx')
    o = pd.read_excel(KOK / 'uc_model_karsilastirma.xlsx', sheet_name='ozet')
    x = pd.read_excel(KOK / 'onnx_olcumu.xlsx')
    birlesik = h.merge(o[['model', 'dice_lezyonlu']], on='model').merge(
        x[['model', 'onnx_gpu_ms']], on='model')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, sut, baslik in [(ax[0], 'gpu_fps', 'PyTorch, uctan uca'),
                           (ax[1], 'onnx_gpu_ms', 'ONNX, yalnizca ag')]:
        for _, r in birlesik.iterrows():
            k = [k for k, c in MODELLER.items() if c['ad'] == r['model']][0]
            deger = r[sut] if sut == 'gpu_fps' else 1000 / r[sut]
            a.scatter(deger, r.dice_lezyonlu, s=110, color=RENK[k], zorder=3)
            a.annotate(r['model'], (deger, r.dice_lezyonlu), fontsize=8.5,
                       textcoords='offset points', xytext=(7, -3))
        a.axvline(30, ls='--', lw=1, color='#888')
        a.text(31, a.get_ylim()[0], '30 FPS', fontsize=8, color='#888', va='bottom')
        a.set_xlabel('FPS (RTX 2060, batch=1)'); a.set_ylabel('Test lezyonlu Dice')
        a.set_title(baslik); a.grid(alpha=0.3); a.margins(x=0.22, y=0.15)
    plt.tight_layout()
    plt.savefig(FIG / 'kalite_hiz.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  kalite_hiz.png')


def eslestirilmis_figur():
    t = pd.read_excel(KOK / 'uc_model_karsilastirma.xlsx', sheet_name='eslestirilmis')
    fig, a = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(t))[::-1]
    a.errorbar(t.fark, y, xerr=[t.fark - t.ga_alt, t.ga_ust - t.fark], fmt='o',
               color='#2c5aa0', capsize=4, lw=1.4, ms=6)
    a.axvline(0, color='#c0392b', lw=1.2)
    a.set_yticks(y); a.set_yticklabels([f'{r.a}  −  {r.b}' for r in t.itertuples()],
                                       fontsize=8.5)
    a.set_xlabel('Test lezyonlu Dice farki (%95 bootstrap GA)')
    a.set_title('Eslestirilmis karsilastirma — tum araliklar sifiri kapsiyor', fontsize=10)
    a.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIG / 'eslestirilmis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  eslestirilmis.png')


def boyut_figur():
    x = KOK / 'uc_model_karsilastirma.xlsx'
    fig, a = plt.subplots(figsize=(8.5, 3.8))
    kovalar = None
    for k, cfg in MODELLER.items():
        sayfa = cfg['ad'].replace('-', '_')[:31]
        gb = pd.read_excel(x, sheet_name=sayfa)
        b = dg.boyut_kirilimi(gb.assign(esik=cfg['esik']), cfg['esik'])
        kovalar = b.kova.tolist()
        a.plot(b.kova, b.dice, marker='o', color=RENK[k], label=cfg['ad'], lw=1.6)
    a.set_xlabel('Lezyon esdegeri cap (px)'); a.set_ylabel('Test Dice')
    a.set_title('Lezyon boyutuna gore Dice', fontsize=10)
    a.legend(fontsize=8.5); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'boyut_kirilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  boyut_kirilimi.png')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print('egitim egrileri:')
    for k, cfg in MODELLER.items():
        egri(k, cfg)

    print('karsilastirma figurleri:')
    kalite_hiz()
    eslestirilmis_figur()
    boyut_figur()

    print('vaka gorselleri:')
    v = json.loads((KOK / 'ortak' / 'eval_cases.json').read_text(encoding='utf-8'))['vakalar']
    lut = {r.case_id: r for r in dg.manifest_oku().itertuples()}
    for k, cfg in MODELLER.items():
        if k == 'rtmdet_512':
            continue          # 256 ile gorsel olarak ayirt edilemiyor, rapora biri giriyor
        vakalar(k, cfg, v, lut)


if __name__ == '__main__':
    main()
