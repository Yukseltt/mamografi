"""Faz 2 rapor figurleri: uc mimari x uc tohum.

Faz 1'in figurleri tek kosu varsayiyordu; buradaki her figur tohum dagilimini
gosteriyor cunku raporun ana bulgusu tam olarak o dagilimin mimari farklarini
yutmasi.

Egitim egrileri icin `loglar_faz2/` altinda dokuz log bekleniyor:
    rtmdet_s{42,43,44}.xlsx      (loglari_topla_faz2.py yerelde uretiyor)
    segformer_s{42,43,44}.xlsx   (Drive'dan indirilir)
    yolo_s{42,43,44}.xlsx        (Drive'dan indirilir)
Eksik olan sessizce atlanir, uyari basilir.
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

FIG = BURASI / 'figurler_faz2'
LOG = BURASI / 'loglar_faz2'
XLSX = KOK / 'faz2_karsilastirma.xlsx'
TOHUMLAR = (42, 43, 44)

MIMARILER = {
    'RTMDet-Ins': dict(kisa='rtmdet', renk='#c0392b'),
    'SegFormer-B0': dict(kisa='segformer', renk='#2c5aa0'),
    'YOLOv11n-Seg': dict(kisa='yolo', renk='#27ae60'),
}
KOVA_SIRA = ['<80', '80-140', '140-220', '>220']


def kur(mimari, tohum):
    if mimari == 'SegFormer-B0':
        return md.segformer_kur(KOK / f'faz2_ckpt/segformer_s{tohum}.pt', 256, 'cuda:0')[1]
    if mimari == 'YOLOv11n-Seg':
        return md.yolo_kur(KOK / f'faz2_ckpt/yolo_s{tohum}.pt', 256, 0)[1]
    from metrik_ayrismasi import kosu_bul
    return md.rtmdet_kur(*kosu_bul(f'rtmdet_ins_tiny_256_orta_s{tohum}'), cihaz='cuda:0')[1]


# ---- 1. tohum varyansi: raporun ana figuru ----

def tohum_varyansi():
    t = pd.read_excel(XLSX, sheet_name='test_kosu')
    v = pd.read_excel(XLSX, sheet_name='tohum_varyansi')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    for a, sut, baslik in [(ax[0], 'dice_lezyonlu', 'Test Dice — lezyonlu görüntüler'),
                           (ax[1], 'dice_tum', 'Test Dice — tüm görüntüler')]:
        for i, (m, cfg) in enumerate(MIMARILER.items()):
            d = t[t.mimari == m][sut].values
            a.scatter(np.full(len(d), i), d, s=70, color=cfg['renk'], zorder=3,
                      label=m if a is ax[0] else None)
            a.hlines(d.mean(), i - 0.22, i + 0.22, color=cfg['renk'], lw=2.2, zorder=4)
            # std bandi: mimariler arasi farkin bu bandin icinde kalip kalmadigi
            # figurun tek soyledigi sey
            a.fill_between([i - 0.3, i + 0.3], d.mean() - d.std(ddof=1),
                           d.mean() + d.std(ddof=1), color=cfg['renk'], alpha=0.13)
        a.set_xticks(range(len(MIMARILER)))
        a.set_xticklabels([m.replace('-', '\n') for m in MIMARILER], fontsize=9)
        a.set_ylabel('Dice'); a.set_title(baslik, fontsize=10); a.grid(alpha=0.3, axis='y')
    plt.suptitle('Üç mimari, her biri üç kez eğitildi — noktalar tek koşular, bant ±1 std',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / 'kosu_dagilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  kosu_dagilimi.png')


# ---- 2. eslestirilmis fark ----

def eslestirilmis_figur():
    t = pd.read_excel(XLSX, sheet_name='eslestirilmis_ort')
    fig, a = plt.subplots(figsize=(8.5, 2.9))
    y = np.arange(len(t))[::-1]
    renk = ['#c0392b' if s == 'EVET' else '#7f8c8d' for s in t.anlamli]
    for yi, r, c in zip(y, t.itertuples(), renk):
        a.errorbar(r.fark, yi, xerr=[[r.fark - r.ga_alt], [r.ga_ust - r.fark]],
                   fmt='o', color=c, capsize=4, lw=1.5, ms=7)
    a.axvline(0, color='#333', lw=1.2)
    a.set_yticks(y)
    a.set_yticklabels([f'{r.a}  −  {r.b}' for r in t.itertuples()], fontsize=9)
    a.set_xlabel('Test lezyonlu Dice farkı (%95 bootstrap GA)')
    a.set_title('Kırmızı: sıfırı kapsamıyor', fontsize=9.5)
    a.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIG / 'eslestirilmis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  eslestirilmis.png')


# ---- 3. boyut kirilimi ----

def boyut_figur():
    k = pd.read_excel(XLSX, sheet_name='boyut_kirilimi')
    fig, a = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(KOVA_SIRA))
    for m, cfg in MIMARILER.items():
        d = k[k.mimari == m].pivot_table(index='kova', values='dice',
                                         aggfunc=['mean', 'std'], observed=True)
        d.columns = ['ort', 'std']
        d = d.reindex(KOVA_SIRA)
        a.errorbar(x, d.ort, yerr=d['std'], marker='o', color=cfg['renk'],
                   label=m, lw=1.7, capsize=3)
    n = k[k.tohum == 42].groupby('kova', observed=True).n.first().reindex(KOVA_SIRA)
    a.set_xticks(x)
    a.set_xticklabels([f'{k_}\n(n={int(v)})' for k_, v in zip(KOVA_SIRA, n)], fontsize=9)
    a.set_xlabel('Lezyon eşdeğer çapı (px)'); a.set_ylabel('Test Dice')
    a.set_title('Lezyon boyutuna göre Dice — üç koşunun ortalaması ±1 std', fontsize=10)
    a.legend(fontsize=8.5); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'boyut_kirilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  boyut_kirilimi.png')


# ---- 4. kalite / hiz ----

def kalite_hiz():
    h = pd.read_excel(KOK / 'hiz_olcumu.xlsx')
    o = pd.read_excel(KOK / 'onnx_olcumu.xlsx')
    v = pd.read_excel(XLSX, sheet_name='tohum_varyansi')
    ad = {'RTMDet-Ins 256': 'RTMDet-Ins', 'SegFormer-B0': 'SegFormer-B0',
          'YOLOv11n-Seg': 'YOLOv11n-Seg'}
    h['mimari'] = h.model.map(ad)
    o['mimari'] = o.model.map(ad)
    b = h.merge(o[['mimari', 'onnx_gpu_ms', 'onnx_cpu_ms']], on='mimari').merge(v, on='mimari')

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for a, sut, baslik in [
            (ax[0], 'gpu_fps', 'PyTorch GPU, uçtan uca'),
            (ax[1], 'onnx_gpu_ms', 'ONNX GPU, yalnızca ağ'),
            (ax[2], 'onnx_cpu_ms', 'ONNX CPU, yalnızca ağ')]:
        for _, r in b.iterrows():
            c = MIMARILER[r.mimari]['renk']
            fps = r[sut] if sut == 'gpu_fps' else 1000 / r[sut]
            a.errorbar(fps, r.dice_lezyonlu_mean, yerr=r.dice_lezyonlu_std,
                       fmt='o', ms=9, color=c, capsize=3, zorder=3)
            a.annotate(r.mimari, (fps, r.dice_lezyonlu_mean), fontsize=8.5,
                       textcoords='offset points', xytext=(8, -3))
        a.axvline(30, ls='--', lw=1, color='#888')
        a.set_xlabel('FPS'); a.set_title(baslik, fontsize=10)
        a.grid(alpha=0.3); a.margins(x=0.3, y=0.25)
    ax[0].set_ylabel('Test lezyonlu Dice (3 koşu ort ±1 std)')
    plt.suptitle('Sıralama ortama göre değişiyor', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / 'kalite_hiz.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  kalite_hiz.png')


# ---- 5. egitim egrileri (val loss ile) ----

def egriler():
    KALITE_ADI = {'dice_tum': 'Dice (tüm görüntüler)', 'mask_mAP': 'maske mAP',
                  'segm_mAP': 'maske mAP'}
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    bulunan = 0
    for j, (m, cfg) in enumerate(MIMARILER.items()):
        kayip_degerleri, kalite_adi = [], 'validation kalitesi'
        for tohum in TOHUMLAR:
            yol = LOG / f"{cfg['kisa']}_s{tohum}.xlsx"
            if not yol.exists():
                continue
            d = pd.read_excel(yol)
            bulunan += 1
            alfa = 0.45 + 0.25 * TOHUMLAR.index(tohum)
            e = d.epoch
            ax[0, j].plot(e, d.train_loss, color=cfg['renk'], lw=1.3, alpha=alfa,
                          label=f'koşu {tohum - 41} eğitim')
            kayip_degerleri.append(d.train_loss)
            vk = next((c for c in ('val_kayip', 'val_loss') if c in d), None)
            if vk:
                ax[0, j].plot(e, d[vk], color=cfg['renk'], lw=1.3, ls='--', alpha=alfa,
                              label=f'koşu {tohum - 41} val')
                kayip_degerleri.append(d[vk])
            kal = next((c for c in ('dice_tum', 'mask_mAP', 'segm_mAP') if c in d), None)
            if kal:
                kalite_adi = KALITE_ADI.get(kal, kal)
                ax[1, j].plot(e, d[kal], color=cfg['renk'], lw=1.3, alpha=alfa,
                              label=f'koşu {tohum - 41}')
        # YOLO'nun val kaybi ilk epochlarda 22'ye kadar sicriyor ve grafigin geri
        # kalanini eziyor; olcek ilk 20 epoch disinda kalan araliga gore kuruluyor
        if kayip_degerleri:
            son = pd.concat([k.iloc[20:] for k in kayip_degerleri])
            ax[0, j].set_ylim(0, float(son.max()) * 1.25)
        ax[0, j].set_title(m, fontsize=10); ax[0, j].grid(alpha=0.3)
        ax[0, j].set_xlabel('epoch'); ax[0, j].set_ylabel('kayıp')
        ax[0, j].legend(fontsize=7, ncol=2)
        ax[1, j].grid(alpha=0.3); ax[1, j].set_xlabel('epoch')
        ax[1, j].set_ylabel(kalite_adi); ax[1, j].legend(fontsize=7)
    if not bulunan:
        plt.close()
        print('  UYARI: loglar_faz2/ bos, egri figuru uretilmedi')
        return
    plt.suptitle('Eğitim ve validation kaybı (üst), validation kalitesi (alt) — mimari başına üç koşu',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG / 'egriler.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  egriler.png ({bulunan}/9 log)')


# ---- 6. vaka gorselleri: uc mimari yan yana ----

def vakalar():
    v = json.loads((KOK / 'ortak' / 'eval_cases.json').read_text(encoding='utf-8'))['vakalar']
    lut = {r.case_id: r for r in dg.manifest_oku().itertuples()}
    esik = pd.read_excel(XLSX, sheet_name='test_kosu').set_index(['mimari', 'tohum']).esik

    fn = {m: kur(m, 42) for m in MIMARILER}
    fig, ax = plt.subplots(len(v), 5, figsize=(15, 2.9 * len(v)))
    for i, vk in enumerate(v):
        r = lut[vk['case_id']]
        img, gt = dg.gri_oku(dg.yol_coz(r.image_path)), dg.gt_maskesi(r)
        katmanlar = [(img, 'orijinal'), (gt, 'ground truth')]
        for m in MIMARILER:
            t = esik[(m, 42)]
            pred = fn[m](r) >= t
            katmanlar.append((pred, f"{m}\nDice {dg.metrikler(pred, gt)['dice']:.3f}"))
        for j, (katman, alt) in enumerate(katmanlar):
            ax[i, j].imshow(img, cmap='gray')
            if j:
                ax[i, j].imshow(np.ma.masked_where(katman == 0, katman),
                                cmap='autumn', alpha=0.45)
            ax[i, j].set_title(alt if j else f"{vk['case_id']} [{vk['kova']}]",
                               fontsize=8)
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(FIG / 'vakalar.png', dpi=110, bbox_inches='tight')
    plt.close()
    print('  vakalar.png')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    print('karsilastirma figurleri:')
    tohum_varyansi()
    eslestirilmis_figur()
    boyut_figur()
    kalite_hiz()
    print('egitim egrileri:')
    egriler()
    print('vaka gorselleri:')
    vakalar()


if __name__ == '__main__':
    main()
