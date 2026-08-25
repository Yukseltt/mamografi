"""Rapor figurleri - yerel olcum dosyalarindan uretilenler.

Drive'dan gelen figurler rapor/girdi/ altinda; burada uretilenler gecikme tarafina ait
ve yalnizca yerel xlsx dosyalarina dayaniyor.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
CIKTI = Path(__file__).resolve().parent / 'figurler'
CIKTI.mkdir(exist_ok=True)

RENK = {'on_isleme': '#4c72b0', 'ag': '#dd8452', 'son_isleme': '#55a868'}


def merdiven():
    """Gecikme merdiveni: hangi basamak neyi kazandirdi."""
    hiz = pd.read_excel(KOK / 'hiz_olcumu.xlsx', sheet_name='ozet')
    srv = pd.read_excel(KOK / 'servis' / 'servis_dogrulama.xlsx')

    cpu = hiz[(hiz.model == 'unet3d') & (hiz.onisleme == 'cpu') &
              (hiz.hassasiyet == 'fp32')].iloc[0]
    gpu = hiz[(hiz.model == 'unet3d') & (hiz.onisleme == 'gpu') &
              (hiz.hassasiyet == 'fp32')].iloc[0]
    f16 = srv[srv.arka_uc == 'torch-fp16'].iloc[0]

    adlar = ['PyTorch fp32\nCPU on isleme', 'PyTorch fp32\nGPU on isleme',
             'PyTorch fp16\nGPU on isleme']
    on = [cpu.on_isleme_ms, gpu.on_isleme_ms, f16.on_isleme_ms]
    ag = [cpu.ag_ms, gpu.ag_ms, f16.ag_ms]
    son = [cpu.son_isleme_ms, gpu.son_isleme_ms, f16.son_isleme_ms]

    fig, a = plt.subplots(figsize=(8.4, 4.0))
    x = np.arange(len(adlar))
    a.bar(x, on, 0.55, label='on isleme', color=RENK['on_isleme'])
    a.bar(x, ag, 0.55, bottom=on, label='ag', color=RENK['ag'])
    a.bar(x, son, 0.55, bottom=np.array(on) + np.array(ag), label='son isleme',
          color=RENK['son_isleme'])
    for i, t in enumerate(np.array(on) + np.array(ag) + np.array(son)):
        a.text(i, t + 6, f'{t:.0f} ms', ha='center', fontsize=10)
    a.axhline(1000, color='#c0392b', ls='--', lw=1.2)
    a.text(len(adlar) - 0.45, 1010, 'hedeflenen ust sinir (1 s)', color='#c0392b',
           fontsize=8.5, ha='right')
    a.set_xticks(x)
    a.set_xticklabels(adlar, fontsize=9)
    a.set_ylabel('vaka basina sure (ms)')
    a.set_ylim(0, 1120)
    a.legend(fontsize=9, loc='upper right')
    a.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CIKTI / 'gecikme_merdiveni.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    return 'gecikme_merdiveni.png'


def arka_uc_karsilastirma():
    """Dort arka ucun uctan uca suresi - izole ag olcumunun aksine."""
    srv = pd.read_excel(KOK / 'servis' / 'servis_dogrulama.xlsx')
    srv = srv[srv.durum == 'OK'].sort_values('toplam_ms')

    # Izole ag sureleri olcum dosyalarindan; sabit yazmak iki olcum oturumunu
    # birbirine karistirir.
    onnx = pd.read_excel(KOK / 'onnx_olcumu.xlsx')
    trt = pd.read_excel(KOK / 'tensorrt_olcumu.xlsx')
    o3 = onnx[onnx.model == 'unet3d']
    t3 = trt[(trt.model == 'unet3d') & (trt.durum == 'OK')]
    izole = {
        'torch-fp32': float(o3[o3.hassasiyet == 'fp32'].torch_ms.iloc[0]),
        'torch-fp16': float(o3[o3.hassasiyet == 'fp16'].torch_ms.iloc[0]),
        'onnx-cuda': float(o3[(o3.hassasiyet == 'fp32') &
                              (o3.calisma_zamani == 'ONNX CUDA')].ms.iloc[0]),
        'trt-fp16': float(t3[(t3.hassasiyet == 'fp16') &
                             (t3.calisma_zamani == 'TensorRT')].ms.iloc[0]),
    }

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    y = np.arange(len(srv))
    a1.barh(y, srv.on_isleme_ms, color=RENK['on_isleme'], label='on isleme')
    a1.barh(y, srv.ag_ms, left=srv.on_isleme_ms, color=RENK['ag'], label='ag')
    a1.barh(y, srv.son_isleme_ms, left=srv.on_isleme_ms + srv.ag_ms,
            color=RENK['son_isleme'], label='son isleme')
    for i, t in enumerate(srv.toplam_ms):
        a1.text(t + 3, i, f'{t:.0f}', va='center', fontsize=9)
    a1.set_yticks(y)
    a1.set_yticklabels(srv.arka_uc, fontsize=9)
    a1.set_xlabel('uctan uca sure (ms)')
    a1.set_xlim(0, srv.toplam_ms.max() * 1.18)
    a1.legend(fontsize=8, loc='lower right')
    a1.set_title('Servis hatti, vaka basina', fontsize=10)
    a1.grid(axis='x', alpha=0.3)

    ad = list(srv.arka_uc)
    izo = [izole[k] for k in ad]
    gercek = list(srv.ag_ms)
    x = np.arange(len(ad))
    a2.bar(x - 0.19, izo, 0.38, label='izole olcum', color='#8172b3')
    a2.bar(x + 0.19, gercek, 0.38, label='servis hatti', color='#937860')
    a2.set_xticks(x)
    a2.set_xticklabels(ad, fontsize=8.5, rotation=12)
    a2.set_ylabel('ag suresi (ms)')
    a2.legend(fontsize=8)
    a2.set_title('Ag suresi: izole olcum vs gercek hat', fontsize=10)
    a2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CIKTI / 'arka_uc.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    return 'arka_uc.png'


AD = {'unet3d': '3D U-Net', 'unet_r34_2d': 'U-Net + ResNet34 (2D)',
      'unet_r34_25d': 'U-Net + ResNet34 (2.5D)', 'segformer_b0': 'SegFormer-B0',
      'unet_mbv3': 'U-Net + MobileNetV3', 'A1_flair': 'A1: +FLAIR',
      'A2_tversky': 'A2: focal-Tversky', 'A3_k1': 'A3: k=1', 'A3_k3': 'A3: k=3'}


def dice_boyut(karsilastirma_yolu=None):
    """Kalite ve model boyutu.

    Ilk surum log eksenli sacilim grafigiydi: dokuz modelin besi 24,4 M parametrede
    ust uste biniyor, etiketler cakisiyor ve eksen disina tasiyordu. Yatay nokta
    grafigi ayni bilgiyi cakismadan veriyor; parametre sayisi ikinci panelde.
    """
    k = karsilastirma_yolu or (Path(__file__).resolve().parent / 'girdi' /
                               'karsilastirma_tablosu.xlsx')
    if not k.exists():
        return None
    df = pd.read_excel(k).sort_values('test_dice')
    taban = float(df[df.model_key == 'unet_r34_2d'].test_dice.iloc[0])
    ad = [AD.get(m, m) for m in df.model_key]
    y = np.arange(len(df))
    aday = df.model_key.isin(['unet3d', 'unet_r34_2d']).values
    renk = np.where(aday, '#c44e52', '#4c72b0')

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                 gridspec_kw={'width_ratios': [2.3, 1]})

    a1.axvspan(taban - 0.024, taban + 0.024, color='#b0b0b0', alpha=0.22, zorder=0,
               label='koşu varyansı bandı (±0,024)')
    a1.axvline(taban, color='#c44e52', ls='--', lw=1.0, zorder=1, alpha=0.7)
    a1.hlines(y, df.test_dice.min() - 0.012, df.test_dice, color='#d0d5dd', lw=1.0, zorder=1)
    a1.scatter(df.test_dice, y, s=68, color=renk, zorder=3, edgecolor='white', linewidth=0.9)
    for i, v in enumerate(df.test_dice):
        a1.text(v + 0.0035, i, f'{v:.3f}', va='center', fontsize=8.4)
    a1.set_yticks(y)
    a1.set_yticklabels(ad, fontsize=9)
    a1.set_xlabel('test Dice')
    a1.set_xlim(df.test_dice.min() - 0.012, df.test_dice.max() + 0.016)
    a1.legend(fontsize=8, loc='lower right', framealpha=0.95)
    a1.grid(axis='x', alpha=0.25)
    a1.set_title('Kalite', fontsize=10)

    a2.barh(y, df.parametre / 1e6, height=0.55, color=renk, alpha=0.85)
    for i, v in enumerate(df.parametre / 1e6):
        a2.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=8.4)
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.set_xlabel('parametre (milyon)')
    a2.set_xlim(0, float(df.parametre.max()) / 1e6 * 1.22)
    a2.grid(axis='x', alpha=0.25)
    a2.set_title('Boyut', fontsize=10)

    plt.tight_layout()
    plt.savefig(CIKTI / 'dice_boyut.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    return 'dice_boyut.png'


if __name__ == '__main__':
    for fn in (merdiven, arka_uc_karsilastirma, dice_boyut):
        ad = fn()
        print(('uretildi: ' + ad) if ad else f'atlandi: {fn.__name__} (girdi yok)')
