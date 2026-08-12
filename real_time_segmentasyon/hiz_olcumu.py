"""Gercek zamanli gecikme / FPS olcumu -- uc mimari, tek ortam, tek protokol.

PLAN.md karari: RTMDet yerelde torch 2.6'da, digerleri Colab'da torch 2.11'de
egitildi; farkli surumlerde olculen FPS karsilastirilabilir olmaz. Bu yuzden
**olcum uc model icin de burada**, yerel RTX 2060 uzerinde yapiliyor.

Neyi olcuyoruz: **kare basina uctan uca sure** -- on isleme (letterbox +
normalize) + ileri gecis + son isleme (maskeyi orijinal cozunurluge tasima).
Canli akista is yukunun tamami bu. Diskten PNG okuma **disarida**: canli akista
kare cihazdan gelir, dosya okuma modele ait bir maliyet degil.

Olculen kodun degerlendirilen kodla ayni oldugu `--dogrula` ile kontrol ediliyor:
`kare_fn` ile `harita_fn` ayni goruntude ayni haritayi uretmeli. Aksi halde
"su model hizli ama kalitesi su" cumlesi iki farkli seyi olcer.

Kullanim:
    python hiz_olcumu.py                    # GPU, val goruntuleri
    python hiz_olcumu.py --cpu              # CPU olcumu de ekle (yavas)
    python hiz_olcumu.py --n 30             # daha az goruntu
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'rtmdet'))

import torch_uyum        # noqa: F401
import ozel_transformlar  # noqa: F401
import degerlendirme as dg
import modeller as md

# Faz 2 checkpointleri (tohum 42). Gecikme agirliklara degil mimariye bagli,
# ama raporun butun sayilari ayni artefaktlardan gelsin diye Faz 2'ye cevrildi.
# 512 satiri dusuruldu: Faz 2 calismasi 256'da uc mimari uzerine kurulu.
MODELLER = {
    'SegFormer-B0': dict(tur='segformer', ckpt=KOK / 'faz2_ckpt/segformer_s42.pt',
                         girdi=256),
    'RTMDet-Ins 256': dict(tur='rtmdet', kosu='rtmdet_ins_tiny_256_orta_s42'),
    'YOLOv11n-Seg': dict(tur='yolo', ckpt=KOK / 'faz2_ckpt/yolo_s42.pt', girdi=256),
}
ISINMA = 15


def kur(cfg, cihaz):
    if cfg['tur'] == 'segformer':
        return md.segformer_kur(cfg['ckpt'], girdi=cfg['girdi'], cihaz=cihaz)
    if cfg['tur'] == 'yolo':
        yolo_cihaz = 0 if str(cihaz).startswith('cuda') else 'cpu'
        return md.yolo_kur(cfg['ckpt'], girdi=cfg['girdi'], cihaz=yolo_cihaz)
    from metrik_ayrismasi import kosu_bul
    return md.rtmdet_kur(*kosu_bul(cfg['kosu']), cihaz=cihaz)


def senkron(cihaz):
    import torch
    if str(cihaz).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def olc(kare_fn, kareler, cihaz):
    """Kare basina sure (ms). Isinma sonrasi, her kare tek tek senkronlanarak."""
    for i in range(ISINMA):
        kare_fn(*kareler[i % len(kareler)])
    senkron(cihaz)

    sureler = []
    for img, sekil in kareler:
        senkron(cihaz)
        t0 = time.perf_counter()
        kare_fn(img, sekil)
        senkron(cihaz)
        sureler.append((time.perf_counter() - t0) * 1000)
    return sureler


def saf_ileri(model, tur, girdi, n=100):
    """Yalnizca ag ileri gecisi -- on/son isleme ve cerceve yuku disarida.

    Uctan uca olcum RTMDet'te 256 ile 512'yi ayni gosterdi; girdi dort kat
    kucukken sure degismiyorsa darbogaz ag degil. Bu olcum farki ayristiriyor:
    "mimarinin maliyeti" ile "cercevenin kare basina ek yuku".
    """
    import torch
    x = torch.randn(1, 3, girdi, girdi, device='cuda')
    if tur == 'segformer':
        ileri = lambda t: model(pixel_values=t)
    elif tur == 'yolo':
        model.model.to('cuda').eval()
        ileri = lambda t: model.model(t)
    else:
        ileri = lambda t: model._forward(t)

    with torch.no_grad():
        for _ in range(20):
            ileri(x)
        torch.cuda.synchronize()
        sure = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ileri(x)
            torch.cuda.synchronize()
            sure.append((time.perf_counter() - t0) * 1000)
    return float(np.median(sure))


def tepe_vram(cihaz):
    import torch
    if not str(cihaz).startswith('cuda') or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=60, help='olcumde kullanilan goruntu sayisi')
    ap.add_argument('--cpu', action='store_true', help='CPU olcumunu de yap')
    ap.add_argument('--cpu-n', type=int, default=15)
    ap.add_argument('--dogrula', action='store_true', default=True)
    ap.add_argument('--cikti', default=str(KOK / 'hiz_olcumu.xlsx'))
    args = ap.parse_args()

    import torch
    print('cihaz:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
          '| torch', torch.__version__)

    alt = dg.manifest_oku(args.split).head(args.n)
    kareler = [(md._gri_bgr(r), tuple(md._gri_bgr(r).shape[:2])) for r in alt.itertuples()]
    print(f'{len(kareler)} kare, cozunurluk medyani '
          f'{int(np.median([s[0] for _, s in kareler]))}x'
          f'{int(np.median([s[1] for _, s in kareler]))}')

    satir = []
    for ad, cfg in MODELLER.items():
        print(f'\n[{ad}]', flush=True)
        kare_fn, harita_fn, model, bilgi = kur(cfg, 'cuda:0')

        if args.dogrula:
            # hizi olculen yol ile Dice'i olculen yol ayni sonucu vermeli
            r = next(alt.itertuples())
            a = kare_fn(*kareler[0])
            b = harita_fn(r)
            fark = float(np.abs(a - b).max())
            print(f'  kare_fn vs harita_fn max fark: {fark:.2e}')
            assert fark < 1e-5, (f'{ad}: hiz olculen yol degerlendirilen yoldan '
                                 'farkli sonuc veriyor')

        torch.cuda.reset_peak_memory_stats()
        gpu = olc(kare_fn, kareler, 'cuda:0')
        vram = tepe_vram('cuda:0')
        kayit = dict(model=ad, girdi=bilgi['girdi'],
                     parametre_M=round(bilgi['parametre'] / 1e6, 2),
                     gpu_ms_medyan=round(statistics.median(gpu), 2),
                     gpu_ms_ort=round(statistics.mean(gpu), 2),
                     gpu_ms_p95=round(float(np.percentile(gpu, 95)), 2),
                     gpu_fps=round(1000 / statistics.median(gpu), 1),
                     tepe_vram_MB=round(vram, 1) if vram else None)
        kayit['saf_ileri_ms'] = round(saf_ileri(model, cfg['tur'], bilgi['girdi']), 2)
        kayit['cerceve_yuku_ms'] = round(kayit['gpu_ms_medyan'] - kayit['saf_ileri_ms'], 2)
        kayit['cerceve_yuku_pct'] = round(
            100 * kayit['cerceve_yuku_ms'] / kayit['gpu_ms_medyan'], 1)
        print(f'  GPU {kayit["gpu_ms_medyan"]:.2f} ms/kare -> {kayit["gpu_fps"]} FPS '
              f'| tepe VRAM {kayit["tepe_vram_MB"]} MB', flush=True)
        print(f'  saf ag {kayit["saf_ileri_ms"]:.2f} ms | cerceve yuku '
              f'{kayit["cerceve_yuku_ms"]:.2f} ms (%{kayit["cerceve_yuku_pct"]})', flush=True)

        del kare_fn, harita_fn, model
        torch.cuda.empty_cache()

        if args.cpu:
            kare_fn_c, _, model_c, _ = kur(cfg, 'cpu')
            cpu = olc(kare_fn_c, kareler[:args.cpu_n], 'cpu')
            kayit['cpu_ms_medyan'] = round(statistics.median(cpu), 2)
            kayit['cpu_fps'] = round(1000 / statistics.median(cpu), 2)
            print(f'  CPU {kayit["cpu_ms_medyan"]:.1f} ms/kare -> '
                  f'{kayit["cpu_fps"]} FPS', flush=True)
            del kare_fn_c, model_c

        satir.append(kayit)

    tablo = pd.DataFrame(satir).sort_values('gpu_ms_medyan')
    print('\n=== hiz ozeti (batch=1, isinma sonrasi, uctan uca kare suresi) ===')
    print(tablo.to_string(index=False))
    tablo.to_excel(args.cikti, index=False)
    print('\nkaydedildi:', args.cikti)


if __name__ == '__main__':
    main()
