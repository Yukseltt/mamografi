"""ONNX export + tek calisma zamaninda mimari karsilastirmasi.

Neden: PyTorch olcumunde RTMDet'in gecikmesinin yarisi mmdet'in kare basina
Python yuku cikti (SegFormer'da %12, YOLO'da %33, RTMDet'te %50). "RTMDet 2.5 kat
yavas" cumlesi bu haliyle mimariyi degil cerceveyi olcuyor.

Uc agi da ONNX'e cevirip **ayni calisma zamaninda** (onnxruntime) olcunce cerceve
yuku ortadan kalkiyor ve geriye yalnizca mimarinin maliyeti kaliyor. Ayrica ONNX
gercek bir dagitim yolu: ultrason cihazinin yanindaki makinede PyTorch
kurulu olmak zorunda degil.

Export edilen sey **ag ileri gecisi**, son isleme degil. Son isleme (NMS, maske
birlestirme) cerceveye ozgu ve zaten ayri olculdu.

Sayisal dogruluk her model icin kontrol ediliyor: ONNX ciktisi PyTorch ciktisindan
sapiyorsa hiz olcumu anlamsiz olur.

Kullanim:
    python onnx_olcumu.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'rtmdet'))

import torch_uyum        # noqa: F401
import ozel_transformlar  # noqa: F401
import modeller as md

CIKTI_DIZIN = KOK / 'onnx'
ISINMA, TEKRAR = 20, 100
# ONNX vs PyTorch **bagil** sapma esigi. Mutlak sapma yanlis olcut: RTMDet'in
# bbox regresyon ciktisi 300'e kadar cikiyor, sinif ciktisi 5 civarinda -- ayni
# fp32 yuvarlamasi birinde 1e-3, digerinde 1e-5 mutlak fark uretiyor. Olculen
# bagil hata uc modelde de 2e-6..5e-6 araliginda; 1e-4 esigi yuvarlamaya yer
# birakirken gercekten yanlis bir grafigi (bagil hata O(1)) yakalar.
BAGIL_ESIK = 1e-4


def duzlestir(x):
    """Ic ice tuple/list -> tensor listesi. Uc cerceve uc farkli sekil donuyor."""
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, (list, tuple)):
        return [t for parca in x for t in duzlestir(parca)]
    return []


class SegformerSarma(torch.nn.Module):
    """HF modeli sozluk donuyor; ONNX icin tek tensor cikisina indiriliyor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


class RtmdetSarma(torch.nn.Module):
    """mmdet `_forward` ic ice tuple donuyor; duz bir listeye aciliyor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return tuple(duzlestir(self.model._forward(x)))


def modelleri_kur():
    from metrik_ayrismasi import kosu_bul
    from mmdet.apis import init_detector
    from ultralytics import YOLO

    kayit = []

    # Faz 2 checkpointleri (tohum 42); 512 satiri dusuruldu, calisma 256'da uc mimari.
    sf = md.segformer_kur(KOK / 'faz2_ckpt/segformer_s42.pt', 256, 'cuda:0')[2]
    kayit.append(('SegFormer-B0', 256, SegformerSarma(sf).eval()))

    y = YOLO(str(KOK / 'faz2_ckpt/yolo_s42.pt'))
    kayit.append(('YOLOv11n-Seg', 256, y.model.to('cuda').eval()))

    m = init_detector(*[str(v) for v in kosu_bul('rtmdet_ins_tiny_256_orta_s42')],
                      device='cuda:0')
    kayit.append(('RTMDet-Ins 256', 256, RtmdetSarma(m).eval()))
    return kayit


def disa_aktar(ad, model, girdi, yol):
    x = torch.randn(1, 3, girdi, girdi, device='cuda')
    with torch.no_grad():
        beklenen = model(x)
    torch.onnx.export(model, (x,), str(yol), input_names=['girdi'],
                      opset_version=17, dynamo=False, verbose=False)
    return x, beklenen


def ort_oturum(yol, saglayici):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(yol), o, providers=[saglayici])


def olc(oturum, girdi_adi, x_np):
    for _ in range(ISINMA):
        oturum.run(None, {girdi_adi: x_np})
    sure = []
    for _ in range(TEKRAR):
        t0 = time.perf_counter()
        oturum.run(None, {girdi_adi: x_np})
        sure.append((time.perf_counter() - t0) * 1000)
    return float(np.median(sure))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cikti', default=str(KOK / 'onnx_olcumu.xlsx'))
    args = ap.parse_args()
    CIKTI_DIZIN.mkdir(exist_ok=True)

    satir = []
    for ad, girdi, model in modelleri_kur():
        yol = CIKTI_DIZIN / f'{ad.replace(" ", "_").replace("-", "_")}.onnx'
        print(f'\n[{ad}] girdi {girdi}', flush=True)
        try:
            x, beklenen = disa_aktar(ad, model, girdi, yol)
        except Exception as e:
            print(f'  EXPORT BASARISIZ: {type(e).__name__}: {str(e)[:160]}', flush=True)
            satir.append(dict(model=ad, girdi=girdi, durum='export basarisiz'))
            continue
        print(f'  export edildi: {yol.stat().st_size/1e6:.1f} MB', flush=True)

        x_np = x.cpu().numpy()
        bek = [t.detach().cpu().numpy() for t in duzlestir(beklenen)]

        kayit = dict(model=ad, girdi=girdi, durum='ok',
                     torch_gpu_ms=round(md_saf(model, x), 2))
        for etiket, saglayici in [('onnx_gpu_ms', 'CUDAExecutionProvider'),
                                  ('onnx_cpu_ms', 'CPUExecutionProvider')]:
            try:
                oturum = ort_oturum(yol, saglayici)
                gelen = oturum.run(None, {oturum.get_inputs()[0].name: x_np})
                # ONNX cikis sayisi cerceveye gore degisebilir; ortak olanlar karsilastirilir
                ciftler = [(a, b) for a, b in zip(gelen, bek) if a.shape == b.shape]
                if not ciftler:
                    raise ValueError('ONNX ve PyTorch ciktilari eslesmedi')
                sapma = max(float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-9)
                            for a, b in ciftler)
                if etiket == 'onnx_gpu_ms':
                    kayit['bagil_sapma'] = float(f'{sapma:.1e}')
                    if sapma > BAGIL_ESIK:
                        raise ValueError(
                            f'ONNX ciktisi PyTorch`tan bagil {sapma:.1e} sapiyor')
                kayit[etiket] = round(olc(oturum, oturum.get_inputs()[0].name, x_np), 2)
                print(f'  {etiket}: {kayit[etiket]} ms', flush=True)
            except Exception as e:
                print(f'  {etiket} BASARISIZ: {type(e).__name__}: {str(e)[:140]}', flush=True)
                kayit[etiket] = None
        satir.append(kayit)
        del model
        torch.cuda.empty_cache()

    tablo = pd.DataFrame(satir)
    if 'onnx_gpu_ms' in tablo and 'torch_gpu_ms' in tablo:
        tablo['hizlanma'] = (tablo.torch_gpu_ms / tablo.onnx_gpu_ms).round(2)
    print('\n=== saf ag ileri gecisi: PyTorch vs ONNX (batch=1) ===')
    print(tablo.to_string(index=False))
    tablo.to_excel(args.cikti, index=False)
    print('\nkaydedildi:', args.cikti)


def md_saf(model, x):
    with torch.no_grad():
        for _ in range(ISINMA):
            model(x)
        torch.cuda.synchronize()
        sure = []
        for _ in range(TEKRAR):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            sure.append((time.perf_counter() - t0) * 1000)
    return float(np.median(sure))


if __name__ == '__main__':
    main()
