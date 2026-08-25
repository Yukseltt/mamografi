"""ONNX export + tek calisma zamaninda mimari karsilastirmasi (merdiven basamak 2, 3, 7).

Neden: PyTorch olcumunde gecikmenin bir kismi cerceve yuku (Python veri hatti, kernel
baslatma). BUSI projesinde bu yuk kalkinca mimari siralamasi TERSINE dondu. Burada da
ayni soru var: `unet3d` 2B modellerden hizli cikti, bu mimarinin mi yoksa 2B'nin dilim
basina Python yukunun mu sonucu?

Olculen sey **ag ileri gecisi**; on/son isleme cerceveye ozgu degil ve hiz_olcumu.py'de
zaten ayri olculdu.

Sayisal dogrulama her model icin yapiliyor: ONNX ciktisi PyTorch ciktisindan BAGIL olarak
sapiyorsa hiz olcumu anlamsiz olur. Mutlak sapma yanlis olcut - sigmoid oncesi logitler
genis araliga yayiliyor.

Kullanim:
    python onnx_olcumu.py
    python onnx_olcumu.py --fp16      # fp16 basamagini da ekle
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / 'ortak'))
import modeller as md                        # noqa: E402

# onnxruntime'in CUDA saglayicisi CUDA/cuBLAS DLL'lerini PATH'te ariyor. torch'un CUDA
# derlemesi bunlari kendi lib/ klasorunde tasiyor; arama yoluna eklenmezse ORT saglayiciyi
# yukleyemez ve SESSIZCE CPU'ya duser (ilk kosuda tam bu oldu).
_torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if hasattr(os, 'add_dll_directory') and os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)

DAGITIM = KOK / 'dagitim'
ONNX_DIR = KOK / 'onnx'
ISINMA, TEKRAR = 20, 50
# ONNX vs PyTorch bagil sapma tavani. fp16'nin kendi hassasiyeti ~1e-3 oldugu icin
# fp32 toleransi orada gecerli degil; asil olcut logit sapmasi degil **Dice'a etkisi**,
# o da hiz_olcumu.py --hassasiyet fp16 ile uctan uca olculuyor.
TOLERANS = {'fp32': 1e-3, 'fp16': 2e-2}


def ornek_girdi(konfig, cfg, batch):
    H, W = cfg['girdi_H'], cfg['girdi_W']
    if konfig['mimari'] == 'unet3d':
        return torch.randn(1, 2, H, W, 96)
    return torch.randn(batch, konfig['kanal'], H, W)


def disa_aktar(model, konfig, cfg, yol, fp16=False):
    ONNX_DIR.mkdir(exist_ok=True)
    cihaz = next(model.parameters()).device          # ornek girdi modelle ayni cihazda olmali
    x = ornek_girdi(konfig, cfg, konfig['batch']).to(cihaz)
    m = model
    if fp16:
        m = model.half()
        x = x.half()
    dinamik = None if konfig['mimari'] == 'unet3d' else {'girdi': {0: 'batch'},
                                                         'cikti': {0: 'batch'}}
    torch.onnx.export(m, (x,), str(yol), input_names=['girdi'], output_names=['cikti'],
                      dynamic_axes=dinamik, opset_version=17, dynamo=False)
    return x


def dogrula(model, onnx_yol, x, saglayici, beklenen=None):
    """ONNX ciktisi PyTorch ciktisiyla ayni mi - BAGIL sapma.

    Ayrica GERCEKTEN kullanilan saglayici kontrol ediliyor. onnxruntime istenen saglayici
    yuklenemezse SESSIZCE CPU'ya duser; ilk kosuda tam bu oldu (ORT 1.29 CUDA 13 istiyor,
    kurulu torch cu126) ve "ONNX PyTorch'tan 10 kat yavas" gibi anlamsiz sayilar uretti.
    """
    import onnxruntime as ort
    oturum = ort.InferenceSession(str(onnx_yol), providers=saglayici)
    kullanilan = oturum.get_providers()
    if beklenen and beklenen not in kullanilan:
        raise RuntimeError('istenen saglayici {} yuklenemedi, kullanilan: {}. '
                           'Olcum yapilmiyor - sessizce CPU"ya dusmus sayilar yaniltici olur.'
                           .format(beklenen, kullanilan))
    with torch.no_grad():
        ref = model(x).cpu().numpy()
    cik = oturum.run(None, {'girdi': x.cpu().numpy()})[0]
    bagil = np.abs(cik - ref).max() / (np.abs(ref).max() + 1e-12)
    return oturum, float(bagil)


def olc_ort(oturum, x_np):
    for _ in range(ISINMA):
        oturum.run(None, {'girdi': x_np})
    t0 = time.perf_counter()
    for _ in range(TEKRAR):
        oturum.run(None, {'girdi': x_np})
    return (time.perf_counter() - t0) / TEKRAR


def olc_torch(model, x):
    cuda = x.is_cuda
    with torch.no_grad():
        for _ in range(ISINMA):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TEKRAR):
            model(x)
        if cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / TEKRAR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp16', action='store_true')
    a = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit('onnxruntime yok. Kur: pip install onnx onnxruntime-gpu')

    if not DAGITIM.exists():
        sys.exit(str(DAGITIM) + ' yok - once Colab"daki "Parca 8 hazirligi" hucresini calistir.')

    cfg = json.load(open(DAGITIM / 'olcum_referansi.json', encoding='utf-8'))['cfg']
    mevcut = ort.get_available_providers()
    print('onnxruntime', ort.__version__, '| saglayicilar:', mevcut)
    cuda_var = 'CUDAExecutionProvider' in mevcut and torch.cuda.is_available()
    print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
    print()

    satir = []
    for pt in sorted(DAGITIM.glob('*_cikarim.pt')):
        mk = pt.stem.replace('_cikarim', '')
        for fp16 in ([False, True] if a.fp16 else [False]):
            if fp16 and not cuda_var:
                continue                      # fp16 CPU'da anlamli degil
            cihaz = 'cuda' if cuda_var else 'cpu'
            model, konfig, _ = md.model_yukle(pt, cihaz)
            etiket = 'fp16' if fp16 else 'fp32'
            yol = ONNX_DIR / (mk + '_' + etiket + '.onnx')
            x = disa_aktar(model, konfig, cfg, yol, fp16=fp16)
            model = model.to(cihaz)

            # PyTorch referansi (ayni girdi, ayni cihaz)
            torch_sn = olc_torch(model, x)

            for ep_ad, ep in ([('ONNX CUDA', ['CUDAExecutionProvider', 'CPUExecutionProvider'])]
                              if cuda_var else []) + [('ONNX CPU', ['CPUExecutionProvider'])]:
                if fp16 and ep_ad == 'ONNX CPU':
                    continue
                beklenen = 'CUDAExecutionProvider' if 'CUDA' in ep_ad else 'CPUExecutionProvider'
                try:
                    oturum, bagil = dogrula(model, yol, x, ep, beklenen)
                except Exception as e:
                    print('{:13s} {:5s} {:10s} ATLANDI: {}'.format(mk, etiket, ep_ad,
                                                                   str(e)[:110]))
                    continue
                sn = olc_ort(oturum, x.cpu().numpy())
                satir.append(dict(model=mk, hassasiyet=etiket, calisma_zamani=ep_ad,
                                  saglayici=oturum.get_providers()[0],
                                  batch=int(x.shape[0]), ms=sn * 1000,
                                  torch_ms=torch_sn * 1000, hizlanma=torch_sn / sn,
                                  bagil_sapma=bagil,
                                  parametre=md.parametre_sayisi(model),
                                  boyut_mb=yol.stat().st_size / 1e6))
                tol = TOLERANS[etiket]
                bayrak = '' if bagil <= tol else '  <-- SAPMA YUKSEK'
                print('{:14s} {:5s} {:10s} {:7.2f} ms  (torch {:7.2f})  hizlanma {:4.2f}x'
                      '  bagil sapma {:.2e}{}'.format(
                          mk, etiket, ep_ad, sn * 1000, torch_sn * 1000,
                          torch_sn / sn, bagil, bayrak))
            del model
            if cuda_var:
                torch.cuda.empty_cache()

    if not satir:
        sys.exit('olcum yapilamadi.')
    df = pd.DataFrame(satir)
    df['tolerans'] = df.hassasiyet.map(TOLERANS)
    yuksek = df[df.bagil_sapma > df.tolerans]
    print()
    print('=' * 78)
    print(df.to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))
    if len(yuksek):
        print()
        print('UYARI: {} olcumde bagil sapma toleransin ustunde - o satirlarin hiz'
              ' sayilari guvenilir degil.'.format(len(yuksek)))
    df.to_excel(KOK / 'onnx_olcumu.xlsx', index=False)
    print()
    print('yazildi: onnx_olcumu.xlsx |', ONNX_DIR)


if __name__ == '__main__':
    main()
