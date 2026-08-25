"""Cikarim motoru - ham NIfTI'den klinik ciktiya, tek sinif.

Parca 8'de merdivenin basamaklari ayri ayri olculmustu; burada hepsi tek hatta birlesiyor
ve **arka uc secilebilir** oluyor. Boylece "75 ms/vaka" sayisi bilesenlerden derlenmis
degil, tek kosuda dogrulanmis oluyor.

Arka uclar:
    torch-fp32   taban cizgisi
    torch-fp16   Parca 8 basamak 3
    onnx-cuda    Parca 8 basamak 2
    trt-fp16     Parca 8 basamak 4 - en hizlisi

Klinik cikti (PLAN.md Bolum 7): lezyon hacmi (mL), lezyon sayisi, en buyuk lezyon,
hemisfer. Her adimin suresi ayri tutuluyor - servis loglarinda darbogaz tahmin edilmeyecek.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / 'ortak'))

import hat            # noqa: E402
import hat_gpu        # noqa: E402
import modeller as md  # noqa: E402

from scipy import ndimage  # noqa: E402

D_HEDEF = 96
UYARI = ('Bu cikti bir tibbi tani araci degildir. Arastirma amaclidir ve '
         'uzman degerlendirmesinin yerine gecmez.')


def _trt_dll_yolu():
    """onnxruntime'in CUDA/TensorRT DLL'lerini bulabilmesi icin arama yolu."""
    if not hasattr(os, 'add_dll_directory'):
        return
    for d in (Path(torch.__file__).parent / 'lib',):
        if d.is_dir():
            os.add_dll_directory(str(d))
    try:
        import tensorrt_libs
        os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))
    except Exception:
        pass


class Motor:
    def __init__(self, model_key='unet3d', arka_uc='trt-fp16', cihaz='cuda',
                 dagitim=None, onnx_dizin=None):
        self.model_key = model_key
        self.arka_uc = arka_uc
        self.cihaz = cihaz if torch.cuda.is_available() else 'cpu'
        dagitim = Path(dagitim or KOK / 'dagitim')
        onnx_dizin = Path(onnx_dizin or KOK / 'onnx')

        ref = json.load(open(dagitim / 'olcum_referansi.json', encoding='utf-8'))
        self.cfg = ref['cfg']
        self.referans_dice = ref['referans_dice'].get(model_key, {})

        pt = dagitim / (model_key + '_cikarim.pt')
        self.model, self.konfig, self.secim = md.model_yukle(pt, self.cihaz)
        self.oturum = None

        if arka_uc == 'torch-fp16':
            self.model = self.model.half()
        elif arka_uc in ('onnx-cuda', 'trt-fp16'):
            _trt_dll_yolu()
            import onnxruntime as ort
            hassas = 'fp16' if arka_uc == 'trt-fp16' else 'fp32'
            yol = onnx_dizin / f'{model_key}_{hassas}.onnx'
            if not yol.exists():
                raise FileNotFoundError(f'{yol} yok - once onnx_olcumu.py calistirilmali')
            if arka_uc == 'trt-fp16':
                onbellek = onnx_dizin / 'trt_cache'
                onbellek.mkdir(exist_ok=True)
                saglayicilar = [('TensorrtExecutionProvider',
                                 {'trt_fp16_enable': True,
                                  'trt_engine_cache_enable': True,
                                  'trt_engine_cache_path': str(onbellek)}),
                                'CUDAExecutionProvider', 'CPUExecutionProvider']
                beklenen = 'TensorrtExecutionProvider'
            else:
                saglayicilar = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                beklenen = 'CUDAExecutionProvider'
            self.oturum = ort.InferenceSession(str(yol), providers=saglayicilar)
            kullanilan = self.oturum.get_providers()[0]
            if kullanilan != beklenen:
                # Sessiz CPU'ya dusus Parca 8'de anlamsiz sayilar uretmisti - burada hata.
                raise RuntimeError(f'{beklenen} yuklenemedi, kullanilan: {kullanilan}')
            self.onnx_dtype = np.float16 if hassas == 'fp16' else np.float32
        elif arka_uc != 'torch-fp32':
            raise ValueError(arka_uc)

    def _girdi_gpu(self, xt):
        """GPU'daki (2,h,w,d) tensoru dogrudan model girdisine cevirir.

        Aksi halde on isleme GPU'da bitip CPU'ya iniyor, sonra ag icin tekrar GPU'ya
        cikiyordu - ayni veriyi iki kez PCIe uzerinden tasimak. Yalniz 3B modelde
        uygulanabiliyor (dolgu tek islem); 2B'de dilim yiginlama numpy tarafinda.
        """
        H, W = self.cfg['girdi_H'], self.cfg['girdi_W']
        h, w, d = xt.shape[1], xt.shape[2], xt.shape[3]
        o = [(H - h) // 2, (W - w) // 2, (D_HEDEF - d) // 2]
        # F.pad son eksenden basa dogru: (d_on, d_son, w_on, w_son, h_on, h_son)
        pad = (o[2], D_HEDEF - d - o[2], o[1], W - w - o[1], o[0], H - h - o[0])
        return torch.nn.functional.pad(xt, pad)[None], (h, w, d)

    # --- ag ileri gecisi, arka uca gore ---
    def _ag(self, girdi):
        if torch.is_tensor(girdi):                    # GPU'da kalan yol
            t = girdi.half() if self.arka_uc == 'torch-fp16' else girdi.float()
            with torch.no_grad():
                p = torch.sigmoid(self.model(t))[0, 0]
            return p.float().cpu().numpy()

        if self.oturum is not None:
            x = girdi.astype(self.onnx_dtype)
            cikti = []
            b = 1 if self.konfig['mimari'] == 'unet3d' else self.konfig['batch']
            for i in range(0, len(x), b):
                parca = x[i:i + b]
                if len(parca) < b and self.konfig['mimari'] != 'unet3d':
                    # ONNX grafigi sabit batch ile disa aktarildi; son parcayi doldur
                    dolgu = np.repeat(parca[-1:], b - len(parca), axis=0)
                    cikti.append(self.oturum.run(
                        None, {'girdi': np.concatenate([parca, dolgu])})[0][:len(parca)])
                else:
                    cikti.append(self.oturum.run(None, {'girdi': parca})[0])
            logit = np.concatenate(cikti).astype(np.float32)
            olasilik = 1.0 / (1.0 + np.exp(-logit))
            # 3B: (1,1,H,W,D) -> (H,W,D) | 2B: (N,1,H,W) -> (N,H,W)
            return olasilik[0, 0] if self.konfig['mimari'] == 'unet3d' else olasilik[:, 0]

        t = torch.from_numpy(girdi).to(self.cihaz)
        if self.arka_uc == 'torch-fp16':
            t = t.half()
        with torch.no_grad():
            if self.konfig['mimari'] == 'unet3d':
                p = torch.sigmoid(self.model(t))[0, 0]
            else:
                p = torch.cat([torch.sigmoid(self.model(t[i:i + self.konfig['batch']]))[:, 0]
                               for i in range(0, len(t), self.konfig['batch'])])
        return p.float().cpu().numpy()

    # --- klinik cikti ---
    def klinik(self, maske, spacing=None):
        spacing = spacing or tuple(self.cfg['hedef_spacing'])
        voxel_ml = float(np.prod(spacing)) / 1000.0
        et, n = ndimage.label(maske, np.ones((3, 3, 3)))
        boyut = np.bincount(et.ravel())[1:] if n else np.array([])
        hacim = float(maske.sum()) * voxel_ml

        hemisfer, oran_sag = None, None
        if n:
            # resample_to_output cikti grid'ini kanonik yone getiriyor: eksen 0 sol->sag.
            merkez_lezyon = float(np.nonzero(maske)[0].mean())
            merkez_hacim = maske.shape[0] / 2.0
            sag = np.nonzero(maske)[0] > merkez_hacim
            oran_sag = float(sag.mean())
            if oran_sag > 0.8:
                hemisfer = 'sag'
            elif oran_sag < 0.2:
                hemisfer = 'sol'
            else:
                hemisfer = 'bilateral'
            del merkez_lezyon
        return dict(
            hacim_ml=round(hacim, 3),
            lezyon_sayisi=int(n),
            en_buyuk_lezyon_ml=round(float(boyut.max()) * voxel_ml, 3) if n else 0.0,
            hemisfer=hemisfer,
            sag_hemisfer_orani=round(oran_sag, 3) if oran_sag is not None else None,
            voxel=int(maske.sum()),
            uyari=UYARI,
        )

    def isit(self, tekrar=2):
        """Sentetik veriyle TUM hatti isitir - on isleme dahil.

        Ilk cagride CUDA baglami kurulmasi ve cekirdek secimi gecikmeyi kat kat
        artiriyor. Olculdu: isitmasiz ilk istek 331 ms (ag 168 + on isleme 157),
        kararli hal 102 ms. Yalniz agi isitmak yetmiyordu - grid_sample ve max_pool3d
        cekirdekleri de ilk cagrilarinda yavas. Isitma olmadan bu bedeli ILK HASTA oder.
        """
        import nibabel as nib

        H, W = self.cfg['girdi_H'], self.cfg['girdi_W']
        # gercek hacimlere yakin sentetik NIfTI (ISLES ortalamasi ~112x112x73, 2 mm)
        rng = np.random.default_rng(0)
        hacim = (rng.random((112, 112, 73)).astype(np.float32) * 500)
        hacim[20:90, 20:90, 10:60] += 400          # beyin maskesi bos kalmasin
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        im = nib.Nifti1Image(hacim, affine)
        ham = (im, im, hacim, hacim)

        for _ in range(tekrar):
            if self.cihaz == 'cuda':
                xt, _ = hat_gpu.on_isle_gpu(ham, self.cfg, self.cihaz)
                if self.oturum is None and self.konfig['mimari'] == 'unet3d':
                    girdi, _ = self._girdi_gpu(xt)
                else:
                    girdi, _ = hat.girdi_tensoru(xt.cpu().numpy(), self.konfig,
                                                 self.cfg, D_HEDEF)
            else:
                x, _ = hat.on_isle(ham, self.cfg)
                girdi, _ = hat.girdi_tensoru(x, self.konfig, self.cfg, D_HEDEF)
            self._ag(girdi)
        if self.cihaz == 'cuda':
            torch.cuda.synchronize()
        return self

    # --- tam hat ---
    def calistir(self, dwi_yol, adc_yol, on_isleme='gpu'):
        """Ham NIfTI -> maske + klinik cikti + bilesen bazli sure."""
        sure = {}
        ham, sure['okuma_sn'] = hat.oku(dwi_yol, adc_yol)

        if self.cihaz == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_yolu = (on_isleme == 'gpu' and self.cihaz == 'cuda'
                    and self.oturum is None and self.konfig['mimari'] == 'unet3d')
        if on_isleme == 'gpu' and self.cihaz == 'cuda':
            xt, dilim = hat_gpu.on_isle_gpu(ham, self.cfg, self.cihaz)
            if gpu_yolu:
                girdi, sekil = self._girdi_gpu(xt)      # CPU'ya hic inmiyor
            else:
                x = xt.cpu().numpy()
                girdi, sekil = hat.girdi_tensoru(x, self.konfig, self.cfg, D_HEDEF)
        else:
            x, dilim = hat.on_isle(ham, self.cfg)
            girdi, sekil = hat.girdi_tensoru(x, self.konfig, self.cfg, D_HEDEF)
        if self.cihaz == 'cuda':
            torch.cuda.synchronize()
        sure['on_isleme_sn'] = time.perf_counter() - t0

        t1 = time.perf_counter()
        olasilik = self._ag(girdi)
        if self.cihaz == 'cuda':
            torch.cuda.synchronize()
        sure['ag_sn'] = time.perf_counter() - t1

        t2 = time.perf_counter()
        maske = hat.son_isle(olasilik, sekil, self.konfig, self.cfg, D_HEDEF,
                             self.secim['esik'], self.secim['min_voxel'])
        k = self.klinik(maske)
        sure['son_isleme_sn'] = time.perf_counter() - t2

        sure['toplam_sn'] = sure['on_isleme_sn'] + sure['ag_sn'] + sure['son_isleme_sn']
        return dict(maske=maske, kirpma=dilim, klinik=k, sure=sure,
                    model=self.model_key, arka_uc=self.arka_uc, esik=self.secim['esik'])
