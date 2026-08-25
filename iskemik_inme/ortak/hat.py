"""Cikarim hatti - ham NIfTI'den maskeye, bilesenleri ayri ayri olculebilir sekilde.

Parca 2'deki on isleme ile BIREBIR ayni sira ve parametreler; tek fark burada
her adimin suresi ayri tutuluyor. Bolum 3 protokolu: diskten okuma disarida
(PACS'tan gelen hacim modelin maliyeti degil) ama ayri kolonda raporlaniyor.
"""
import time

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy import ndimage

BAGLANTI = np.ones((3, 3, 3))


def beyin_maskesi(dw, esik_carpani=0.05):
    b = dw > np.percentile(dw, 99) * esik_carpani
    b = ndimage.binary_closing(b, np.ones((3, 3, 3)))
    b = ndimage.binary_fill_holes(b)
    et, n = ndimage.label(b)
    if n > 1:
        b = et == (np.bincount(et.ravel())[1:].argmax() + 1)
    return b


def adc_carpan(p999):
    """Parca 2 olcumu: veri uc ayri birimde geliyor (mm2/s, x10-3, x10-6)."""
    if p999 < 0.05:
        return 1e6
    if p999 < 50:
        return 1e3
    return 1.0


def dolgula_2b(a, H, W):
    h, w = a.shape[-2:]
    ust, sol = (H - h) // 2, (W - w) // 2
    pad = [(0, 0)] * (a.ndim - 2) + [(ust, H - h - ust), (sol, W - w - sol)]
    return np.pad(a, pad)


def dolgula_3b(a, H, W, D):
    h, w, d = a.shape[-3:]
    o = [(H - h) // 2, (W - w) // 2, (D - d) // 2]
    pad = [(0, 0)] * (a.ndim - 3) + [(o[0], H - h - o[0]), (o[1], W - w - o[1]),
                                     (o[2], D - d - o[2])]
    return np.pad(a, pad)


def bilesen_ele(maske, min_voxel):
    if min_voxel <= 0:
        return maske
    et, n = ndimage.label(maske, BAGLANTI)
    if n == 0:
        return maske
    boyut = np.bincount(et.ravel())
    kucuk = np.where(boyut < min_voxel)[0]
    return maske & ~np.isin(et, kucuk[kucuk > 0])


def oku(dwi_yol, adc_yol):
    """Diskten okuma - BOLUM 3 protokolunde gecikme butcesinin DISINDA, ayri olculuyor."""
    t0 = time.perf_counter()
    dwi_im, adc_im = nib.load(dwi_yol), nib.load(adc_yol)
    dw = np.asarray(dwi_im.dataobj).astype(np.float32)
    ad = np.asarray(adc_im.dataobj).astype(np.float32)
    return (dwi_im, adc_im, dw, ad), time.perf_counter() - t0


def on_isle(ham, cfg):
    """resample -> beyin maskesi -> kirpma -> normalizasyon. Parca 2 ile ayni.

    Not: resample_to_output'a dogrudan nibabel goruntusu verilirse `.dataobj` tembel
    oldugu icin hacim DISKTEN TEKRAR okunur - `oku()` zaten okumus olmasina ragmen.
    Olcumde bu, on islemeye ait olmayan bir disk maliyetini iceri sokuyordu ve GPU
    yoluyla karsilastirmayi haksiz kiliyordu (GPU yolu zaten bellekteki diziyi
    kullaniyor). Bellekteki dizilerden yeni goruntu nesnesi kuruluyor.
    """
    dwi_im, adc_im, dw_ham, ad_ham = ham
    hs = tuple(cfg['hedef_spacing'])
    dwi_b = nib.Nifti1Image(dw_ham, dwi_im.affine, dwi_im.header)
    adc_b = nib.Nifti1Image(ad_ham, adc_im.affine, adc_im.header)
    dw = np.asarray(resample_to_output(dwi_b, hs, order=1).dataobj).astype(np.float32)
    ad = np.asarray(resample_to_output(adc_b, hs, order=1).dataobj).astype(np.float32)

    b = beyin_maskesi(dw, cfg['beyin_esik_carpani'])
    idx = np.where(b)
    m = cfg['kirpma_marji']
    dilim = tuple(slice(max(int(i.min()) - m, 0), min(int(i.max()) + 1 + m, s))
                  for i, s in zip(idx, dw.shape))
    dw, ad, b = dw[dilim], ad[dilim], b[dilim]

    ad = ad * adc_carpan(float(np.percentile(ad[b], 99.9)))
    ad = np.clip(ad, 0, cfg['adc_klip']) / cfg['adc_klip']
    ort, std = float(dw[b].mean()), float(dw[b].std()) + 1e-6
    dw = np.clip((dw - ort) / std, -cfg['dwi_z_klip'], cfg['dwi_z_klip'])
    dw[~b] = 0.0
    ad[~b] = 0.0
    return np.stack([dw, ad]).astype(np.float32), dilim


def gt_hazirla(msk_im, cfg, dilim):
    '''GT maskeyi tahminle AYNI uzaya sokar: nearest resample + ayni kirpma.

    Parca 2'de `_y.npy` boyle uretilmisti; yerel olcumun Dice'i notebook'un referansiyla
    karsilastirilabilsin diye burada birebir tekrarlaniyor. Bu adim gecikme butcesinin
    DISINDA - cikarimda GT yok.
    '''
    m = np.asarray(resample_to_output(msk_im, tuple(cfg['hedef_spacing']), order=0).dataobj)
    return (m > 0.5)[dilim]


def girdi_tensoru(x, konfig, cfg, D_HEDEF):
    """(2,h,w,d) -> modele verilecek numpy dizi + geri alma bilgisi."""
    H, W = cfg['girdi_H'], cfg['girdi_W']
    h, w, D = x.shape[1], x.shape[2], x.shape[3]
    if konfig['mimari'] == 'unet3d':
        return dolgula_3b(x, H, W, D_HEDEF)[None], (h, w, D)
    k = konfig['k'] or 0
    yigin = []
    for z in range(D):
        zs = np.clip(np.arange(z - k, z + k + 1), 0, D - 1)
        img = x[:, :, :, zs].transpose(1, 2, 0, 3).reshape(h, w, -1)
        yigin.append(dolgula_2b(img.transpose(2, 0, 1), H, W))
    return np.stack(yigin), (h, w, D)


def son_isle(olasilik, sekil, konfig, cfg, D_HEDEF, esik, min_voxel):
    """Dolguyu geri al -> esikle -> kucuk bileseni ele."""
    h, w, D = sekil
    H, W = cfg['girdi_H'], cfg['girdi_W']
    if konfig['mimari'] == 'unet3d':
        o = [(H - h) // 2, (W - w) // 2, (D_HEDEF - D) // 2]
        p = olasilik[o[0]:o[0] + h, o[1]:o[1] + w, o[2]:o[2] + D]
    else:
        ust, sol = (H - h) // 2, (W - w) // 2
        p = olasilik[:, ust:ust + h, sol:sol + w].transpose(1, 2, 0)
    return bilesen_ele(p >= esik, min_voxel)


def dice(tahmin, gt):
    tp = float(np.logical_and(tahmin, gt).sum())
    payda = float(tahmin.sum()) + float(gt.sum())
    if payda == 0:
        return 1.0
    return 2 * tp / payda
