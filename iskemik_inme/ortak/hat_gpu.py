"""GPU'ya tasinmis on isleme - Parca 8 merdiveninin yeni basamaklari.

Neden: yerel RTX 2060 on olcumunde uctan uca 313 ms'nin **%75'i on isleme**, ag yalniz
%23. Kirilim (10 vaka): beyin maskesi 77 ms, resample DWI 72 ms, resample ADC 71 ms,
normalizasyon+kirpma 9 ms. Ucu de CPU'da scipy/nibabel ile kosuyordu.

Burada:
  - `resample_gpu`   : nibabel'in resample_to_output'unun grid_sample karsiligi.
                       Cikti grid'i (sekil + affine) yine nibabel'in `vox2out_vox`'undan
                       aliniyor, yani hedef grid BIREBIR ayni; degisen yalniz orneklemenin
                       nerede yapildigi.
  - `beyin_maskesi_gpu`: esikleme + morfolojik kapama GPU'da (max_pool3d/min_pool3d).
                       Delik doldurma ve en buyuk bilesen **alt orneklenmis** maskede
                       CPU'da - bu iki adim yalniz kirpma kutusunu ve normalizasyon
                       istatistiklerini etkiliyor, ikisi de ince ayrintiya duyarsiz.

Her fonksiyonun CPU karsiligiyla ayni sonucu urettigi `dogrula.py` ile olculuyor -
hizlandirma sonucu degistiriyorsa kazanc degil hata olur.
"""
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import vox2out_vox
from scipy import ndimage


def cikti_grid(im, voxel_sizes):
    """nibabel ile AYNI cikti grid'i: (sekil, affine)."""
    return vox2out_vox((im.shape, im.affine), voxel_sizes)


def _ornekleme_grid(in_shape, M, out_shape, cihaz, dtype=torch.float32):
    """grid_sample icin normalize koordinat izgarasi.

    M: 4x4, cikti voxel koordinatindan girdi voxel koordinatina donusum
       (inv(in_affine) @ out_affine).
    grid_sample (N,C,D,H,W) girdisinde grid'in son ekseni (x, y, z) sirasinda ve
    sirasiyla W, H, D eksenlerine karsilik geliyor - eksen sirasi burada ters cevriliyor.
    """
    d, h, w = out_shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=cihaz, dtype=dtype),
        torch.arange(h, device=cihaz, dtype=dtype),
        torch.arange(w, device=cihaz, dtype=dtype), indexing='ij')
    M = torch.as_tensor(M, device=cihaz, dtype=dtype)
    i0 = M[0, 0]*zz + M[0, 1]*yy + M[0, 2]*xx + M[0, 3]
    i1 = M[1, 0]*zz + M[1, 1]*yy + M[1, 2]*xx + M[1, 3]
    i2 = M[2, 0]*zz + M[2, 1]*yy + M[2, 2]*xx + M[2, 3]
    # voxel -> [-1, 1]; align_corners=True ile 0..N-1 araligi tam [-1,1]'e esleniyor
    n0, n1, n2 = (max(s - 1, 1) for s in in_shape)
    g = torch.stack([2 * i2 / n2 - 1, 2 * i1 / n1 - 1, 2 * i0 / n0 - 1], dim=-1)

    # Kenar davranisi: scipy affine_transform(mode='constant', cval=0) girdi sinirinin
    # DISINDAKI her noktaya 0 verir; grid_sample ise kenarda ic degerle sifiri harmanlar.
    # Oblik affine'li vakalarda bu, hacim sinirinda binde bir voxelde yuzlerce birimlik
    # fark uretiyordu. Gecerlilik maskesi iki davranisi birebir esitliyor.
    gecerli = ((i0 >= 0) & (i0 <= in_shape[0] - 1) &
               (i1 >= 0) & (i1 <= in_shape[1] - 1) &
               (i2 >= 0) & (i2 <= in_shape[2] - 1))
    return g[None], gecerli


def resample_gpu(im, voxel_sizes, cihaz='cuda', veri=None):
    """resample_to_output'un GPU karsiligi. veri verilirse tekrar okunmaz."""
    out_shape, out_affine = cikti_grid(im, voxel_sizes)
    M = np.linalg.inv(im.affine) @ out_affine
    x = veri if veri is not None else np.asarray(im.dataobj)
    t = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=cihaz)[None, None]
    grid, gecerli = _ornekleme_grid(im.shape[:3], M, out_shape, cihaz)
    y = F.grid_sample(t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return y[0, 0] * gecerli, out_shape, out_affine


def _kapama_gpu(b):
    """binary_closing(3x3x3) = dilate sonra erode. max_pool3d / -max_pool3d(-x)."""
    x = b[None, None].float()
    x = F.max_pool3d(x, 3, stride=1, padding=1)                 # dilate
    x = -F.max_pool3d(-x, 3, stride=1, padding=1)               # erode
    return x[0, 0] > 0.5


def beyin_maskesi_gpu(dw, esik_carpani=0.05):
    """Esikleme + morfolojik kapama GPU'da; delik doldurma ve en buyuk bilesen CPU'da.

    Delik doldurma ve bagli bilesen etiketleme GPU'da ucuz karsiligi olmayan iki islem;
    TAM cozunurlukte CPU'da birakiliyor. Sonuc CPU surumuyle **bit duzeyinde ayni**
    (12 vakada 0 farkli voxel olculdu), maliyet 74 -> 40 ms.

    Ilk denemede bu iki adim 2 kat alt orneklenmis maskede yapiliyordu (8 kat az voxel);
    daha hizliydi ama maskeyi degistirip kirpma kutusunu 1 voxel kaydiriyordu ve cikti
    CPU surumuyle uyusmuyordu. Hiz ugruna sonucu degistirmek kazanc degil hata olur.
    """
    esik = torch.quantile(dw.flatten().float(), 0.99) * esik_carpani
    b = _kapama_gpu(dw > esik).cpu().numpy()
    b = ndimage.binary_fill_holes(b)
    et, n = ndimage.label(b)
    if n > 1:
        b = et == (np.bincount(et.ravel())[1:].argmax() + 1)
    return torch.as_tensor(b, device=dw.device)


def adc_carpan_t(p999):
    if p999 < 0.05:
        return 1e6
    if p999 < 50:
        return 1e3
    return 1.0


def on_isle_gpu(ham, cfg, cihaz='cuda'):
    """hat.on_isle ile ayni cikti, adimlar GPU'da. (2, h, w, d) float32 numpy dondurur."""
    dwi_im, adc_im, dw_ham, ad_ham = ham
    hs = tuple(cfg['hedef_spacing'])
    dw, _, _ = resample_gpu(dwi_im, hs, cihaz, veri=dw_ham)
    ad, _, _ = resample_gpu(adc_im, hs, cihaz, veri=ad_ham)

    b = beyin_maskesi_gpu(dw, cfg['beyin_esik_carpani'])
    idx = torch.nonzero(b, as_tuple=True)
    m = cfg['kirpma_marji']
    dilim = tuple(slice(max(int(i.min()) - m, 0), min(int(i.max()) + 1 + m, s))
                  for i, s in zip(idx, b.shape))
    dw, ad, b = dw[dilim], ad[dilim], b[dilim]

    ad = ad * adc_carpan_t(float(torch.quantile(ad[b].float(), 0.999)))
    ad = torch.clamp(ad, 0, cfg['adc_klip']) / cfg['adc_klip']
    ort, std = dw[b].mean(), dw[b].std() + 1e-6
    dw = torch.clamp((dw - ort) / std, -cfg['dwi_z_klip'], cfg['dwi_z_klip'])
    dw = torch.where(b, dw, torch.zeros((), device=dw.device))
    ad = torch.where(b, ad, torch.zeros((), device=ad.device))
    return torch.stack([dw, ad]), dilim
