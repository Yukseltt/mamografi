"""Yerel mmdet ortaminin egitim oncesi dogrulamasi.

Sirayla: surumler -> CUDA -> mmcv CUDA operatorleri -> RTMDet-Ins model kurulumu
-> gercek bir BUSI goruntusu uzerinde ileri gecis -> VRAM olcumu.

Herhangi bir adim patlarsa egitime baslamanin anlami yok.
"""
import time
from pathlib import Path

import numpy as np
import torch

ORNEK = r'd:\mamografi\real_time_segmentasyon\Dataset_BUSI_with_GT\benign\benign (1).png'
CFG_ADI = 'rtmdet-ins_tiny_8xb32-300e_coco.py'


def baslik(s):
    print(f'\n--- {s} ---')


baslik('surumler')
import mmcv, mmengine, mmdet
print('torch   ', torch.__version__)
print('mmcv    ', mmcv.__version__)
print('mmengine', mmengine.__version__)
print('mmdet   ', mmdet.__version__)
print('numpy   ', np.__version__)

baslik('CUDA')
assert torch.cuda.is_available(), 'CUDA gorunmuyor'
print('cihaz     ', torch.cuda.get_device_name(0))
print('kapasite  ', torch.cuda.get_device_capability(0))
print('toplam VRAM %.2f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))

baslik('mmcv CUDA operatorleri')
from mmcv.ops import nms, RoIAlign
b = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]]).cuda()
s = torch.tensor([0.9, 0.8]).cuda()
print('nms      ->', nms(b, s, 0.5)[1].tolist())
ra = RoIAlign(output_size=7, spatial_scale=1.0, sampling_ratio=0).cuda()
feat = torch.randn(1, 4, 32, 32).cuda()
rois = torch.tensor([[0., 0., 0., 16., 16.]]).cuda()
print('RoIAlign ->', tuple(ra(feat, rois).shape))

baslik('RTMDet-Ins model kurulumu')
cfg_yolu = Path(mmdet.__file__).parent / '.mim' / 'configs' / 'rtmdet' / CFG_ADI
assert cfg_yolu.exists(), cfg_yolu
print('config:', cfg_yolu.name)

from mmdet.apis import init_detector, inference_detector

torch.cuda.reset_peak_memory_stats()
model = init_detector(str(cfg_yolu), None, device='cuda:0')   # rastgele agirlik, sadece yapi testi
n_par = sum(p.numel() for p in model.parameters())
print('parametre: %.2f M' % (n_par / 1e6))
print('kurulum sonrasi VRAM: %.0f MB' % (torch.cuda.max_memory_allocated() / 1e6))

baslik('ileri gecis (gercek BUSI goruntusu)')
assert Path(ORNEK).exists(), ORNEK
sonuc = inference_detector(model, ORNEK)          # ilk cagri: cekirdek derlemesi dahil

t0 = time.perf_counter()
for _ in range(5):
    sonuc = inference_detector(model, ORNEK)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 5

pred = sonuc.pred_instances
print('cikti alanlari :', [k for k in pred.all_keys()])
print('instance sayisi:', len(pred))
if 'masks' in pred:
    print('maske tensoru  :', tuple(pred.masks.shape), pred.masks.dtype)
print('sure/goruntu   : %.1f ms  (rastgele agirlik, 640 pipeline, isinma sonrasi)' % (dt * 1000))
print('tepe VRAM      : %.0f MB' % (torch.cuda.max_memory_allocated() / 1e6))

baslik('SONUC')
print('Ortam calisiyor. Egitime gecilebilir.')
