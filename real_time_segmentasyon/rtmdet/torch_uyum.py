"""torch 2.6 + mmengine 0.10.7 checkpoint uyumu.

torch 2.6, `torch.load` varsayilanini `weights_only=True` yapti. mmengine kendi
checkpointlerine `message_hub` icinde `HistoryBuffer` gibi nesneler yaziyor; bu
yuzden **kendi urettigimiz** checkpointler yuklenemiyor -- resume ve egitim sonrasi
degerlendirme kirilir. (COCO on-egitimli agirlik yalnizca tensor icerdigi icin
`load_from` etkilenmiyordu, sorun ancak kendi checkpointimizi okurken cikti.)

Yalnizca kendi urettigimiz dosyalari yukledigimiz icin varsayilan geri aliniyor.
Bu modulu checkpoint okuyan her giris noktasinda import etmek yeterli.
"""
import torch

_orijinal_load = torch.load


def _load_weights_only_kapali(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orijinal_load(*args, **kwargs)


if getattr(torch.load, '__name__', '') != '_load_weights_only_kapali':
    torch.load = _load_weights_only_kapali
