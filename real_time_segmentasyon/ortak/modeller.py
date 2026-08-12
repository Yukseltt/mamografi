"""Uc mimarinin cikarim yolu, tek arayuz.

Her kurucu iki sey donduruyor:
  `kare_fn(img_bgr, ori_shape) -> skor haritasi`   diskten okuma yok
  `harita_fn(satir) -> skor haritasi`              degerlendirme protokolunun bekledigi

Ikisi ayni hesabi yapiyor, tek fark goruntunun nereden geldigi. Hiz olcumu
`kare_fn` uzerinden yapiliyor cunku canli ultrason akisinda kare cihazdan gelir,
PNG okuma maliyeti modele ait degildir. Kalite olcumu `harita_fn` uzerinden.
Ikisinin ayni sonucu verdigi `hiz_olcumu.py` icinde assert ile kontrol ediliyor --
yoksa hizi olculen kod ile Dice'i olculen kod ayrisir.

Letterbox kurali uc modelde de ayni (bkz. PLAN.md hata 6): en-boy korunarak
GIRDI'ye sigacak sekilde kucult, sag-alta doldur. RTMDet bunu mmdet pipeline'inda,
YOLO Ultralytics icinde, SegFormer burada yapiyor.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI))

import degerlendirme as dg

MIN_SKOR = 0.005      # instance modellerde haritaya giren alt sinir


def _gri_bgr(satir):
    return cv2.cvtColor(dg.gri_oku(dg.yol_coz(satir.image_path)), cv2.COLOR_GRAY2BGR)


def _harita_sar(kare_fn):
    """kare_fn -> harita_fn: degerlendirme protokolunun bekledigi imza."""
    def harita_fn(satir):
        img = _gri_bgr(satir)
        return kare_fn(img, img.shape[:2])
    return harita_fn


# ---- letterbox (SegFormer icin; digerleri kendi cercevesinde yapiyor) ----

def letterbox_olcu(h, w, girdi):
    r = min(girdi / h, girdi / w)
    return r, int(h * r + 0.5), int(w * r + 0.5)   # mmcv rescale_size konvansiyonu


def letterbox_uygula(img, girdi):
    h, w = img.shape[:2]
    _, nh, nw = letterbox_olcu(h, w, girdi)
    out = np.full((girdi, girdi, 3), 114, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def letterbox_geri(harita, h, w, girdi):
    """(girdi, girdi) tensor -> (h, w). Dolgu kirpilir, sonra orijinale buyutulur."""
    import torch.nn.functional as F
    _, nh, nw = letterbox_olcu(h, w, girdi)
    return F.interpolate(harita[..., :nh, :nw][None, None], size=(h, w),
                         mode='bilinear', align_corners=False)[0, 0]


# ---- model basina kurucular ----

def segformer_kur(ckpt, girdi=256, cihaz='cuda:0'):
    """Egitim notebook'undaki `skor_haritasi` ile birebir ayni yol."""
    import albumentations as A
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b0', num_labels=1, ignore_mismatched_sizes=True)
    durum = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    model.load_state_dict(durum['model'], strict=True)   # isim uyusmazligi sessiz gecmesin
    model = model.to(cihaz).eval()
    NORM = A.Normalize()          # uint8 girdi -> varsayilan max_pixel_value=255 dogru

    @torch.no_grad()
    def kare_fn(img, ori_shape):
        h, w = ori_shape
        x = torch.from_numpy(NORM(image=letterbox_uygula(img, girdi))['image'])
        x = x.permute(2, 0, 1)[None].float().to(cihaz)
        # logit girdi cozunurluguna buyutulur, sigmoid en sonda -- notebook ile ayni
        logit = F.interpolate(model(pixel_values=x).logits, size=(girdi, girdi),
                              mode='bilinear', align_corners=False)[0, 0]
        return torch.sigmoid(letterbox_geri(logit, h, w, girdi)).cpu().numpy()

    bilgi = dict(ad='SegFormer-B0', tip='semantic', girdi=girdi,
                 parametre=sum(p.numel() for p in model.parameters()),
                 epoch=durum.get('epoch'))
    return kare_fn, _harita_sar(kare_fn), model, bilgi


def yolo_kur(ckpt, girdi=256, cihaz=0):
    """retina_masks=True: maske girdi uzayinda degil orijinal boyutta uretilir."""
    from ultralytics import YOLO

    model = YOLO(str(ckpt))

    def kare_fn(img, ori_shape):
        r = model.predict(img, imgsz=girdi, conf=MIN_SKOR, iou=0.7, retina_masks=True,
                          verbose=False, device=cihaz)[0]
        sekil = tuple(int(v) for v in ori_shape)
        if r.masks is None or len(r.masks) == 0:
            return np.zeros(sekil, np.float32)
        return dg.instans_haritasi(r.masks.data.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy(), sekil)

    bilgi = dict(ad='YOLOv11n-Seg', tip='instance', girdi=girdi,
                 parametre=sum(p.numel() for p in model.model.parameters()), epoch=None)
    return kare_fn, _harita_sar(kare_fn), model, bilgi


def rtmdet_kur(config, ckpt, girdi=None, cihaz='cuda:0'):
    from mmdet.apis import inference_detector, init_detector
    from mmengine.config import Config

    model = init_detector(str(config), str(ckpt), device=cihaz)
    if girdi is None:
        girdi = int(Config.fromfile(str(config)).get('GIRDI', 0)) or None

    def kare_fn(img, ori_shape):
        pred = inference_detector(model, img).pred_instances
        skor = pred.scores.cpu().numpy()
        sec = skor >= MIN_SKOR
        sekil = tuple(int(v) for v in ori_shape)
        if not sec.any():
            return np.zeros(sekil, np.float32)
        return dg.instans_haritasi(pred.masks.cpu().numpy()[sec], skor[sec], sekil)

    bilgi = dict(ad=f'RTMDet-Ins-tiny {girdi}', tip='instance', girdi=girdi,
                 parametre=sum(p.numel() for p in model.parameters()), epoch=None)
    return kare_fn, _harita_sar(kare_fn), model, bilgi
