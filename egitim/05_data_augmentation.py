# Veri Artirma (Data Augmentation): tek goruntuden varyasyonlar

import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DICOM_KLASORU = Path(r"D:\mamografi\INbreast_dataset\AllDICOMs")
dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]

ds = pydicom.dcmread(dosya)
ham = ds.pixel_array
norm = ((ham - ham.min()) / (ham.max() - ham.min()) * 255).astype(np.uint8)
img = cv2.resize(norm, (512, 512), interpolation=cv2.INTER_AREA)


def dondur(g, aci):
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), aci, 1.0)
    return cv2.warpAffine(g, M, (w, h))


def parlaklik(g, carpan):
    return np.clip(g.astype(float) * carpan, 0, 255).astype(np.uint8)


def yakinlastir(g, oran):
    h, w = g.shape
    yh, yw = int(h / oran), int(w / oran)
    y0, x0 = (h - yh) // 2, (w - yw) // 2
    kirp = g[y0:y0 + yh, x0:x0 + yw]
    return cv2.resize(kirp, (w, h), interpolation=cv2.INTER_LINEAR)


# (baslik, goruntu, guvenli mi?)
varyasyonlar = [
    ("ORIJINAL", img, True),
    ("Yatay cevir (mirror)", cv2.flip(img, 1), True),
    ("Dondur +12 derece", dondur(img, 12), True),
    ("Dondur -12 derece", dondur(img, -12), True),
    ("Parlaklik x1.3", parlaklik(img, 1.3), True),
    ("Parlaklik x0.7", parlaklik(img, 0.7), True),
    ("Yakinlastir x1.2", yakinlastir(img, 1.2), True),
    ("Dikey cevir", cv2.flip(img, 0), False),
    ("Renk TERSLEME", 255 - img, False),
]

fig, eksenler = plt.subplots(3, 3, figsize=(14, 14))
for eksen, (baslik, g, guvenli) in zip(eksenler.ravel(), varyasyonlar):
    eksen.imshow(g, cmap="gray")
    renk = "green" if guvenli else "red"
    eksen.set_title(baslik, color=renk, fontsize=12)
    eksen.axis("off")

plt.tight_layout()
cikti = Path(r"D:\mamografi\egitim\cikti")
plt.savefig(cikti / "05_augmentation.png", dpi=80, bbox_inches="tight")
print(f"Kaydedildi -> {cikti / '05_augmentation.png'}")
print("Yesil = guvenli artirma, Kirmizi = teshisi bozan artirma")
