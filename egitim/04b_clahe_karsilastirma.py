# CLAHE tileGridSize karsilastirmasi: 4x4 vs 8x8 vs 16x16

import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DICOM_KLASORU = Path(r"D:\mamografi\INbreast_dataset\AllDICOMs")
dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]

ds = pydicom.dcmread(dosya)
ham = ds.pixel_array

# Normalize + 512x512 (onceki adimla ayni hazirlik)
norm = ((ham - ham.min()) / (ham.max() - ham.min()) * 255).astype(np.uint8)
kucuk = cv2.resize(norm, (512, 512), interpolation=cv2.INTER_AREA)

# Farkli tile boyutlari (clipLimit sabit)
tile_boyutlari = [None, (4, 4), (8, 8), (16, 16)]  # None = CLAHE yok (referans)

fig, eksenler = plt.subplots(1, 4, figsize=(20, 6))
for eksen, tile in zip(eksenler, tile_boyutlari):
    if tile is None:
        sonuc = kucuk
        baslik = "CLAHE YOK (referans)"
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile)
        sonuc = clahe.apply(kucuk)
        baslik = f"tileGridSize={tile}"
    eksen.imshow(sonuc, cmap="gray")
    eksen.set_title(baslik)
    eksen.axis("off")

plt.tight_layout()
cikti = Path(r"D:\mamografi\egitim\cikti")
plt.savefig(cikti / "04b_clahe_karsilastirma.png", dpi=90, bbox_inches="tight")
print(f"Kaydedildi -> {cikti / '04b_clahe_karsilastirma.png'}")

# Sayisal olarak da karsilastir: standart sapma = kontrast olcusu
print("\nKontrast olcusu (std sapma, yuksek = daha kontrastli):")
print(f"  CLAHE yok : {kucuk.std():.1f}")
for tile in [(4, 4), (8, 8), (16, 16)]:
    s = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile).apply(kucuk)
    print(f"  {str(tile):8s}: {s.std():.1f}")
