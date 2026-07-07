# Veri On Isleme: ham -> normalize -> boyutlandir -> kontrast

import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DICOM_KLASORU = Path("D:\\mamografi\\INbreast_dataset\\AllDICOMs")
dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]

ds = pydicom.dcmread(dosya)
ham = ds.pixel_array   # uint16, 0..2970 arasi
print(f"HAM  -> dtype={ham.dtype}, min={ham.min()}, max={ham.max()}, shape={ham.shape}")

# 1) NORMALIZASYON: her degeri 0..255 araligina cek, uint8'e cevir
# Formul: (deger - min) / (max - min) * 255
normalize = (ham - ham.min()) / (ham.max() - ham.min()) * 255
normalize = normalize.astype(np.uint8)
print(f"NORM -> dtype={normalize.dtype}, min={normalize.min()}, max={normalize.max()}")

# 2) BOYUTLANDIRMA: dev goruntuyu modele uygun kucuk kareye getir
hedef = (512, 512)
kucuk = cv2.resize(normalize, hedef, interpolation=cv2.INTER_AREA)
print(f"RESIZE -> shape={kucuk.shape}")

# 3) KONTRAST IYILESTIRME (CLAHE): lokal kontrasti artirir, dokuyu belirginlestirir
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
kontrastli = clahe.apply(kucuk)

# 4) Karsilastirma gorseli: 4 asama yan yana
cikti = Path("D:\\mamografi\\egitim\\cikti")
cikti.mkdir(exist_ok=True)

# Ham veriyi "olceklemeden" gostermek icin vmin/vmax=uint8 araligi zorla
fig, eksenler = plt.subplots(1, 4, figsize=(20, 6))
eksenler[0].imshow(ham, cmap="gray", vmin=0, vmax=255)   # olcekleme YOK -> karanlik
eksenler[0].set_title("1) HAM (0-255 gibi gosterilirse\nKAPKARANLIK)")
eksenler[1].imshow(normalize, cmap="gray")
eksenler[1].set_title("2) NORMALIZE (0-255)")
eksenler[2].imshow(kucuk, cmap="gray")
eksenler[2].set_title("3) 512x512 BOYUTLANDIR")
eksenler[3].imshow(kontrastli, cmap="gray")
eksenler[3].set_title("4) CLAHE KONTRAST")
for e in eksenler:
    e.axis("off")
plt.tight_layout()
plt.savefig(cikti / "04_on_isleme_8x8.png", dpi=90, bbox_inches="tight")
print(f"\nKarsilastirma kaydedildi -> {cikti / '04_on_isleme_8x8.png'}")
