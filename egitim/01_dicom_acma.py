# Bir DICOM Goruntusunu Acmak

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DICOM_KLASORU = Path("D:\\mamografi\\INbreast_dataset\\AllDICOMs")

# Ornek olarak ilk .dcm dosyasi
ilk_dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]
print(f"Aciliyor: {ilk_dosya.name}\n")

# Metadata ve piksel verisini yukle
ds = pydicom.dcmread(ilk_dosya)

# Piksel verisi: 2 boyutlu NumPy dizisi
goruntu = ds.pixel_array

print("GORUNTU BILGILERI")
print(f"Veri tipi (dtype) : {goruntu.dtype}")
print(f"Boyut (yukseklik, genislik) : {goruntu.shape}")
print(f"Toplam piksel : {goruntu.size:,}")
print(f"En koyu / en parlak : {goruntu.min()} / {goruntu.max()}")
print(f"Ortalama parlaklik : {goruntu.mean():.1f}")

# PNG olarak kaydet
cikti = Path("D:\\mamografi\\egitim\\cikti")
cikti.mkdir(exist_ok=True)

plt.figure(figsize=(6, 8))
plt.imshow(goruntu, cmap="gray")
plt.title(f"Mamografi: {ilk_dosya.name[:25]}...")
plt.axis("off")
plt.tight_layout()
kayit_yolu = cikti / "01_ilk_goruntu.png"
plt.savefig(kayit_yolu, dpi=100, bbox_inches="tight")
print(f"\nGoruntu kaydedildi - {kayit_yolu}")
