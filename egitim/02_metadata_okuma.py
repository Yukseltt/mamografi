# Metadata Okuma

import pydicom
from pathlib import Path

DICOM_KLASORU = Path("D:\\mamografi\\INbreast_dataset\\AllDICOMs")
ilk_dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]

# stop_before_pixels=True -> pikselleri yukleme, sadece metadata oku (cok hizli)
ds = pydicom.dcmread(ilk_dosya, stop_before_pixels=True)

# DICOM'da her bilgi bir "etiket" (tag) ile durur. Erisim: ds.EtiketAdi
print("HASTA / CEKIM BILGILERI")
print("Hasta ID        :", ds.get("PatientID", "yok"))
print("Cinsiyet        :", ds.get("PatientSex", "yok"))
print("Modalite        :", ds.get("Modality", "yok"))       # MG = Mammography
print("Cekim yonu      :", ds.get("ViewPosition", "yok"))   # CC / MLO
print("Taraf (L/R)     :", ds.get("ImageLaterality", "yok"))
print("Cihaz ureticisi :", ds.get("Manufacturer", "yok"))
print("Cihaz modeli    :", ds.get("ManufacturerModelName", "yok"))

print("\nTEKNIK PIKSEL BILGILERI")
print("Satir (yukseklik):", ds.get("Rows", "yok"))
print("Sutun (genislik) :", ds.get("Columns", "yok"))
print("Bit derinligi    :", ds.get("BitsStored", "yok"), "bit")
print("Piksel araligi   :", ds.get("PixelSpacing", "yok"), "mm")  # 1 pikselin mm karsiligi

# Tum etiketleri gormek istersen (ilk 25 tanesi):
print("\nTUM ETIKETLER (ilk 25)")
for eleman in list(ds)[:25]:
    print(f"  {eleman.name:35s}: {eleman.value}")
