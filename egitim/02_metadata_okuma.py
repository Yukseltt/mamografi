# Metadata Okuma

import pydicom
from pathlib import Path

DICOM_KLASORU = Path("D:\\mamografi\\INbreast_dataset\\AllDICOMs")
ilk_dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]

# stop_before_pixels=True ile pikselleri yukleme, sadece metadata oku
ds = pydicom.dcmread(ilk_dosya, stop_before_pixels=True)

# DICOM'da her bilginin bir etiketi vardir.
print("HASTA / CEKIM BILGILERI")
print("Hasta ID :", ds.get("PatientID", "yok"))
print("Cinsiyet :", ds.get("PatientSex", "yok"))
print("Modalite :", ds.get("Modality", "yok"))
print("Cekim yonu :", ds.get("ViewPosition", "yok"))
print("Taraf (L/R) :", ds.get("ImageLaterality", "yok"))
print("Cihaz ureticisi :", ds.get("Manufacturer", "yok"))
print("Cihaz modeli :", ds.get("ManufacturerModelName", "yok"))

print("\nTEKNIK PIKSEL BILGILERI")
print("Satir (yukseklik):", ds.get("Rows", "yok"))
print("Sutun (genislik) :", ds.get("Columns", "yok"))
print("Bit derinligi    :", ds.get("BitsStored", "yok"), "bit")
print("Piksel araligi   :", ds.get("PixelSpacing", "yok"), "mm")  # 1 pikselin mm karsiligi

# İlk 25 etiket:
print("\nTUM ETIKETLER (ilk 25)")
for eleman in list(ds)[:25]:
    print(f"  {eleman.name:35s}: {eleman.value}")
