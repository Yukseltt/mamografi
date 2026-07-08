# Veri Seti Istatistikleri - 412 goruntuyu tarayip pandas tablosu cikar

import pydicom
import pandas as pd
from pathlib import Path

DICOM_KLASORU = Path("D:\\mamografi\\INbreast_dataset\\AllDICOMs")
dosyalar = sorted(DICOM_KLASORU.glob("*.dcm"))
print(f"Toplam DICOM: {len(dosyalar)}\n")

# Her dosya adindan bilgi ayristir: 20586908_HASTAID_MG_R_CC_ANON.dcm
kayitlar = []
for dosya in dosyalar:
    parcalar = dosya.stem.split("_")   # stem = uzantisiz ad
    kayitlar.append({
        "dosya": dosya.name,
        "goruntu_id": parcalar[0],
        "hasta_id": parcalar[1],
        "taraf": parcalar[3],          # R / L
        "yon": parcalar[4],            # CC / ML / MLO
    })

# Listeyi pandas DataFrame'e cevir
df = pd.DataFrame(kayitlar)
print("Tablonun ilk 5 satiri:")
print(df.head(), "\n")

# value_counts -> bir sutundaki her degerin kac kez gectigini sayar
print("TARAF DAGILIMI (sag/sol)")
print(df["taraf"].value_counts(), "\n")

print("CEKIM YONU DAGILIMI (CC/ML/MLO)")
print(df["yon"].value_counts(), "\n")

print("BENZERSIZ HASTA SAYISI")
print(f"{df['hasta_id'].nunique()} hasta, {len(df)} goruntu")
print(f"Hasta basina ortalama {len(df) / df['hasta_id'].nunique():.1f} goruntu\n")

# Taraf + yon kombinasyonu (capraz tablo)
print("TARAF x YON")
print(pd.crosstab(df["taraf"], df["yon"]))

# Tabloyu CSV olarak kaydet
cikti = Path("D:\\mamografi\\egitim\\cikti")
cikti.mkdir(exist_ok=True)
df.to_csv(cikti / "03_goruntu_tablosu.csv", index=False)
print(f"\nTablo kaydedildi -> {cikti / '03_goruntu_tablosu.csv'}")
