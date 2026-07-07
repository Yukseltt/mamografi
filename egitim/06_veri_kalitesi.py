# Veri Kalitesi ve Class Imbalance: BIRADS teshislerini analiz et

import pandas as pd
from pathlib import Path

# 1) Onceki adimda urettigimiz goruntu tablosu
goruntu = pd.read_csv(r"D:\mamografi\egitim\cikti\03_goruntu_tablosu.csv")

# 2) Teshis (BIRADS) tablosu. Ayrac ';' , File Name = goruntu_id ile eslesir
teshis = pd.read_csv(r"D:\mamografi\INbreast_dataset\INbreast.csv", sep=";")
teshis["File Name"] = teshis["File Name"].astype(str)
goruntu["goruntu_id"] = goruntu["goruntu_id"].astype(str)

# 3) MERGE: iki tabloyu ortak anahtar uzerinden birlestir
df = goruntu.merge(
    teshis[["File Name", "Bi-Rads", "ACR"]],
    left_on="goruntu_id", right_on="File Name", how="inner",
)
print(f"Birlestirilen goruntu: {len(df)}\n")

# 4) Bi-Rads bazi degerler '4a','4b','4c' olabilir -> ilk rakami al, sayiya cevir
df["birads_sayi"] = df["Bi-Rads"].astype(str).str.extract(r"(\d)").astype(int)

print("=== BIRADS DAGILIMI ===")
print(df["birads_sayi"].value_counts().sort_index(), "\n")

# 5) Ikili etiket: 1-3 = Negatif, 4-6 = Pozitif (kanser suphesi)
df["etiket"] = df["birads_sayi"].apply(lambda b: "POZITIF" if b >= 4 else "NEGATIF")

print("=== IKILI ETIKET (CLASS IMBALANCE) ===")
sayim = df["etiket"].value_counts()
print(sayim)
oran = sayim.min() / sayim.max()
print(f"\nDengesizlik orani (azinlik/cogunluk): {oran:.2f}")
print(f"Pozitif oran: %{100 * (df['etiket'] == 'POZITIF').mean():.1f}\n")

# 6) ACR = meme yogunlugu (1=yagli ... 4=cok yogun). Yogun meme teshisi zorlastirir
print("=== ACR (MEME YOGUNLUGU) DAGILIMI ===")
print(df["ACR"].value_counts().sort_index())

df.to_csv(r"D:\mamografi\egitim\cikti\06_etiketli_tablo.csv", index=False)
print("\nEtiketli tablo kaydedildi -> cikti/06_etiketli_tablo.csv")
