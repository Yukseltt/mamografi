# Class imbalance gorsellestirme

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(r"D:\mamografi\egitim\cikti\06_etiketli_tablo.csv")

fig, (sol, sag) = plt.subplots(1, 2, figsize=(13, 5))

# Sol: BIRADS dagilimi (6 sinif)
birads = df["birads_sayi"].value_counts().sort_index()
renkler = ["#2e7d32" if b < 4 else "#c62828" for b in birads.index]
sol.bar(birads.index.astype(str), birads.values, color=renkler)
sol.set_title("BIRADS Dagilimi (yesil=negatif, kirmizi=pozitif)")
sol.set_xlabel("BIRADS skoru")
sol.set_ylabel("Goruntu sayisi")
for i, v in zip(birads.index.astype(str), birads.values):
    sol.text(i, v + 2, str(v), ha="center")

# Sag: ikili etiket (imbalance)
etiket = df["etiket"].value_counts()
sag.bar(etiket.index, etiket.values, color=["#2e7d32", "#c62828"])
sag.set_title(f"Ikili Etiket - Dengesizlik (%{100*(df['etiket']=='POZITIF').mean():.0f} pozitif)")
sag.set_ylabel("Goruntu sayisi")
for i, v in enumerate(etiket.values):
    sag.text(i, v + 3, str(v), ha="center")

plt.tight_layout()
cikti = Path(r"D:\mamografi\egitim\cikti")
plt.savefig(cikti / "06b_dengesizlik.png", dpi=90, bbox_inches="tight")
print(f"Kaydedildi -> {cikti / '06b_dengesizlik.png'}")
