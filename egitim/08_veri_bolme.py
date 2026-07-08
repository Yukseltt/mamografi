import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

CIKTI = Path(r"D:\mamografi\egitim\cikti")

df = pd.read_csv(CIKTI / "06_etiketli_tablo.csv")

# hasta bazinda etiket: herhangi bir gorseli pozitifse hasta pozitif sayilir
hasta_etiket = df.groupby("hasta_id")["etiket"].apply(
    lambda x: "POZITIF" if "POZITIF" in x.values else "NEGATIF"
)
df["hasta_etiket"] = df["hasta_id"].map(hasta_etiket)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

df["fold"] = -1
for fold, (_, val_idx) in enumerate(
    sgkf.split(df, df["hasta_etiket"], groups=df["hasta_id"])
):
    df.loc[val_idx, "fold"] = fold

df.to_csv(CIKTI / "08_fold_tablosu.csv", index=False)

ozet = df.groupby(["fold", "etiket"]).size().unstack(fill_value=0)
print(ozet)
print(f"\n{len(df)} goruntu | {df['hasta_id'].nunique()} hasta | 5 fold")

# dogrulama: ayni hasta farkli foldlarda olmamali
assert df.groupby("hasta_id")["fold"].nunique().max() == 1, "Hasta sizdirmasi var!"
print("Hasta sizdirma kontrolu: OK")
