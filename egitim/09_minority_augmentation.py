import pydicom
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

DICOM_DIR  = Path(r"D:\mamografi\INbreast_dataset\AllDICOMs")
CIKTI      = Path(r"D:\mamografi\egitim\cikti")
ISLEN_DIR  = Path(r"D:\mamografi\egitim\islenmis")
AUG_DIR    = ISLEN_DIR / "aug"
ISLEN_DIR.mkdir(parents=True, exist_ok=True)
AUG_DIR.mkdir(parents=True, exist_ok=True)


def yukle_ve_isle(goruntu_id):
    dcm = next(DICOM_DIR.glob(f"{goruntu_id}_*.dcm"), None)
    if dcm is None:
        return None
    ham = pydicom.dcmread(dcm).pixel_array.astype(float)
    norm = ((ham - ham.min()) / (ham.max() - ham.min()) * 255).astype(np.uint8)
    img = cv2.resize(norm, (512, 512), interpolation=cv2.INTER_AREA)

    _, maske = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, etiketler, istat, _ = cv2.connectedComponentsWithStats(maske)
    if n <= 1:
        return img
    en_buyuk = 1 + np.argmax(istat[1:, cv2.CC_STAT_AREA])
    temiz = np.where(etiketler == en_buyuk, 255, 0).astype(np.uint8)
    kor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    temiz = cv2.morphologyEx(temiz, cv2.MORPH_CLOSE, kor)
    return cv2.bitwise_and(img, img, mask=temiz)


def augmentasyonlar(img):
    h, w = img.shape
    result = []

    result.append(("fliph", cv2.flip(img, 1)))

    for aci in [-15, -10, 10, 15]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), aci, 1.0)
        result.append((f"rot{aci:+d}", cv2.warpAffine(img, M, (w, h))))

    for k in [0.8, 1.2]:
        aug = np.clip(img.astype(float) * k, 0, 255).astype(np.uint8)
        result.append((f"bright{k}", aug))

    for z in [1.15, 1.25]:
        yh, yw = int(h / z), int(w / z)
        y0, x0 = (h - yh) // 2, (w - yw) // 2
        kirp = img[y0:y0 + yh, x0:x0 + yw]
        result.append((f"zoom{z}", cv2.resize(kirp, (w, h))))

    return result


df = pd.read_csv(CIKTI / "08_fold_tablosu.csv")

print("Tum goruntuler isleniyor ve PNG olarak kaydediliyor...")
png_yollari = {}
for i, (_, satir) in enumerate(df.iterrows(), 1):
    gid = satir["goruntu_id"]
    png_yol = ISLEN_DIR / f"{gid}.png"
    if not png_yol.exists():
        img = yukle_ve_isle(gid)
        if img is not None:
            cv2.imwrite(str(png_yol), img)
    png_yollari[gid] = str(png_yol)
    if i % 50 == 0:
        print(f"  {i}/{len(df)}")

df["png_yolu"] = df["goruntu_id"].map(png_yollari)
print(f"{len(df)} goruntu islendi -> {ISLEN_DIR}\n")

for fold in range(5):
    satirlar = []

    for _, satir in df.iterrows():
        yeni = satir.to_dict()
        yeni["split"] = "val" if satir["fold"] == fold else "train"
        yeni["augmented"] = False
        satirlar.append(yeni)

    egitim_pos = df[(df["fold"] != fold) & (df["etiket"] == "POZITIF")]

    for _, satir in egitim_pos.iterrows():
        img = cv2.imread(satir["png_yolu"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for aug_ad, aug_img in augmentasyonlar(img):
            dosya_ad = f"fold{fold}_{satir['goruntu_id']}_{aug_ad}.png"
            cv2.imwrite(str(AUG_DIR / dosya_ad), aug_img)
            yeni = satir.to_dict()
            yeni["png_yolu"] = str(AUG_DIR / dosya_ad)
            yeni["split"] = "train"
            yeni["augmented"] = True
            satirlar.append(yeni)

    fold_df = pd.DataFrame(satirlar)
    fold_df.to_csv(CIKTI / f"09_fold{fold}_dataset.csv", index=False)

    train = fold_df[fold_df["split"] == "train"]
    val   = fold_df[fold_df["split"] == "val"]
    tn = (train["etiket"] == "NEGATIF").sum()
    tp = ((train["etiket"] == "POZITIF") & ~train["augmented"]).sum()
    ta = ((train["etiket"] == "POZITIF") &  train["augmented"]).sum()
    vn = (val["etiket"] == "NEGATIF").sum()
    vp = (val["etiket"] == "POZITIF").sum()
    print(f"Fold {fold} | train: {tn}N + {tp}P + {ta}P_aug | val: {vn}N + {vp}P")

print("\nPer-fold dataset CSV'leri kaydedildi.")
