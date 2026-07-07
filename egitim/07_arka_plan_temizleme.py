# Arka Plan Temizleme: memeyi ayikla, etiket/artefakt/boslugu at

import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DICOM_KLASORU = Path("D:\\mamografi\\INbreast_dataset\\AllDICOMs")
dosya = sorted(DICOM_KLASORU.glob("*.dcm"))[0]

ds = pydicom.dcmread(dosya)
ham = ds.pixel_array
norm = ((ham - ham.min()) / (ham.max() - ham.min()) * 255).astype(np.uint8)
img = cv2.resize(norm, (512, 512), interpolation=cv2.INTER_AREA)

# 1) ESIKLEME: Otsu yontemi esigi otomatik secer -> doku=beyaz, arka plan=siyah
esik, maske = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu esigi: {esik}")

# 2) En buyuk bagli bileseni bul (meme), kucuk lekeleri/etiketleri at
sayi, etiketler, istatistik, _ = cv2.connectedComponentsWithStats(maske)
# 0. bileseni (arka plan) atla, en genis alanli bileseni sec
en_buyuk = 1 + np.argmax(istatistik[1:, cv2.CC_STAT_AREA])
temiz_maske = np.where(etiketler == en_buyuk, 255, 0).astype(np.uint8)

# 3) Maske icindeki kucuk delikleri doldur (morfolojik kapama)
cekirdek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
temiz_maske = cv2.morphologyEx(temiz_maske, cv2.MORPH_CLOSE, cekirdek)

# 4) Maskeyi orijinale uygula: sadece meme kalir, gerisi siyah
temiz = cv2.bitwise_and(img, img, mask=temiz_maske)

# Ne kadar piksel atildi?
atilan = 100 * (1 - temiz_maske.mean() / 255)
print(f"Atilan (arka plan) piksel orani: %{atilan:.1f}")

# Gorsellestir
fig, eksenler = plt.subplots(1, 4, figsize=(20, 6))
for eksen, (g, b) in zip(eksenler, [
    (img, "1) Girdi"),
    (maske, "2) Otsu maskesi (ham)"),
    (temiz_maske, "3) En buyuk bilesen (temiz maske)"),
    (temiz, "4) Temizlenmis meme"),
]):
    eksen.imshow(g, cmap="gray")
    eksen.set_title(b)
    eksen.axis("off")
plt.tight_layout()
cikti = Path(r"D:\mamografi\egitim\cikti")
plt.savefig(cikti / "07_arka_plan_temizleme.png", dpi=90, bbox_inches="tight")
print(f"Kaydedildi -> {cikti / '07_arka_plan_temizleme.png'}")
