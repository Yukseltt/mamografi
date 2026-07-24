# Mamografi Kitle (Mass) Lezyon Tespiti — İlerleme Raporu

Bu metin, projede şu ana kadar yapılan işlerin özetidir. Detaylı teknik
gerekçeler ve referans kodlar için `PROJE_DOKUMANI.md` dosyasına bakınız.

---

## 1. Proje Kapsamı

Mamografi görüntülerinde kitle (mass) lezyonlarının nesne tespiti (object
detection) ile otomatik tespiti. Üç farklı model mimarisi (YOLO11, YOLO26,
RF-DETR) aynı veri seti üzerinde adil biçimde eğitilip karşılaştırılacak.
Geliştirme ortamı Google Colab, GPU olarak T4 kullanılıyor.

Veri seti: VinDr-Mammo + Teknofest birleşimi (`Kitle-Vindr&Tekno`), tek sınıf
(`mass`), YOLO formatında etiketli, toplam 3340 görüntü / 4021 kutu
(train 2671/3203, val 333/418, test 336/400).

---

## 2. Keşifsel Veri Analizi (EDA)

Veri seti incelendiğinde iki önemli bulgu ortaya çıktı:

- Görüntü başına ortalama ~1.2 lezyon kutusu; çoğu görüntüde tek lezyon var.
- **Küçük nesne baskınlığı:** Kutuların yaklaşık **%72'si** görüntü alanının
  %1'inden küçük. Medyan bbox alan oranı ~0.00552 — yani tipik bir lezyon,
  görüntünün binde 5'i kadar bir alan kaplıyor. Bu bulgu, hem model giriş
  boyutu (imgsz) seçimini hem de sonradan eklenen kırpma ön işlemesini
  doğrudan yönlendirdi.

---

## 3. Deney 1 — Kırpmasız Baseline (YOLO11s)

İlk yaklaşımda orijinal, kırpılmamış görüntüler doğrudan kullanıldı; sadece
model giriş boyutu (imgsz=1024) büyütülerek küçük lezyonların kaybolması
engellenmeye çalışıldı.

### Eğitim ayarları

| Parametre | Değer |
|---|---|
| Model | YOLO11s (COCO ön eğitimli) |
| imgsz | 1024 |
| batch | 8 |
| epochs (limit) | 100 |
| patience (early stopping) | 20 |
| seed | 42 |

Augmentation olarak yatay flip açıktı; ancak farkında olmadan Ultralytics'in
varsayılan mozaik (1.0) ve random erasing (0.4) ayarları da aktif kaldı —
bu, Deney 2'de bilinçli olarak gözden geçirildi ve kapatıldı.

### Test Sonuçları (YOLO11s, kırpmasız)

| Metrik | Değer |
|---|---|
| mAP@50 | **0.6928** |
| mAP@50-95 | **0.3820** |
| Precision | 0.7475 |
| Recall | 0.6365 |
| F1-Score | 0.6876 |

### Değerlendirme

- Test sonucu (0.6928) validation sonucuna (0.701) çok yakın — **overfitting
  görülmedi.**
- Early stopping en iyi sonucu epoch 61'de buldu, epoch 81'de eğitimi
  durdurdu. 81 epoch, Drive'dan yavaş görüntü okuma nedeniyle yaklaşık 6.6
  saat sürdü.
- Eğitim eğrisinde ilk ~30 epoch boyunca train ve val loss birlikte düştü;
  sonrasında hafif bir ayrışma oldu (train düşmeye devam etti, val loss
  platoya girdi) ama val mAP artmaya devam ettiği için bu ayrışma zararsız
  kabul edildi.
- Literatürde VinDr-Mammo benzeri setlerde tipik mAP@0.5 aralığı 0.50-0.70
  olarak kabul ediliyor; bu sonuç aralığın **üst ucunda**, sağlam bir
  başlangıç noktası.

Bu model ve sonuçları **korunuyor, silinmiyor** — ilerideki kırpmalı deneyle
karşılaştırma için doğal bir referans (aynı model, kırpmalı vs kırpmasız)
oluşturuyor.

---

## 4. Deney 2 — Kırpma Ön İşlemesi (devam ediyor)

### Gözlem ve gerekçe

Mamografi görüntülerinde meme dokusu, görüntünün ancak %30-40'ını kaplıyor;
geri kalanı siyah arka plan. Görüntüler model giriş boyutuna küçültülürken bu
boş alan da küçültülüyor, bu da zaten küçük olan lezyonları birkaç piksele
kadar düşürüyor. Meme bölgesini kırpıp boş alanı atmak, efektif çözünürlüğü
2-3 kat artırarak modelin küçük lezyonlara daha iyi odaklanmasını
sağlayabilir.

**Karar:** Üç model de (YOLO11 dahil, baştan) kırpılmış veriyle yeniden
eğitilecek. Böylece modeller arası mAP farkının ne kadarının mimariden, ne
kadarının kırpmadan geldiği net biçimde ayrılabilecek.

### Kırpma yöntemi (özet)

- Sabit eşik (`BG_THRESH=10`, Otsu değil) ile meme dokusu arka plandan
  ayrılıyor.
- Morfolojik açma ile küçük artefaktlar ve kenar yazıları (L-CC, R-MLO gibi)
  temizleniyor.
- En büyük bağlı bileşen (meme) hem x hem y ekseninde, %5 kenar payıyla
  kırpılıyor.
- Etiket kutuları yeni kırpılmış çerçeveye göre yeniden hesaplanıyor
  (koordinat dönüşümü doğrulandı).
- Güvenlik koşulu: tespit edilen meme genişliği görüntünün %15'inden azsa
  kırpma iptal ediliyor, orijinal görüntü korunuyor.

### Durum

- Kırpma tamamlandı: `Kitle-Vindr-Cropped` Drive'da, kutu sayıları
  orijinalle birebir aynı (train 3203, val 418, test 400). Drive'a toplu
  kopyalama sırasında train'de 563 label dosyası sessizce eksik kalmıştı
  (Colab'ın Drive FUSE mount'unda bilinen bir davranış); kaynak veriden
  yeniden üretilip tamamlandı.
- Kırpma sonrası EDA: küçük nesne oranı (%1'den küçük bbox) **%72'den
  %39.3'e** düştü — kırpmanın etkisi doğrulandı.

### Deney 2a — YOLO11s, kırpılmış, patience=5 (tamamlandı)

Deney 1'de (kırpmasız) 40. epoch'tan sonra overfit gözlenmişti. Buna karşı:
`optimizer=AdamW` (lr0=0.001, cos_lr=True), `weight_decay=0.001`
(varsayılanın 2 katı), `patience=5` (öncekinde 20), `mosaic=0.3`
(Deney 1'de farkında olmadan 1.0'dı).

**Sonuç: 20 epoch'ta durdu, en iyi model epoch 15.** Train/val loss paralel
düştü — overfitting bu ayarlarla önlendi.

| Metrik | Deney 1 (kırpmasız, 81 epoch) | Deney 2a (kırpılmış, 20 epoch) |
|---|---|---|
| mAP@50 | 0.6928 | 0.6901 |
| mAP@50-95 | 0.3820 | 0.3669 |
| Precision | 0.7475 | 0.6823 |
| Recall | 0.6365 | 0.6325 |
| F1 | 0.6876 | 0.6565 |

Sonuç Deney 1'e çok yakın, hafifçe düşük — kırpmanın beklenen faydası bu
turda görülmedi. mAP eğrisi hâlâ genel yükseliş eğilimindeyken patience=5
tetiklendi; val seti küçük olduğundan (332-333 görüntü) epoch-to-epoch
gürültü erken durmaya yol açmış olabilir (overfitting değil, erken durma
asıl sınırlayıcı görünüyor).

**Karar:** Deney 2b olarak `patience=10` ile ayrı bir deneme yapılacak,
diğer tüm ayarlar aynı kalacak, sonuçlar ayrı klasörde (`yolo11s_cropped_p10`)
tutulacak.

### Deney 2b — YOLO11s, kırpılmış, patience=10 (tamamlandı)

Deney 2a ile birebir aynı kurulum, tek fark `patience=5 → 10`. Amaç: 2a'nın
erken durma hipotezini test etmek.

**Sonuç: 25 epoch'ta durdu, en iyi model yine epoch 15.** Test sonuçları
Deney 2a ile bire bir aynı (mAP@50 0.6901, mAP@50-95 0.3669, Precision
0.6823, Recall 0.6325, F1 0.6565) — 10 epoch daha (16-25) şans tanımak
hiçbir iyileşme getirmedi, sadece ~1.1 saat fazladan sürdü.

**Hipotez çürütüldü:** Erken durma değil, epoch 15 gerçek bir yerel tavan.
`patience=5` zaten yeterliymiş. YOLO11s + kırpılmış veri + bu ayarlarla
mAP@50 tavanı **~0.69** — Deney 1'in (kırpmasız) 0.6928'ine pratikte eşit.
Kırpmanın beklenen faydası bu iki denemede de görülmedi; gerçek fayda üç
model bitince toplu karşılaştırmada değerlendirilecek.

**Karar:** YOLO11s + kırpma tarafı bu konfigürasyonla tavana çarptı kabul
edildi, sıradaki adım **YOLO26**.

### Deney 2e — YOLO11s, kırpılmış, `optimizer='SGD'` (şimdiye kadarki en iyi sonuç)

Deney 2a/2b'nin AdamW tavanı (0.6901), Deney 1'in kırpmasız baseline'ının
(0.6928) hafifçe altındaydı. Tek değişken olarak `optimizer='SGD'`,
`lr0=0.001`, diğer her şey aynı (cos_lr, weight_decay=0.001, patience=5,
mosaic=0.3) denendi.

**Sonuç: overfitting yok, ve şimdiye kadarki en iyi sonuç:**

| Metrik | Deney 1 (kırpmasız) | Deney 2a/2b (AdamW) | **Deney 2e (SGD)** |
|---|---|---|---|
| mAP@50 | 0.6928 | 0.6901 | **0.7561** |
| mAP@50-95 | 0.3820 | 0.3669 | **0.4121** |
| Precision | 0.7475 | 0.6823 | **0.7802** |
| Recall | 0.6365 | 0.6325 | **0.7000** |
| F1 | 0.6876 | 0.6565 | **0.7379** |

**Karar:** AdamW bu problem için doğru seçim değilmiş; SGD (uygun düşük
`lr0` ile) belirgin şekilde daha iyi. Bu, projenin şu anki en iyi YOLO11
sonucu — bundan sonraki karşılaştırmalarda referans olarak bu kullanılacak.

### Deney 2c — YOLO26s, kırpılmış (tamamlandı)

Deney 2a/2b'nin doğruladığı ayarlarla birebir aynı (patience=5, AdamW,
cos_lr, weight_decay=0.001, mosaic=0.3) — tek değişken mimari (`yolo26s.pt`).
Eğitim sırasında Colab bağlantısı koptu; `epoch10.pt`'den resume edilip
epoch 15'te en iyi sonucu buldu, epoch 20'de durdu.

**Test sonuçları:** mAP@50 0.6494, mAP@50-95 0.3483, Precision 0.6551,
Recall 0.6316, F1 0.6432 — **YOLO11s'in her iki versiyonundan da (kırpmalı/
kırpmasız) düşük.** STAL'ın beklenen küçük-nesne avantajı bu veri setinde
görülmedi; muhtemelen YOLO11 için doğrulanmış hiperparametreler YOLO26'ya
doğrudan aktarıldı, ayrıca doğrulanmadı.

**Yan bulgu:** Grad-CAM hücresi ilk çalıştırmada hata verdi — YOLO26 NMS'siz
(end2end) `Detect` başı kullanıyor, YOLO11'in `cv3` dalı yerine
`one2one_cv3` var. Kaynak katman (16, P3) aynı kaldığı için tek satırlık
düzeltmeyle (`cv3` → `one2one_cv3`) çözüldü.

**Karar:** YOLO26 tarafı da kaydedildi, sıradaki adım **RF-DETR**.

### Deney 2d — YOLO26s, kırpılmış, imgsz=1280 (tamamlandı)

Deney 2c'de YOLO26s YOLO11s'in altında kalmıştı ama overfitting yoktu; tek
değişken olarak `imgsz: 1024 → 1280` denendi, gerisi Deney 2c ile aynı.

**Sonuç: beklenenin tersine daha kötü.** mAP@50 0.6220, mAP@50-95 0.3367,
Precision 0.6772, Recall 0.5875, F1 0.6292 — Deney 2c'nin (0.6494) altında.
Confusion matrix'te kaçırılan lezyon sayısı arttı (FN 158 vs 128). Eğitim
sırasında görülen val mAP zirvesi (~0.66) ile test sonucu (0.622) arasında
belirgin bir fark vardı, val/test tutarsızlığına işaret ediyor.

**Karar:** Çözünürlük artırma hipotezi çürütüldü. YOLO26 tarafında sıradaki
lever çözünürlük değil, **optimizer** (Deney 2f: SGD).

### Deney 2f — YOLO26s, kırpılmış, `optimizer='SGD'` (tamamlandı)

Deney 2e'nin YOLO11s için SGD'nin AdamW'yi geçtiğini göstermesi üzerine
aynı düzeltme burada denendi: `imgsz` 1024'e (Deney 2c) geri döndü,
`optimizer='SGD'`, `lr0=0.001` — doğrudan bu degerden başlandı (Deney
2e'nin `lr0=0.01` kararsızlaştırdığı bulgusuna dayanarak).

**Sonuç: ilk denemede hiç kararsızlık yaşanmadı, overfitting yok.**

| Metrik | Deney 2c (AdamW) | Deney 2d (AdamW, 1280) | **Deney 2f (SGD)** |
|---|---|---|---|
| mAP@50 | 0.6494 | 0.6220 | **0.6914** |
| mAP@50-95 | 0.3483 | 0.3367 | **0.3686** |
| Precision | 0.6551 | 0.6772 | **0.7011** |
| Recall | 0.6316 | 0.5875 | **0.6392** |
| F1 | 0.6432 | 0.6292 | **0.6687** |

**Karar:** SGD > AdamW hipotezi YOLO26'da da doğrulandı (+4.2 puan), ama
YOLO11'deki sıçramanın (+6.6 puan) gerisinde kaldı. YOLO26, en iyi
ayarıyla bile YOLO11s'in en iyisinin (0.7561) altında — **YOLO11s + SGD
projenin genel en iyi sonucu olmaya devam ediyor.** Sıradaki adım
**RF-DETR**.

### Deney 3 — RF-DETR-Small, kırpılmış, COCO formatı (tamamlandı)

Üçüncü ve son planlanan mimari. YOLO'dan tamamen farklı bir kütüphane
(`rfdetr`, PyTorch Lightning tabanlı, transformer/DINOv2 omurga) ve veri
formatı (COCO) kullanıyor. `Kitle-Vindr-Cropped` (YOLO format) dokunulmadan,
ayrı bir `Kitle-Vindr-Cropped-COCO` kopyası üretildi. Model: `RFDETRSmall`
(YOLO11s/YOLO26s ile "s" parite), `batch_size=4, grad_accum_steps=4`
(T4 için efektif batch 16), `lr=1e-4`, `early_stopping_patience=5` (diğer
iki modelle aynı kural).

**Sonuç: eğitim sadece 9 epoch'ta erken durdu** (val mAP EMA 0.438'de
platoya girdi) — YOLO'nun 16-19 epoch'una göre çok daha hızlı yakınsadı.

| Metrik | YOLO11s (Deney 2e) | YOLO26s (Deney 2f) | **RF-DETR-S (Deney 3)** |
|---|---|---|---|
| mAP@50 | **0.7561** | 0.6914 | 0.7551 |
| mAP@50-95 | **0.4121** | 0.3686 | 0.4109 |
| Precision | **0.7802** | 0.7011 | 0.6163 |
| Recall | 0.7000 | 0.6392 | **0.8150** |
| F1 | **0.7379** | 0.6687 | 0.7018 |

**Değerlendirme:** mAP'te YOLO11s + SGD'ye neredeyse eşit (fark <0.15
puan), ama çok daha az epoch'ta. Precision/recall dengesi belirgin
farklı: RF-DETR recall'ı önceliklendiriyor (0.815, projedeki en yüksek),
precision düşük (0.616, en düşük) — daha fazla lezyon yakalıyor ama daha
çok yanlış alarm üretiyor. Tarama amaçlı bir görevde yüksek recall
(kaçırılan lezyon riskini azaltma) klinik olarak değerli olabilir, F1'de
geride kalsa da (0.7018 vs 0.7379). Grad-CAM kapsam dışı bırakıldı
(transformer mimarisi, YOLO'nun katman-hook yaklaşımı doğrudan
uygulanamaz).

**Genel tablo (üç model, en iyi sonuçlar):**

| Model | En iyi mAP@50 | Ayar |
|---|---|---|
| **YOLO11s** | **0.7561** | SGD, lr0=0.001 (Deney 2e) |
| RF-DETR-S | 0.7551 | batch4×grad_accum4, lr=1e-4 (Deney 3) |
| YOLO26s | 0.6914 | SGD, lr0=0.001 (Deney 2f) |

---

## 5. Sıradaki Adımlar

1. ~~Kırpma işlemini bitirip veri setini doğrulamak.~~ TAMAM.
2. ~~Kırpılmış veride EDA + augmentation görsel kontrolü.~~ TAMAM.
3. ~~YOLO11s'i kırpılmış veriyle eğitip Deney 1 ile karşılaştırmak.~~ TAMAM
   (Deney 2a, patience=5).
4. ~~Deney 2b: aynı ayarlarla `patience=10` deneyip erken durma hipotezini
   test etmek.~~ TAMAM — hipotez çürütüldü, patience=5 yeterliymiş.
5. ~~YOLO26'yı aynı kırpılmış veri ve ortak ayarlarla eğitmek.~~ TAMAM
   (Deney 2c) — mAP@50 0.6494, YOLO11'in altında.
6. ~~Deney 2d: YOLO26'da imgsz=1280 denemek.~~ TAMAM — hipotez çürütüldü,
   sonuç Deney 2c'den kötü (0.6220).
7. ~~Deney 2e: YOLO11'de `optimizer='SGD'` denemek.~~ TAMAM — şimdiye
   kadarki en iyi sonuç (mAP@50 0.7561).
8. ~~Deney 2f: YOLO26'da da `optimizer='SGD'` denemek.~~ TAMAM — 0.6914,
   YOLO11'in altında ama Deney 2c/2d'den iyi.
9. ~~RF-DETR'yi eğitmek.~~ TAMAM (Deney 3) — mAP@50 0.7551, YOLO11s'e
   neredeyse eşit, çok daha az epoch'ta, belirgin şekilde daha yüksek
   recall (0.815) düşük precision (0.616) ile.
10. **Nihai üç-model karşılaştırma raporu** (sırada) — YOLO11/YOLO26 için
    Grad-CAM zaten var; RF-DETR'de kapsam dışı bırakıldı (transformer
    mimarisi ayrı bir yaklaşım gerektiriyor, bkz. Deney 3).
