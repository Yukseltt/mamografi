# Mamografi Kitle (Mass) Lezyon Tespiti Projesi

Bu doküman, mamografi görüntülerinde kitle lezyonlarının nesne tespiti üzerine
geliştirilen projenin tüm detaylarını içerir. Amaç, üç farklı mimariyi
(YOLO11, YOLO26, RF-DETR) aynı veri seti üzerinde adil biçimde eğitip
karşılaştırmak, overfitting oluşturmadan mümkün olan en yüksek mAP değerine
ulaşmaktır.

Geliştirme ortamı Google Colab, GPU olarak T4 kullanılıyor.

---

## 1. Proje Hedefi

Kitle (mass) lezyonlarının nesne tespiti. Her model için:

- Uygun eğitim stratejileri, veri artırma (augmentation) ve hiperparametre
  ayarlarıyla overfitting olmadan en yüksek mAP.
- Tüm deneylerin raporlanması: kullanılan model ve parametreler, eğitim ve
  test metrikleri (mAP, Precision, Recall, F1-Score), eğitim eğrileri
  (loss, mAP, Precision, Recall).
- Grad-CAM analizleri.
- Confusion matrix.
- Hangi yaklaşımın daha başarılı olduğunun değerlendirilmesi.

Metrik olarak hem mAP@0.5 hem mAP@0.5:0.95 raporlanacak.

ROC eğrisi çizilmeyecek. Gerekçe: ROC bir sınıflandırma metriğidir ve nesne
tespitinde "negatif örnek" kavramı doğrudan tanımlı olmadığı için uygun
değildir. Nesne tespitinde standart olan Precision-Recall eğrisidir; mAP@0.5
zaten bu eğrinin altındaki alandır. PR eğrisi Ultralytics tarafından otomatik
üretiliyor.

---

## 2. Veri Seti

- Kaynak: VinDr-Mammo + Teknofest verisinin birleşimi. Orijinal klasör adı
  `Kitle-Vindr&Tekno`.
- Google Drive'da tutuluyor, Colab'a mount ediliyor.
- Klasör yapısı klasik YOLO düzeni: `train/`, `val/`, `test/`, her birinde
  `images/` ve `labels/`.
- Format: `.png` görüntüler, YOLO formatında `.txt` etiketler.
- Görüntüler tek kanal, gri tonlama, çok yüksek çözünürlük (örnekler:
  2364x2964, 2812x2012, 2800x3518 civarı).

### Sınıf yapısı

- Tek sınıf: `mass`. `nc = 1`, sınıf id `0`.
- Yani saf lokalizasyon problemi. Benign/malignant veya BI-RADS alt sınıfı yok.
- Sınıf id'leri temiz (0..n-1 aralığında), class-ordering bug riski yok.

### Split boyutları (orijinal veri)

| Split | Görüntü | Kutu |
|-------|---------|------|
| train | 2671    | 3203 |
| val   | 333     | 418  |
| test  | 336     | 400  |

Toplam 3340 görüntü, 4021 kutu. Yaklaşık 80/10/10 dağılım.

### Keşif (EDA) bulguları

- Görüntü başına ortalama ~1.2 kutu (çoğu görüntüde tek lezyon).
- **Kritik bulgu: küçük nesne baskın.** Kutuların yaklaşık %72'si görüntü
  alanının %1'inden küçük. Medyan bbox alan oranı ~0.00552 (tipik lezyon
  görüntünün binde 5'i kadar).
- Her split'in labels klasöründe bir `classes.txt` dosyası var; eğitim bunu
  yok sayar, sorun değil.
- Görüntü-label eşleşmesinde ciddi sorun yok.

Bu küçük nesne baskınlığı, hem imgsz seçimini hem de sonradan eklenen kırpma
(crop) ön işlemesini doğrudan yönlendirdi.

---

## 3. Literatür Özeti

### Gerçekçi mAP hedefi

Kitle tespiti zor bir problemdir. VinDr-Mammo ve benzeri setlerde literatürdeki
tipik sonuçlar mAP@0.5 için kabaca **0.50-0.70** aralığındadır. Bu aralığın
üstü çok iyi, altı normal kabul edilir. Bu referans, rapor değerlendirmesinde
kullanılacak.

### Transfer learning

Üç model için de en önemli strateji. Literatür, mamografi ön eğitiminin mAP ve
F1'i belirgin artırdığını gösteriyor. Biz COCO ön eğitimli ağırlıklarla
başlıyoruz (üç model için de standart).

### Augmentation (medikale özel)

- Yatay flip: anatomik olarak makul (sol/sağ meme simetrisi), açık.
- Dikey flip: meme dokusunun göğüs duvarından meme ucuna doğru olan doğal
  yönelimini bozar. Literatürde mamografi augmentation listelerinde belirgin
  değil. Kapalı.
- Renk augmentation'ları (hue, saturation): gri tonlamalı görüntüde anlamsız.
  Kapalı.
- Mozaik: küçük veride aynı görüntüyü tekrar kullanabilir, küçük lezyonları
  parçalayabilir. Bu projede kapatıldı (aşağıda augmentation bölümüne bakınız).
- Augmentation yalnızca train setine uygulanır; val ve test'e asla. Bu, adil
  değerlendirme için şarttır.

### Overfitting kontrolü

Train ve val loss eğrileri birlikte ve düzenli düşüyorsa sağlıklıdır; makas
açılırsa (train düşerken val platoya girer/yükselir) overfitting başlamıştır.
Ana araç early stopping (patience) ve train-val metrik farkının izlenmesidir.

### Modeller

- **YOLO11:** Olgun, kararlı CNN. Baseline.
- **YOLO26:** Ocak 2026'da çıktı. STAL (small-target-aware label assignment)
  içerir, küçük nesnelerde avantajlı. Ultralytics paketiyle geliyor.
- **RF-DETR:** Transformer tabanlı (DINOv2 backbone), NMS'siz. Küçük veri ve
  domain adaptasyonunda güçlü, T4'te çalışabiliyor. Ultralytics dışı, kendi
  eğitim API'si var. Küçük veri setlerinde yakınsama için 100+ epoch
  gerektirebilir.

Not: Model adı **RF-DETR** (Roboflow), RT-DETR değil. İkisi farklı modeller.

---

## 4. İki Deney Yapısı

Proje iki ayrı deney içeriyor. İkincisi, birincinin eksiğini gidermek için
eklendi.

### Deney 1 — Kırpmasız (baseline)

İlk yaklaşım. Orijinal görüntüler doğrudan kullanıldı, sadece imgsz büyütüldü.

- YOLO11s bu düzende eğitildi ve tamamlandı.
- **Test sonuçları (YOLO11s, kırpmasız):**
  - mAP@50: 0.6928
  - mAP@50-95: 0.3820
  - Precision: 0.7475
  - Recall: 0.6365
  - F1-Score: 0.6876
- Overfitting yok: test (0.6928) ile validation (0.701) neredeyse aynı.
- Early stopping epoch 61'de en iyi sonucu buldu, 81'de durdu. 81 epoch ~6.6
  saat (Drive'dan yavaş okuma nedeniyle uzun).
- Eğitim eğrisi: ilk ~30 epoch train ve val loss birlikte düştü; sonra hafif
  ayrışma (train düşmeye devam, val plato), ama val mAP artmaya devam etti,
  bu yüzden overfitting zararsız kaldı.

Bu sonuç literatürün üst ucunda ve sağlam bir baseline. **Deney 1 korunuyor,
silinmeyecek.** Raporda "kırpmanın etkisi" için doğal referans olacak
(aynı model, kırpmalı vs kırpmasız).

### Deney 2 — Kırpmalı (asıl deney)

Kullanıcının gözlemiyle eklendi: meme dokusu görüntünün ancak %30-40'ını
kaplıyor, gerisi siyah arka plan. Bu dev görüntüler küçültülünce küçük
lezyonlar birkaç piksele iniyor. Otsu benzeri bir eşikle meme bölgesini kırpmak
efektif çözünürlüğü 2-3 kat artırır, model küçük lezyonlara daha iyi odaklanır.

**Karar: üç modeli de (YOLO11 dahil) kırpılmış veriyle baştan eğitmek.**
Gerekçe: eğer sadece bazı modeller kırpmalı olursa, modeller arası mAP farkının
ne kadarının mimariden, ne kadarının kırpmadan geldiği ayrılamaz. Tam adil
karşılaştırma için üçü de aynı kırpılmış veriyle eğitilecek.

Bu doküman esas olarak Deney 2'nin sağlam kurulumunu anlatır.

### Deney 2a — YOLO11s, kırpılmış, patience=5 (ilk deneme)

Kırpılmış veriyle yapılan ilk eğitim. Deney 1'de (kırpmasız) 40. epoch'tan
sonra overfit gözlenmişti; buna karşı üç ek önlem alındı: `optimizer=AdamW`
(lr0=0.001, cos_lr=True), `weight_decay=0.001` (varsayılanın 2 katı), ve
agresif early stopping `patience=5` (öncekinde 20). Mozaik de kısmen
açıldı (`mosaic=0.3`, `close_mosaic=10`) — Deney 1'de farkında olmadan 1.0
idi, ilk Deney 2 planı 0.0 öngörmüştü, ikisinin ortası denendi. Diğer
ayarlar (imgsz=1024, batch=8, epochs limiti 100, seed=42, diğer
augmentation) Deney 1 ile aynı.

**Sonuç: eğitim 20 epoch'ta durdu** (patience=5), en iyi model **epoch
15**'te bulundu. Train/val loss eğrileri paralel düştü, ayrışma yok —
overfitting hedefi bu ayarlarla karşılandı.

**Test sonuçları (YOLO11s, kırpılmış, patience=5):**
- mAP@50: 0.6901
- mAP@50-95: 0.3669
- Precision: 0.6823
- Recall: 0.6325
- F1-Score: 0.6565

| Metrik | Deney 1 (kırpmasız, 81 epoch) | Deney 2a (kırpılmış, 20 epoch) |
|---|---|---|
| mAP@50 | 0.6928 | 0.6901 |
| mAP@50-95 | 0.3820 | 0.3669 |
| Precision | 0.7475 | 0.6823 |
| Recall | 0.6365 | 0.6325 |
| F1 | 0.6876 | 0.6565 |

**Değerlendirme:** Sonuç Deney 1'e çok yakın, hafifçe düşük — kırpmanın
beklenen faydası (küçük nesne oranı %72→%39.3, efektif çözünürlük artışı)
bu turda mAP'e yansımadı. Sebep muhtemelen overfitting değil **erken
durma**: mAP eğrisi epoch 20'ye kadar gürültülü ama genel yükseliş
eğilimindeydi, val seti küçük olduğundan (332-333 görüntü) epoch-to-epoch
mAP dalgalanması patience=5'i erken tetiklemiş olabilir. Confusion
matrix'te de Deney 1'e göre daha fazla false positive var (208 vs 169),
modelin kalibre olmaya yeterli süresi olmadığına işaret ediyor.

**Karar: Deney 2b olarak `patience=10` ile ayrı bir deneme yapılacak**,
diğer tüm ayarlar (AdamW, cos_lr, weight_decay=0.001, mosaic=0.3) aynı
kalacak. Overfitting kontrolü bu ayarlarla zaten işe yaradığı için
patience'ı gevşetmek güvenli kabul edildi. Sonuçlar ayrı model klasöründe
(`yolo11s_cropped_p10`) tutulacak, Deney 2a'nın (`yolo11s_cropped`) üzerine
yazılmayacak.

### Deney 2b — YOLO11s, kırpılmış, patience=10 (doğrulama denemesi)

Deney 2a ile birebir aynı kurulum, tek fark `patience=5 → 10`. Amaç: 2a'nın
erken durma hipotezini test etmek (val mAP hâlâ yükseliş eğilimindeyken
durmuştu, küçük val setinin gürültüsü şüpheleniliyordu).

**Sonuç: eğitim 25 epoch'ta durdu**, en iyi model yine **epoch 15**'te —
patience'ı iki katına çıkarmak 10 epoch daha (16-25) şans tanıdı ama
hiçbiri epoch 15'i geçemedi. Test sonuçları Deney 2a ile **bire bir aynı**
çıktı (seed=42 ile epoch 15'e kadar eğitim deterministik, aynı checkpoint
tekrar en iyi seçildi):

- mAP@50: 0.6901, mAP@50-95: 0.3669, Precision: 0.6823, Recall: 0.6325,
  F1: 0.6565.

25 epoch 2.7 saat sürdü (2a'nın 20 epoch/1.6 saatine göre ~1.1 saat
fazladan, kazanç sıfır).

**Sonuç: erken durma hipotezi çürütüldü.** Epoch 15 gerçek bir yerel
tavan — küçük val setinin gürültüsü değil, modelin bu ayarlarla
ulaşabildiği gerçek sınır. `patience=5` zaten yeterliydi, `patience=10`
gereksiz zaman kaybı oldu (gelecekte benzer küçük val setlerinde
`patience=5-7` aralığı tercih edilmeli).

**Genel değerlendirme (Deney 2a + 2b):** YOLO11s + kırpılmış veri + bu
optimizasyon/augmentation ayarlarıyla mAP@50 tavanı **~0.69** — Deney 1'in
(kırpmasız) 0.6928'ine pratikte eşit, üstüne çıkamadı. Kırpmanın beklenen
faydası (küçük nesne oranı %72→%39.3) bu iki denemede de mAP'e yansımadı.
YOLO11s + kırpma tarafı bu konfigürasyonla tavana çarptı kabul edilip
**sıradaki model olan YOLO26'ya geçilecek** (bkz. bölüm 11 model eğitim
sırası). Kırpmanın gerçek faydası, üç modelin tamamı bitince toplu
karşılaştırmada daha net değerlendirilecek.

### Deney 2c — YOLO26s, kırpılmış

Deney 2a/2b'nin doğruladığı ayarlarla birebir aynı (`patience=5`, AdamW,
lr0=0.001, cos_lr=True, weight_decay=0.001, mosaic=0.3, diğer augmentation
aynı) — tek değişken mimari (`yolo26s.pt`, YOLO11s ile aynı 's' boyut
sınıfı). Eğitim sırasında Colab bağlantısı koptu; `epoch10.pt`'den (numaralı
save_period checkpoint) resume edildi, epoch 12'den devam edip epoch 15'te
en iyi sonucu buldu, epoch 20'de `patience=5` ile durdu — YOLO11'deki
desenle aynı.

**Test sonuçları (YOLO26s, kırpılmış):**
- mAP@50: 0.6494
- mAP@50-95: 0.3483
- Precision: 0.6551
- Recall: 0.6316
- F1-Score: 0.6432

| Metrik | Deney 1 (YOLO11s, kırpmasız) | Deney 2a/2b (YOLO11s, kırpılmış) | Deney 2c (YOLO26s, kırpılmış) |
|---|---|---|---|
| mAP@50 | 0.6928 | 0.6901 | 0.6494 |
| mAP@50-95 | 0.3820 | 0.3669 | 0.3483 |
| Precision | 0.7475 | 0.6823 | 0.6551 |
| Recall | 0.6365 | 0.6325 | 0.6316 |
| F1 | 0.6876 | 0.6565 | 0.6432 |

**Değerlendirme:** YOLO26s, YOLO11s'in her iki versiyonundan da tüm
metriklerde düşük çıktı. Beklenenin aksine STAL'ın (small-target-aware
label assignment) bu veri setinde avantaj sağladığı görülmedi. Olası
sebepler: (1) hiperparametreler YOLO11 için doğrulanmıştı, YOLO26'nın
farklı mimarisine doğrudan aktarıldı, doğrulanmadı; (2) aynı erken durma
deseni (epoch 15 tavan, patience=5) burada da tekrarladı — modelin gerçek
potansiyeli patience=10 ile test edilmedi (Deney 2b YOLO11 için bunun
gereksiz olduğunu göstermişti, ama bu YOLO26'ya genellenemez).

**Grad-CAM mimari notu:** YOLO26, YOLO11'den farklı olarak **NMS'siz
(end2end)** `Detect` başı kullanıyor — `cv2`/`cv3` dalları `None`, yerine
`one2one_cv2` (kutu regresyonu) ve `one2one_cv3` (sınıf logit'i) var.
Kaynak katmanlar (`[16, 19, 22]`, P3/P4/P5) aynı kaldığı için Grad-CAM
kodunda tek değişiklik `cv3[0]` → `one2one_cv3[0]` oldu, hedef katman (16)
aynen kullanıldı. Bu, RF-DETR'ye geçildiğinde de mimari farklılıkların
Grad-CAM hook noktalarını etkileyebileceğinin bir hatırlatıcısı (bkz.
bölüm 9).

### Deney 2d — YOLO26s, kırpılmış, imgsz=1280

Deney 2c'de YOLO26s, YOLO11s'in altında kalmıştı (0.6494 vs 0.6901) ama
overfitting yoktu — sorunun regularizasyon değil kapasite/çözünürlük
olabileceği düşünüldü. Kırpma sonrası bile küçük nesne oranı ~%39
olduğundan, tek değişken olarak **`imgsz: 1024 → 1280`** denendi, diğer
her şey (patience=5, AdamW, cos_lr, weight_decay=0.001, mosaic=0.3) aynı
kaldı.

**Sonuç: beklenenin tersine, çözünürlük artışı işe yaramadı.** Eğitim 28
epoch sürdü, overfitting yine yoktu, ama test sonucu Deney 2c'den **daha
kötü** çıktı:

| Metrik | Deney 2c (imgsz=1024) | Deney 2d (imgsz=1280) |
|---|---|---|
| mAP@50 | 0.6494 | **0.6220** |
| mAP@50-95 | 0.3483 | **0.3367** |
| Precision | 0.6551 | **0.6772** |
| Recall | 0.6316 | **0.5875** |
| F1 | 0.6432 | **0.6292** |

Confusion matrix: TP=242, FP=154, FN=158 — kaçırılan lezyon sayısı Deney
2c'ye göre arttı (158 vs 128), recall düştü. Eğitim sırasında görülen val
mAP zirvesi (~0.66) ile gerçek test sonucu (0.622) arasında da belirgin
bir fark vardı — bu, val/test genellemesinde bir tutarsızlığa işaret
ediyor, küçük test setinde (336 görüntü) örnekleme gürültüsü olabilir.

**Değerlendirme:** Çözünürlük artırma hipotezi çürütüldü. Olası sebepler:
imgsz artışının `mosaic=0.3`/`scale=0.3` gibi diğer augmentation'larla
etkileşimi, ya da basitçe bu mimari/veri seti kombinasyonunda 1024'ün
zaten yeterli olması. **Karar: YOLO26 tarafında çözünürlük değil,
optimizer (bkz. Deney 2f) denenecek.**

### Deney 2e — YOLO11s, kırpılmış, `optimizer='SGD'` (şimdiye kadarki en iyi sonuç)

Deney 2a/2b'nin AdamW tavanı (mAP@50 0.6901), Deney 1'in kırpmasız
baseline'ının (0.6928, `optimizer=auto`) hafifçe altında kalmıştı — AdamW'nin
bu problem için en iyi seçim olduğu net değildi. Tek değişken olarak
`optimizer='SGD'`, `lr0=0.001` (AdamW'nin ölçeğiyle aynı, temkinli bir SGD
başlangıcı), `cos_lr=True`, `weight_decay=0.001`, `patience=5` denendi;
diğer her şey (imgsz=1024, batch=8, mosaic=0.3) sabit kaldı.

**Sonuç: eğitim 19 epoch'ta durdu, train/val loss paralel düştü —
overfitting yok.**

**Test sonuçları (YOLO11s, kırpılmış, SGD):**
- mAP@50: **0.7561**
- mAP@50-95: **0.4121**
- Precision: **0.7802**
- Recall: **0.7000**
- F1-Score: **0.7379**

| Metrik | Deney 1 (kırpmasız) | Deney 2a/2b (AdamW) | **Deney 2e (SGD)** |
|---|---|---|---|
| mAP@50 | 0.6928 | 0.6901 | **0.7561** |
| mAP@50-95 | 0.3820 | 0.3669 | **0.4121** |
| Precision | 0.7475 | 0.6823 | **0.7802** |
| Recall | 0.6365 | 0.6325 | **0.7000** |
| F1 | 0.6876 | 0.6565 | **0.7379** |

Her metrikte önceki tüm denemelerin belirgin şekilde üzerinde (mAP@50'de
+6.6 puan). Confusion matrix: TP=302, FP=129, FN=98 — Deney 1'e göre hem
daha az kaçırılan lezyon hem daha az yanlış alarm. **Sonuç: AdamW bu
problem için doğru seçim değilmiş; SGD (uygun düşük `lr0` ile) belirgin
şekilde daha iyi. Bu, projenin şu anki en iyi YOLO11 sonucu.**

### Deney 2f — YOLO26s, kırpılmış, `optimizer='SGD'`

Deney 2c/2d'nin AdamW tavanı (0.6494 / 0.6220), Deney 2e'nin YOLO11s için
SGD'nin (uygun düşük `lr0` ile) AdamW'yi açık farkla geçtiğini göstermesi
üzerine aynı düzeltme burada da denendi: `imgsz` Deney 2c'nin 1024'üne
geri döndü, `optimizer='SGD'`, `lr0=0.001` — doğrudan bu değerden
başlandı (Deney 2e'nin `lr0=0.01`'in kararsızlaştırdığı bulgusuna
dayanarak), diğer her şey aynı kaldı.

**Sonuç: eğitim 16 epoch'ta durdu, ilk denemede hiç kararsızlık
yaşanmadı (train/val loss düzgün paralel düştü) — overfitting yok.**

**Test sonuçları (YOLO26s, kırpılmış, SGD):**
- mAP@50: 0.6914
- mAP@50-95: 0.3686
- Precision: 0.7011
- Recall: 0.6392
- F1-Score: 0.6687

| Metrik | Deney 2c (AdamW, 1024) | Deney 2d (AdamW, 1280) | Deney 2f (SGD, 1024) |
|---|---|---|---|
| mAP@50 | 0.6494 | 0.6220 | **0.6914** |
| mAP@50-95 | 0.3483 | 0.3367 | **0.3686** |
| Precision | 0.6551 | 0.6772 | **0.7011** |
| Recall | 0.6316 | 0.5875 | **0.6392** |
| F1 | 0.6432 | 0.6292 | **0.6687** |

**Değerlendirme:** SGD > AdamW hipotezi YOLO26'da da doğrulandı (+4.2 puan
mAP@50), ama YOLO11'deki sıçramanın (+6.6 puan) gerisinde kaldı. Confusion
matrix'te FP belirgin düştü (90), precision iyileşti; recall artışı daha
mütevazı. **YOLO26, en iyi ayarlarıyla bile YOLO11s'in en iyi sonucunun
(0.7561) altında kalıyor** — artık optimizer'dan değil, muhtemelen
mimarinin bu veri setine (küçük, tek sınıf, gri tonlama) daha az uygun
olmasından kaynaklanıyor.

**Genel tablo (iki model, en iyi sonuçlar):**

| Model | En iyi mAP@50 | Ayar |
|---|---|---|
| **YOLO11s** | **0.7561** | SGD, lr0=0.001 (Deney 2e) |
| YOLO26s | 0.6914 | SGD, lr0=0.001 (Deney 2f) |

### Deney 3 — RF-DETR-Small, kırpılmış (COCO)

Üçüncü ve son planlanan mimari. YOLO'dan tamamen farklı bir kütüphane
(`rfdetr`, PyTorch Lightning tabanlı) ve veri formatı (COCO, `_annotations.coco.json`)
kullandığından, kod yazmadan önce resmi dokümantasyon araştırıldı (bkz.
bölüm 9'daki not, RF-DETR ≠ RT-DETR ayrımı bölüm 3'te). `Kitle-Vindr-Cropped`
(YOLO format) dokunulmadan `Kitle-Vindr-Cropped-COCO` adıyla ayrı bir COCO
kopyası üretildi (train/valid/test, `_annotations.coco.json`).

**Model:** `RFDETRSmall` — YOLO11s/YOLO26s ile "s" boyut paritesi için.
**Ayarlar:** `batch_size=4, grad_accum_steps=4` (efektif batch 16, T4 için
resmi öneri), `lr=1e-4`, `early_stopping_patience=5` (diğer iki modelle
aynı kural), `epochs=100` (üst sınır, gerçekte erken durdu).

**Sonuç: eğitim sadece 9 epoch'ta erken durdu** (val mAP@50-95 EMA 0.438'de
platoya girdi, patience=5 tetiklendi) — YOLO'nun 16-19 epoch'una göre çok
daha hızlı yakınsadı, güçlü ön-eğitimli transformer omurgasının (DINOv2)
avantajına işaret ediyor.

**Test sonuçları (RFDETRSmall, kırpılmış, COCO):**
- mAP@50: **0.7551**
- mAP@50-95: **0.4109**
- Precision: 0.6163
- Recall: **0.8150**
- F1-Score: 0.7018

| Metrik | YOLO11s (Deney 2e) | YOLO26s (Deney 2f) | **RF-DETR-S (Deney 3)** |
|---|---|---|---|
| mAP@50 | **0.7561** | 0.6914 | 0.7551 |
| mAP@50-95 | **0.4121** | 0.3686 | 0.4109 |
| Precision | **0.7802** | 0.7011 | 0.6163 |
| Recall | 0.7000 | 0.6392 | **0.8150** |
| F1 | **0.7379** | 0.6687 | 0.7018 |

Confusion matrix (IoU=0.5, conf=0.25): TP=326, FP=203, FN=74 (400 test
kutusu). AP alan kırılımı (pycocotools): küçük nesne yok (kırpma sonrası
zaten neredeyse hiç yok, bkz. bölüm 5 EDA), orta AP=0.273, büyük AP=0.432.

**Değerlendirme:** mAP@50/mAP@50-95'te YOLO11s + SGD'ye (Deney 2e, şu ana
kadarki en iyi) neredeyse eşit (fark <0.15 puan) — ama **çok daha az
epoch'ta**. Precision/recall dengesi belirgin şekilde farklı: RF-DETR
recall'ı önceliklendiriyor (0.815, projedeki en yüksek), precision'ı
düşük (0.616, en düşük) — yani daha fazla lezyonu yakalıyor ama daha çok
yanlış alarm üretiyor. Kütle tespiti gibi tarama amaçlı bir görevde
yüksek recall (kaçırılan lezyon riskini azaltma) klinik olarak genelde
daha değerli kabul edilir, dolayısıyla F1'de YOLO11s'in gerisinde kalsa
da (0.7018 vs 0.7379) RF-DETR'nin profili kullanım amacına göre tercih
edilebilir. `CONF_THR=0.25` üç modelde de aynı tutuldu; RF-DETR'ye özgü
bir eşik ayarı (precision/recall dengesini kaydırmak için) denenmedi,
olası bir sonraki adım.

**Genel tablo (üç model, en iyi sonuçlar):**

| Model | En iyi mAP@50 | Ayar |
|---|---|---|
| **YOLO11s** | **0.7561** | SGD, lr0=0.001 (Deney 2e) |
| RF-DETR-S | 0.7551 | batch4×grad_accum4, lr=1e-4 (Deney 3) |
| YOLO26s | 0.6914 | SGD, lr0=0.001 (Deney 2f) |

---

## 5. Kırpma (Crop) Ön İşlemesi — Deney 2

Kırpma bir **ön işleme** adımıdır: her görüntüye bir kez uygulanır,
deterministiktir, diske kaydedilir. Augmentation değildir.

### Yöntem

Kullanıcının daha önce başka bir projede (TwoViewDensityNet / Nguyen
notebook'ları) sahada test ettiği kanıtlanmış yöntem alındı ve nesne tespitine
uyarlandı.

- Sabit eşik (`BG_THRESH = 10`) ile meme dokusu siyah arka plandan ayrılır
  (Otsu değil; sabit eşik daha öngörülebilir).
- 9x9 disk ile morfolojik açma (opening): küçük artefaktları ve ayrık metin
  işaretlerini (L-CC, R-MLO gibi) temizler.
- `connectedComponentsWithStats` ile en büyük bağlı bileşen = meme. Bu adım
  köşedeki yazı işaretlerini otomatik dışlar.
- **Hem x hem y ekseninde** kırpma (bounding box). Yalnızca x kırpmak üstteki
  köşe yazılarını bırakabildiği için y de kırpılıyor.
- %5 kenar payı (padding) eklenir, lezyon kenarda kesilmesin diye.
- Güvenlik koşulu: bulunan meme genişliği görüntünün **%15'inden** azsa kırpma
  iptal edilir, orijinal döndürülür. (Başlangıçta %35 idi; bazı gerçek dar
  memeler — yağlı/küçük meme, class a tipi — %34 gibi değerlerde yanlışlıkla
  iptal ediliyordu. %15'e düşürülünce gerçek dar memeler de kırpılıyor, ama
  tamamen hatalı çok küçük kırpmalar yine engelleniyor.)

### Bounding box koordinat dönüşümü (kritik)

Nesne tespitinde kırpma yaparken etiket kutularının koordinatları da yeni
kırpılmış çerçeveye göre güncellenmelidir, yoksa kutular kayar. Bu, kullanıcının
eski sınıflandırma projelerinde olmayan (gerek olmayan) bir adımdır ve bu
projeye özel olarak eklenip test edilmiştir.

- Normalize xywh kutu, orijinaldeki mutlak piksele çevrilir.
- Kırpma ofseti (x1, y1) çıkarılır, yeni kırpma boyutuna (nw, nh) yeniden
  normalize edilir.
- Hem x hem y dönüşür (çünkü hem x hem y kırpılıyor).

Doğrulama: kırpma sonrası kutu merkezinin gerçekten lezyonun (parlak bölge)
üstüne düştüğü, yerelde piksel değeri kontrolüyle test edildi (değer ~200,
lezyon parlak). Geri dönüşüm testi de tam tuttu.

### Kırpma fonksiyonu (referans kod)

Üç notebook'un da (ve ön işleme adımının) kullanacağı fonksiyon:

```python
import cv2
import numpy as np

BG_THRESH = 10       # sabit arka plan esigi
MORPH_KERNEL = 9     # acilma disk boyutu
PAD_RATIO = 0.05     # meme etrafi pay

def crop_breast_xy(img_gray, padding_ratio=PAD_RATIO):
    # en buyuk bagli bilesen kutusu ile iki eksende kirp
    # dondurur: x1, x2, y1, y2, kirpildi_bayragi
    h_img, w_img = img_gray.shape
    _, binary = cv2.threshold(img_gray, BG_THRESH, 255, cv2.THRESH_BINARY)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, disk)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        return 0, w_img, 0, h_img, False
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component = (labels == largest).astype(np.uint8)
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return 0, w_img, 0, h_img, False
    x1v, x2v = xs.min(), xs.max()
    y1v, y2v = ys.min(), ys.max()
    w = x2v - x1v
    if w < 0.15 * w_img:
        return 0, w_img, 0, h_img, False
    px = int(w * padding_ratio)
    py = int((y2v - y1v) * padding_ratio)
    x1 = max(0, x1v - px)
    x2 = min(w_img, x2v + px)
    y1 = max(0, y1v - py)
    y2 = min(h_img, y2v + py)
    return x1, x2, y1, y2, True

def transform_bbox_xy(box_norm, x1, x2, y1, y2, ow, oh):
    # kirpma sonrasi hem x hem y koordinati donusur
    cx = box_norm[0] * ow
    cy = box_norm[1] * oh
    bw = box_norm[2] * ow
    bh = box_norm[3] * oh
    nw = x2 - x1
    nh = y2 - y1
    return (cx - x1)/nw, (cy - y1)/nh, bw/nw, bh/nh
```

### İşleme stratejisi

Drive'dan tek tek okuma çok yavaş (~2.8 sn/görüntü; ayrıca binlerce küçük
label dosyasını Drive'da açmak dakikalar sürüyor). Bu yüzden Nguyen
notebook'undaki strateji kullanılıyor:

1. Tüm veriyi hızlı yerel diske (`/content/cropped_local`) işle.
2. `shutil.copytree` ile Drive'a toplu kopyala.

Kırpılmış veri seti hedefi: `Kitle-Vindr-Cropped` (orijinal `Kitle-Vindr&Tekno`
korunuyor).

İşleme sonrası doğrulama raporu: kaç görüntü işlendi, kaç tanesi kırpıldı, kaç
tanesi güvenlik nedeniyle atlandı, kaç kutu korundu. Kutu sayıları orijinalle
aynı olmalı (train 3203, val 418, test 400).

---

## 6. Augmentation Politikası

**Önemli ilke: augmentation diske kaydedilmez, eğitim sırasında canlı
uygulanır.** Sadece kırpma diske kaydedilir.

Gerekçe:
- Augmentation'ın değeri her epoch'ta farklı varyasyon göstermektir; kaydedilirse
  bu çeşitlilik kaybolur.
- Augmentation yalnızca train'e uygulanmalı; kaydedilirse val/test'e sızma
  riski doğar.
- **Farklı model aileleri:** YOLO (Ultralytics) ve RF-DETR augmentation'ı kendi
  hatlarında uygular. Önceden kaydedilmiş augmentation, modelin kendi
  augmentation'ının üstüne biner ve kontrolsüz/bozuk sonuç verir. Doğrusu:
  veriye sadece kırpma uygulanır, augmentation her modelin kendi eğitim
  ayarında tanımlanır. Politika üç modelde de aynı tutulur (adil karşılaştırma).

### Bu deneyde augmentation kararları

Augmentation'lar tek tek, bilinçli seçiliyor. Sorun çıkarabilecekler
kısıtlanıyor. Deney 1'de farkında olmadan Ultralytics varsayılanları
(random erasing 0.4, mozaik 1.0) aktifti; Deney 2'de bunlar bilinçli olarak
gözden geçiriliyor.

- **Random erasing: KAPALI (0.0).** Deney 1'de farkında olmadan 0.4 idi.
  Küçük lezyonları kısmen silme riski var.
- **Mozaik: KAPALI.** Küçük lezyonları parçalayabilir.
- Yatay flip (fliplr): AÇIK (0.5).
- Dikey flip (flipud): KAPALI (0.0), anatomik gerekçe.
- Hue, saturation: KAPALI (gri tonlama).
- Hafif parlaklık (hsv_v): düşük (~0.2).
- Hafif rotasyon, translasyon, ölçek: düşük değerlerde.
- MixUp: KAPALI (iki lezyonu karıştırmak yanıltıcı).

Not: Kırpılmış görüntüler üzerinde augmentation'lar Aşama C'de görsel olarak
tek tek kontrol edilecek (gerçek Ultralytics çıktısında, tahminle değil).

---

## 7. Ortak Eğitim Ayarları (üç model için sabit)

Adil karşılaştırmanın temeli: mimari dışında her şey sabit.

- **imgsz: 1024.** Gerekçe: küçük nesne oranı %72. Orijinal ~2364x2964, medyan
  lezyon ~175x175 piksel. 640'a küçültünce lezyon ~47 piksele iner, kaçırılır.
  1024'te ~75 piksel kalır. T4 belleğinin kaldırabildiği ve üç modelin de
  altından kalkabildiği denge. (Not: Deney 2'de kırpma sonrası efektif
  çözünürlük daha da artacak.)
- **batch: 8** (T4 OOM verirse 4'e düşülür).
- **epochs: 100**, YOLO'lar için. RF-DETR için daha uzun gerekebilir (100+).
- **patience: 20** (early stopping).
- **seed: 42.**
- COCO ön eğitimli ağırlıklar (transfer learning).
- Metrikler: mAP@50, mAP@50-95, Precision, Recall, F1.

Model boyutu: YOLO11 için `yolo11s` seçildi (Deney 1). Deney 2'de de küçük/small
varyantlarla devam edilecek (T4 kısıtı).

---

## 8. Sağlam Altyapı (üç notebook'ta ortak)

Üç model farklı çıktı üretir; hepsi tek bir standart şemaya çevrilir, böylece
grafikler ve raporlar birebir karşılaştırılabilir olur. Bu, kullanıcının
"grafiklerde standart istiyorum" talebinin karşılığıdır.

### 8.1 Standart metrik şeması

Tüm modeller şu sabit kolonlara sahip bir Excel üretir:

```
epoch, train_loss, val_loss, precision, recall, mAP50, mAP50_95, f1
```

### 8.2 Canlı Excel logging (her epoch)

- Her epoch sonunda Excel'e yazılır (kullanıcının önceki alışkanlığı).
- Drive'a yazılır, Colab koparsa kaybolmaz.
- Resume'da tekrar eden satırlar temizlenir (`resume_epoch` ve sonrası atılır).
- YOLO tarafında Ultralytics callback ile (`on_fit_epoch_end`).
- Metrik eşlemesi: Ultralytics `metrics/precision(B)`, `metrics/recall(B)`,
  `metrics/mAP50(B)`, `metrics/mAP50-95(B)` anahtarlarını standart şemaya çevirir.
- **val_loss:** Ultralytics tek değer olarak sunmaz, ama `trainer.metrics`
  içindeki `val/box_loss + val/cls_loss + val/dfl_loss` toplamından alınır.
  Böylece ayrı bir val geçişi gerekmez, ek süre yükü minimum. Bu sayede
  train-val loss makası (overfitting göstergesi) grafikte görünür olur.

### 8.3 Checkpoint ve resume

- YOLO: Ultralytics `save_period=10` (her 10 epoch checkpoint) + `resume=True`.
- `project` Drive yolu olduğu için checkpoint'ler kalıcı.
- Eğitim koparsa ilgili hücrede `RESUME=True` yapıp aynı hücre çalıştırılır,
  `last.pt`'den devam eder.

### 8.4 Standart eğitim eğrileri (4 grafik)

Rapor talebindeki dört eğri: **loss (train+val), mAP (50 ve 50-95), Precision,
Recall.** 2x2 tek figürde, tek bir `plot_training_curves` fonksiyonuyla. Üç
modelde de aynı fonksiyon, aynı görünüm. `marker="o"`, `grid(alpha=0.3)`,
`dpi=150` (kullanıcının stil tercihi).

### 8.5 Confusion matrix (kullanıcının stilinde)

- Ultralytics `plots=True` ile kendi confusion matrix'ini de üretir, ama rapor
  tutarlılığı için kullanıcının stilinde (`imshow(cmap="Blues")`, hücre içi
  sayılar, dpi=150) yeniden çiziliyor.
- Tek sınıf olduğu için 2x2: mass vs background.
- Tüm test seti taranıp TP/FP/FN, IoU eşleştirmesiyle (IoU eşiği 0.5,
  conf eşiği 0.25) toplanır. Matristen hesaplanan precision/recall, test
  metrikleriyle yakın çıkmalı (doğrulama).
- background-background hücresi nesne tespitinde tanımsız, 0 gösterilir.

### 8.6 Test değerlendirmesi

En iyi checkpoint (`best.pt`) TEST split'inde değerlendirilir (`split='test'`).
`box.map50`, `box.map` (50-95), `box.mp` (precision), `box.mr` (recall);
F1 bunlardan hesaplanır. Küçük bir özet Excel'i yazılır.

---

## 9. Grad-CAM (nesne tespiti için, elle hook)

Grad-CAM kullanıcının EfficientNet örneğindeki mantığın nesne tespitine
uyarlanmış hali: elle hook, bağımlılık yok, kontrol tam. Bu kısım çok sayıda
teşhis ve düzeltme gerektirdi; aşağıda kök sebepler ve nihai çözüm var.

### Çözülen sorunlar ve kök sebepleri

1. **Cihaz uyumsuzluğu (CUDA vs CPU):** Model GPU'da, girdi CPU'daydı. Çözüm:
   modeli `.to(device)` ile GPU'ya sabitle, girdiyi `.to(device)` ile taşı,
   `predict` çağrılarından sonra da cihazı tekrar teyit et.

2. **Boş/anlamsız ısı haritası — yanlış katman:** YOLO11 Detect başı üç ölçekten
   beslenir: layer 16 (P3, küçük nesne), layer 19 (P4, orta), layer 22 (P5,
   büyük). Bir tespit hangi ölçekten geldiyse gradyan sadece ona akar. Bu veride
   nesneler küçük, dolayısıyla P3'ten (layer 16) tespit ediliyor. Teşhis:
   layer 19 ve 22'ye gradyan hiç akmıyor (gradyan abs sum = 0), sadece layer
   16'ya akıyor. **Doğru hedef katman: layer 16.** (Layer 22 seçmek en büyük
   hataydı; küçük lezyonlar o ölçekte temsil edilmiyor.)

3. **Zayıf gradyan — sigmoid doygunluğu:** Sigmoid sonrası skordan (0.52) geri
   yaymak gradyanı sigmoid'in düz bölgesinde eziyor (abs sum ~5.9). Çözüm: Detect
   başının `cv3[0]` dalından **sigmoid öncesi ham logit** alınır. Gradyan ~113'e,
   sonra top-k logit toplamıyla ~7900'e çıktı (20-70 kat güçlenme).

4. **Kopuk Grad-CAM — tespitle bağlantısızlık:** Rastgele en güçlü logit'ten
   yaymak, ısı haritasını modelin gerçek tespitine bağlamıyordu. **Nihai çözüm:
   Grad-CAM'i gerçek tespit kutusuna bağlamak.** Doğru tespitlerde modelin tespit
   ettiği kutunun merkezindeki P3 hücresinden, kaçırılan durumlarda gerçek tümör
   kutusunun merkezindeki P3 hücresinden geri yayılır. Böylece ısı haritası tam
   olarak o tespiti/lezyonu temsil eder.

5. **Görsel temizlik:** forward hook `None` döndürmeli (tuple döndüren lambda
   çıktıyı bozuyordu); `retain_grad()` gerekli; `torch.enable_grad()` bloğu
   gerekli; model parametrelerinde `requires_grad` açık olmalı. Gaussian blur
   (sigma ~20) noktasal benekleri yumuşak ısı bulutlarına çevirir. Doku maskesi
   (`gray > 15`) ısı haritasını memede tutar, siyah arka plandaki sahte sıcak
   bölgeleri siler. CAM eşiği ~0.25 (blur sonrası genişleme için).

### Grad-CAM görselleştirme

- Test setinden otomatik seçim: her görüntü predict edilip IoU eşleştirmesiyle
  `dogru` / `kacirilan` / `yanlis_alarm` olarak sınıflanır.
- 9'lu grid (3x3): karışık örnekler (doğru + kaçırılan + yanlış alarm).
- **Gerçek tümör kutuları parlak yeşil çiziliyor.** Böylece modelin baktığı yer
  (ısı haritası) ile gerçek lezyon konumu (yeşil kutu) yan yana görünür. Doğru
  tespitte örtüşür, kaçırılan/yanlış alarmda ayrışır — yorumlanabilirlik için
  değerli.

### Grad-CAM referans mantığı (compute_overlay özü)

```python
# hedef: tespit kutusu merkezi (varsa), yoksa gercek tumor merkezi
# hook: layer 16 aktivasyon + cv3[0] ham logit
# logits[0, :, gy, gx].sum() ten geri yay  (gy,gx = hedef merkezin P3 hucresi)
# cam = relu( (grad.mean * activation).sum )  -> normalize
# gaussian blur (sigma ~20) -> doku maskesi (gray>15) -> JET overlay -> yesil GT kutu
```

Not: RF-DETR transformer olduğu için Grad-CAM YOLO'daki gibi doğrudan
çalışmayacak; hedef katman ve logit erişimi RF-DETR'ye sıra gelince ayrıca
kurulup test edilecek. Yukarıdaki her şey YOLO11/YOLO26 (ikisi de Ultralytics)
için geçerli — ama birebir aynı değil: YOLO26 NMS'siz (end2end) `Detect`
başı kullanıyor, `cv2`/`cv3` dalları `None`, yerine `one2one_cv2`/
`one2one_cv3` var (bkz. Deney 2c, bölüm 4). Kaynak katman numarası (16)
aynı kaldı, sadece logit dalının adı değişti. RF-DETR'ye geçilince benzer
bir isim/yapı farkı beklenmeli, körü körüne kopyalamadan önce
`type(model)`/`print(model)` ile mimariyi teşhis et.

---

## 10. Klasör ve Çıktı Düzeni

- Orijinal veri: `/content/drive/MyDrive/Kitle-Vindr&Tekno` (korunuyor).
- Kırpılmış veri: `/content/drive/MyDrive/Kitle-Vindr-Cropped`.
- Model çıktıları: her model ayrı Drive klasörüne. Deney 1 örneği:
  `/content/drive/MyDrive/mass_detection/yolo11s/`.
- Her model klasöründe: `<model>_log.xlsx` (canlı log), `<model>_mass/weights/`
  (checkpoint'ler: last.pt, best.pt), `<model>_curves.png` (4 eğri),
  `<model>_confusion_matrix.png`, `<model>_gradcam.png`,
  `<model>_test_summary.xlsx`.

### data.yaml (tek sınıf)

```yaml
path: <veri seti koku>
train: train/images
val: val/images
test: test/images
nc: 1
names: ['mass']
```

---

## 11. Yol Haritası ve Durum

### Aşamalar (Deney 2)

- **Aşama A — Kırpma EDA:** TAMAM. Kırpma görsel olarak doğrulandı (tümör
  kutuları doğru, yazılar temizleniyor, dar memeler de kırpılıyor).
- **Aşama B — Tüm veriyi kırpma:** TAMAM. `Kitle-Vindr-Cropped` yazıldı;
  Drive'a toplu kopyalama sırasında bazı label dosyaları (train'de 563 adet)
  sessizce eksik kalmıştı, kaynak veriden yeniden üretilip tamamlandı. Kutu
  sayıları doğrulandı: train 3203, val 418, test 400 — orijinalle birebir
  aynı.
- **Aşama C — Kırpma sonrası EDA + augmentation kontrolü:** TAMAM. Kırpılmış
  veride küçük nesne oranı %72'den %39.3'e düştü. Augmentation'lar gerçek
  Ultralytics pipeline'ıyla (`build_yolo_dataset`) görsel olarak önizlendi.
- **Aşama D — Sağlam altyapı:** TAMAM (YOLO11 için). Standart şema, Excel,
  grafik, confusion matrix, Grad-CAM `mamosis_mass_lezyon_yolo11s_kirpilmis.ipynb`
  içinde kuruldu.
- **Aşama E — Üç modeli sırayla eğitme:** TAMAM (üçü de). YOLO11 en iyi
  sonuç Deney 2e (SGD), mAP@50 0.7561. YOLO26 Deney 2c/2f, mAP@50 0.6914
  (SGD) — YOLO11'in altında. RF-DETR-S (Deney 3), mAP@50 0.7551 — YOLO11s'e
  neredeyse eşit ama çok daha az epoch'ta (9), belirgin şekilde daha yüksek
  recall (0.815) düşük precision (0.616) ile.

### Model eğitim sırası

1. ~~YOLO11 (baseline, en oturmuş).~~ TAMAM — Deney 1 (kırpmasız, 0.6928),
   Deney 2a/2b (AdamW, kırpılmış, 0.69 civarı), **Deney 2e (SGD, kırpılmış,
   0.7561 — en iyi sonuç)**.
2. ~~YOLO26 (aynı Ultralytics altyapısı).~~ TAMAM — Deney 2c (kırpılmış,
   patience=5). mAP@50 0.6494, YOLO11'in her iki versiyonundan da düşük.
   Grad-CAM için `cv3` → `one2one_cv3` mimari farkı teşhis edilip düzeltildi.
3. ~~RF-DETR (COCO formatına dönüşüm gerektirir).~~ TAMAM — Deney 3,
   mAP@50 0.7551, sadece 9 epoch'ta erken durdu. Grad-CAM kapsam dışı
   bırakıldı (transformer/DINOv2 mimarisi, YOLO'nun katman-hook yaklaşımı
   doğrudan uygulanamaz, bkz. bölüm 9).

Üçü de tamamlandı. Sıradaki: nihai üç-model karşılaştırma raporu (bkz.
ILERLEME_RAPORU.md, Sıradaki Adımlar).

---

## 12. Çalışma Prensipleri

Bu proje boyunca benimsenen ilkeler:

- Koddan önce düşün, emin değilsen sor, sessiz varsayım yapma.
- Önce sadelik, minimum kod, overengineering yok.
- Cerrahi müdahale: sadece gerekli yeri değiştir.
- Deneme yanılmadan önce kök sebebi anla (özellikle Grad-CAM'de teşhis
  hücreleriyle sorunun kaynağı bulunup öyle çözüldü).
- Her adımdan görsel/sayısal olarak emin olmadan bir sonrakine geçme.
- EDA ve görsel kontrol eğitimden önce yapılır.
- Yorum satırları: İngilizce açıklama / Türkçe açıklama, sade metin, süsleme yok.
