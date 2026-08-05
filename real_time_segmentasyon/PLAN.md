# Gerçek Zamanlı Meme Ultrason Segmentasyonu — Proje Planı

## Veri Özeti (doğrulandı)

- Kaynak: `real_time_segmentasyon/Dataset_BUSI_with_GT` (BUSI — Breast Ultrasound Images)
- 780 görüntü: benign 437, malignant 210, normal 133
- Format: PNG, 3 kanal (içerik gri tonlamalı), `uint8`
- Maske: binary `{0, 255}`, lezyon vs arkaplan. `normal` sınıfının maskesi tamamen boş
- Çözünürlük: 190×310 – 1048×719 arası, **780 görüntüde 639 farklı çözünürlük**, medyan 564×474
- 17 vakada birden fazla maske dosyası (`_mask_1.png` vb.) → **birleşim (union)** ile tek binary maskeye indirgeniyor
- Anomali yok: her görüntünün maskesi var, boyutlar birebir uyuşuyor
- Ortam: Google Colab (VSCode üzerinden bağlanılan runtime)

## EDA Bulguları

`rt_seg_veri_analizi.ipynb` çıktısı. Kararların tamamı bu sayılara dayanıyor.

### Lezyon morfolojisi

| | benign | malignant |
|---|---|---|
| Alan p5 | %0.60 | %2.81 |
| Alan medyan | %3.79 | %12.06 |
| Alan p95 | %23.65 | %37.20 |
| Eşdeğer çap medyan | 115 px | 209 px |
| Çok lezyonlu vaka | 24 | 10 |

### Gerçek zamanlı çözünürlük analizi

Girdi boyutu doğrudan FPS'i belirliyor, ama küçültünce lezyon piksel olarak kaybolabilir. Ölçüm:

| Girdi | Medyan lezyon | p5 lezyon | Min lezyon | Çapı <16 px olan |
|---|---|---|---|---|
| 192×192 | 2124 px | 272 px | 103 px | %2.9 |
| **256×256** | **3777 px** | **483 px** | **184 px** | **%0.2** |
| 320×320 | 5902 px | 755 px | 287 px | %0.0 |
| 512×512 | 15110 px | 1935 px | 737 px | %0.0 |

**Sonuç**: bu veri setinde lezyonlar büyük; 256×256 girdide lezyonların %99.8'i 16 px çapın üzerinde kalıyor. 512'ye çıkmanın segmentasyon karşılığı zayıf, FPS maliyeti doğrudan. Ana deneyler 256, ablasyon 384.

Kare resize'ın getirdiği en/boy bozulması: medyan 1.20×, p95 1.54×, max 2.05×.

### Kontrast

- Vakaların **%75.4'ü hipoekoik** (lezyon arkaplandan koyu). Kontrast medyanı benign −0.47, malignant −0.31
- Görüntü ortalama parlaklığı 30–170 arası — cihaz/gain farkı veri setinde zaten mevcut

### Kopya / yinelenen görüntüler (kritik bulgu)

İki aşamalı tespit: dHash (64 bit, hamming ≤ 6) ile aday üretimi → 128×128 z-normalize küçük kopyalar arası Pearson korelasyonu + maske IoU ile teyit.

- **115 teyitli kopya çifti** (korelasyon > 0.95)
- **179 görüntü (%23)** çok görüntülü kopya gruplarının içinde; 80 grup, en büyük grup 4
- 6 çift sınıflar arası
- **11 çift "aynı görüntü, çelişkili maske"** (korelasyon yüksek, maske IoU < 0.5):

| A | B | corr | maske IoU |
|---|---|---|---|
| benign (433) | malignant (145) | 1.000 | 0.739 |
| benign (233) | benign (299) | 0.999 | 0.149 |
| malignant (116) | malignant (18) | 0.998 | 0.116 |
| benign (437) | normal (34) | 0.978 | 0.000 |
| benign (269) | normal (1) | 0.977 | 0.000 |

Son iki satır aynı görüntünün biri "lezyon var", diğeri "lezyon yok" diye etiketlenmiş hali.

**Neden önemli**: bu görüntüler rastgele split'te train ve val/test'e dağılırsa model ezberlediği kareyi tekrar görür, Dice yapay olarak şişer. BUSI ile yapılan çalışmaların çoğunun sessizce düştüğü tuzak bu.

## Kararlar

Tamamı kullanıcıyla birlikte, EDA sayıları önde olacak şekilde verildi.

| Konu | Karar | Gerekçe |
|---|---|---|
| `normal` sınıfı | **Dahil** (negatif örnek) | Canlı ultrason akışında karelerin çoğunda lezyon yok; boş maskeler false-positive baskılamayı öğretir |
| Kopya politikası | **Grup-aware split + çelişkili çift temizliği** | Sızıntı ve gürültülü etiket birlikte çözülür, veri kaybı 11 görüntüyle sınırlı |
| Split | **70 / 15 / 15** train / val / test | 3 mimari karşılaştırılacak; model seçimi val'de, final sayılar hiç dokunulmamış test'te |
| Stratify | sınıf × lezyon boyutu (3 bin), **birim = kopya grubu** | Sınıf ve zorluk dağılımı üç bölümde de aynı kalsın |
| Girdi boyutu | **256×256** ana, **384×384** ablasyon | Yukarıdaki çözünürlük analizi |
| Resize | Kare resize (letterbox değil) | Üç mimaride de aynı uygulanır, karşılaştırma adil kalır |
| Augmentasyon | Speckle noise + motion blur dahil, **hafif vs orta ablasyonu** | Real-time dağıtım robustluğu; 769 görüntü küçük, yoğunluk deneyle belirlenecek |
| Modeller | **SegFormer-B0 + YOLOv11-Seg + RTMDet-Ins** | Kullanıcının referans listesi. RTMDet-Ins için mmcv/mmdet kurulumuna ayrı hazırlık oturumu ayrılacak |

### Temizlik kuralı (deterministik, kayıt altında)

Çelişkili çiftlerde tek temsilci bırakılır:

1. Bir üyenin maskesi boşsa o elenir — kopyasında lezyon işaretliyken boş maske kesin hatalı
2. İkisi de doluysa küçük anotasyonlu olan elenir (eksik işaretleme, fazla işaretlemeden daha olası)

`MANUAL_DROP` / `MANUAL_KEEP` listeleriyle elle geçersiz kılınabilir. Elenen her görüntü gerekçesiyle `temizlik_kayit.xlsx`'e yazılır.

Sonuç: **780 → 769** görüntü (benign 430, malignant 208, normal 131).

### Split sonucu (seed 42)

| | benign | malignant | normal | TOPLAM | % |
|---|---|---|---|---|---|
| train | 300 | 146 | 92 | 538 | 70.0 |
| val | 65 | 31 | 20 | 116 | 15.1 |
| test | 65 | 31 | 19 | 115 | 15.0 |

Sınıf oranları: train %55.8/%27.1/%17.1, val %56.0/%26.7/%17.2, test %56.5/%27.0/%16.5.

Doğrulama (notebook'ta `assert` ile): birden fazla bölüme dağılan grup **0**, bölümler arası bölünmüş kopya çifti **0**.

#### Kopya gruplarının dağılımı — val/test kopyasız

| | grup | görüntü | çok görüntülü grup | benzersiz sahne | tekrar oranı |
|---|---|---|---|---|---|
| train | 450 | 538 | 70 (158 görüntü) | 450 | 1.20× |
| val | 116 | 116 | **0** | 116 | 1.00× |
| test | 115 | 115 | **0** | 115 | 1.00× |

70 kopya grubunun tamamı train'e düşüyor. Bu algoritmanın kaçınılmaz sonucu: her stratum'da gruplar büyükten küçüğe işleniyor ve başlangıçta en büyük açık her zaman train'de (0.70×T vs 0.15×T), dolayısıyla büyük gruplar train dolana kadar oraya gidiyor; sıra val/test'e geldiğinde elde yalnızca tekil gruplar kalıyor.

**Bu istenen özellik**: val ve test tamamen kopyasız, her değerlendirme vakası ayrı bir sahne — metrikler içsel tekrarla şişmiyor. Bedeli train tarafında 1.20×'lik örtük ağırlıklandırma (70 sahne 2–4 kez görünüyor), ihmal edilebilir düzeyde. Sınıf ve lezyon boyutu dağılımları stratify sayesinde üç bölümde de dengeli kalıyor.

Gruplar bölümlere yayılsın istenirse `stratified_group_split` içindeki `np.argsort(-size)` sıralaması kaldırılıp gruplar rastgele sırayla işlenebilir — ama o zaman val/test içinde de kopya bulunur.

### Augmentasyon tasarımı

| Karar | Gerekçe |
|---|---|
| `HorizontalFlip` var | Prob yönü zaten değişken |
| **`VerticalFlip` yok** | Ultrasonda derinlik ekseni sabit (cilt üstte); dikey çevirme fiziksel olarak imkânsız görüntü üretir. Density projesinde vardı ama mamografide dikey simetri anlamlıydı |
| `Affine` ±15° / ±%15 ölçek / ±%10 kaydırma | Prob açısı ve cihaz derinlik ayarı farkları |
| `ElasticTransform` alpha=30, sigma=6 | Prob basıncı altında doku deformasyonu. **alpha anlamlı seviyede** — density projesinde alpha=1 pratikte no-op'tu |
| `RandomBrightnessContrast` ±0.2, `RandomGamma` 80–120 | Gain farkı veri setinde zaten var |
| **CLAHE yok** | Vakaların %75.4'ü hipoekoik; kontrast standardizasyonu bu sinyali bozar |
| **`MultiplicativeNoise` (speckle)** | Ultrasonun çarpımsal gürültü fiziği — toplamsal Gaussian'dan gerçekçi |
| **`MotionBlur`** | Canlı akışta prob hareketinden gelen bulanık kareler; real-time hedefin doğrudan gereği |
| **Akustik gölge yok** | Lezyon üstüne denk geldiğinde GT ile görüntü çelişir |

Seviyeler: `hafif` (flip + hafif affine + hafif brightness) ve `orta` (tam set). Ablasyon tek bir mimaride yapılıp kazanan diğerlerine uygulanacak.

**Normalizasyon**: görüntü `uint8` (0–255) okunuyor, dolayısıyla `A.Normalize` varsayılan `max_pixel_value=255.0` ile **doğru**. Bu, density projesinin tersi — orada görüntü Dataset içinde `[0,1]`'e çekildiği için `max_pixel_value=1.0` gerekiyordu ve varsayılan bırakılması dinamik aralığı 255 kat daraltmıştı. Augmentasyon notebook'u tensör aralığını `assert` ile kontrol ediyor.

## Instance Segmentation Formatı ve Değerlendirme Protokolü

SegFormer-B0 semantic segmentation (ikili maske ile eğitilir), YOLOv11-Seg ve RTMDet-Ins ise **instance segmentation** çerçeveleri — ikisi de ikili maske kabul etmiyor, poligon anotasyonu bekliyor (RTMDet: COCO JSON, YOLO: normalize poligon `.txt`).

Bu dönüşümün kayıplı olması, o iki modelin ulaşabileceği Dice'ı baştan sınırlardı ve karşılaştırma mimariyi değil format farkını ölçerdi. 647 lezyonlu maske üzerinde ölçüldü:

| Ölçüm | Dice medyan | min |
|---|---|---|
| gt256 vs poligon doğrudan 256'da rasterize | 0.9760 | 0.8949 |
| **Poligon kaybı izole** (orijinal çözünürlükte) | **0.9969** | 0.9782 |
| İki rasterize yönteminin farkı (kontrol) | 0.9765 | 0.8941 |

İlk satırdaki düşüş poligondan değil, "nearest ile küçültme" ile "poligonu doğrudan 256'da rasterize etme" arasındaki yöntem farkından geliyor — hizalama testindeki tuzağın aynısı, kontrol ölçümüyle ayrıştırıldı. **Poligon temsili pratikte kayıpsız.**

Bundan çıkan protokol kararı:

> **Değerlendirme orijinal çözünürlükte yapılır.** Üç modelin tahmini de orijinal görüntü boyutuna büyütülüp orijinal ikili maskeyle karşılaştırılır. 256'da karşılaştırmak SegFormer'a yapay avantaj verirdi, çünkü onun GT'si zaten nearest ile 256'ya indirilmiş maskedir.

### Çıktı-stride tavanı — girdi boyutu kararının asıl dayanağı

Poligon endişesi ölçülünce ters çıktı: asıl darboğaz poligon değil, **modellerin maskeyi ürettiği ızgara çözünürlüğü.** Her mimari maskeyi girdinin bir bölümünde üretip büyütüyor, bu da "mükemmel tahmin" halinde bile bir tavan koyuyor.

Tavan, maskeyi G×G ızgaraya indirip (alan ortalaması) orijinale bilineer büyütüp 0.5'te eşikleyerek ölçüldü (638 lezyonlu görüntü):

| Izgara | Dice medyan | ortalama | p5 |
|---|---|---|---|
| 32 | 0.9814 | 0.9735 | 0.9289 |
| 48 | 0.9890 | 0.9851 | 0.9616 |
| 64 | 0.9922 | 0.9898 | 0.9754 |
| 96 | 0.9958 | 0.9946 | 0.9869 |
| 128 | 0.9972 | 0.9965 | 0.9916 |

Mimarilerin efektif çıktı stride'ı:

| Model | çıktı stride | kaynak |
|---|---|---|
| SegFormer-B0 | 4 | MLP decode head logitleri girdi/4'te üretir |
| YOLOv11-Seg | 4 | maske prototipleri girdi/4'te |
| RTMDet-Ins | **8** | `mask_feat` stride 8'de; `mask_loss_stride=4` ama tahmin ×2 büyütülerek denetleniyor, bilgi içeriği stride 8 |

Bundan çıkan tablo:

| Model | girdi | ızgara | tavan |
|---|---|---|---|
| SegFormer-B0 / YOLOv11-Seg | 256 | 64 | **0.9922** |
| SegFormer-B0 / YOLOv11-Seg | 384 | 96 | 0.9958 |
| RTMDet-Ins | 256 | 32 | **0.9814** |
| RTMDet-Ins | 384 | 48 | 0.9890 |
| RTMDet-Ins | **512** | 64 | **0.9922** |

Aynı 256 girdide RTMDet %1.1 düşük tavana sahip; küçük lezyonlarda fark açılıyor (çap 40–80 px: 0.948 vs 0.980).

> **Karar (kullanıcı onayı ile): RTMDet-Ins hem 256 hem 512'de eğitilecek.** 256 diğer iki modelle eşit girdi bütçesini, 512 eşit maske tavanını (0.9922) temsil ediyor. Böylece rapor iki tespiti de sayıyla gösterebilir: aynı girdide RTMDet daha kaba maske üretiyor, ve aynı tavanı yakalamak için 4 kat piksel işlemesi gerekiyor — FPS bedeli ölçülecek. Yerel GPU'da kota kısıtı olmadığı için ek maliyet yalnızca duvar saati.

### Poligon çıkarma parametreleri

- `cv2.approxPolyDP` **epsilon = 1.0 px** — Dice 0.9968 (medyan), görüntü başına 44 nokta. epsilon=0 Dice 1.0000 verir ama 241 nokta/görüntü, dosyalar gereksiz şişer; epsilon=2.0'da Dice 0.9936'ya iner.
- **Delikli maskeler**: 647'nin 34'ünde (%5.3) iç delik var. COCO poligon formatı deliği temsil edemez (çoklu poligonlar birleşim olarak yorumlanır, XOR değil), delikler doluyor. Maliyeti ihmal edilebilir: epsilon=0'da orijinal çözünürlükte min Dice 0.9991 — o kayıp zaten yalnızca deliklerden geliyor.
- **Poligona çevrilemeyen minik bileşenler**: 22 görüntüde 10 pikselin altında bileşenler var, toplam **40 piksel**. En kötü tek görüntüde Dice etkisi 0.99977. Ayrı bir eşitleme filtresi uygulanmadı, ölçülüp kayda geçirildi.
- **pycocotools rasterize konvansiyonu**: aynı poligonlar cv2 ile rasterize edildiğinde Dice 0.9968, pycocotools ile 0.9901 çıkıyor. Fark deliklerden değil (delikli maskeler aslında biraz daha yüksek skorluyor), pycocotools'un tarama konvansiyonunun poligonu yarım piksel daraltmasından geliyor. Sabit offset ile telafi denendi, kapatmadı (+0.5 offset 0.9883 → 0.9903). Doğrulama bilerek pycocotools ile yapılıyor: mmdet eğitimde aynı konvansiyonu kullandığı için **modelin gerçekte göreceği GT bu**.

### COCO dönüşümü sonucu (`rt_seg_coco_donusum.py`)

| split | görüntü | instance | boş görüntü | çoklu instance | geri-rasterize Dice medyan | min |
|---|---|---|---|---|---|---|
| train | 538 | 455 | 92 | 9 | 0.99010 | 0.94850 |
| val | 116 | 100 | 20 | 4 | 0.98996 | 0.96800 |
| test | 115 | 98 | 19 | 2 | 0.99040 | 0.96346 |

Boş maskeli 131 `normal` görüntü sıfır anotasyonlu görüntü olarak korundu — negatif örnek işlevi için şart. Çıktılar: `veri/coco/busi_{train,val,test}.json`. YOLO formatı daha sonra aynı JSON'dan türetilecek ki iki instance modeli birebir aynı anotasyonu görsün.

## Yerel Eğitim Ortamı (RTMDet-Ins)

Colab'ın torch 2.11 / cu128 / Python 3.12 ortamına uyan mmcv wheel'i yok (resmî wheel'ler torch 2.1'de, topluluk wheel'leri 2.6'da bitiyor). RTMDet-Ins bu yüzden **yerelde** eğitiliyor: kurulum bir kez yapılır ve kalıcı olur, her Colab oturumunda tekrarlanmaz.

Donanım: RTX 2060 6 GB (Turing, sm_75), sürücü 591.86 / CUDA 13.1.

| Bileşen | Sürüm | Not |
|---|---|---|
| Python | 3.10 | makinede 3.12 yok; mmcv wheel'i cp310 için mevcut, cp313 için hiç yok |
| torch / torchvision | 2.6.0 / 0.21.0 (cu124) | mmcv wheel'inin derlendiği sürüm |
| mmcv | 2.2.0+a8073c7pt2.6.0cu124 | topluluk prebuilt wheel'i, derleme yok |
| mmengine | 0.10.7 | |
| mmdet | 3.3.0 | `mmcv<2.2.0` sürüm kontrolü yamalanır |

Venv: `d:\mamografi\.venv_mmdet` (gitignore'da).

Notlar:

- **numpy 1.26.4'e düşürmek gerekmedi.** Literatürdeki yaygın tavsiye numpy 1.x'e düşmek yönünde (mmcv'nin numpy 1 ABI'sine karşı derlendiği varsayımıyla), ancak bu topluluk build'i numpy 2.2.6 ile sorunsuz çalıştı — CUDA `nms` operatörü GPU'da doğrulandı.
- **pip SSL**: makinedeki TLS araya girme (antivirüs/proxy) nedeniyle pip sertifika hatası veriyordu; venv'e `pip.ini` ile `trusted-host` listesi yazıldı. Sorun yalnızca PyPI'ye özgü — `download.openmmlab.com`'dan ön-eğitimli ağırlık indirme sertifika hatası vermiyor.
- Yerel makinede MSVC yok — prebuilt wheel kullanıldığı için gerekmiyor.

Kurulum `yerel_ortam/kurulum.ps1` ile tekrarlanabilir, doğrulama `yerel_ortam/ortam_dogrulama.py` ile yapılır.

### Ortam doğrulama sonucu

| Kontrol | Sonuç |
|---|---|
| GPU | RTX 2060, sm_75, 6.44 GB |
| mmcv CUDA op'ları | `nms` ✓ `RoIAlign` ✓ |
| RTMDet-Ins-tiny kurulumu | 5.64 M parametre |
| Gerçek BUSI görüntüsünde çıkarım | maske tensörü `(100, 471, 562)` — orijinal çözünürlükte ✓ |
| Hız / tepe VRAM | 36.3 ms/görüntü, **367 MB** (640 pipeline, rastgele ağırlık) |
| Ön-eğitimli ağırlık indirme | ✓ (22.7 MB) |

VRAM tarafında geniş pay var; eğitimde batch 8'in sığması bekleniyor.

### Hız ölçümü nerede yapılacak

RTMDet yerelde torch 2.6'da, diğer ikisi Colab'da torch 2.11'de eğitilecek. Farklı torch sürümlerinde ölçülen FPS karşılaştırılabilir olmaz. Bu yüzden **final gecikme/FPS ölçümü üçü için de tek ortamda, yerel RTX 2060 üzerinde** yapılacak (batch=1, warmup sonrası, aynı protokol). Eğitim nerede yapıldığından bağımsız.

Yan fayda: RTX 2060 sınıfı bir kart, T4/A100 gibi veri merkezi kartlarına göre gerçek bir ultrason cihazının yanındaki makineye daha yakın — real-time iddiası için daha savunulabilir zemin. ONNX/TensorRT o zaman "adaleti sağlayan araç" olmaktan çıkıp asıl olması gereken şeye dönüşür: raporlanan bir optimizasyon deneyi.

## Kod Yapısı

Density projesindeki notebook-başına-deney deseni korunuyor; veri tarafı ise tek seferlik ve ortak.

```
real_time_segmentasyon/
├── PLAN.md
├── rt_seg_veri_analizi.ipynb      # EDA — envanter, morfoloji, kontrast, kopya tespiti, çözünürlük analizi
├── rt_seg_veri_hazirlik.ipynb     # temizlik + grup-aware split -> veri_manifest.csv
├── rt_seg_augmentasyon.ipynb      # transform + Dataset tanımı, görsel/sayısal doğrulama
└── rt_seg_egitim_<MODEL_KEY>.ipynb   # model başına eğitim (henüz yazılmadı)
```

Drive çıktı yapısı:

```
/content/drive/MyDrive/real_time_segmentasyon/
├── Dataset_BUSI_with_GT/          # veri (elle yüklendi)
├── eda/                           # EDA çıktıları + figurler/
├── veri/                          # veri_manifest.csv, splits.json, temizlik_kayit.xlsx
├── augmentasyon/                  # doğrulama görselleri ve raporu
└── <MODEL_KEY>/                   # model başına eğitim çıktıları
```

Yerel ayna: `/content/outputs/<alt_klasor>/` — her Drive yazımından sonra `mirror_to_local()`. Density projesindeki desenin aynısı.

`veri_manifest.csv` bundan sonraki tüm eğitim notebook'larının okuyacağı **tek kaynak**: `case_id, cls, split, group_id, image_path, mask_paths, height, width, lesion_px, lesion_ratio, n_components, eq_diam`.

Veri seti her oturumda Drive'dan `/content/dataset`'e kopyalanıyor — 780 küçük PNG'yi her epoch Drive üzerinden okumak eğitimin en yavaş kısmı olurdu (density projesinde aynı darboğaz `.npy` cache ile çözülmüştü).

## Kayıt / Log Standardı

`mass_lezyon_projesi` ve `mendeley_data_density` ile birebir aynı desen: **Excel tabanlı epoch logu + Drive kalıcı depo + yerel mirror + resume-safe checkpoint**.

- `<MODEL_KEY>_log.xlsx` — epoch bazlı canlı log, `append_epoch_row(..., resume_epoch=...)` ile resume'da tekrar satır oluşmaz
- `checkpoints/last.pt` (her epoch) + `best.pt` (val Dice iyileştiğinde) — model + optimizer + scheduler + scaler + epoch + best_dice + `epochs_without_improve` + RNG state
- `<MODEL_KEY>_curves.png`, `<MODEL_KEY>_cases/`, `<MODEL_KEY>_test_summary.xlsx`
- Sabit değerlendirme vakaları tüm modellerde ortak (`eval_cases.json`), modeller arası görsel karşılaştırma adil olsun diye
- Tüm metrikler manuel TP/FP/FN üzerinden, **görüntü başına** hesaplanıp batch içinde ortalanır (`f1 = dice` binary'de özdeş)

### Real-time'a özgü ek ölçümler

Bu proje density'den burada ayrılıyor — segmentasyon kalitesi tek başına yeterli değil:

- Çıkarım gecikmesi (ms/kare) ve FPS — batch=1, warmup sonrası, GPU ve CPU ayrı
- Parametre sayısı ve FLOPs
- Optimizasyon sonrası ölçümler (ONNX / TensorRT / yarı hassasiyet)
- Dice–FPS ödünleşim grafiği: üç mimari + iki girdi boyutu

## Boş maskeli örneklerde metrik

`normal` sınıfı dahil edildiği için Dice'ın özel ele alınması gerekiyor: GT boş ve tahmin de boşken Dice tanımsız (0/0). Bu vakalar `dice = 1` sayılacak ve raporlarda **lezyonlu / boş maskeli kırılım ayrı verilecek** — aksi halde tek bir ortalama Dice, 131 boş maskeli görüntünün etkisiyle yanıltıcı olur.

## Şu Anki Durum

- [x] **`rt_seg_veri_analizi.ipynb` Colab'da çalıştırıldı** — tüm çıktılar Drive'da, sayılar yerel doğrulamayla birebir örtüşüyor
- [x] **`rt_seg_veri_hazirlik.ipynb` Colab'da çalıştırıldı** — 769 görüntü, split 538/116/115, sızıntı 0, `veri_manifest.csv` üretildi
- [x] **`rt_seg_augmentasyon.ipynb` Colab'da çalıştırıldı** (albumentations 2.0.8) — tensör aralığı `[-2.118, 2.64]`, ikili ihlal 0, lezyon kaybı 0, **hizalama IoU 1.0 / merkez mesafesi 0.0 px**
- [x] **Veri fazı kapandı** — üç notebook da çalıştı, çıktılar Drive'da
- [x] **RTMDet-Ins yerel ortamı kuruldu ve doğrulandı** — `yerel_ortam/`
- [x] **COCO dönüşümü** — `rt_seg_coco_donusum.py`, 538/116/115 görüntü, geri-rasterize Dice 0.990
- [x] **RTMDet eğitim config'i yazıldı ve doğrulandı** — `rtmdet/rtmdet_ins_busi.py`

Config doğrulama sonucu (256, batch 8, `orta` augmentasyon):

| Kontrol | Sonuç |
|---|---|
| Tüm veri seti pipeline testi | 538/538 (446 dolu + 92 boş), hata 0 |
| Girdi tensörü / GT maske | (3,256,256) uint8 / (1,256,256) |
| Adım süresi | 0.175 sn |
| Epoch süresi | ~12 sn (68 adım) |
| 100 epoch tahmini | **~20 dakika** |
| Tepe VRAM | 352 MB / 6144 MB |

Kayıp sağlığı (ön-eğitimli ağırlıkla 150 iterasyon): `loss_cls` ilk 10 iterasyon ortalaması 86 → son 25 iterasyon ortalaması 7.9; `loss_bbox` ~0.50, `loss_mask` ~0.24 baştan stabil. (Rastgele ağırlıkla ölçüldüğünde `loss_cls` 2201 çıkıyordu — bu yalnızca başlatma artefaktı.)

- [ ] Excel epoch logu + resume-safe checkpoint hook'u (proje standardı)
- [ ] Orijinal çözünürlükte Dice değerlendirmesi (üç model için ortak)
- [ ] RTMDet-Ins eğitimi: 256 ve 512
- [ ] SegFormer-B0 ve YOLOv11-Seg eğitim notebook'ları (Colab)
- [ ] Gerçek zamanlı çıkarım + optimizasyon
- [ ] Final rapor (HTML + PDF, density projesindeki `rapor/build_report.py` deseniyle)

### Karşılaşılan ve düzeltilen hatalar

**1. Kopya gruplarında hizalama hatası (EDA, `c14` + grup hücresi).** `pairs` çerçevesi korelasyona göre sıralanıyordu ama kenar kurulumunda kullanılan `I`/`J` dizileri sıralanmamış orijinal düzende kalıyordu; `I[kop]` maskesi sıralı çerçeveden geldiği için **yanlış çiftler** birleştiriliyordu. Kopya sayısı (115) doğru göründüğü için gözden kaçabilirdi — belirti yalnızca "gruplanan görüntü 189" ile "etkilenen benzersiz görüntü 179" arasındaki tutarsızlıktı.

Sonuç yanlış grup yapısıydı (678 grup / 189 görüntü, doğrusu 681 / 179) ve doğrudan split'e giriyordu: birbirinin kopyası 10 görüntü yanlış gruplara atanacağı için train/test arasına dağılabilirdi.

Düzeltme: `i`/`j` satır indisleri `pairs`'in kolonu yapıldı, sıralamayla birlikte taşınıyor. Ayrıca kalıcı kontrol eklendi:

```python
assert gruplanan == etkilenen, f'grup kurulumu tutarsiz: {gruplanan} != {etkilenen}'
```

**2. `md_table` tip yükseltmesi (hazırlık notebook'u).** `iterrows()` her satırı tek dtype'lı Series'e çevirdiği için int kolonlar float'a yükseliyor, rapor tablolarında sayımlar `538.0` diye yazılıyordu. Kolon kolon okumaya geçildi.

**3. float32 gösterimi.** Korelasyon değerleri float32 olduğu için `round(3)` sonrası bile `0.9990000128746033` basılıyordu. `astype(np.float64)` ile çözüldü.

**4. Hizalama testinin kendisi hatalıydı (augmentasyon notebook'u).** Görüntü–maske hizalamasını ölçmek için görüntü kanalına maskenin kendisi besleniyor ve çıkan görüntü eşiklenip çıkan maskeyle IoU'su alınıyordu. Ancak gerçek pipeline görüntüde bilineer, maskede nearest interpolasyon kullanıyor (doğrusu da bu) — bu fark tek başına IoU'yu düşürüyor:

| Senaryo | IoU medyan | min |
|---|---|---|
| Bilineer görüntü + eşik vs nearest maske | 0.9702 | 0.8206 |
| Her iki akış da nearest | **1.0000** | **1.0000** |

647 lezyonlu maske üzerinde ölçüldü; hiçbir augmentasyon olmadan yalnızca 256'ya resize ile. Küçük lezyonlarda kayıp daha büyük (çap <40 px olanlarda medyan 0.935), çünkü sınır bandı alana oranla büyük.

Yani test geometri uyuşmazlığını değil interpolasyon farkını ölçüyordu ve 0.97 eşiği yanlış alarm veriyordu. Düzeltme: test kopyasında her iki akış da nearest kullanıyor (beklenen değer tam 1.0), ayrıca interpolasyondan bağımsız ikinci ölçü olarak lezyon merkezi mesafesi eklendi. Sayılar `assert`'ten önce basılıyor — önceki sürümde hata anında hiçbir değer görünmüyor, ayrıca `hizalama` tanımsız kaldığı için rapor hücresi de `NameError` veriyordu.

Bu ölçüm ayrıca kendi başına faydalı: maske için nearest dışında bir interpolasyon kullanılırsa GT'nin ne kadar bozulacağını gösteriyor.

**5. mmdet'in `Albu` transformu boş anotasyonlu örneklerde çöküyor.** `_postprocess_results` maske boyutunu `results['masks'][0].shape` üzerinden okuyor (mmdet 3.3.0, `transforms.py:1770`); hiç maske kalmadığında `IndexError`. Hemen altındaki `elif` dalında doğru davranış zaten var, sadece bu dalda kullanılmamış.

Etkisi büyüktü: **538 eğitim görüntüsünün 92'si** boş maskeli `normal` vakalar, yani bilerek dahil ettiğimiz negatif örneklerin tamamı. Belirti sinsiydi — hata ancak o örnekler batch'e düştüğünde ortaya çıkıyordu, aynı doğrulama iki kez çalıştırıldığında biri geçip diğeri patlıyordu.

Çözüm: kütüphaneyi düzenlemek yerine `rtmdet/ozel_transformlar.py` içinde `AlbuBosGuvenli` alt sınıfı yazıldı (maske boyutunu augmentasyon sonrası görüntüden alıyor). Config `custom_imports` ile onu kullanıyor.

Kalıcı kontrol: `config_dogrulama.py` artık **tüm 538 örneği tek tek pipeline'dan geçiriyor** — rastgele batch'e güvenmiyor.

**6. `keep_ratio=False` maske hizalamasını bozuyordu — projedeki en pahalı hata.** RTMDet-Ins'in maske geri-ölçekleme kodu tek ölçek katsayısı varsayıyor; kare resize ile x/y katsayıları farklı olunca (örn. 473×323 görüntü için 1.85 vs 1.26) maske görüntüyle hizalanmıyordu.

Belirtiler ve yanlış teşhisim: `segm_mAP` her epoch tam 0 çıkıyordu, `bbox_mAP` ise 0.578'e tırmanmıştı. Bunu "maske dalı zayıf" diye yorumladım — **yanlıştı**. Ölçüm kanıtladı:

| Ölçüm | Sonuç |
|---|---|
| Maskenin kendi tahmin kutusu içinde kalan oranı | medyan 0.592, min 0.000 |
| Dice — gerçek maskeler | 0.488 |
| Dice — sadece tahmin kutusunu doldurmak | **0.804** |

Kutuyu doldurmanın gerçek maskeyi geçmesi, kaybın model kalitesinden değil taşımadan geldiğinin kanıtıydı.

Çözüm: `Resize(keep_ratio=True)` + kareye `Pad` (letterbox). Tek ölçek katsayısı olduğu için sorun kaynağında kalkıyor. Sonuç, **15 epoch'ta** (öncekinde 44 epoch'tu):

| | keep_ratio=False (ep 44) | letterbox (ep 15) |
|---|---|---|
| Lezyonlu Dice | 0.488 | **0.787** |
| IoU | 0.368 | **0.705** |
| Precision | 0.455 | **0.798** |
| Recall | 0.550 | **0.844** |
| `segm_mAP` | 0.000 | 0.390 |

> **Karar değişikliği**: girdi hazırlama kare resize yerine **letterbox** oldu. EDA'daki kare-resize kararı (bozulma faktörü medyan 1.20) bu bulgu karşısında geçersiz kaldı. YOLOv11-Seg zaten letterbox kullandığı için modeller arası tutarlılık da arttı; SegFormer tarafında da aynı hazırlık uygulanmalı ki karşılaştırma adil kalsın.

**7. Eğitimi rastgele bir epoch'ta öldüren `IndexError`.** Albu bir örneğin tek instance'ını düşürdüğünde (Affine görüntüden çıkarıyor / `min_visibility` eliyor) `gt_bboxes` sıfıra iniyor ama `gt_ignore_flags` eski uzunluğunda kalıyor — `label_fields` listesinde olmadığı için filtrelenmiyor. `PackDetInputs` bundan `valid_idx` üretip boş diziyi indeksliyor (`formatting.py:102`).

İlk koşu bu yüzden epoch 47'de düştü. Aralıklı olduğu için erken doğrulamalarda görünmedi. Düzeltme `AlbuBosGuvenli` içinde bayrakların da `idx_mapper` ile filtrelenmesi; doğrulama 3 tur × 538 örnek = **1614 örnekte 0 hata**.

### Notlar

- Bu repodaki `.ipynb` dosyaları ile Colab'da fiilen çalışan oturum ayrı şeyler — buradaki değişiklikler Colab'a otomatik yansımıyor.
- Sıralı çalıştırma zorunlu: `veri_analizi` → `veri_hazirlik` → `augmentasyon`. İkincisi birincinin Excel çıktılarını, üçüncüsü ikincinin manifest'ini okuyor.
