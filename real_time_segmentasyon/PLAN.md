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
├── rt_seg_coco_donusum.py         # manifest -> COCO JSON (RTMDet + YOLO'nun ortak anotasyon kaynağı)
├── rt_seg_yolo_donusum.py         # COCO JSON -> YOLO-seg etiketleri (geri-rasterize Dice 1.0)
├── rt_seg_egitim_segformer.ipynb  # Colab
├── rt_seg_egitim_yolo.ipynb       # Colab
├── ortak/                         # üç modelin paylaştığı zemin
│   ├── degerlendirme.py           # Dice protokolü — tek tanım, mmdet/ultralytics bağımlılığı yok
│   ├── eval_vakalari.py           # sabit değerlendirme vakalarını seçer
│   └── eval_cases.json            # 11 vaka: sınıf × boyut kovası + 3 boş maskeli
└── rtmdet/                        # yerel eğitim (mmdet)
```

`ortak/degerlendirme.py` üç mimarinin ortak soyutlaması: her model için
`harita_fn(satır) -> (H, W) skor haritası` yazılır, eşik taraması bunun üzerinde döner.

- SegFormer: lezyon sınıfının piksel olasılığı
- instance modeller: her pikselde, onu kaplayan instance'ların maksimum skoru

Instance modellerde bir haritayı `t`'de eşiklemek "skoru ≥ t olan instance'ların birleşimi"ne
birebir eşit, dolayısıyla tek bir eşik taraması üç modelde de aynı anlama geliyor.

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

- [x] **Ortak Dice değerlendirmesi** — `rtmdet/dice_degerlendirme.py`, orijinal çözünürlükte, boş maske kırılımı ayrı
- [x] **Excel epoch logu + eğitim eğrileri** — `rtmdet/egitim_grafikleri.py`
- [x] **RTMDet-Ins eğitimi tamamlandı: 256 ve 512**
- [x] **Metrik ayrışması çözüldü** — `rtmdet/metrik_ayrismasi.py`; sebep mmdet'teki 1 px maske hatası, düzeltme `CocoMetricHizali`. Ana metrik: orijinal çözünürlükte Dice
- [x] **Ortak değerlendirme zemini** — `ortak/degerlendirme.py` + `ortak/eval_cases.json` (11 sabit vaka)
- [x] **YOLO formatı** — `rt_seg_yolo_donusum.py`, COCO JSON'dan türetildi, geri-rasterize Dice **1.0** (normalize etme kayıpsız)
- [x] **SegFormer-B0 ve YOLOv11-Seg Colab notebook'ları yazıldı**
- [x] **YOLOv11n-Seg eğitildi** (Colab, 100 epoch) — val 0.7697, test 0.6936
- [x] **SegFormer-B0 eğitildi** (Colab, 100 epoch) — val 0.8195, test 0.7365
- [x] **Üç mimarinin de eğitimi ve test değerlendirmesi tamamlandı**
- [x] **Sabit vaka görselleri üç modelde de üretildi** + hizalama dördü için ortak fonksiyonla ölçüldü
- [x] **Eşleştirilmiş karşılaştırma** — `uc_model_karsilastirma.py`, Colab sayıları yerelde birebir yeniden üretildi
- [x] **Gerçek zamanlı ölçüm** — `hiz_olcumu.py` (GPU + CPU + saf ağ) ve `onnx_olcumu.py`
- [x] **Final rapor** — `rapor/rt_seg_sonuc_raporu.pdf`

### Colab oturumlarının tasarımı

| Konu | SegFormer-B0 | YOLOv11-Seg |
|---|---|---|
| Kütüphane | `transformers`, `nvidia/mit-b0` encoder + taze head | `ultralytics`, `yolo11n-seg.pt` |
| Parametre | 3.8 M | 2.9 M (RTMDet-Ins-tiny 5.6 M) |
| Çıktı stride | 4 (notebook `assert` ile doğruluyor) | 4 |
| Girdi | 256 letterbox | 256 letterbox (Ultralytics varsayılanı) |
| Anotasyon | ikili maske | `veri/yolo/labels/` ← COCO JSON |
| Model seçimi | **her epoch orijinal çözünürlükte Dice** | Ultralytics'in mask mAP'i, sonra Dice ile eşik seçimi |

SegFormer'da model seçimi bilerek raporlanan metriğin kendisiyle yapılıyor — RTMDet'te
seçimin `segm_mAP` üzerinden olması ve o metriğin bozuk çıkması pahalıya patlamıştı.

**YOLO augmentasyon paritesi**: Ultralytics'in `Albumentations` sarmalayıcısı maskeleri
taşımıyor, o yüzden geometrik dönüşümler Ultralytics parametrelerinden
(`degrees=15, translate=0.1, scale=0.15, fliplr=0.5` = bizim `Affine`'imiz), fotometrikler
albumentations enjeksiyonundan geliyor. `mosaic/mixup/copy_paste/hsv_*` kapalı.
Eşlenemeyen tek kalem **`ElasticTransform`** (p=0.25) — geometrik olduğu için maske
taşınamıyor; raporda not düşülecek.

## Üç Mimarinin Karşılaştırması

Ortak zemin: aynı split (538/116/115, grup-aware, kopyasız val/test), aynı letterbox
kuralı, aynı augmentasyon zinciri, aynı Dice protokolü (orijinal çözünürlük, boş maske
kırılımı ayrı), eşik val'de `dice_tum` ile seçilip test'e sabit uygulandı.

| | SegFormer-B0 | RTMDet-Ins 256 | RTMDet-Ins 512 | YOLOv11n-Seg |
|---|---|---|---|---|
| Tip | semantic | instance | instance | instance |
| Parametre | 3.8 M | 5.6 M | 5.6 M | **2.9 M** |
| Eğitim yeri | Colab | yerel RTX 2060 | yerel RTX 2060 | Colab |
| Seçilen eşik | 0.4 | 0.4 | 0.2 | 0.2 |
| **Val lezyonlu Dice** | **0.8195** | 0.8084 | 0.8109 | 0.7697 |
| **Test lezyonlu Dice** | **0.7365** | 0.6998 | 0.7101 | 0.6936 |
| Test Dice (tüm) | **0.7800** | 0.7407 | 0.7406 | 0.7268 |
| Test IoU (lezyonlu) | **0.6515** | 0.6261 | 0.6331 | 0.6105 |
| Test precision / recall | 0.7603 / 0.7686 | 0.7241 / 0.7186 | 0.7198 / 0.7615 | 0.6664 / 0.7982 |
| Test boş görüntü doğru | **19/19** | 18/19 | 17/19 | 17/19 |
| Hizalama kayması (kazanç) | **0.0002** | (0,0) zirve | — | 0.0012 |

SegFormer-B0 ortalamada önde, **ama eşleştirilmiş test bunu desteklemiyor** — aşağıya
bakınız. Sıralamayı ortalamalara bakarak yapmak bu fark büyüklüğünde yanıltıcı.

Dikkat çeken: SegFormer boş görüntülerin **hepsinde** temiz (19/19), diğerleri 17–18/19.
Semantic model instance önerisi üretmediği için yanlış pozitif üretme eğilimi daha düşük;
gerçek zamanlı akışta karelerin çoğu lezyonsuz olduğu için bu pratik bir avantaj.

### Eşleştirilmiş karşılaştırma: test'te hiçbir model ayrışmıyor

`uc_model_karsilastirma.py` dört konfigürasyonu **yerelde, tek ortamda** çalıştırıp aynı
görüntüler üzerinde eşleştirilmiş test yapıyor. Önce Colab sayıları yeniden üretiliyor;
dördünde de **sapma 0.0000**, yani yerel çıkarım yolu Colab ile birebir aynı.

Test (96 lezyonlu görüntü), altı çiftin tamamı:

| A | B | fark | %95 GA | Wilcoxon p | A iyi / B iyi | ayrışıyor mu |
|---|---|---|---|---|---|---|
| SegFormer | RTMDet 256 | +0.037 | −0.008 … +0.088 | 0.62 | 38 / 50 | hayır |
| SegFormer | RTMDet 512 | +0.026 | −0.020 … +0.075 | 1.00 | 41 / 48 | hayır |
| SegFormer | YOLOv11n | +0.043 | −0.006 … +0.092 | 0.010 | 54 / 35 | hayır |
| RTMDet 256 | RTMDet 512 | −0.010 | −0.061 … +0.038 | 0.64 | 43 / 45 | hayır |
| RTMDet 256 | YOLOv11n | +0.006 | −0.058 … +0.068 | 0.11 | 55 / 34 | hayır |
| RTMDet 512 | YOLOv11n | +0.017 | −0.034 … +0.066 | 0.10 | 53 / 35 | hayır |

> **Test setinde dört konfigürasyon istatistiksel olarak ayırt edilemiyor.** Altı çiftin
> hiçbirinde güven aralığı sıfırı dışlamıyor. Val'de üç çift ayrışıyordu (hepsi YOLO
> aleyhine), test'te o da kayboluyor.

Yani "SegFormer en iyi" **kanıtlanmış değil**; ortalamada önde olması tek koşuluk bir
sıralama. Rapor bunu böyle yazmalı.

### Modeller ortalamada değil, başarısızlık biçiminde ayrışıyor

Ortalama ile medyan ters yönü gösteriyor — bu tek başına bilgi:

| | ortalama | medyan | std | Dice < 0.3 |
|---|---|---|---|---|
| **val** | | | | |
| SegFormer-B0 | **0.8195** | 0.9068 | **0.219** | **6** |
| RTMDet-Ins 512 | 0.8109 | **0.9169** | 0.245 | 7 |
| RTMDet-Ins 256 | 0.8084 | 0.9158 | 0.269 | 9 |
| YOLOv11n-Seg | 0.7697 | 0.8916 | 0.278 | 10 |
| **test** | | | | |
| SegFormer-B0 | **0.7365** | 0.8635 | **0.301** | **14** |
| RTMDet-Ins 512 | 0.7101 | **0.8871** | 0.333 | 16 |
| RTMDet-Ins 256 | 0.6998 | 0.8843 | 0.348 | 18 |
| YOLOv11n-Seg | 0.6936 | 0.8630 | 0.330 | 17 |

**RTMDet tipik vakada biraz daha iyi** (medyan her iki bölümde de önde), **SegFormer
kuyrukta daha sağlam** (en düşük std, en az ağır başarısızlık, val p5'te 0.242'ye karşı
RTMDet 256'nın 0.0006'sı). SegFormer'ın ortalama öndeliği buradan geliyor: çoğu
görüntüde biraz geride, birkaç görüntüde çok önde.

Klinik okuma: canlı akışta bir lezyonu tamamen kaçırmak, sınırını birkaç piksel kaba
çizmekten pahalı. Bu ölçüte göre SegFormer tercih edilir — ama fark ortalamada anlamlı
olmadığı için bunu "kanıt" değil "eğilim" olarak yazmak gerekir.

### Val → test düşüşü dört koşuda da benzer

| Model | Val | Test | Düşüş |
|---|---|---|---|
| SegFormer-B0 | 0.8195 | 0.7365 | −0.083 |
| RTMDet-Ins 256 | 0.8084 | 0.6998 | −0.109 |
| RTMDet-Ins 512 | 0.8109 | 0.7101 | −0.101 |
| YOLOv11n-Seg | 0.7697 | 0.6936 | −0.076 |

Düşüş modele özgü değil; 115 görüntülük test bölümü daha zor. Eşikler val'de seçilip
test'e sabit uygulandı, test üzerinde hiçbir ayar yapılmadı.

### Hizalama: üç pipeline da temiz

`kaydirma_taramasi` üç modelde de çalıştırıldı. Kaydırmayla kazanılan Dice:

| Model | Kazanç | Zirve | Yorum |
|---|---|---|---|
| RTMDet-Ins 256 | **0.0000** | (0, 0) | mmdet kendi içinde tutarlı |
| RTMDet-Ins 512 | 0.0001 | (1, 0) | aynı |
| SegFormer-B0 | 0.0002 | (0, 1) | iki uç da merkez-doğru (`NEAREST_EXACT` + `align_corners=False`) |
| YOLOv11n-Seg | 0.0012 | (1, 2) | poligon analitik ölçekleniyor, rasterize en sonda |
| RTMDet-Ins (yamalı, reddedildi) | 0.0043 | (2, 2) | tek ucu "düzeltmek" tutarlılığı bozmuştu |

Dördü de `ortak/degerlendirme.kaydirma_taramasi` ile, aynı koddan ölçüldü. RTMDet
tarafında `rtmdet/vaka_gorselleri.py` çalıştırıyor. Yan fayda bir çapraz kontrol:
taramanın (0,0) hücresi 256'da 0.8084, 512'de 0.8109 çıkıyor — `dice_degerlendirme.py`'nin
verdiği sayılarla birebir aynı, yani iki bağımsız kod yolu aynı sonucu üretiyor.

Testin ayırt etme gücü son satırdan geliyor: bilinen bozuk konfigürasyonu yakalıyor,
sağlam olanları temiz geçiriyor.

### RTMDet-Ins final sonuçları (100 epoch, `CocoMetricHizali`, mmdet varsayılan `Resize`)

| | 256 | 512 |
|---|---|---|
| En iyi epoch | 64 | 76 |
| Eşik (val'de `dice_tum` ile) | 0.4 | 0.2 |
| **Val lezyonlu Dice** | 0.8084 | **0.8109** |
| Val Dice (tüm) | 0.8242 | **0.8263** |
| **Test lezyonlu Dice** | 0.6998 | **0.7101** |
| Test Dice (tüm) | 0.7407 | 0.7406 |
| `segm_mAP` (hizalanmış) | **0.6424** | 0.6384 |
| `segm_mAP` (ham, bozuk ölçüm) | 0.4033 | 0.6290 |
| 1 px kısa maske (val) | 24/116 | 2/116 |
| Tepe VRAM | **606 MB** | 2102 MB |
| Eğitim süresi | **52 dk** | 58 dk |

### "256 mı 512 mi" sorusu kapandı: fark yok

Her model kendi val eşiğinde, aynı 96 lezyonlu görüntüde eşleştirilmiş karşılaştırma:

| Metrik | 256 | 512 | fark | %95 GA | Wilcoxon p |
|---|---|---|---|---|---|
| Birleşim Dice | 0.8084 | 0.8109 | −0.0025 | −0.029 … +0.020 | 0.83 |
| Oracle instance Dice | 0.8846 | 0.8784 | +0.0062 | −0.008 … +0.021 | 0.84 |
| Sınır bandı Dice | 0.6327 | 0.6440 | −0.0113 | −0.032 … +0.007 | 0.18 |

**Hiçbir metrikte anlamlı fark yok.** `ham` satırı orijinal ayrışmanın tamamının
ölçüm hatası olduğunu da gösteriyor: 256'nın ham `segm_mAP`'i 0.4033, hizalanmışı
0.6424; 512'de 0.6290 → 0.6384. "512, 256'yı %51 geçiyor" tespiti buradan geliyordu.

> **Karar: ana RTMDet konfigürasyonu 256.** PLAN'ın başındaki hipotez 512'nin eşit
> maske tavanı (0.9922 vs 0.9814) sayesinde öne geçeceğiydi. Tavan bağlayıcı
> değilmiş — modeller tavanın 18 puan altında çalışıyor, dolayısıyla 4 kat piksel
> ve 3.5 kat VRAM karşılığında ölçülebilir bir kazanç yok. 512 raporda bu tespitin
> kanıtı olarak kalıyor.

### Test seti daha zor — üç modelde de aynı

| Model | Val lezyonlu Dice | Test | Düşüş |
|---|---|---|---|
| RTMDet-Ins 256 | 0.8084 | 0.6998 | −0.109 |
| RTMDet-Ins 512 | 0.8109 | 0.7101 | −0.101 |
| YOLOv11n-Seg | 0.7697 | 0.6936 | −0.076 |

YOLO'daki val→test düşüşü ilkin ona özgü bir aşırı uyum olabilir diye işaretlenmişti;
RTMDet iki koşuda da **daha büyük** düşüş gösteriyor. Yani sebep modelde değil, 115
görüntülük test bölümünün daha zor olmasında. Eşikler val'de seçilip test'e sabit
uygulandı, test üzerinde hiçbir ayar yapılmadı — `dice_degerlendirme.py` artık test
için `--esik` zorunlu kılıyor ki bu yanlışlıkla ihlal edilemesin.

### YOLOv11n-Seg sonuçları (256, `orta` aug, 100 epoch)

| | val | test |
|---|---|---|
| Eşik (val'de seçildi, test'e sabit) | **0.20** | 0.20 |
| Lezyonlu Dice | 0.7697 | 0.6936 |
| Dice (tüm, boş dahil) | 0.8008 | 0.7268 |
| IoU (lezyonlu) | 0.6856 | 0.6105 |
| Precision / Recall | 0.7257 / 0.8534 | 0.6664 / 0.7982 |
| Boş görüntü doğru | 19/20 | 17/19 |
| mask mAP (Ultralytics) | ~0.57 (zirve ep60) | — |

Hizalama kontrolü (`kaydirma_taramasi`, val): yüzey ±2 px aralığında 0.7618–0.7709
arasında, zirvenin (0,0)'a göre kazancı **0.0012**. Karşılaştırma için RTMDet'in
yanlış hizalanmış koşusunda aynı kazanç 0.0043'tü. **YOLO'nun poligon yolu temiz** —
Ultralytics polikonu analitik ölçekleyip en sonda rasterize ettiği için nearest
yeniden ölçekleme bias'ı taşımıyor; bu artık varsayım değil ölçüm.

> Not: ilk yazdığım kontrol yalnızca "zirve (0,0) mı" diye bakıyordu ve düz bir
> yüzeyde `argmax` gürültü olduğu için YOLO'da yanlış alarm verdi. `hizalama_ozeti()`
> artık **kaydırmayla kazanılan Dice'a** bakıyor (eşik 0.003, iki gözleme dayanıyor
> ve geçici). Sınırda kalan sonuç `kayma_testi.py` ile teyit edilmeli.

### Eşik seçim ölçütü: `dice_tum`

YOLO'nun ilk taraması ölçütün yanlış olduğunu ortaya çıkardı. `dice_lezyonlu` ile
seçilen eşik 0.05'ti ve o eşikte 20 normal görüntünün **8'inde yanlış pozitif** vardı;
0.20'de 1'inde. Boş görüntüleri hesaba katmayan bir ölçüt, projenin `normal` sınıfını
dahil etme gerekçesini (false-positive baskılama) geri alıyor.

| eşik | dice_lezyonlu | dice_tüm | boş doğru |
|---|---|---|---|
| 0.05 | **0.7919** | 0.7588 | 12/20 |
| 0.20 | 0.7697 | **0.8008** | 19/20 |

Ölçüt `dice_tum` yapıldı. **RTMDet'i etkilemiyor** — orada iki ölçüt de 0.4'ü veriyor,
yani değişiklik yalnızca farkın önemli olduğu yerde etkili.

Aynı taramada ikinci bir kusur çıktı: seçilen eşik ızgaranın **en alt ucuydu**, yani
gerçek optimum aralık dışında kalmış olabilirdi. Izgara 0.01–0.9 arasına genişletildi
(13 değer) ve `en_iyi_esik()` artık optimum uçta çıkarsa **hata veriyor** — sınırdaki
bir optimumu sessizce raporlamak yanlış olur.

### Açık karar: maske küçültmede yarım piksel kayması

`cv2.INTER_NEAREST` çıktı pikselini `floor` ile eşliyor ve maskeyi **her eksende +0.5 px**
kaydırıyor; görüntü bilinear (merkez-doğru) küçüldüğü için hedef, görüntü içeriğine göre
kayıyor. `INTER_NEAREST_EXACT`'te kayma 0.002 px.

Val'in 96 lezyonlu görüntüsünde ölçülen temsil tavanı (küçült → geri büyüt → 0.5'te eşikle):

| Girdi | `INTER_NEAREST` | `INTER_NEAREST_EXACT` |
|---|---|---|
| 256 | 0.9810 | **0.9915** |
| 512 | 0.9906 | **0.9977** |

Üç model bu konuda aynı yerde değil:

| Model | Durum |
|---|---|
| RTMDet-Ins | mmdet maskeleri `INTER_NEAREST` ile ölçekliyor → **kayma var** |
| YOLOv11-Seg | poligonu analitik ölçekleyip en sonda rasterize ediyor → kayma yok |
| SegFormer | `MASKE_INTERP` config değişkeni; şu an `INTER_NEAREST_EXACT` |

İlk okumam "RTMDet yamalanmalı, yoksa ~0.01 Dice geride başlar" oldu ve yama yazıldı.

> **Sonuç: yama ölçümle reddedildi ve geri alındı.** mmdet'in çıkarım tarafı ters
> yönde bir kayma taşıyor ve nearest bias'ı onu telafi ediyormuş; tek ucu düzeltmek
> hizalamayı bozdu. Tüm sayılar ve mekanizma ölçümü "Karşılaşılan ve düzeltilen
> hatalar" 9. maddede. Doğru kural: **her pipeline kendi içinde tutarlı olmalı**,
> ve bu teoriyle değil `ortak/degerlendirme.kaydirma_taramasi()` ile doğrulanır.

### Epoch bütçesi: 70 (üç model için ortak)

Düzeltilmiş metrikle yapılan 100 epoch'luk 256 koşusunun eğrisinden seçildi:

| Bütçe | 256'nın ulaştığı en iyi `segm_mAP` | zirvenin yüzdesi |
|---|---|---|
| 20 | 0.630 | %98.1 |
| 50 | 0.634 | %98.8 |
| 70 | **0.642** | %100 |
| 100 | 0.642 | %100 |

Eğri epoch 20'den sonra düz; 0.634–0.642 aralığı gürültü düzeyinde. 70, 256'nın
zirvesini (ep64) ve 512'ninkini (ep24) kapsıyor, 100'e göre %30 süre kazandırıyor.
Bütçe küçülünce cosine takvimi de sıkıştığı için model daha erken yakınsıyor —
bu bir kırpma değil, takvimin yeniden ölçeklenmesi.

### RTMDet-Ins sonuçları (val, 116 görüntü)

Checkpoint seçimi `save_best='coco/segm_mAP'` ile yapıldı — o metriğin bozuk olduğu sonradan ortaya çıktı, aşağıya bakınız.

| | 256 | 512 |
|---|---|---|
| Maske ızgarası | 32×32 | 64×64 |
| Temsil tavanı | 0.9814 | 0.9922 |
| `segm_mAP` — eğitimde loglanan (bozuk) | 0.412 (epoch 40) | 0.623 (epoch 24) |
| **`segm_mAP` — hizalanmış** | **0.626** | **0.644** |
| `segm_mAP@50` (hizalanmış) | 0.889 | 0.873 |
| `bbox_mAP` | 0.625 | 0.606 |
| **Lezyonlu Dice** (eşik 0.2) | 0.803 | 0.784 |
| Instance başına Dice | 0.891 | 0.876 |
| Sınır bandı Dice (±5 px) | 0.634 | **0.645** |
| Tepe VRAM | 606 MB | 2102 MB |
| Eğitim süresi (100 epoch) | 48 dk | ~60 dk |

Lezyonlu Dice 0.803, BUSI literatürünün (0.75–0.82) üst bandında.

### Metrik ayrışması çözüldü — artefaktmış

`segm_mAP`'in 512'yi +%51 öne koyup Dice'ın hiç fark görmemesinin sebebi mimari değil ölçüm hatasıydı: RTMDet-Ins maskeyi `ori_shape`'ten 1 px kısa üretebiliyor, COCOeval iki RLE'nin boyutu uyuşmadığında IoU'yu geçersiz sayıyor ve o görüntü `segm_mAP`'e **sıfır katkıyla** giriyor. Etkilenen görüntü oranı girdi boyutuna bağlı:

| Girdi | `ori_shape` ile uyuşmayan maske | `segm_mAP` loglanan | `segm_mAP` hizalanmış |
|---|---|---|---|
| 256 | 25/116 (**%21.6**) | 0.412 | **0.626** |
| 512 | 2/116 (%1.7) | 0.623 | **0.644** |

Hizalandığında +%51'lik fark +%2.8'e iniyor. Ayrıntı ve kök neden aşağıda, "Karşılaşılan ve düzeltilen hatalar" 8. madde.

`rtmdet/metrik_ayrismasi.py` bunun yanında ayrışmanın kalan bileşenlerini de ölçtü (val, eşik 0.2, 96 lezyonlu görüntü):

| Ölçüm | 256 | 512 | Yorum |
|---|---|---|---|
| Birleşim Dice | 0.803 | 0.784 | görüntü düzeyi, ana metrik adayı |
| Instance başına Dice (eşleşen) | 0.891 | 0.876 | birleşimden ~0.09 yüksek |
| Oracle instance Dice (skor eşiği yok) | 0.885 | 0.888 | modelin üretebildiği en iyi maske |
| Sınır bandı Dice (±5 px) | 0.634 | 0.645 | ince sınır kalitesi |
| `segm_mAP` — oracle maske | 0.951 | 0.944 | maske mükemmel olsaydı |
| `segm_mAP` — oracle skor | 0.618 | 0.665 | skor kalibrasyonu mükemmel olsaydı |

Çıkan tablo:

- **Birleşim Dice ile instance Dice arasındaki 0.09'luk fark maskeden değil fazladan instance'tan geliyor**: lezyonlu görüntülerde eşiği geçen 27 (256) / 32 (512) fazla tahmin var, ayrıca 20 boş görüntünün 7'sinde (256: 10) yanlış pozitif. Bunlar birleşime giriyor, instance eşlemesinde ise eşleşmeyen tahmin olarak kalıyor.
- **512 gerçekten daha ince sınır üretiyor, ama az**: sınır bandı Dice +0.011 (eşleştirilmiş Wilcoxon p=0.004, 96 görüntünün 62'sinde 512 daha iyi). Çıktı-stride tavanının öngördüğü etki bu — yönü doğru, büyüklüğü küçük.
- **Skor kalibrasyonu sorun değil**: skorları gerçek IoU ile değiştirmek mAP'i 256'da 0.626 → 0.618, 512'de 0.644 → 0.665 yapıyor, yani sıralama zaten iyi. Kalan mAP açığının tamamı maske kalitesinden (oracle maske ile mAP 0.95).

#### 0.803 vs 0.784 gürültü mü — evet

Aynı 96 lezyonlu görüntüde eşleştirilmiş karşılaştırma:

| Metrik | fark (256 − 512) | %95 GA (bootstrap) | Wilcoxon p | 256 iyi / 512 iyi |
|---|---|---|---|---|
| Birleşim Dice | +0.019 | −0.010 … +0.052 | 0.89 | 42 / 51 |
| Oracle instance Dice | −0.003 | −0.022 … +0.021 | 0.053 | 37 / 59 |
| Sınır bandı Dice | −0.011 | −0.031 … +0.012 | **0.004** | 31 / 62 |

Birleşim Dice'ta 256'nın önde görünmesi birkaç aykırı görüntüden geliyor; görüntü sayısı olarak 512 önde (51 vs 42) ve Wilcoxon p = 0.89. **256 ile 512 arasında segmentasyon kalitesi farkı yok** demek doğru okuma; ölçülebilir tek fark sınır bandında ve 512 lehine.

### Karar: ana metrik orijinal çözünürlükte Dice

1. `segm_mAP` bu kurulumda kırılgan olduğunu gösterdi — mmdet'in bir uygulama detayı girdi boyutuna bağlı olarak metriği %34 düşürüyordu ve bunu sessizce yapıyordu. Üç ayrı çerçeve karşılaştırılacakken bu risk kabul edilemez.
2. SegFormer-B0 semantic segmentation; instance üretmiyor, `segm_mAP` onun için tanımlı bile değil. Üç modelde ortak tanımlanan tek metrik Dice.
3. Klinik hedef lezyon alanı uyuşması, sıralama kalitesi değil.

`segm_mAP` **ikincil metrik** olarak iki instance modelinde raporlanmaya devam edecek (tespit kalitesi hakkında Dice'ın söylemediği bir şey söylüyor: fazladan instance sayısı), ama model seçimi ve karşılaştırma Dice üzerinden.

### Epoch bütçesi

256 epoch 40'ta, 512 epoch 24'te zirve yaptı — **ancak ikisi de bozuk `segm_mAP` üzerinden seçildi**, yani 256 için "en iyi epoch" büyük ölçüde rastlantısal (görüntülerin %21.6'sı metriğe sıfır katkıyla giriyordu). Düzeltilmiş metrikle 256 yeniden eğitilmeden epoch bütçesi kararı verilmemeli; koşu 48 dakika. 512 tarafında etki %1.7 ile sınırlı olduğu için oradaki seçim büyük ölçüde geçerli.

## Gerçek Zamanlı Ölçüm (yerel RTX 2060, batch=1)

Üçü de aynı ortamda (torch 2.6), aynı protokolle: ısınma 15 kare, her kare tek tek
senkronlanarak, 60 val görüntüsü kendi doğal çözünürlüğünde. Ölçülen **uçtan uca kare
süresi** — ön işleme (letterbox + normalize) + ileri geçiş + son işleme (maskeyi orijinal
çözünürlüğe taşıma). Diskten PNG okuma dışarıda: canlı akışta kare cihazdan gelir.

`hiz_olcumu.py` hızı ölçülen kodun Dice'ı ölçülen kodla aynı olduğunu `assert` ile
doğruluyor (`kare_fn` ile `harita_fn` aynı haritayı üretmeli) — dördünde de fark 0.

| Model | Girdi | Param | ms/kare | **GPU FPS** | Saf ağ | Çerçeve yükü | CPU ms | CPU FPS | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| **SegFormer-B0** | 256 | 3.71 M | **10.4** | **96** | 9.2 | 1.2 ms (%12) | 36.8 | 27 | 63 MB |
| YOLOv11n-Seg | 256 | 2.84 M | 16.0 | 63 | 10.7 | 5.2 ms (%33) | **24.8** | **40** | 34 MB |
| RTMDet-Ins 512 | 512 | 5.62 M | 26.0 | 38 | 13.6 | 12.4 ms (%48) | 94.6 | 11 | 78 MB |
| RTMDet-Ins 256 | 256 | 5.62 M | 27.2 | 37 | 13.3 | 13.8 ms (%51) | 48.2 | 21 | 89 MB |

Tekrarlanan koşularda GPU sayıları %5–10 oynuyor (SegFormer üç koşuda 9.3 / 9.8 / 10.4 ms);
modeller arasındaki farklar bunun çok üzerinde, ama küçük farklar anlamlı okunmamalı.

**CPU'da sıralama değişiyor**: YOLOv11n en hızlı (40 FPS), RTMDet 512 en yavaş (11 FPS).
Ve CPU'da girdi boyutu fark yaratıyor (RTMDet 512, 256'nın iki katı süre) — GPU'da
yaratmıyordu. Bu, aşağıdaki "kernel başlatma yükü" açıklamasının doğrudan teyidi:
CPU hesap yüküne bağlı, GPU bu boyutlarda değil.

Dördü de 30 FPS'in üzerinde, yani hepsi gerçek zamanlı sayılabilir. **SegFormer-B0 hem
en hızlı hem kalite tarafında geride değil** — kalite farkları zaten anlamlı değildi.

### İki beklenmedik bulgu

**1. Girdi boyutu gecikmeyi neredeyse hiç etkilemiyor.** RTMDet 256 ile 512 aynı süreyi
veriyor (25.3 vs 24.5 ms), saf ağ ölçümünde de öyle (12.8 vs 12.9 ms) — dört kat piksele
rağmen. Sebep: bu boyutlarda batch=1'de GPU **kernel başlatma yükünde takılı**, hesap
yükünde değil. Planın başındaki "512'nin FPS maliyeti doğrudan" varsayımı bu donanımda
geçersiz çıktı.

Sonuç: 256 ile 512 arasında ne Dice farkı var, ne FPS farkı. 512'nin tek somut maliyeti
eğitimdeki VRAM (2102 MB vs 606 MB). Ana konfigürasyon olarak 256 kararı geçerli ama
gerekçesi "FPS" değil, "eğitim maliyeti".

**2. RTMDet'in gecikmesinin yarısı mmdet'in kare başına Python yükü.** Saf ağ 12.8 ms,
uçtan uca 25.3 ms. SegFormer'da bu oran %12, YOLO'da %31. Yani "RTMDet 2.5 kat yavaş"
cümlesi mimariyi değil, `inference_detector`'ın veri hattı kurma maliyetini ölçüyor.
Saf ağ karşılaştırmasında sıralama çok daha yakın: SegFormer 8.6 ms, YOLO 9.9 ms,
RTMDet 12.8 ms.

ONNX/TensorRT adımının asıl değeri burada: çerçeve yükünü kaldırıp mimarileri kendi
başlarına karşılaştırmayı sağlayacak.

## ONNX: çerçeve yükü kalkınca sıralama tersine dönüyor

Üç ağ da ONNX'e aktarılıp **aynı çalışma zamanında** (onnxruntime 1.23) ölçüldü.
Sayısal doğruluk her model için bağıl hata ile kontrol edildi: **5e-7 … 1.3e-5**,
yani fp32 yuvarlama düzeyinde.

| Model | PyTorch GPU | **ONNX GPU** | Hızlanma | ONNX CPU | ONNX CPU FPS |
|---|---|---|---|---|---|
| **RTMDet-Ins 256** | 13.4 | **4.40** | **3.05×** | 11.9 | 84 |
| SegFormer-B0 | 9.5 | 4.48 | 2.11× | 22.0 | 45 |
| YOLOv11n-Seg | 13.6 | 5.59 | 2.43× | **7.4** | **135** |
| RTMDet-Ins 512 | 13.6 | 6.12 | 2.23× | 44.2 | 23 |

> **RTMDet-Ins-tiny, ağ olarak en hızlısı.** PyTorch'ta 2.6 kat yavaş görünmesinin
> tamamı mmdet'in kare başına Python yüküymüş. Mimari sıralaması hakkında PyTorch
> ölçümüne bakarak yapılacak her yorum yanlış olurdu.

Girdi boyutu ONNX'te fark yaratmaya başlıyor (RTMDet 4.40 vs 6.12 ms) — çalışma
zamanının başlatma yükü düşünce hesap yükü görünür hale geliyor.

**CPU'da YOLOv11n-Seg açık ara önde**: 7.4 ms, yani GPU'suz bir cihazda bile 135 FPS.
Gerçek zamanlı hedef için GPU'nun zorunlu olmadığını gösteriyor.

### Bu sayılar "dağıtılabilir FPS" değil

ONNX grafiği **yalnızca ağı** içeriyor. Instance modellerde son işleme (NMS + maske
birleştirme) grafiğin dışında ve ölçülmüş durumda: RTMDet'te 2.8 ms (256) / 1.6 ms (512).
SegFormer'ın son işlemesi eşikleme + yeniden boyutlandırmadan ibaret (~1.2 ms).

Yani uçtan uca dağıtım maliyetinde SegFormer'ın avantajı ONNX tablosunun gösterdiğinden
büyük, RTMDet'inki küçük. Tam dağıtım hattını ölçmek ayrı bir iş; bu tablo mimarilerin
kendi maliyetini karşılaştırmak için.

## Final Rapor

`rapor/` altında, `mendeley_data_density/rapor` deseniyle: tek dosyalık HTML (figürler base64
gömülü) + Chrome headless ile PDF.

| Dosya | Ne |
|---|---|
| `loglari_topla.py` | Dört koşunun epoch logunu normalize eder. SegFormer'ınki notebook çıktısındaki epoch satırlarından geri kazanılıyor |
| `figurleri_uret.py` | Eğitim eğrileri, vaka görselleri, karşılaştırma figürleri — üçü de aynı koddan |
| `build_report.py` | HTML üretir |
| `pdf_olustur.ps1` | HTML → PDF |
| `rt_seg_sonuc_raporu.pdf` | 5.1 MB, 8 bölüm, 9 figür, 3 tablo |

Vaka görselleri Colab'da üretilmişti ama rapor için yerelde yeniden üretildi: aynı çizim kodu,
aynı vakalar, aynı eşikler — rapora yan yana konulacakları için görsel farkın modelden gelmesi
gerekiyor, çizim ayarından değil.

### Validation kaybı bilerek yok

SegFormer ve RTMDet'te validation adımı kayıp değil kalite metriği üretiyor (Dice / mAP): mmdet
validation'ı tahmin modunda çalıştırıyor, SegFormer döngüsü de aynı yapıyı izliyor. Ultralytics
val kaybını hesaplıyor ama **rapora alınmadı** — tek modelde gösterip diğer ikisinde göstermemek
eğrileri karşılaştırılamaz hale getirirdi.

Ayrıca val kaybı modeller arası zaten okunamaz: üç mimarinin kayıp fonksiyonu farklı
(Dice+BCE / cls+bbox+mask / box+seg+cls+dfl). Aşırı öğrenme takibi işini validation kalite eğrisi
daha doğrudan yapıyor ve üç eğride de mevcut.

> Not: bu, density projesinin şemasından bilinçli bir sapma. Orada val döngüsü aynı kaybı
> hesapladığı için `val_loss` sütunu vardı.

### Eksik: YOLO eğitim eğrisi

Ultralytics `results.csv` Drive'da kaldı; o dosya olmadan YOLO'nun eğitim eğrisi üretilemiyor.
Rapor bu eksikliği metinde açıkça belirtiyor. `real_time_segmentasyon/yolo_results.csv` olarak
konulursa `figurleri_uret.py` figürü otomatik üretiyor ve `build_report.py` rapora alıyor.

## Faz 2: sınırlılıkları kapatma

### Kod değişiklikleri tamamlandı ve doğrulandı

| Model | Val loss | Checkpoint seçimi | Tohum |
|---|---|---|---|
| RTMDet-Ins | `ValLossHook` — ayrı dataloader, `mode='loss'` | `busi/dice_tum` (`BusiDiceMetric`) | `TOHUM`, work_dir'e giriyor |
| SegFormer-B0 | val döngüsü, **eğitimdekiyle aynı kayıp ve uzay** | `dice_tum`, eşik taranarak | `TOHUM`, MODEL_KEY'e giriyor |
| YOLOv11n-Seg | Ultralytics hesaplıyor (rapora alınmıyor) | `save_period=1` + eğitim sonrası Dice ile seçim | `TOHUM`, MODEL_KEY'e giriyor |

Ek olarak SegFormer kaybından **letterbox dolgusu çıkarıldı** (geçerli piksel oranı medyan
0.824, en uçta 0.551 — yani en uç en-boy oranında piksellerin %45'i sabit gri dolguymuş).

#### Kod yazarken çıkan üç tuzak

**1. Hook önceliği.** `ValLossHook` 48'de yazılmıştı; `RuntimeInfoHook` (10) metrikleri
message_hub'a ondan önce itiyor, dolayısıyla eklenen anahtarlar hiçbir yere ulaşmıyordu.
Öncelik 9'a alındı — EMAHook'un (49) parametre takasından da önce, yani kayıp val
metrikleriyle aynı ağırlıklarda ölçülüyor.

**2. `loss` içeren anahtarlar düzleştiriliyor.** mmengine LogProcessor adında `loss` geçen
skalerleri pencere ortalamasıyla raporluyor ve bu `scalars.json`'a da yansıyordu (epoch 2'de
gerçek 1.8666 iken kayda 2.1161). Anahtarlar `val_kayip*` yapılınca epoch değeri olduğu gibi
kaydediliyor.

**3. `egitim_grafikleri.py` val satırını kaçırıyordu.** Satır sınıflandırması yalnızca `coco/`
anahtarına bakıyordu; model henüz eşiği geçen tahmin üretmediğinde o epoch'un val verisi train
satırı sayılıp kayboluyordu. `busi/` ve `val_kayip` de sayılıyor artık.

#### Kapsam dışı ama düzeltilen bir kusur

SegFormer kaybının Dice terimi, **boş maskeli görüntüde tahmin mükemmelken bile 0.74**
veriyordu: `smooth=1` iken ~54 bin geçerli pikselde biriken minik olasılıklar paydayı şişiriyor.
Eğitim setinin %17'si (92 boş görüntü) neredeyse indirgenemez bir kayıp taşıyordu ve bu, kendi
değerlendirme kuralımızla ("GT boş + tahmin boş = Dice 1") çelişiyordu. Dice terimi lezyonlu
görüntülerle sınırlandı; aynı testte kayıp 0.0000'a indi. Bu kusur önceki SegFormer koşusunda
da etkiliydi.

### Kalan: eğitimler



Karar: **aynı üç mimari, 3 tohum** ile yeniden eğitilecek ve aşağıdaki dört düzeltme birlikte
uygulanacak. Yeni mimari eklenmeyecek — tek koşu sorunu kapanmadan yeni model eklemek ana
sınırlılığı kapatmıyor.

#### Uygulanan düzeltmeler (referans)

**1. Val loss — üç modelde de kaydedilecek**
- SegFormer: val döngüsüne Dice+BCE hesabı eklenir (eğitim uzayında, letterbox'lı hedefle)
- RTMDet: mmdet validation'ı tahmin modunda çalışıyor, kayıp üretmiyor → `mode='loss'` ile
  val üzerinde ileri geçiş yapan özel bir hook gerekiyor. EMAHook'un parametre takası ile
  hook önceliği çakışmasına dikkat; kısa bir koşuyla (3 epoch) doğrulanmalı
- YOLO: Ultralytics zaten hesaplıyor, `results.csv` saklanacak

**2. Checkpoint seçim ölçütü birleştirilecek**
Şu an SegFormer `dice_tum`, RTMDet `segm_mAP`, YOLO Ultralytics `fitness` ile seçiyor. Üçü de
**orijinal çözünürlükte Dice** ile seçecek. RTMDet için epoch başına Dice hesaplayan hook,
YOLO için `save_period` ile epoch ağırlıklarını saklayıp sonradan seçme gerekiyor.

**3. SegFormer kaybından letterbox dolgusu çıkarılacak**
Piksellerin medyanda ~%16'sı sabit gri dolgu ve BCE'ye giriyor; RTMDet'in bölge tabanlı kaybı
içermiyor. Dataset'te geçerlilik maskesi üretilip kayıp ona göre maskelenecek.

**4. Planlanan iki ablasyon yapılacak**
- `hafif` vs `orta` augmentasyon (tek mimaride, kazanan diğerlerine uygulanır)
- 384 girdi

### Bütçe

| İş | Nerede | Süre |
|---|---|---|
| RTMDet 256 × 3 tohum | yerel | ~2.6 saat |
| SegFormer × 3 tohum | Colab | ~1.5 saat |
| YOLO × 3 tohum | Colab | ~1.5 saat |
| Ablasyonlar | ikisi | +2-3 saat |

### Sıra

1. Kod değişiklikleri (val loss, seçim ölçütü, dolgu maskesi) — kısa koşularla doğrulanmadan
   uzun eğitim başlatılmayacak
2. 3 tohumlu koşular
3. Tohumlar arası varyans ölçülüp mevcut eşleştirilmiş karşılaştırma yenilenecek
4. Ablasyonlar
5. Rapor yeniden üretilecek

> Bu faz tamamlandığında mevcut rapordaki sayılar geçersiz olacak; rapor baştan üretilecek.

### Koşu sonuçları (val, orijinal çözünürlük)

| Koşu | Tohum | Best epoch | Eşik | dice_tum | dice_lezyonlu | boş doğru |
|---|---|---|---|---|---|---|
| RTMDet-Ins 256 orta | 42 | 61 | 0.30 | 0.8290 | 0.8037 | 19/20 |
| RTMDet-Ins 256 orta | 43 | 56 | 0.30 | 0.8538 | 0.8234 | 20/20 |
| RTMDet-Ins 256 orta | 44 | 93 | 0.40 | 0.8366 | 0.8026 | 20/20 |
| SegFormer-B0 256 orta | 42 | 63 | — | 0.8275 | 0.8020 | — |
| SegFormer-B0 256 orta | 43 | 65 | 0.50 | 0.8361 | 0.8123 | 19/20 |
| SegFormer-B0 256 orta | 44 | 68 | 0.40 | 0.8235 | 0.7971 | 19/20 |
| YOLOv11n-Seg 256 orta | 42 | 75 (mAP) | 0.40 | 0.7985 | 0.7669 | 19/20 |
| YOLOv11n-Seg 256 orta | 43 | 79 (mAP) | 0.10 | 0.8269 | 0.7908 | 20/20 |
| YOLOv11n-Seg 256 orta | 44 | 60 (mAP) | 0.30 | 0.8136 | 0.7956 | 18/20 |

SegFormer test: s43 dice_tum 0.7442 / lezyonlu 0.7249 (boş 16/19), s44 0.7607 / 0.7341
(17/19). YOLO test: s42 0.7326 / 0.6901 (18/19), s43 0.7171 / 0.7132 (14/19), s44 0.7347 / 0.6926 (18/19).
Hizalama üç YOLO koşusunda da temiz (kazanç 0.0008 / 0.0005 / 0.0004). Hizalama taraması iki koşuda da temiz (kazanç 0.0000 ve 0.0003).

#### Tohumlar arası varyans — üç mimari (9 koşu tamam)

| Model | val dice_tum | val lezyonlu | seçilen eşikler |
|---|---|---|---|
| RTMDet-Ins | 0.8398 ± 0.0127 | 0.8099 ± 0.0117 | 0.30 / 0.30 / 0.40 |
| SegFormer-B0 | 0.8290 ± 0.0064 | 0.8038 ± 0.0078 | — / 0.50 / 0.40 |
| YOLOv11n-Seg | 0.8130 ± 0.0142 | 0.7844 ± 0.0154 | 0.40 / 0.10 / 0.30 |

İkili farklar (dice_tum / lezyonlu, parantezde ortak std):
- RTMDet − SegFormer: +0.0108 / +0.0061 (0.0101) → **ayırt edilemiyor**
- SegFormer − YOLO: +0.0160 / +0.0194 (0.0110) → sınırda
- RTMDet − YOLO: +0.0268 / +0.0255 (0.0135) → **std'nin ~2 katı, tek gerçek ayrım adayı**

RTMDet ile SegFormer arasındaki fark ikisinin de tohum gürültüsünün içinde. Faz 1'in
"SegFormer en iyi" okuması bir koşunun gürültüsüymüş. YOLO ise her iki modelden de geride
ve fark tohum bandını aşıyor — eşleştirilmiş testle doğrulanacak.

RTMDet'in varyansı SegFormer'ınkinin iki katı; ön eğitimli encoder'ın (mit-b0) başlangıç
noktasını sabitlemesi beklenen bir etki (RTMDet sadece backbone'da ön eğitimli, head'i
sıfırdan). Pratik sonucu: RTMDet'i tek koşuyla değerlendirmek en riskli olanı.

### Test sonuçları — 9 koşu, eşleştirilmiş karşılaştırma

`faz2_karsilastirma.py`. Eşik her koşu için val'de baştan seçilip test'e sabit uygulandı;
dokuz koşunun val değeri yerelde birebir yeniden üretildi (en büyük sapma 0.0001), yani
çıkarım yolu ayrışmamış.

| Mimari | test dice_tum | test lezyonlu |
|---|---|---|
| RTMDet-Ins | 0.7615 ± 0.0089 | 0.7212 ± 0.0160 |
| SegFormer-B0 | 0.7601 ± 0.0156 | 0.7335 ± 0.0083 |
| YOLOv11n-Seg | 0.7281 ± 0.0096 | 0.6986 ± 0.0127 |

Tohum-ortalamalı eşleştirilmiş test (lezyonlu, n=96):

| Çift | Fark | %95 GA | Wilcoxon p | Anlamlı |
|---|---|---|---|---|
| RTMDet − SegFormer | −0.0123 | [−0.053, +0.026] | 0.846 | hayır |
| RTMDet − YOLO | +0.0226 | [−0.020, +0.063] | 0.058 | hayır |
| SegFormer − YOLO | **+0.0348** | [+0.004, +0.067] | **0.003** | **EVET** |

Dokuz çapraz tohum çiftinde: SegFormer−YOLO'nun **dokuzunda da** işaret aynı yönde
(3'ü tek başına anlamlı). Diğer iki çiftte işaret tohuma göre değişiyor.

#### Ne söylenebilir, ne söylenemez

**Söylenebilir:** SegFormer-B0, YOLOv11n-Seg'i lezyonlu Dice'ta geçiyor. Tek sağlam
model ayrımı bu.

**Söylenemez:** RTMDet ile SegFormer arasında fark yok (p=0.85, GA sıfırı içeriyor).
Faz 1'in "SegFormer en iyi" sonucu tek koşunun gürültüsüymüş.

**RTMDet − YOLO anlamlı çıkmadı** (p=0.058) — fark +0.0226 olmasına rağmen, çünkü
RTMDet'in tohumlar arası std'si (0.0160) SegFormer'ınkinin (0.0083) iki katı. Aynı
büyüklükteki fark, varyansı düşük modelde anlamlı çıkıyor, yüksek olanda çıkmıyor.

#### Val ve test sıralaması ters dönüyor

Val'de RTMDet önde (0.8398 vs 0.8290), test'te lezyonlu Dice'ta SegFormer önde
(0.7335 vs 0.7212). 115 görüntülük test setinde bu ölçekte bir dönüş beklenen bir şey;
tek başına "hangisi daha iyi" sorusuna cevap sayılmamalı. Rapor sıralama iddiası değil,
ayırt edilemezlik bulgusu üzerine kurulmalı.

#### Lezyon boyutu kırılımı — dokuz koşu, test (n: 16 / 28 / 29 / 23)

Test Dice, üç tohumun ortalaması:

| Kova (eş-çap px) | RTMDet-Ins | SegFormer-B0 | YOLOv11n-Seg |
|---|---|---|---|
| <80 | 0.6478 | **0.7762** | 0.7069 |
| 80–140 | 0.7993 | **0.8315** | 0.7528 |
| 140–220 | 0.6637 | 0.6552 | 0.6709 |
| >220 | **0.7496** | 0.6830 | 0.6618 |

İki ayrı bulgu çıkıyor:

**1. 140–220 çukuru mimariden bağımsız.** Üç mimaride de (0.655–0.671) ve dokuz koşunun
hepsinde aynı yerde. Model kaynaklı olsaydı mimariler arasında oynardı. Veriye ait bir
zorluk; o kovadaki 29 görüntünün ayrıca incelenmesi gerekiyor. Monotonik olmadığı için
"büyük lezyon zordur" açıklaması geçersiz.

**2. RTMDet ile SegFormer'ın güçlü olduğu yerler ters.** Küçük lezyonlarda SegFormer
RTMDet'i 0.13 geçiyor (0.776 vs 0.648); en büyük kovada RTMDet SegFormer'ı 0.067
geçiyor (0.750 vs 0.683). Mimari olarak beklenen bir davranış: RTMDet'in maske başı
RoI tabanlı ve düşük çözünürlükte çalışıyor, SegFormer'ın decoder'ı stride 4'te yoğun.

Bu, ikisinin toplam ortalamalarının neden ayırt edilemediğini de açıklıyor: farklı
yerlerde iyiler, ortalamada birbirlerini götürüyorlar. **Rapor "hangisi daha iyi"
yerine "hangisi nerede iyi" üzerine kurulmalı** — klinik kullanımda lezyon boyutu
bilinen bir bağlam.

#### YOLO'nun eşik seçimi kırılgan

Seçilen eşik YOLO'da 0.40 / 0.10 / 0.30 arasında savruluyor; diğer iki mimaride oynama
bir kademe. Sebebi s43'ün val tablosunda görünüyor: 0.05'te boş görüntülerin 17/20'si
doğruyken 0.10'da 20/20 oluyor, `dice_tum` de tam o sıçramaya oturuyor — dar bir optimum.
Test'e taşınmıyor: s43 val'de 20/20 boş doğruyken test'te 14/19 (s42 ve s44 ise 18/19).

Protokol hatası değil (eşik val'de seçilip test'e sabit uygulanıyor, doğrusu bu), ama
YOLO'nun skor kalibrasyonunun diğer ikisinden kırılgan olduğunu gösteriyor. Rapora girmeli.

#### Hata 10 — YOLO notebook'unda eski hizalama mantığı kalmıştı

Hizalama kontrolü üç notebook'ta aynı olmalıydı; YOLO'da `dg.hizalama_ozeti`'ye
geçirilmemiş, "zirve (0,0)'da değilse ön işleme bozuk" diyen ilk sürüm duruyordu.
YOLO s42'de zirve dx=1'de çıktı ve **"DIKKAT: tahmin sistematik kaymış"** bastı —
oysa kazanç 0.7677 − 0.7669 = **0.0008**, eşiğin (0.003) dörtte biri. Yanlış alarm.

Aynı koşuda SegFormer s44 de dx=1'de zirve yapmıştı ama `hizalama_ozeti` kullandığı
için doğru şekilde "hizalama tamam" dedi. Tek piksellik kaymada Dice zaten ±0.001
oynuyor; zirve konumuna bakmak her koşuyu yanlış işaretler.

Düzeltildi (notebook + üreteç). s42'nin eğitimini etkilemiyor — hücre yalnızca rapor
üretiyor, ölçülen kazanç zaten eşiğin altında.

#### Boyut kırılımında tekrarlayan örüntü

140–220 px kovası üç SegFormer koşusunun **hem val'inde hem test'inde** en kötü kova
(val ~0.707, test ~0.64–0.66) — hem daha küçük hem daha büyük lezyonlardan belirgin düşük.
Monotonik değil, yani "büyük lezyon zor" ile açıklanmıyor. İki ayrı bölünmede ve üç tohumda
tekrarladığı için tesadüf değil; rapora girmeli.

Eğitim sırasında seçim ızgarasının (0.3–0.7) üst sınırı ilk ~15 epoch'ta seçiliyordu — sınırda
optimum riski. Zirve epoch'unda eşik 0.5 çıktığı için seçim etkilenmedi; ızgara değiştirilmedi
(değiştirmek s42 ile s43'ü farklı deney yapardı). Raporlanan sayılar zaten `dg.esik_tablosu`nun
13 değerlik ızgarasıyla üretiliyor ve `en_iyi_esik` sınırda optimumda hata fırlatıyor.

**RTMDet üç tohum:** dice_tum ort **0.8398**, std **0.0127**, aralık **0.0248**;
lezyonlu ort 0.8099, std 0.0117, aralık 0.0208.

Bu aralık, Faz 1'de dört konfigürasyon arasında ölçtüğümüz — hiçbiri anlamlı çıkmayan —
farklardan büyük. Tek koşuyla model karşılaştırması yapılamayacağı doğrulandı: Faz 1'deki
"model A model B'den iyi" tipi her okuma, tohum gürültüsünün içinde kalıyor.

Seçilen eşik tohuma göre 0.30–0.40 arasında oynuyor, üçünde de iç optimum (sınırda değil).
Best epoch 56/61/93 — geniş dağılım, erken durdurma bütçesinin (100 epoch) dar olmadığını
gösteriyor.

**SegFormer s42'nin kayıp sürümü — doğrulandı.** Kayıp fonksiyonu bu oturumda iki kez değişti
(dolgu maskesi, ardından Dice teriminin lezyonlu görüntülerle sınırlanması) ve koşunun hangi
sürümle yapıldığı zaman damgalarından çözülemiyordu (Colab UTC / yerel UTC+3). `best.pt`'nin
val kaybı güncel kodla yeniden hesaplandı: logdaki 0.27918, yeniden hesaplanan 0.27918 —
fark 0. Koşu güncel kayıpla yapılmış, tohum 43/44 ile aynı deney.

> Genel kural: bir koşunun hangi kod sürümüyle üretildiği şüpheliyse, kaydedilmiş ağırlıkla
> logdaki bir metriği güncel kodla yeniden hesaplayıp karşılaştır. Zaman damgasından
> çıkarım yapma.

Not: Faz 1 SegFormer val lezyonlu Dice 0.8195, Faz 2 s42'de 0.8020. Aynı tohum, aynı mimari —
fark kaybın iki düzeltmesinden geliyor. Yani teorik olarak doğru olan değişiklikler ölçümde
geri gitmiş görünüyor. Tek koşuyla karar verilmeyecek; üç tohum bitince bakılacak.

### Faz 2 raporu üretildi

`rapor/build_report_faz2.py` → `rt_seg_raporu_faz2.html` → `rt_seg_sonuc_raporu_faz2.pdf`.
Figürler `rapor/figurleri_uret_faz2.py` ile, loglar `rapor/loglari_topla_faz2.py` ile.

Faz 1 raporundan farkı: sıralama iddiası yok. Ana kurgu (1) tohum varyansının mimari
farklarını yutması, (2) modellerin lezyon boyutuna göre ters yönde ayrışması,
(3) hız sıralamasının çalışma ortamına göre değişmesi.

Eksik: eğitim eğrileri figüründe 9 logdan yalnızca 3'ü (RTMDet) var. SegFormer ve YOLO
logları Drive'da; `rapor/loglar_faz2/ham/` içine konulup `loglari_topla_faz2.py` ve
`figurleri_uret_faz2.py` yeniden koşulunca Şekil 6 tamamlanır.

Ablasyonlar (hafif augmentasyon, 384 girdi) kullanıcı kararıyla yapılmadı; raporun
sınırlılıklar bölümünde belirtildi.

## Bilinen sınırlılıklar ve açık riskler

Sonuçları okurken ve rapora yazarken hesaba katılması gerekenler. Hiçbiri kapatılmadı.

### 1. Her model tek kez eğitildi — "SegFormer en iyi" henüz kanıtlanmadı

Tohum tekrarı yok. Ölçtüğümüz eşleştirilmiş gürültü bandı ±0.03 Dice (RTMDet 256↔512
karşılaştırmasından). SegFormer'ın test öndeliği bir sonrakine **+0.026**, yani bandın
içinde. Sıralama şu an *işaret* düzeyinde; kesinleştirmek için ya görüntü başına
eşleştirilmiş test ya da 3 tohumluk tekrar gerekiyor.

### 2. Checkpoint seçim ölçütü üç modelde farklı

| Model | `best` neye göre seçiliyor |
|---|---|
| SegFormer-B0 | `dice_tum`, taranmış eşik — raporlanan metriğin kendisi |
| RTMDet-Ins | `coco/segm_mAP` |
| YOLOv11n-Seg | Ultralytics `fitness` (mAP tabanlı) |

SegFormer'da seçim bilerek raporlanan metriğe bağlandı (hata 8'in dersi), diğer ikisinde
mAP tabanlı kaldı. Bu SegFormer lehine küçük bir avantaj olabilir: en iyi mAP epoch'u ile
en iyi Dice epoch'u aynı olmak zorunda değil. Büyüklüğü **ölçülmedi ve geriye dönük
ölçülemez** — `max_keep_ckpts=1` yüzünden ara epoch'lar saklanmadı.

### 3. Augmentasyon paritesi tam değil

`ElasticTransform` (alpha=30, p=0.25) YOLO'da yok — Ultralytics'in Albumentations
sarmalayıcısı maskeleri taşımadığı için geometrik dönüşüm enjekte edilemiyor. YOLO aynı
zamanda en zayıf model; farkın ne kadarının bundan geldiği ölçülmedi.

SegFormer'ın kaybı letterbox dolgusunu içeriyor (medyanda piksellerin ~%16'sı, en uç
en-boy oranında %31), RTMDet'in bölge tabanlı maske kaybı içermiyor. İkinci derece bir
asimetri, ölçülmedi.

### 4. Planlanan iki ablasyon yapılmadı

- **`hafif` vs `orta` augmentasyon** (PLAN "Augmentasyon tasarımı"): tek mimaride yapılıp
  kazanan diğerlerine uygulanacaktı. Yapılmadı, üç model de `orta` ile eğitildi.
- **384×384 girdi ablasyonu** (PLAN "Kararlar" tablosu): yerini 256 vs 512 karşılaştırması
  aldı. Plan metnindeki 384 ifadeleri güncellenmedi, eskimiş durumda.

### 5. Küçük sayılar üzerinde konuşulan farklar

Test'te boş maskeli görüntü yalnızca **19 tane**. "19/19 vs 17/19" farkı iki görüntü
demek; tek başına mimari yorumu taşımaz.

### 6. Veri türevleri sürüm kontrolünde değil

`.gitignore` `real_time_segmentasyon/veri`'yi hariç tutuyor, yani `veri_manifest.csv`,
COCO JSON'ları ve YOLO etiketleri repoda yok. COCO ve YOLO betiklerden yeniden
üretilebilir ama **manifest bir Colab notebook çalıştırmasının çıktısı** — kaybolursa
`rt_seg_veri_hazirlik.ipynb` yeniden koşturulmalı ve split'in birebir aynı çıkması
seed'e bağlı.

### 7. Teyit edilmemiş tek sayı

SegFormer'ın **val lezyonlu Dice 0.8195** değeri, eşik taraması tablosundan değil
hizalama taramasının merkez hücresinden okundu (ikisi aynı şeyi hesaplıyor: seçilen
eşikte, lezyonlu görüntülerde Dice). Doğru bir çıkarım ama val eşik taraması tablosu
görülerek teyit edilmeli.

## Yapılacaklar

1. **YOLO `results.csv`'yi Drive'dan getir** — tek eksik figür, rapor onunla tamamlanır

Opsiyonel, rapora değer katacak ama zorunlu olmayanlar:

2. **TensorRT** — onnxruntime'da `TensorrtExecutionProvider` mevcut; ONNX'in üstüne ek
   hızlanma ölçülebilir
3. **Uçtan uca ONNX dağıtım hattı** — şu anki ONNX ölçümü yalnızca ağı içeriyor, son
   işleme dışarıda; gerçek dağıtım FPS'i için ikisi birleştirilmeli
4. **Tohum tekrarı** — kalite sıralamasını istatistiksel olarak kapatmanın tek yolu

### Bir sonraki oturuma not

Kod ve doküman güncel. Metrik ayrışması kapandı (ana metrik orijinal çözünürlükte Dice), yarım piksel yaması ölçümle reddedilip geri alındı, epoch bütçesi 100'de kaldı (70 denendi, geri alındı). Üç mimari de eğitildi, değerlendirildi ve hız tarafı ölçüldü.

`rtmdet/calisma/` altındaki koşu sürümleri:

| Klasör | Ne |
|---|---|
| **`..._256_orta`** | **final** — `CocoMetricHizali`, yarım piksel yaması yok, 100 epoch, Dice 0.8084 |
| `..._256_orta_bozukmetrik` | v1 — bozuk `segm_mAP`, Dice 0.803 |
| `..._256_orta_70ep` | v3 — yamalı + 70 epoch, bütçe kararı hatalıydı |
| `..._256_orta_yarimpiksel_reddedildi` | v4 — yamalı + 100 epoch, yamanın kontrollü testi; yama reddedildi |

Klasörler bir kez yeniden adlandırıldı: final koşu bir süre `_metrikduzeltildi` altındaydı ve
kanonik ad reddedilen v4'ü tutuyordu — `metrik_ayrismasi.py` gibi varsayılanı `..._256_orta`
olan araçlar sessizce yanlış koşuyu ölçüyordu.

Devam noktası: **final rapor**. Ölçüm tarafı bitti.

Özet bulgu: kalitede dört konfigürasyon istatistiksel olarak ayrışmıyor; PyTorch'ta SegFormer-B0 en hızlı görünüyor ama ONNX'e geçilince RTMDet-Ins-tiny 256 öne çıkıyor (4.40 ms) — PyTorch'taki fark mimariden değil mmdet'in kare başına yükünden geliyormuş. CPU'da ise YOLOv11n-Seg açık ara önde (135 FPS, ONNX). Yani "en iyi model" sorusunun cevabı dağıtım hedefine bağlı ve rapor bunu böyle sunmalı.

Eğitim çıktıları `rtmdet/calisma/rtmdet_ins_tiny_{256,512}_orta/` altında (Excel log, eğriler, Dice tabloları, checkpoint'ler); ayrışma analizinin tüm tabloları `rtmdet/calisma/metrik_ayrismasi.xlsx` içinde (özet, mAP varyantları, boyut kırılımı, eşleştirilmiş test, eşik taraması, görüntü başına ham değerler).
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

**8. `segm_mAP`'i girdi boyutuna bağlı olarak düşüren 1 pikselik maske hatası.** RTMDet-Ins maskeyi girdi uzayında üretip `1/scale_factor` ile büyütüyor, sonra `[..., :ori_h, :ori_w]` ile kırpıyor (`rtmdet_ins_head.py:500-507`). Ters çevrilmiş `scale_factor`'un **indisleri takas edilmiş**: yüksekliğe `w_scale`, genişliğe `h_scale` uygulanıyor.

```python
size=[math.ceil(mask_logits.shape[-2] * scale_factor[0]),   # yükseklik ← w_scale
      math.ceil(mask_logits.shape[-1] * scale_factor[1])]   # genişlik  ← h_scale
```

`keep_ratio=True` bunu çözmüyor: Resize tek oran kullansa da yeni kenar uzunlukları tamsayıya yuvarlandığı için gerçekleşen `w_scale` ile `h_scale` birbirinden azıcık farklı kalıyor. 688×806 bir görüntüde 256'ya küçültme `w_scale=0.317618`, `h_scale=0.318314` veriyor; büyütme genişliği `ceil(256/0.318314) = 805` px yapıyor ve kırpma 806'ya ulaşamıyor.

COCOeval iki RLE'nin `size`'ı uyuşmadığında IoU'yu geçersiz sayıyor, dolayısıyla o görüntü `segm_mAP`'e sıfır katkıyla giriyor. Hata girdi boyutuna bağlı çünkü ölçek oranındaki yuvarlama hatası ~1/(2·girdi):

| Girdi | kısa maske | loglanan | hizalanmış |
|---|---|---|---|
| 256 | 25/116 (%21.6) | 0.412 | **0.626** |
| 512 | 2/116 (%1.7) | 0.623 | **0.644** |

Belirti tam olarak "512 mimari olarak çok daha iyi" gibi görünüyordu ve o okuma yanlış olurdu. Yakalanma yolu: aynı checkpoint'in `segm_mAP`'i mmdet'in kendi test döngüsünde 0.412, tahminler dışarıda toplanıp COCOeval'e verildiğinde 0.626 çıktı. İkisinin tahminleri bit birebir aynıydı — fark yalnızca bizim değerlendirmemizin maskeyi `ori_shape`'e ölçeklemesindeydi. O ölçekleme aslında `keep_ratio=False` döneminden kalma bir düzeltmeydi ve bu hatayı da tesadüfen kapatıyordu.

Düzeltme: `ozel_transformlar.CocoMetricHizali` — maskeyi encode'dan önce `ori_shape`'e getiren `CocoMetric` alt sınıfı; config `val_evaluator` / `test_evaluator` bunu kullanıyor. Doğrulama: aynı checkpoint'lerde mmdet test döngüsü artık 0.626 / 0.644 raporluyor, çevrimdışı ölçümle birebir.

Aynı sebeple **Dice değerlendirmesi etkilenmiyordu** — maske orada zaten orijinal boyuta ölçekleniyor. İki metriğin ayrışması da tam olarak buydu.

Bozuk metriğin ikinci bedeli **checkpoint seçimi** oldu: `save_best='coco/segm_mAP'` görüntülerin %21.6'sı sıfır katkıyla girerken karar veriyordu. Düzeltilmiş metrikle aynı config yeniden eğitildiğinde seçilen epoch 40 → **64**, lezyonlu Dice 0.803 → **0.8084** (eşik 0.2 → 0.4). Metriği düzeltmek modeli de düzeltti.

**9. Yarım piksel "düzeltmesi" — ölçüm tarafından reddedildi.** Bu madde bir hatanın değil, **yanlış çıkan bir düzeltmenin** kaydı.

Tespit doğruydu: `cv2.INTER_NEAREST` çıktı pikselini `floor` ile eşliyor ve maskeyi küçültürken her eksende +0.5 px kaydırıyor (`INTER_NEAREST_EXACT`'te kayma 0.002 px). Görüntü bilinear, yani merkez-doğru küçüldüğü için hedef maske görüntü içeriğine göre kayıyor. Ölçülen temsil tavanı 256'da 0.9810, `NEAREST_EXACT` ile 0.9915.

Buradan "mmdet'i yamalayalım, ~0.01 Dice kazanırız" sonucunu çıkardım. **Yanlıştı.** Aynı config 100 epoch, tek değişken yama:

| | v2 yamasız | v4 yamalı |
|---|---|---|
| Lezyonlu Dice (eşik 0.2) | **0.7965** | 0.7864 |
| Sınır bandı Dice | **0.6449** | 0.6265 |
| `segm_mAP` zirvesi | **0.642** | 0.621 |

Eşleştirilmiş test yamayı reddetti (sınır bandı p=0.0001, 96 görüntünün 65'inde yamasız daha iyi).

Sebebi mekanizma ölçümüyle bulundu (`rtmdet/kayma_testi.py`) — tahmin merkezi eksi GT merkezi, orijinal çözünürlükte:

| koşu | dx medyan | dy medyan |
|---|---|---|
| v2 yamasız | −1.01 | −0.50 |
| v4 yamalı | −1.96 | −1.95 |

mmdet'in **çıkarım tarafındaki geri-ölçeklemesi ters yönde ~−2 px kayma taşıyor**; eğitimdeki nearest bias'ı (+1 px) onu kısmen götürüyormuş. Ben bir ucu "düzeltince" telafiyi kaldırıp kaymayı ikiye katlamışım.

Kesin kanıt, tahmini kaydırıp Dice'ı yeniden ölçmek:

| kaydırma | v2 yamasız | v4 yamalı |
|---|---|---|
| (0, 0) | **0.7965** | 0.7864 |
| (+1, +1) | 0.7956 | 0.7904 |
| (+2, +2) | 0.7910 | **0.7907** |

v2 zaten en iyi hizada — her düzeltme onu bozuyor. v4 ~+1.5 px kaymış ve düzeltilince farkın bir kısmı geri geliyor.

**Ders:** doğru olan "her zaman `NEAREST_EXACT`" değil, **girdi hazırlığı ile çıkarım geri-ölçeklemesinin aynı piksel-merkezi konvansiyonunu kullanması.** mmdet kendi içinde tutarlıydı; tek ucunu düzeltmek tutarlılığı bozdu. Bir ön işleme değişikliği teorik olarak doğru diye kabul edilmez, ölçülür.

Ayrıca yakalanışı da bir ders: ilk yazdığım letterbox kontrolü IoU tabanlıydı ve **hata 4'ün aynısını tekrarlıyordu** — maskeyi nearest ile küçültüp bilinear ile büyütüp IoU'ya bakıyordu, yani geometri kaymasını değil interpolasyon kaybını ölçüyordu (0.963 medyan) ve `assert > 0.97` yanlış alarm verecekti. **Maske hizalaması IoU ile ölçülmez**; interpolasyondan bağımsız bir ölçü gerekir.

Kalıcı sonuç:

- `ResizeHassas` kaldırıldı, config mmdet varsayılanında (`Resize`) kaldı
- `rtmdet/kayma_testi.py` — eğitilmiş modelin tahmin-GT merkez farkını ölçer
- `ortak/degerlendirme.kaydirma_taramasi()` — **model-bağımsız** hizalama testi: tahmini kaydırıp Dice'ın nerede zirve yaptığına bakar. Zirve (0,0)'da değilse ön işleme bozuk demektir. Her üç modelin notebook'unda "Hizalama kontrolü" bölümü olarak çalışıyor; bilinen 2 px'lik yapay kaymayla doğrulandı.

RTMDet'te kalan ~−1 px'lik artık kayma düzeltilmedi: kaydırma taraması v2'nin zirvesini (0,0)'da gösteriyor, yani Dice açısından bedeli yok.

### Notlar

- Bu repodaki `.ipynb` dosyaları ile Colab'da fiilen çalışan oturum ayrı şeyler — buradaki değişiklikler Colab'a otomatik yansımıyor.
- Sıralı çalıştırma zorunlu: `veri_analizi` → `veri_hazirlik` → `augmentasyon`. İkincisi birincinin Excel çıktılarını, üçüncüsü ikincinin manifest'ini okuyor.
