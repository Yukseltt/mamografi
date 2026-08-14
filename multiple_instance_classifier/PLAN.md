# GMIC — CMMD2022 Üzerinde Global Farkındalıklı Çoklu Örnek Sınıflandırıcı — Deney Planı

**Kaynak makale:** Shen ve ark., *Globally-Aware Multiple Instance Classifier for Breast Cancer
Screening*, MLMI 2019 (`978-3-030-32692-0_3.pdf`, Türkçe özeti: [MAKALE_TR.md](MAKALE_TR.md))
**Veri seti:** CMMD2022 — The Chinese Mammography Database
(https://www.kaggle.com/datasets/tommyngx/cmmd2022, orijinal kaynak TCIA)
**Amaç:** Makaledeki ön-işleme, augmentasyon, global–lokal yapı, ROI seçimi ve MIL yaklaşımını
mümkün olduğunca birebir uygulamak; yalnızca CMMD2022'nin gerektirdiği uyarlamaları yapmak ve
sonuçları görselleştirilmiş bir raporla sunmak.

---

## 1. Veri Özeti — **ölçüldü** (Parça 1)

Kaynak: `archive/TheChineseMammographyDatabase` (yerel, 22 GB) — TCIA hiyerarşisi
`CMMD/<hasta>/<tetkik>/<seri>/1-N.dcm`. Aşağıdaki sayılar 5.202 DICOM header'ının ve
`CMMD_clinicaldata_revision.xlsx`'in tamamı taranarak ölçüldü.

### Görüntüler

| Ölçüm | Değer |
|---|---|
| DICOM sayısı | **5.202** |
| Hasta | **1.775** (her hastada 1 tetkik, 1 seri) |
| Görüntülenmiş meme | **2.601** |
| Çözünürlük | **2294 × 1914** — istisnasız hepsi aynı |
| `PhotometricInterpretation` | `MONOCHROME2` (hepsi) → **invert gerekmiyor** |
| Bit derinliği (`BitsStored`) | **8** (5.200 görüntü), **16** (2 görüntü — `D1-1343` L memesi) |
| `ImagerPixelSpacing` | 0,0941 mm (hepsi) |
| Görüntüleme yönü | **`ViewPosition` tag'i tamamen boş**; yön yalnızca `ViewCodeSequence[0].CodeMeaning` içinde: `cranio-caudal` (2.601) / `medio-lateral oblique` (2.601) |
| Meme başına görüntü | **her memede tam olarak 2: CC + MLO** (2.601 memenin hepsi) |
| Hasta başına görüntü | 2 görüntü (tek meme): 949 hasta — 4 görüntü (iki meme): 826 hasta |
| `Manufacturer` | boş (cihaz bilgisi yok) |
| `PatientOrientation` | `taraf` + `view`'den birebir türetilebiliyor, ek bilgi taşımıyor |

### Klinik veri (`CMMD_clinicaldata_revision.xlsx`)

Kolonlar doğrulandı: `ID1`, `LeftRight`, `Age`, `number`, `abnormality`, `classification`,
`subtype`. **1.872 satır**, her satır bir meme (tekrar yok). `number` her satırda `2` —
yani meme başına görüntü sayısı, ek bilgi değil.

| Ölçüm | Değer |
|---|---|
| Etiketli meme | **1.872** → **3.744 görüntü eğitilebilir** |
| Etiketsiz ama görüntülenmiş meme | **729** (1.458 görüntü) — hepsi iki memesi de görüntülenmiş hastalardan |
| `classification` | `Benign` **556** / `Malignant` **1.316** → **%29,7 / %70,3 dengesiz** |
| `abnormality` | `mass` 1.149 / `both` 461 / `calcification` 262 |
| `subtype` | 749 memede dolu (Luminal B 376, Luminal A 152, HER2-enriched 135, triple negative 86) — **yalnızca malign vakalarda** |
| `Age` | 69 farklı değer; benign ortalaması malignden düşük (Bölüm 9'da alt grup analizi) |
| İki memesi de etiketli hasta | **97** — bunların **30'unda iki memenin sınıfı farklı** |
| Alt küme × sınıf | `D1-`: 544 benign / 563 malign (dengeli) — `D2-`: **12 benign / 753 malign** |

### Bu ölçümlerin plana etkisi (yeni kararlar)

1. **Etiketsiz 729 meme eğitime alınmıyor.** İki memesi görüntülenmiş hastalarda yalnızca
   lezyonlu meme etiketli; kontralateral meme *"sağlıklı"* varsayılamaz (biyopsi doğrulaması
   yok, makale tüm etiketlerin biyopsi doğrulamalı olmasına dayanıyor). Kullanılabilir veri
   **1.872 meme / 3.744 görüntü**.
2. **Sınıf dengesizliği %30/70** → `WeightedRandomSampler` gerekli (Bölüm 2'deki koşullu karar
   böylece kesinleşti). Makaledeki "pozitifler + eşit sayıda negatif" dengeli örneklemenin
   karşılığı bu.
3. **`D2-` alt kümesi confounder (confounder):** neredeyse tamamı malign (753/765). Model
   hastalık yerine alt kümeyi öğrenebilir. Split **`D1`/`D2` × sınıf** üzerinden stratify
   edilecek ve Bölüm 9'da AUC **alt küme kırılımıyla** ayrıca raporlanacak.
4. **30 karışık sınıflı hasta** (bir memesi benign, diğeri malign): hasta-seviyesi stratified
   split bunları tek bir sınıf etiketiyle katmanlayamaz → hasta için katman etiketi
   "karışık" adında üçüncü bir kategori olacak.
5. **Görüntüler 8-bit** (12/16-bit değil): `WindowCenter 128 / Width 256`, `VOI LUT SIGMOID`,
   yani veri zaten pencerelenmiş ve 8 bit'e indirilmiş. Makalenin "ince detay" tezi için
   dinamik aralık kaybı burada **veri setinden geliyor**, bizim ön işlememizden değil —
   raporda belirtilecek. `D1-1343` L memesinin 2 adet 16-bit dosyası Parça 2'de
   `im / im.max() * 255` ile aynı ölçeğe indiriliyor (901 → 256 gri seviye); 3.734 görüntünün
   2'sini etkileyen kuantizasyon, z-score sonrası etkisi ihmal edilebilir, raporda dipnot.
6. **Çözünürlük istisnasız 2294 × 1914** → pad/kırpma varyasyonu yok, `GLOBAL_SIZE` kademe 0
   bu boyut.
7. **Görüntüleme yönü `ViewCodeSequence`'den okunmak zorunda** — `ViewPosition` boş, ona
   güvenen kod sessizce boş view üretir.

### Yinelenen piksel hash kontrolü — **TCIA uyarısı doğrulandı**

5.202 görüntünün tamamının piksel md5'i alındı: **5.197 benzersiz hash**, 10 görüntü çakışıyor,
**hepsi farklı hasta ID'lerine dağılmış** (4 hasta). İki ayrı vaka:

| Çakışma | Kapsam | Klinik etiketler | Sorun |
|---|---|---|---|
| `D1-0202` ↔ `D2-0284` | **4 görüntünün tamamı** (L/R × CC/MLO) birebir aynı, yaş ikisinde de 38 | D1-0202 L: `calcification / Malignant`, R: `calcification / Benign`; D2-0284 L: **`both` / Malignant / Luminal A**, R kaydı yok | Aynı hasta iki kez, **farklı `abnormality` etiketiyle**; biri D1 biri D2 alt kümesinde |
| `D1-0808` ↔ `D1-1292` | yalnızca **R-CC** görüntüsü aynı (R-MLO'ları farklı), yaş ikisinde de 45 | ikisi de `mass`, ama **`Benign` vs `Malignant`** | Birebir aynı piksel verisine **zıt sınıf** etiketi — biri kesinlikle hatalı |

**Karar: bu 4 hasta tamamen çıkarılıyor.** Hangi etiketin doğru olduğu bilinemiyor (TCIA de
belirleyemiyor). Tutulmaları iki ayrı zarar veriyor: (a) aynı görüntü farklı split'lere
düşerse test seti sızıntısı olur, (b) zıt etiketli birebir aynı görüntü çifti eğitim sinyalini
doğrudan kirletir. Liste `ozet/haric_hastalar.json` içinde, Parça 2 bunu okuyup uyguluyor.

### Kullanılabilir veri (nihai)

| | Ham | Dışlamalardan sonra |
|---|---|---|
| Hasta | 1.775 | **1.771** |
| Etiketli meme | 1.872 | **1.867** |
| Görüntü | 3.744 | **3.734** |
| Benign / Malign | 556 / 1.316 | **554 / 1.313** |

### Confounder taban çizgileri — **AUC'ler buna karşı okunacak**

Modelin görmediği ama görüntülerin dolaylı taşıdığı değişkenlerin tek başına malign AUC'si
(1.867 meme üzerinde ölçüldü):

| Yalnızca bu bilgiyle | Malign AUC |
|---|---|
| Yaş | **0,682** |
| `D2` alt kümesinde olmak | **0,776** |
| Yaş + D2 | **0,847** |

Makaledeki GMIC 0,900 alıyor. Bizim modelin 0,85 raporlaması hastalığı öğrendiğinin kanıtı
**değil**: `D2` görüntüleri farklı bir toplama partisinden geliyorsa CNN bu imzayı öğrenip
%98,4'ü malign olan bir kümeyi işaretleyerek bedavaya 0,776 alır; yaşın da görsel karşılıkları
var (meme yoğunluğu, involüsyon). Bu üç sayı raporda referans çizgisi olarak verilecek.

**Kohort kararı:** eğitim tüm veride yapılır (veri az), ama **birincil raporlanan sayı `D1` test
alt kümesindeki AUC**'dir; genel ve `D2` ayrı ayrı da verilir. `D1` içinde malign oranı %50,9
(1.103 meme) — asıl deney kohortu bu. `D2`'de ayrım yapılacak neredeyse bir şey yok (764 memede
12 benign). Bu karar hiçbir kapıyı kapatmıyor: sonradan yalnızca D1 ile eğitmeye geçilirse ön
işleme yeniden koşulmaz.

### Anormallik tipi ters yönde sinyal taşıyor

`D1` içinde: `mass` **%56 benign**, `calcification` %63,5 malign, `both` %65,5 malign. Lezyon tipi
sınıfla ilişkili ve yön sezgiye ters (mass daha çok benign). Beklenti: **MIL dalı kalsifikasyonda
(çok küçük ROI — `K=6` yama tam bu iş için) mass'tan daha fazla kazandırır.** Makalenin lokal dal
tezinin bu veri setindeki en net testi bu; Bölüm 9'daki alt grup analizi bunu ölçecek.

### Split büyüklüğü — test seti küçük

%15 test ≈ 280 meme ama içinde **yalnızca ~83 benign**. Bootstrap güven aralıkları geniş olacak;
0,02'lik AUC farkları "kazanç" olarak raporlanamaz. Ablasyon tablosunda **sıralama ve CI**
konuşacak, üçüncü ondalık hane konuşmayacak.

Hasta seviyesinde katmanlar: `Malignant_D2` 736, `Malignant_D1` 542, `Benign_D1` 464,
`karisik_D1` 17, `karisik_D2` 12. Son ikisi çok küçük (%15'i 1,8 hasta) → **tek `karisik`
katmanında birleştirilecek.**

### Piksel istatistikleri (40 rastgele görüntü)

- `dtype` `uint8`, min **her görüntüde 0**, max 196–255 (ortalama 238)
- Görüntü ortalaması 9–34 (çok koyu) — **piksellerin %72,2'si < 10**, yani kare büyük ölçüde
  arka plan. Parça 2'deki meme maskesi/kırpma adımının kazancı bu kadar: kırpma sonrası
  görüntünün anlamlı kısmı ~%28.
- Otsu yerine **sabit düşük eşik (≈10) + en büyük bağlı bileşen** yeterli görünüyor; Parça 2'de
  ikisi karşılaştırılıp seçilecek.

---

## 2. Makale → CMMD2022 Uyarlamaları

Uyarlanan her nokta ve gerekçesi. Bunların dışındaki her şey makaledeki gibi bırakılıyor.

| Konu | Makale | CMMD2022 | Karar |
|---|---|---|---|
| Ölçek | 229.426 tetkik / 1M görüntü | 1.775 hasta / **1.872 etiketli meme / 3.744 görüntü** | 100 model × 40 epoch random search yapılamaz → daraltılmış arama (bkz. Bölüm 7). Veri ~120× küçük |
| Etiket uzayı | `y ∈ {0,1}²`: benign var/yok, malign var/yok; **negatif tetkikler mevcut** | her meme kaydı ya benign ya malign; **sağlıklı meme yok** | Mimari birebir korunuyor: iki sigmoid başlık + iki saliency map, iki BCE terimi. Görev "malign vs benign ayrımı"na dönüşüyor; pratikte iki başlık birbirinin tümleyeni |
| Dengeli örnekleme | her iterasyonda pozitifler + eşit sayıda rastgele negatif | negatif sınıf yok; genel oran %29,7/%70,3 ama **dengesizlik D2'den geliyor**, D1 içinde denge tam | `WeightedRandomSampler`, ağırlıklar **sınıf × alt küme** üzerinden. Yalnızca sınıfa göre ağırlıklamak benign'i çekerken farkında olmadan D1'i de yukarı çeker ve alt küme dağılımını bozar |
| Girdi çözünürlüğü | 2944 × 1920, tam çözünürlük | **2294 × 1914, istisnasız hepsi aynı** | **Öncelik: native çözünürlük** (makalenin ana tezi bu). Bellek yetmezse `GLOBAL_SIZE` ile kademeli düşürme (Bölüm 6), yama boyutu 256 **her durumda sabit** |
| Meme-seviyesi tahmin | CC + MLO ortalaması | aynı yapı | Aynen: mevcut görüntülerin ortalaması; meme tek görüntülüyse o görüntünün tahmini |
| Lokalizasyon metriği | sürekli F1/P/R, piksel maskeleriyle | **piksel maskesi yok** | Nicel lokalizasyon raporlanamaz. Yerine: (a) Şekil 4'ün birebir niteliksel karşılığı (annotated input yerine ham girdi \| benign SM \| malign SM \| patch map \| 6 ROI yaması + dikkat skorları), (b) SM aktivasyonunun meme maskesi içinde kalma oranı — maskesiz uygulanabilen vekil ölçüt |
| Test seti | ayrı 14.148 tetkik | tek havuz | **Hasta seviyesinde** stratified split: train %70 / val %15 / test %15 (~1.310 / 281 / 281 meme). Katman etiketi **sınıf × alt küme (D1/D2)**, karışık sınıflı 30 hasta üçüncü kategori. Aynı hastanın iki memesi ve tüm görüntüleri **tek bölünmede** kalır |
| Baseline | ResNet-22 (Wu ve ark. 2019) | aynı mimari yeniden kurulacak | GMIC'in global dalı (`f_d`) + GAP + FC = baseline. Aynı kodda `DENEY_KEY='resnet22_baseline'` |
| Ön-eğitim | `f_t` ResNet-18 ImageNet; `f_d` sıfırdan | veri ~120× küçük | `f_t` ImageNet ön-eğitimli, `f_d` sıfırdan — **ikisi de makaleyle aynı**. ResNet-22'nin yayınlanmış ImageNet ağırlığı olmadığı için `f_d`'yi tohumlamak mümkün değil; aynı soru `gmic_ft_scratch` ablasyonuyla tersinden ölçülüyor |
| Etiketsiz meme | yok (tüm tetkikler etiketli) | **729 görüntülenmiş meme etiketsiz** (iki memesi çekilmiş hastalarda kontralateral) | Eğitime **alınmıyor** — biyopsi doğrulaması olmadan negatif varsayılamaz |
| Ön işleme / augmentasyon | makalede detay yok, Wu ve ark. 2019'a atıf | — | Yazarların yayınladığı koddan alındı (`nyukat/GMIC`): meme etrafını kırp, **sağ memeleri çevir**, görüntü başına z-score, augmentasyon olarak yalnızca kırpma penceresi gürültüsü |
| Veri bütünlüğü | — | **4 hastada yinelenen piksel + çelişen etiket** (TCIA uyarısı doğrulandı) | Dört hasta da çıkarıldı; liste `ozet/haric_hastalar.json` |
| Moleküler alt tip | yok | 749 hasta için mevcut | Eğitimde **kullanılmıyor**; rapordaki vaka görsellerinde bağlam olarak yazılıyor (`mendeley_data_density`'de density yüzdesinin kullanıldığı gibi) |

---

## 3. Model Mimarisi (makaleye birebir)

Detaylı formüller [MAKALE_TR.md](MAKALE_TR.md) Bölüm 2'de; burada uygulama karşılıkları.

### Lokalizasyon modülü — `f_d` (**koddan doğrulandı**, `nyukat/GMIC/src/modeling/modules.py`)

| Parametre | Değer |
|---|---|
| `input_channels` | **1** — gri tonlu, 3 kanala kopyalanmıyor |
| `num_filters` / `growth_factor` | 16 / 2 → kanallar 16, 32, 64, 128, 256 |
| İlk katman | 7×7 conv, stride 2, padding 3 |
| İlk pool | 3×3 max, stride 2 |
| `blocks_per_layer_list` | [2, 2, 2, 2, 2] (pre-activation ResNet**V2** blokları) |
| `block_strides_list` | [1, 2, 2, 2, 2] |
| Saliency map | `Conv2d(256, 2, 1×1, bias=False)` + sigmoid |
| Parametre sayısı | **2,80 M** (ölçüldü) |

Toplam downsampling **64** (2 conv × 2 pool × 16 block stride). Makalenin 2944×1920 girdisinde
46×30'luk saliency map verir; bizim 2304×1280 girdimizde **36×20** (test edildi).

`input_channels=1` bir düzeltme: ilk taslakta `f_d`'ye de 3 kanal verileceği varsayılmıştı.
3 kanala kopyalama yalnızca `f_t` (ImageNet ön-eğitimli ResNet-18) için geçerli.

### Tespit modülü — Algoritma 1 (**uygulandı ve test edildi**)
- Sınıf başına min-max normalizasyon → `Â = Σ_c Ã_c`
- Greedy döngü: `F.avg_pool2d` ile tüm pencere ortalamaları tek seferde, `argmax` ile en yüksek
  toplamlı pencere. Saliency map üzerindeki pencere boyutu `round(256 · h/H) = 4×4` (kodla aynı).
- `K = 6`, yama 256 × 256, **girdinin orijinal çözünürlüğünden** kırpılır.
- Turevlenebilir değil → `no_grad` + `detach`.

> **Reset kuralında küçük bir sapma (gerekçesi test tarafından bulundu).** Makale ve NYU kodu
> seçilen bölgeyi `0` yapıyor. Ama min-max normalizasyon sonrası **arka plan zaten tam 0**;
> tepeler sıfırlanınca harita her yerde 0'a eşitleniyor ve `argmax` hep aynı konumu (0,0)
> döndürüyor — yani **aynı yama K kez MIL'e giriyor.** Sentetik testte tam bu görüldü
> (6 yamanın 4'ü aynı konum). Çözüm: seçilen bölge `-1` yapılıyor; böylece seçilmiş bölgeler
> her zaman dokunulmamış bölgelerin altında kalır. Harita pozitif değer içerdiği sürece
> davranış `0` ile birebir aynı, yalnızca dejenere durumda farklı. Test: düz (sabit) bir
> saliency map'te bile 6 farklı yama üretiliyor.

> **Depo sürümü uyarısı:** `nyukat/GMIC` deposu makalenin **sonraki (dergi) sürümünü** içeriyor —
> `gmic.py`'de `y_fusion` ve `concat_vec` var, yani global ve lokal temsilleri birleştiren
> **üçüncü bir dal**. Bizim hedefimiz MLMI 2019: yalnızca `ŷ_loc` ve `ŷ_mil`, çıkarımda
> ortalaması. Bu yüzden **mimari makaleden, uygulama detayları koddan** alınıyor.

### MIL modülü (**koddan doğrulandı**)

- `f_t` = ResNet-18 (ImageNet), yama özelliği **512-boyutlu** (avgpool çıkışı)
- Yamalar **1 kanal** olarak veriliyor (kodda `...unsqueeze(1)`), 3 kanala kopyalanmıyor.
  ImageNet ön-eğitimi korunsun diye `conv1` ağırlıkları RGB ekseninde toplanıp tek kanala
  indiriliyor — ön-eğitimli filtreler korunur, girdi akışı kodla aynı kalır.
- Gated Attention: `mil_attn_V = Linear(512, 128)`, `mil_attn_U = Linear(512, 128)`,
  `mil_attn_w = Linear(128, 1)`, hepsi `bias=False`
- `α_k = softmax(w^T (tanh(V h_k) ⊙ sigm(U h_k)))`, `z = Σ_k α_k h_k` (512-boyutlu)
- Sınıflandırıcı: `Linear(512, 2, bias=False)` + sigmoid → `ŷ_mil`

> **Makale ile kod arasında fark:** Makale "yamalar `h̃_k ∈ R^L` vektörlerine dönüştürülür,
> `L = 128`" diyor. Yayınlanan kodda yama özelliği **512**, `128` ise **attention'ın gizli
> katman boyutu**. Yazarların fiilen koştuğu yapı bu olduğu için kod izleniyor; fark burada
> kayda geçiyor.

### Kayıp ve çıkarım
- `f_agg(A_c)` = `A_c` içindeki **top-t%** değerin ortalaması. Kodda `top_t = round(H*W*t)`,
  `topk(...).mean()`. Doğrulandı: `t=1` → GAP, `t→0` → GMP (makalenin iki ucu)
- `L_reg(A_c) = Σ |A^c_{i,j}|^β` — **toplam**, ortalama değil

> **Çözünürlük tuzağı:** `L_reg` bir toplam olduğu için saliency map alanıyla ölçekleniyor.
> `t` bir oran olduğundan çözünürlükten bağımsız, ama sabit bir `λ` farklı çözünürlüklerde
> farklı regularization baskısı demek: 36×20 → 720 konum, `gmic_lowres`'te 18×10 → 180 konum,
> yani **4× fark**. Makalede girdi boyutu sabit olduğu için bu sorun hiç doğmuyor.
> Karar: `gmic_lowres` koşusunda `λ` alan oranıyla (×4) ölçeklenir, aksi halde ablasyon
> çözünürlüğü değil regularization gücünü ölçer.
>
> **Ölçüldü (Parça 6 smoke testleri):** aynı kod 576×320'de `reg = 42.9`, 2304×1280'de
> `reg = 694` verdi — oran tam 16, alan oranının (16×) birebir karşılığı. Yani `L_reg`'in
> alanla doğrusal ölçeklendiği ölçümle doğrulandı, ×4 kararı tahmin değil.
> **β < 1 tekilliği — ölçüldü ve düzeltildi (Parça 7, koşu 06).** `Σ|A|^β`'nın türevi
> `β·|A|^(β−1)`; β < 1 iken A → 0'da sonsuza gidiyor. AMP altında sigmoid, yeterince büyük
> negatif logitte fp16'da **tam olarak 0**'a underflow ediyor, o anda gradyan `inf` oluyor,
> ağırlıklar NaN'a düşüyor ve bir sonraki forward'da `binary_cross_entropy`'nin "girdi [0,1]
> içinde" kernel assert'i patlıyor. Device-side assert CUDA context'ini öldürdüğü için tek bir
> aday tüm aramayı düşürüyordu.
> Çözüm: `L_reg`'de `clamp_min(1e-6)` (gradyan sonlu kalır; kayba etkisi 678 üzerinden 0,0016)
> ve BCE öncesi sonluluk kontrolü → `Iraksama` (Python hatası, CUDA context sağlam kalır, arama
> sonraki adaya geçer). β aralığı **daraltılmadı** — makalenin `β ∈ e^[-1.6,1.6]` aralığı aynen
> taranıyor, sorun sayısaldı.
- `L = Σ_c [BCE(y_c, ŷ_loc^c) + BCE(y_c, ŷ_mil^c)] + λ L_reg`
- Çıkarım: `ŷ = (ŷ_loc + ŷ_mil) / 2`
- Tespit modülü türevlenebilir değil → gradyan MIL dalından `f_d`'ye akmaz; iki dal `f_agg`
  üzerinden **eş zamanlı** eğitilir (makaledeki şema).

---

## 4. Deneyler

Makaledeki Tablo 1 ablasyonu birebir + CMMD'ye özgü iki ek deney.

| # | `DENEY_KEY` | Tanım | Makale karşılığı |
|---|---|---|---|
| 1 | `resnet22_baseline` | `f_d` + GAP + FC | ResNet-22 baseline |
| 2 | `gmic` | tam model, `ŷ = (ŷ_loc + ŷ_mil)/2` | GMIC |
| 3 | `gmic_loc` | yalnızca `ŷ_loc` (aynı eğitilmiş modelden okunur) | GMIC-loc |
| 4 | `gmic_mil` | yalnızca `ŷ_mil` (aynı modelden okunur) | GMIC-mil |
| 5 | `gmic_noattn` | tüm yamalara eşit dikkat (`α_k = 1/K`) | GMIC-noattn |
| 6 | `gmic_random` | MIL modülü rastgele seçilen yamalarda | GMIC-random |
| 7 | `gmic_loc_random` | `ŷ = (ŷ_loc + ŷ_random)/2` | GMIC-loc-random |
| 8 | `gmic_ft_scratch` | `f_t` ImageNet **olmadan** (sıfırdan) | — (CMMD ölçeğine özgü ek; ön-eğitimin katkısını ölçer) |
| 9 | `gmic_lowres` | `GLOBAL_SIZE` yarıya indirilmiş | — (makalenin "downsampling zarar verir" tezinin doğrudan testi) |
| 10 | `gmic_augment` | ana tarife augmentasyon eklenmiş (flip yok; rotasyon ±15°, parlaklık/kontrast, hafif affine) | — (NYU pipeline'ında augmentasyon yok; küçük veride maliyeti ölçmek için) |

**Üç deney ayrı eğitim gerektirmez**, değerlendirme aşamasında türetilir:
- `gmic_loc` ve `gmic_mil` → `gmic` koşusunun iki çıktı başlığından okunur
- `gmic_loc_random` → `gmic`'in `ŷ_loc`'u ile `gmic_random`'ın `ŷ_random`'ının ortalaması
  (makalede de iki ayrı modelin tahminlerinin birleşimi)

Dolayısıyla **7 eğitim koşusu** var: 1, 2, 5, 6, 8, 9, 10.

---

## 5. Kod Yapısı

`mendeley_data_density` deseni: **tek parametrik notebook**, `DENEY_KEY` ile deney seçimi.

- **`gmic_cmmd_egitim.ipynb`** — ortak şablon, sırada eğitilecek deneye ayarlı tutulur.
- **`gmic_cmmd_<DENEY_KEY>.ipynb`** — biten her deneyin çıktıları gömülü arşiv kopyası
  (Colab'da "Save a copy as"). `mass_lezyon_projesi` ve `mendeley_data_density`'deki
  notebook-başına-deney deseninin aynısı.
- **`gmic_cmmd_veri_hazirlik.ipynb`** — Faz 0/1: DICOM okuma, klinik XLSX eşleştirme, meme
  maskesi + kırpma, split üretimi, `.npy`/`.png` önbellek yazımı. Bir kez koşar, çıktısı Drive'da.
- **`hazir/onisleme.py`** — Parça 2 tarafından üretilen paylaşılan ön işleme modülü (`pencere_al`, `z_score`, `kirp_ve_cevir`, `best_center`, `en_buyuk_bilesen`). Eğitim notebook'u bunu import eder; fonksiyonlar iki yerde kopyalanmaz. Augmentasyon ve normalizasyon sabitleri de `hazir/config.json`'dan okunur — tek kaynak.
- **`butunluk.py`** — `hazir/` klasörünün bütünlük kontrolü. `uret` modu yerelde dosya başına
  boyut + md5 referansı (`hazir/butunluk.json`) yazar; `kontrol` modu bulunduğu ortamda bunu
  doğrular. Kökü kendi bulur, yani yerelde ve Colab'da aynı komutla koşar. Yedi şeyi kontrol
  eder: md5 + boyut, meta dosyaların varlığı, manifest ile diskteki PNG kümesinin birebir
  örtüşmesi, split'in hasta seviyesinde sızıntısız ve manifesti tam kaplaması, bir memenin iki
  view'inin etiket tutarlılığı, her PNG'nin açılıp manifestteki şekil/tipe uyması,
  `eval_cases` + `config` tutarlılığı. `hizli` argümanı md5'i atlar ve 50 örnek PNG açar.
- **`rapor/`** — `figurleri_uret.py` + `build_report.py`, `real_time_segmentasyon/rapor`
  deseniyle (figürler base64 gömülü tek HTML → Chrome ile PDF).

Senkronizasyon notu: bu repodaki `.ipynb` ile Colab'da fiilen çalışan oturum ayrı şeyler;
buradaki değişiklikler Colab'a otomatik yansımıyor.

---

## 6. Ön İşleme ve Augmentasyon — NYU'nun kendi pipeline'ına göre

Makale ön işlemeyi Wu ve ark. (2019)'a atıfla geçiyor ve augmentasyon listesi vermiyor. Bu
yüzden adımlar **yazarların kendi yayınladığı uygulamadan** (`github.com/nyukat/GMIC`,
`nyukat/breast_cancer_classifier`) alındı — makaleye en sadık kaynak bu:

> "A mammography image that is cropped to 2944 x 1920 and are saved as 16-bit png files."
> "In their original formats, images from `L-CC` and `L-MLO` views face right, and images from
> `R-CC` and `R-MLO` views face left. **We horizontally flipped `R-CC` and `R-MLO` images so that
> all four views face right.**"

Koddan doğrulananlar: `crop_mammogram.py` memenin etrafını kırpıp arka planı atıyor;
`standard_normalize_single_image()` **görüntü başına z-score** uyguluyor (ortalama çıkar, std'ye
böl, `eps=1e-5`); eğitimdeki tek augmentasyon `random_augmentation_best_center()` — kırpma
penceresinin **konumuna ve boyutuna gürültü** ekliyor (`max_crop_noise`, `max_crop_size_noise`).
**Flip, rotasyon veya yoğunluk augmentasyonu yok.**

### Adımlar (CMMD karşılıkları)

1. **DICOM oku** → `pydicom`, `pixel_array` (`uint8`, 2294 × 1914)
2. **Invert gerekmiyor** — tüm görüntüler `MONOCHROME2` (Bölüm 1'de ölçüldü)
3. **Meme maskesi + arka plan kırpma** — `crop_mammogram.py` karşılığı. Piksellerin %72,2'si
   `< 10` olduğu için sabit düşük eşik + en büyük bağlı bileşen yeterli görünüyor; Parça 2'de
   Otsu ile karşılaştırılıp seçilecek
4. **Sağ memeleri yatay çevir** → tüm görüntülerde meme sağa bakar. **Yazarların pipeline'ında
   birebir var** (yukarıdaki alıntı), bizim eklememiz değil
5. **Sabit pencere**: `GLOBAL_SIZE`, meme merkezine ("best center") oturtulur, taşan kenar
   `shift_window_inside_image` mantığıyla içeri kaydırılır
6. **Normalizasyon: görüntü başına z-score** (`(x - mean) / max(std, 1e-5)`) — NYU'nun
   `standard_normalize_single_image` karşılığı. **Min-max veya ImageNet mean/std kullanılmıyor.**
   `f_t` ImageNet ön-eğitimli olsa da girdi z-score'lu tek kanal, 3 kanala kopyalanarak veriliyor.
   **Tek sapma: istatistikler yalnızca meme dokusu (piksel > 10) üzerinden alınıyor** — gerekçesi
   ölçüldü, aşağıda
7. **Önbellek**: işlenmiş görüntüler `.npy` yazılır; her epoch DICOM açmak eğitimin en yavaş
   kısmı olur (aynı projedeki `load_volume` deseni, atomik `os.replace`)

> Not: Bir önceki taslakta buraya `A.Normalize(max_pixel_value=1.0)` uyarısı yazılmıştı
> (`mendeley_data_density`'de yaşanan çift-bölme hatası). ImageNet normalizasyonu artık hiç
> kullanılmadığı için o tuzak bu projede yok. Yerine konan doğrulama: Dataset çıkışında
> z-score sonrası ortalama ≈ 0, std ≈ 1 print'i.

### `GLOBAL_SIZE` — **ölçüldü: 2304 × 1280**

Kırpma sonrası boyut dağılımı (3.734 görüntü): yükseklik medyan 2043 / p99 2294,
genişlik medyan 759 / p95 1111 / p99 1278 / maks 1762. Pencere **p99'dan** seçildi ve ağa uygun
olsun diye 32'nin katına yuvarlandı: **2304 × 1280**.

- Penceresine sığmayan görüntü: **35 / 3.734 (%0,9)** — bunlarda kenar kırpılır
- Pencereden küçük olan: **3.698 / 3.734** — sıfırla doldurulur

Pencereyi küçültmek (örn. p90 → 2304 × 1024) compute'u ~%20 azaltırdı ama %10 görüntüde meme
dokusu keserdi. Lezyon her yerde olabileceği için **doku kaybı yerine padding tercih edildi.**

| Kademe | Boyut | Not |
|---|---|---|
| 0 | **2304 × 1280** | ölçülen p99, ana koşu |
| 1 | 1536 × 864 | OOM olursa |
| 2 | 1152 × 640 | `gmic_lowres` ablasyonu bu kademede |

Yama boyutu (256) hiçbir kademede değişmez — makalenin lokal dalı orijinal çözünürlükten
beslenir, aksi halde ablasyon anlamını yitirir.

### z-score istatistikleri neden maske içinden alınıyor (ölçüm)

Pencerenin alan olarak ortalama **~%46'sı padding** ve padding oranı görüntüden görüntüye
değişiyor (doku doluluk oranı %10–%82). İstatistikler pencere geneli alınırsa:

| | Doluluk oranı ile korelasyon |
|---|---|
| Pencere geneli ortalama | **+0,83** |
| Maske içi ortalama | −0,27 |

Yani pencere geneli z-score, normalizasyonu **dokuya değil padding oranına** bağlıyor. Aynı
görüntü iki farklı pencere boyutunda z-score'lanınca doku ortalaması +1,14 ile +0,57 arasında
kayıyor — iki kat fark, tamamen padding kaynaklı.

NYU'da bu sorun yok çünkü onların penceresi (2944 × 1920) kırpma boyutuna yakın, padding küçük.
Bizde padding baskın olduğu için **bu sapma zorunlu**. Padding pikselleri ham hâlde 0 kalıyor,
maske içi istatistikle normalize edilince tüm görüntülerde tutarlı bir negatif sabite
(≈ −1,7) dönüşüyor.

### Augmentasyon — yalnızca kırpma penceresi gürültüsü

Makaleye sadık tek augmentasyon:

- **Kırpma penceresi konum gürültüsü**: `max_crop_noise = (100, 100)` piksel (native boyutun
  ~%4-5'i), her kenara bağımsız `U(-1,1)` çarpanıyla
- **Kırpma penceresi boyut gürültüsü**: `max_crop_size_noise = 100` piksel
- Validation/test'te gürültü **kapalı** (NYU'nun yayınlanmış çıkarım varsayılanı `(0,0)` / `0`)

**Uygulanmayanlar ve gerekçesi:** flip (yön normalizasyonunu geri bozar — 4. adımın amacını yok
eder), rotasyon, parlaklık/kontrast jitter, elastic, agresif crop/zoom. Hiçbiri NYU
pipeline'ında yok.

> **Bilinen risk:** 1.867 memede augmentasyonsuz eğitim, 1M görüntüde olmayan bir overfitting
> baskısı yaratır. Bunu ana tarifi bozarak çözmüyoruz; yerine **ek ablasyon** olarak
> `gmic_augment` deneyi eklendi (Bölüm 4, #10): aynı model + mamografiye uygun augmentasyon
> seti (flip yok, rotasyon ±15°, parlaklık/kontrast ±0,15, hafif affine). Böylece hem makaleye
> sadık sonuç hem de augmentasyonun bu veri ölçeğinde ne kattığı raporlanır.

---

## 7. Hiperparametreler

| Parametre | Değer | Kaynak |
|---|---|---|
| `K` (yama sayısı) | 6 | makale |
| Yama boyutu | 256 × 256 | makale |
| `L` (özellik boyutu) | 128 | makale |
| `f_d` | ResNet-22 | makale |
| `f_t` | ResNet-18, ImageNet | makale |
| Optimizer | Adam | makale |
| Epoch (max) | 40 | makale |
| Early stopping patience | 10 | CMMD ölçeğine ek (makalede sabit 40 epoch × 100 model) |
| Batch size | 4, gradient accumulation ile efektif 16 | bkz. aşağıdaki ölçüm notu |
| AMP | açık (`torch.amp.autocast('cuda')`) | pratik |
| Gradient clipping | `max_norm=1.0`, `scaler.unscale_` sonrası | pratik |
| `SEED` | 42 | pratik |

### Kalibrasyon — **ölçüldü** (Colab, Tesla T4 15,6 GB, 2304 × 1280)

| | Drive üzerinden | Yerel diskten (`/content/hazir`) |
|---|---|---|
| Isınma (4 batch) | 6,7 sn | **1,4 sn** |
| Eğitim | — | 0,22 sn/batch |
| Doğrulama | — | 0,20 sn/batch |
| Epoch (654 + 139 batch) | ~12 dk (tahmini) | **2,8 dk** |
| Tepe bellek | — | 1,41 GB tahsis / 1,57 GB rezerve = **%10** |

Sonuçlar:

- **Darboğaz GPU değil I/O'ydu.** Drive FUSE üzerinden 3.734 küçük PNG okumak epoch süresini ~5×
  şişiriyordu. `hazir/`'ı bir kez `/content/hazir`'a kopyalamak (33 dk) epoch'u 2,8 dk'ya indirdi.
  Bu makaleden sapma değil, yalnızca I/O. `/content` geçici: her yeni VM'de kopya yenilenir.
- **T4 yeterli, A100 gereksiz** — bellek kullanımı %10, GPU beklemede.
- **Batch 4 kalıyor.** Bellek gerekçesi ortadan kalktı ama batch, hiperparametre aramasının
  içinde sabit tutulmalı: aramanın yarısını batch 4, yarısını batch 16 ile koşmak aday
  karşılaştırmasını bozar (ayrıca ResNet-22'nin BatchNorm istatistikleri batch boyutuna bağlı,
  gradient accumulation bunu birleştirmiyor). Seçilen `lr` batch 4 altında bulunduğu için
  **Parça 8 de batch 4 ile koşmak zorunda.**
- Bütçe: arama 8,5 saat, tam eğitimler 13,3 saat (early stopping ile daha az), toplam ~21,8 saat.

### Hiperparametre araması

Makale: 4 parametrede logaritmik random search, 100 model. CMMD'de bu bütçe yok →
**aynı aralıklarda 12 koşuluk random search**, yalnızca `gmic` deneyi üzerinde, kısaltılmış
bütçeyle (15 epoch), seçilen konfigürasyon tüm ablasyonlarda **sabitlenir**:

- `η ∈ 10^[-5.5, -3.8]`
- `λ ∈ 10^[-5, -2.8]`
- `β ∈ e^[-1.6, 1.6]`
- `t ∈ e^[-5, -1.5]`

Seçim ölçütü makaleyle aynı: **val malign AUC**. Makaledeki "top-5 ensemble" karşılığı olarak
aramadan çıkan **en iyi 3 konfigürasyon** tam bütçeyle eğitilip ensemble raporlanır (12 koşuluk
aramada top-5 aramanın yarısı olurdu, top-3 daha anlamlı).

#### Arama sonuçları — **tamamlandı**, 12/12 koşu (val malign AUC'ye göre sıralı)

| koşu | `lr` | `λ` | `β` | `t` | val malign AUC | süre |
|---|---|---|---|---|---|---|
| **05** | **6,15e-05** | **6,03e-05** | **4,510** | **0,1535** | **0,8133** | 42,9 dk |
| **04** | 2,77e-05 | 1,38e-05 | 2,853 | 0,0615 | **0,7767** | 42,9 dk |
| **10** | 1,75e-05 | 6,79e-04 | 1,898 | 0,0201 | **0,7583** | 51,0 dk |
| 11 | 8,22e-05 | 5,89e-04 | 0,698 | 0,0185 | 0,7464 | 51,6 dk |
| 03 | 3,93e-05 | 6,46e-04 | 0,834 | 0,0149 | 0,7358 | 30,8 dk |
| 00 | 6,54e-05 | 9,24e-05 | 3,150 | 0,0774 | 0,7222 | 25,9 dk |
| 07 | 5,78e-06 | 3,18e-04 | 2,189 | 0,1991 | 0,7194 | 44,4 dk |
| 06 | 6,66e-05 | 2,68e-05 | 0,899 | 0,0079 | 0,7184 | 44,6 dk |
| 02 | 5,22e-06 | 9,79e-05 | 0,661 | 0,1727 | 0,7157 | 42,9 dk |
| 09 | 5,26e-06 | 1,11e-04 | 0,417 | 0,0703 | 0,6799 | 44,2 dk |
| 08 | 1,13e-05 | 6,53e-05 | 0,907 | 0,0131 | 0,6764 | 29,6 dk |
| 01 | 4,57e-06 | 1,40e-03 | 2,306 | 0,1055 | 0,6741 | 20,1 dk |

**Seçim:** koşu 05 → `HP = dict(lr=6.148217e-05, lam=6.025001e-05, beta=4.5097, t=0.15350)`,
hücre 3'e işlendi. Ensemble için top-3: koşu **05, 04, 10**.
Kayıt: `arama/secim.json`, `arama/arama_ozeti.xlsx`, `arama/arama_dagilim.png`.

Toplam süre 7,7 saat (12 koşu, ortalama 39 dk).

#### Aramanın okunması — üç not

1. **`lr` en güçlü eğilimi gösteriyor**, dağılım grafiğinde AUC ile birlikte artıyor. Ama en iyi
   `lr` 6,15e-05 ve **örneklenen en büyük değer** 8,22e-05 iken aralığın üst sınırı 1,58e-04
   (10^-3,8). Yani aralığın üst yarısı 12 çekilişle iyi taranamadı; optimum örneklenen bölgenin
   üstünde olabilir.
2. **`β` sınırda kazandı**: 4,510, makalenin aralığının üst ucu 4,95 (e^1,6). Kenar etkisi.
   β > 1 iken `|A|^β`'nın gradyanı `A → 1`'de en büyük, `A → 0`'da ihmal edilebilir; yani kazanan
   konfigürasyon düzgün bir küçültme değil, **doygunluğa karşı hedefli** bir baskı uyguluyor.
   Kayıt için: koşu 05'te `mean(A)` 0,408 → 0,157'ye indi, `attn_entropi` 1,780 → 1,156.
3. **Top-3 birbirinden ayrılamaz.** 0,8133 / 0,7767 / 0,7583, doğrulama kümesi 266 meme;
   bu aralık bootstrap gürültüsünün içinde. "En iyi konfigürasyon" ifadesi makalenin protokolüne
   uyularak (val malign AUC ile seçim) kullanılıyor, istatistiksel bir üstünlük iddiası değil.
   Ensemble'ın gerekçesi de bu.

Aralıkları genişletmek makaleden sapma olurdu (`η ∈ 10^[-5.5,-3.8]`, `β ∈ e^[-1.6,1.6]` aynen
makaleden); bu yüzden aralık **değiştirilmedi**, kenar etkisi raporda not olarak geçecek.

Loss fp32'de hesaplanacak (`logits.float()`): `mendeley_data_density`'de AMP altında soft-Dice
terimi fp16 tavanını aşıp sessizce gradyan üretmeyi bırakmıştı. Burada Dice yok ama `L_reg`
tam SM üzerinden toplam alıyor (aynı taşma riski) → `A` üzerindeki toplam da fp32'de.

---

## 8. Loglama Standardı — Notebook İçi Canlı Log

`mendeley_data_density/density_segmentation_train.ipynb` deseni birebir: **Excel tabanlı epoch
logu + Drive kalıcı depo + yerel mirror + resume-safe checkpoint**. Gerekçe: tekrarlanabilirlik
(her koşunun tam geçmişi tek dosyada) ve Colab oturum kopmalarına dayanıklılık.

### Klasör yapısı (Drive, kalıcı)

```
/content/drive/MyDrive/gmic_cmmd/
├── eval_cases.json                     # tum deneylerde ortak, 6 sabit test vakasi
├── split.json                           # hasta-seviyesi train/val/test bolunmesi (bir kez uretilir)
├── comparison_table.xlsx                # tum deneylerin final metrikleri (ablation tablosu)
└── <DENEY_KEY>/                          # orn. gmic
    ├── <DENEY_KEY>_log.xlsx              # epoch bazli canli log (SCHEMA asagida)
    ├── checkpoints/
    │   ├── last.pt                       # her epoch uzerine yazilir
    │   └── best.pt                       # val malign AUC en yuksek oldugunda
    ├── <DENEY_KEY>_curves.png            # 2x3 egitim egrisi paneli
    ├── <DENEY_KEY>_cases/                # 6 sabit test vakasi: SM + patch map + yamalar
    ├── <DENEY_KEY>_test_summary.xlsx     # final test metrikleri (tek satir)
    └── <DENEY_KEY>_test_tahminler.xlsx   # goruntu ve meme seviyesi tahminler (ROC/DeLong icin)
```

Yerel ayna: `/content/outputs/<DENEY_KEY>/` — her Drive yazımından sonra `mirror_to_local()`.

### Excel SCHEMA (tüm deneylerde sabit, karşılaştırılabilirlik için)

```
epoch,
train_loss, train_bce_loc, train_bce_mil, train_reg,
val_loss, val_bce_loc, val_bce_mil, val_reg,
auc_malign, auc_benign,
auc_malign_loc, auc_malign_mil,
acc, precision_malign, recall_malign, f1_malign,
attn_entropi, sm_ort_aktivasyon, sm_meme_ici_oran,
lr, grad_norm
```

Neden bu kolonlar:

- **Kayıp bileşenleri ayrı** (`bce_loc`, `bce_mil`, `reg`): üç terim tek toplamda izlenirse
  `λ L_reg` terimi baskın hale gelip SM'yi tamamen sıfıra bastırdığında bunu göremeyiz.
  GMIC'in bilinen başarısızlık modu bu.
- **`auc_malign` birincil ölçüt** — makale checkpoint ve model seçimini malign val AUC ile
  yapıyor. `best.pt` bu kolona bakar, early stopping de bunu izler.
- **`auc_malign_loc` / `auc_malign_mil` her epoch ayrı** — Tablo 1'in loc/mil satırları ek
  eğitim gerektirmiyor, aynı koşudan okunuyor; ayrıca iki dalın **hangi epoch'ta ayrıştığı**
  yalnızca burada görülür.
- **`attn_entropi`** = `-Σ α_k log α_k`, val ortalaması. Dikkat çöktüğünde (tek yamaya
  kilitlenme) veya hiç seçim yapmadığında (`≈ log 6`, yani `gmic_noattn` davranışı) bunu
  loss'tan göremeyiz. GMIC-noattn ablasyonunun canlı teşhis kolonu.
- **`sm_ort_aktivasyon`** = `mean(A)`, val ortalaması. `L_reg`'in SM'yi ne kadar bastırdığını
  ölçer; sıfıra giderse `ŷ_loc` bilgisiz hale gelir.
- **`sm_meme_ici_oran`** = SM aktivasyonunun meme maskesi içinde kalan payı. Piksel maskesi
  olmadığı için lokalizasyon vekil ölçütü; model arka plana bakıyorsa burada düşer.
- **`grad_norm`** — clipping eşiği 1.0; iki dalın birleşik kaybı gradyan sıçramasına yatkın.
- Eşik bağımlı metrikler (`acc`, `precision`, `recall`, `f1`) **0.5 eşiğinde**, meme
  seviyesinde. AUC eşikten bağımsız olduğu için birincil, bunlar tamamlayıcı.

Tüm val metrikleri **meme seviyesinde** (CC + MLO tahmin ortalaması) — makale böyle
tanımlıyor. Görüntü seviyesi tahminler `test_tahminler.xlsx`'te ayrıca saklanır.

### Yazma ve resume mantığı

- `append_epoch_row(excel_path, row, resume_epoch=None)` — `mendeley_data_density`'deki
  fonksiyonun aynısı: dosya varsa okur, `resume_epoch` verildiyse o epoch ve sonrasını atar,
  yeni satırı ekler, tekrar yazar. Resume'da satır tekrarı oluşmaz.
- `RESUME=False`'ta eski `<DENEY_KEY>_log.xlsx` silinir — yarım kalmış önceki koşunun satırları
  yeni koşuyla karışmasın.
- Checkpoint içeriği: `model`, `optimizer`, `scaler`, `epoch`, `best_auc`,
  `epochs_without_improve`, `torch_rng_state`, `numpy_rng_state`.
  `epochs_without_improve` checkpoint'te tutulur — yoksa resume'da early stopping sayacı
  sıfırlanır (aynı projede yaşanmış hata).
- `find_resumable_checkpoint()`: `last.pt` içinde `epoch` ve `optimizer` yoksa **açıkça hata
  verir**, sessizce sıfırdan başlamaz.
- Final değerlendirme ve tüm görseller **`best.pt`** ile üretilir.

### Eğitim eğrileri — `plot_training_curves()`, 2×3 panel

| Panel | İçerik | Not |
|---|---|---|
| (0,0) | Loss: train + val (kalın), `bce_loc`/`bce_mil`/`reg` (ince, alpha) | üç terimin dengesi burada okunur |
| (0,1) | `auc_malign` + `auc_malign_loc` + `auc_malign_mil`, zirve epoch işaretli | Tablo 1'in canlı hali |
| (0,2) | `auc_benign` | |
| (1,0) | `precision_malign`, `recall_malign`, `f1_malign` (eşik 0.5) | |
| (1,1) | `lr` (log ölçek) + `grad_norm` (ikincil eksen, log), clip eşiği 1.0 kesikli çizgi | |
| (1,2) | `attn_entropi` + `sm_ort_aktivasyon` (ikincil eksen), `log 6` referans çizgisi | GMIC'e özgü teşhis paneli |

- AUC panellerinde **sabit `ylim=(0.5, 1.0)`**, P/R/F1 panelinde **`ylim=(0, 1)`** — 10 deneyin
  figürü rapora yan yana konulacak; otomatik ölçek küçük dalgalanmayı kararsızlık gibi
  gösteriyor (`mendeley_data_density`'de `METRIC_YLIM` aynı gerekçeyle sabitlendi).
- Her epoch sonunda tek satır konsol çıktısı: `epoch | train_loss | val_loss | auc_mal |
  auc_ben | best_auc` + iyileşme yıldızı.

---

## 9. Değerlendirme ve Rapor Figürleri

### Nicel

- **Ablation tablosu** (Tablo 1 karşılığı): 10 deney × malign AUC / benign AUC, test seti,
  meme seviyesinde.
- **Hata payı**: her deney için bootstrap (2000 tekrar) AUC %95 güven aralığı. Makale top-5
  ortalaması veriyor; burada 3 konfigürasyonun ortalaması + bootstrap CI.
- **Eşleştirilmiş karşılaştırma**: `gmic` vs `resnet22_baseline` ve `gmic` vs `gmic_loc` için
  **DeLong testi** (aynı test memeleri üzerinde). `mendeley_data_density`'deki eşleştirilmiş
  per-case karşılaştırmanın sınıflandırma karşılığı.
- **Ensemble**: en iyi 3 konfigürasyonun tahmin ortalaması → tek satır.
- **Confounder taban çizgileri**: yaş (0,682), D2 göstergesi (0,776), yaş+D2 (0,847) her AUC
  tablosunun altına referans olarak yazılır. Bir deneyin AUC'si bu çizgilerin altındaysa model
  görüntüden hastalığa dair bir şey öğrenmemiş demektir.
- **Kohort kırılımı**: her deney için AUC üç kez — **D1 (birincil)**, genel, D2. D2'de 12 benign
  olduğu için o kolonun AUC'si gürültülü, yine de confoundernın etkisini görmek için verilir.

### Görsel

1. **ROC eğrileri** — malign, tüm deneyler tek eksende + baseline.
2. **Eğitim eğrileri** — 10 deneyin `_curves.png` paneli.
3. **Şekil 4'ün karşılığı** (raporun ana görseli): 6 sabit test vakası, satır başına
   `ham girdi | benign SM | malign SM | patch map (6 mavi kare) | 6 ROI yaması + dikkat skoru`.
   Vakalar `eval_cases.json`'da sabit, tüm deneylerde aynı — modeller arası fark görselden
   okunabilsin. Alt tip bilgisi olan vakalarda başlığa yazılır.
4. **Karışıklık matrisi** — `gmic`, eşik 0.5, meme seviyesinde.
5. **Alt grup kırılımı** — `abnormality` (mass / calcification) ve `Age` bandına göre malign
   AUC. Makalede yok; CMMD'de `abnormality` kolonu var ve MIL'in kalsifikasyon (çok küçük ROI)
   ile mass'ta farklı davranması beklenir — planın en bilgilendirici ek analizi.
6. **Çözünürlük ablasyonu** — `gmic` vs `gmic_lowres`, makalenin "downsampling zarar verir"
   tezinin CMMD üzerindeki testi.

Rapor: `rapor/figurleri_uret.py` tüm figürleri tek koddan üretir (aynı eşik, aynı çizim ayarı),
`rapor/build_report.py` base64 gömülü tek HTML üretir, Chrome ile PDF'e basılır.

---

## 10. Aşamalar

Süreç **10 parçaya** bölündü. Her parça tek bir somut çıktı üretir ve bitmeden sonrakine
geçilmez. Sıradaki parça açılırken bu tablo güncellenir (durum kolonu).

| # | Parça | İş | Çıktı | Durum |
|---|---|---|---|---|
| **1** | Veri keşfi | Kaggle'dan indirme, DICOM envanteri: sayım, çözünürlük, `ViewPosition`, `PhotometricInterpretation`, klinik XLSX kolonları, benign/malign oranı, meme başına görüntü sayısı, yinelenen piksel hash kontrolü | `gmic_cmmd_veri_kesif.ipynb` + Bölüm 1 tablosu doğrulanmış | **TAMAM** — yerel veride koşuldu, Bölüm 1 ölçülen sayılarla güncellendi, `ozet/` çıktıları + `haric_hastalar.json` üretildi |
| **2** | Ön işleme + split | Invert, meme maskesi, arka plan kırpma, yön normalizasyonu, `GLOBAL_SIZE` pad, `.npy` önbellek; hasta-seviyesi stratified `split.json`, `eval_cases.json` | `gmic_cmmd_veri_hazirlik.ipynb` + hazır veri | **TAMAM** — 3.734 görüntü `hazir/` altında (kırpılmış + yön normalize, kayıpsız PNG), `manifest.csv`, `split.json`, `eval_cases.json`, `config.json` üretildi |
| **3** | Log / checkpoint altyapısı | `SCHEMA`, `append_epoch_row`, `save/restore_checkpoint`, `find_resumable_checkpoint`, `mirror_to_local`, `plot_training_curves` (2×3 panel) — veriden bağımsız, tek başına test edilebilir | eğitim notebook'unun log hücreleri | **TAMAM** — `gmic_cmmd_egitim.ipynb` (Parça 3 bölümü), sentetik 40 epoch logla test edildi |
| **4** | Global dal | ResNet-22 (`f_d`), 1×1 conv + sigmoid → `A`, `f_agg` (top-t%), `L_reg`; `resnet22_baseline` (GAP + FC) aynı koddan | model hücreleri, şekil/boyut testleri geçmiş | **TAMAM** — ResNet-22 (2,80 M), saliency map 36×20 doğrulandı, `f_agg` elle hesapla karşılaştırıldı, gradyan akışı ve baseline test edildi |
| **5** | Lokal dal | Algoritma 1 (greedy ROI çekme, örtüşme sıfırlama), `f_t` = ResNet-18, Gated Attention, `ŷ_mil`; birleşik kayıp ve `ŷ = (ŷ_loc + ŷ_mil)/2` | model hücreleri tamam | **TAMAM** — GMIC 14,10 M parametre; Algoritma 1, gated attention, birleşik kayıp ve gradyan akışı test edildi |
| **6** | Eğitim döngüsü + smoke test | Metrikler (meme seviyesi AUC, `attn_entropi`, `sm_ort_aktivasyon`, `sm_meme_ici_oran`), AMP + clipping, early stopping; 2 epoch × küçük alt küme | çalışan `gmic_cmmd_egitim.ipynb` | **TAMAM** — yerelde CPU'da (576×320) ve Colab GPU'da **tam çözünürlükte (2304×1280)** baştan sona geçti: 22/22 log kolonu dolu, checkpoint + grafik + resume doğrulandı, OOM yok |
| **6b** | Drive bütünlük kontrolü | `butunluk.py` ile Colab'daki `hazir/` klasörünün yerel referansa karşı doğrulanması | `hazir/butunluk.json` + geçen kontrol | **TAMAM** — 3.739 dosyanın tamamı yereldekiyle bit bazında aynı (md5), 3.734 PNG'nin hepsi açıldı ve manifestteki şekil/tipe uydu, split ve etiket tutarlılığı geçti |
| **7** | Hiperparametre araması | 12 koşu × 15 epoch, `η/λ/β/t` log-uniform; seçim ölçütü val malign AUC | seçilen konfigürasyon + en iyi 3 | **TAMAM** — 12/12 koşu başarılı (7,7 saat), en iyi koşu 05 val malign AUC 0,8133, `HP` hücre 3'e işlendi, top-3 = 05/04/10 |
| **8** | Tam eğitimler | **7 koşu** (3, 4, 7 numaralılar türetilir), 40 epoch, patience 10 | 7 × log / curves / summary / cases | bekliyor |
| **9** | Değerlendirme | Bootstrap AUC CI, DeLong testleri, alt grup kırılımı (`abnormality`, `Age`), ensemble | `comparison_table.xlsx` | bekliyor |
| **10** | Rapor | `rapor/figurleri_uret.py` + `build_report.py`, base64 gömülü tek HTML → Chrome ile PDF | `rapor/gmic_cmmd_raporu.pdf` | bekliyor |

Sıra notu: 3 numaralı parça veriden bağımsız olduğu için 1–2 ile paralel yürütülebilir, ancak
karışıklık olmaması için sırayla gidiliyor.

---

## 11. Başka Makinede Devam Etme

İki depo var ve ikisi de gerekli: **git** kodu ve küçük çıktıları taşır, **Drive** veriyi ve
eğitim çıktılarını taşır. Notebook'lar VSCode'dan Colab kernel'ine bağlanarak koşuyor, yani
notebook dosyası yerelde (git'ten), veri ve loglar Drive'da.

### Git'te olan (~2,1 MB)

| Dosya | Ne |
|---|---|
| `gmic_cmmd_veri_kesif.ipynb` | Parça 1 |
| `gmic_cmmd_veri_hazirlik.ipynb` | Parça 2 — `hazir/`'ı üreten notebook |
| `gmic_cmmd_egitim.ipynb` | Parça 3–7 (model, eğitim döngüsü, arama) |
| `PLAN.md`, `MAKALE_TR.md` | plan ve makalenin Türkçe versiyonu |
| `978-3-030-32692-0_3.pdf` | makale |
| `butunluk.py` | `hazir/` bütünlük kontrolü |
| `hazir/butunluk.json` | 3.739 dosyanın boyut + md5 referansı |
| `ozet/` | Parça 1–2 ölçüm çıktıları (xlsx + png), raporda kullanılacak |

### Drive'da olan (`/content/drive/MyDrive/multiple_instance_classifier/`)

| Klasör | Ne | Kaybolursa |
|---|---|---|
| `archive/` | ham CMMD2022 DICOM'ları | Kaggle'dan yeniden indirilir |
| `hazir/` | 3.734 kırpılmış PNG (2,78 GB) + `manifest.csv`, `split.json`, `eval_cases.json`, `config.json`, `onisleme.py` | Parça 2 `archive/`'den yeniden üretir |
| `arama/` | Parça 7 koşu logları + `sonuc.json` + `secim.json` | **yeniden üretilemez** — 8,5 saatlik koşu |
| `gmic/`, diğer deney klasörleri | Parça 8 logları, checkpointler, grafikler | yeniden üretilemez |

Git'te **olmayan ve olmaması gerekenler**: `hazir/`'ın PNG'leri (2,8 GB), `archive/`,
`_smoke/` (324 MB test checkpointleri), `_altyapi_testi/`, `hazir_duman/`, `outputs/`, `arama/`.

### Yeni makinede kurulum

1. `git clone` → `multiple_instance_classifier/` klasörü kodla birlikte gelir.
2. Aynı Google hesabıyla Colab'a bağlan; Drive'daki veri olduğu yerde durur, yüklemeye gerek yok.
3. VSCode'da Colab kernel'ine bağlan, `gmic_cmmd_egitim.ipynb`'i aç.
4. `butunluk.py kontrol` ile Drive'daki `hazir/`'ı doğrula (referans git'ten geldi).
5. Hücre 43 `hazir/`'ı `/content/hazir`'a kopyalar (~33 dk, yeni VM'de her seferinde).
6. Arama yarımsa hücre 48'i koş — bitmiş koşular `sonuc.json` sayesinde atlanır.

Tek kısıt: Drive hesabı aynı olmalı. Farklı hesapla devam edilecekse `archive/` + `hazir/`
(toplam ~10 GB) yeni hesaba kopyalanmalı, ya da `archive/` kopyalanıp Parça 2 yeniden koşulmalı.

---

## 12. Açık Riskler

- ~~**Bellek**~~ — **kapandı**: Colab GPU'da 2304 × 1280, batch 4 + AMP + gradient accumulation
  ile smoke test OOM vermeden geçti. `GLOBAL_SIZE` kademe düşürmeye gerek yok, native
  çözünürlük şartından sapılmıyor.
- **Küçük test seti**: ~280 meme / ~83 benign → geniş bootstrap CI. Ablasyon sıralaması
  yorumlanabilir, ondalık farklar yorumlanamaz.
- **Confounderlar**: yaş+D2 tek başına 0,847 AUC veriyor. Modelin bunu aşıp aşmadığı ancak D1
  kırılımıyla anlaşılır — bu yüzden birincil metrik D1 AUC.
- **Veri ölçeği**: 1.771 hasta ile makaledeki AUC seviyelerine (0.900) ulaşılması beklenmiyor.
  Hedef mutlak AUC değil, **ablasyon sıralamasının** (GMIC > GMIC-loc ≈ GMIC-mil > baseline,
  noattn ve random varyantların düşmesi) yeniden üretilmesi.
- **Negatif sınıf yokluğu**: görev "malign vs benign" ayrımı; makaledeki "bulgu var/yok"
  taramasından farklı ve daha zor bir ayrım. AUC'lerin makaleden düşük çıkması bu yüzden de
  beklenir, raporda ayrıca belirtilecek.
- **`L_reg` çökmesi**: `λ` fazla büyükse SM sıfıra bastırılır ve `ŷ_loc` bilgisiz kalır.
  `sm_ort_aktivasyon` kolonu bunun canlı nöbetçisi.
- **TCIA yinelenen hash uyarısı**: çakışan görüntüler çıkarılmazsa split sızıntısı olur;
  Faz 0'da kontrol edilecek.
