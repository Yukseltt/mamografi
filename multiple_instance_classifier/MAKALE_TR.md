# Meme Kanseri Taraması için Global Farkındalıklı Çoklu Örnek Sınıflandırıcı (GMIC)

**Orijinal başlık:** Globally-Aware Multiple Instance Classifier for Breast Cancer Screening
**Yazarlar:** Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras (NYU)
**Yayın:** MLMI 2019, LNCS 11861, s. 18–26. DOI: 10.1007/978-3-030-32692-0_3
**Yerel kopya:** `978-3-030-32692-0_3.pdf`

---

## Özet (Abstract)

Doğal görüntüler üzerinde görsel sınıflandırma için tasarlanmış derin öğrenme modelleri medikal
görüntü analizinde yaygınlaştı. Ancak medikal görüntüler doğal görüntülerden pek çok yönden
ayrılır: çözünürlük çok daha yüksektir, ilgi bölgeleri (ROI) çok daha küçüktür. Ayrıca medikal
görüntü analizinde hem **global yapı** hem de **lokal detay** önemlidir.

Yazarlar bu özelliklere cevap olarak, meme kanseri lezyonlarını **hem global bir belirginlik
haritasından (saliency map) hem de birden fazla lokal yamadan (patch)** gelen bilgiyi kullanarak
sınıflandıran bir sinir ağı önerir. Model, ResNet tabanlı temel modeli (baseline) geride bırakır
ve tarama mamografisi yorumlamada radyolog seviyesinde performansa ulaşır. Model yalnızca
**görüntü-seviyesi etiketlerle** eğitilmesine rağmen, olası malign bulguların yerini gösteren
**piksel-seviyesi saliency mapler** üretebilir.

**Anahtar kelimeler:** derin öğrenme, sinir ağları, meme kanseri taraması, zayıf gözetimli
lokalizasyon, yüksek çözünürlüklü görüntü sınıflandırma.

---

## 1. Giriş

Meme kanseri ABD'de kadınlarda kanser kaynaklı ölümlerin ikinci nedenidir. Tarama mamografisinin
mortaliteyi belirgin şekilde düşürdüğü gösterilmiş olsa da kusursuz bir araç değildir. Bu
sınırların aşılması için doğal görüntü görevleri için tasarlanmış evrişimli sinir ağları (CNN)
mamografiye uyarlanmıştır: örneğin ImageNet sınıflandırması için tasarlanan VGGNet meme yoğunluğu
sınıflandırmasına, Faster R-CNN ise mamografide şüpheli bulguların lokalizasyonuna uygulanmıştır.

### Doğal görüntü modellerinin medikal görüntüye uymayan yanları

1. **Çözünürlük:** Medikal görüntüler tipik doğal görüntülerden çok daha yüksek çözünürlüklü
   olduğu için doğal görüntüler için iyi çalışan derin CNN'ler GPU bellek kısıtı nedeniyle
   doğrudan uygulanamaz.
2. **ROI boyutu:** Lezyon ve kalsifikasyon gibi ilgi bölgeleri, doğal görüntülerdeki nesnelere
   kıyasla oransal olarak çok daha küçüktür. Etiketi bazen sadece birkaç piksellik ince
   detaylar, bazen de radyodens dokunun uzamsal dağılımı gibi global özellikler belirler.
3. **Downsampling kaybı:** Doğal görüntüler agresif küçültmeye rağmen sınıflandırma için gerekli
   bilgiyi korurken, medikal görüntülerde küçültme ile ciddi bilgi kaybı olur ve doğru tanı
   imkânsız hâle gelebilir.

### Makalenin katkıları

- Global bilgiyi bir **saliency mapnda (SM)** koruyan, önemli detayları ise **Çoklu Örnek
  Öğrenme (MIL)** çerçevesiyle bir araya toplayan yeni bir model.
- Piksel-seviyesi lezyon anotasyonuna dayanan yaklaşımların aksine **yalnızca görüntü-seviyesi
  gözetim** gerektirir; buna karşın şüpheli lezyonları işaretleyen piksel-seviyesi belirginlik
  haritaları üretir.
- **Dikkat (attention) mekanizması** ile bilgilendirici yamaları seçer; böylece sınıflandırma
  süreci yorumlanabilir hâle gelir.
- 1 milyondan fazla yüksek çözünürlüklü tarama tetkiki üzerinde eğitilip değerlendirildiğinde
  ResNet tabanlı temel modeli aşar ve radyolog seviyesinde performans elde eder.

### İlgili çalışmalar

Önceki yaklaşımlar meme kanseri tespitine MIL ve 3D CNN gibi tekniklerle yaklaşmıştır. Bu model
zayıf gözetimli nesne tespiti çalışmalarından esinlenir: görüntü-seviyesi etiketlerle eğitilen
CNN sınıflandırıcıların piksel seviyesinde semantik segmentasyon yapabildiği gösterilmiştir. Bu
iki adımda gerçekleşir: (1) bir omurga CNN girdi görüntüsünü ayırt edici bölgeleri öne çıkaran
bir saliency mapna çevirir; (2) bir global havuzlama (pooling) operatörü bu haritayı
skaler tahminlere indirir, böylece model uçtan uca eğitilebilir olur. Mevcut modellerin çoğu
görüntü-seviyesi tahmin için yalnızca saliency mapna dayanır ve ince detayları kaçırır.
GMIC ise buna ek olarak, özel bir **yama-seviyesi sınıflandırıcı** ile ROI önerilerinden gelen
lokal bilgiyi de kullanır.

---

## 2. Yöntem

Görev **çok-etiketli sınıflandırma** olarak kurulur. Gri tonlu yüksek çözünürlüklü bir görüntü
`x ∈ R^{H,W}` verildiğinde, `y` etiketi tahmin edilir; `y_c`, `c ∈ C` sınıfının bulunup
bulunmadığını gösterir.

GMIC üç modülden oluşur:

| # | Modül | İşlevi |
|---|-------|--------|
| i | **Lokalizasyon modülü** | `x`'i işleyip ROI'lerin yaklaşık konumunu gösteren saliency map `A`'yı üretir |
| ii | **Tespit modülü** | `A`'yı kullanarak `x`'ten `K` adet yama çeker; bunlar rafine ROI önerileridir |
| iii | **MIL modülü** | Çekilen yamalardan gelen bilgiyi birleştirip nihai tahmini üretir |

### 2.1 Lokalizasyon modülü

Önce bir CNN `f_d(·)` ile `x`'ten ilgili özellikler çıkarılır. Bellek kısıtı nedeniyle girdi
görüntüleri genelde `f_d(·)` öncesinde küçültülür; ancak mamografide küçültme lezyon sınırlarını
bozar ve küçük ROI'leri bulanıklaştırır. **Orijinal çözünürlüğü korumak için** `f_d(·)`, bir
**ResNet-22** olarak parametrelendirilir ve global average pooling ile tam bağlı katmanları
kaldırılır. Bu mimari orijinal ResNet'lere göre her katmanda **daha az filtreye** sahiptir;
böylece görüntü tam çözünürlükte işlenirken GPU bellek tüketimi yönetilebilir kalır.

Son artık bloktan (residual block) sonraki özellik haritaları, **1×1 evrişim + sigmoid**
doğrusalsızlığı ile saliency map `A ∈ R^{h,w,|C|}`'ye dönüştürülür. `A`'nın her elemanı
`A^c_{i,j} ∈ [0,1]`, `(i,j)` uzamsal konumunun girdiyi `c` sınıfı olarak sınıflandırmaya
katkısını gösteren bir skordur.

### 2.2 Tespit modülü

`f_d(·)`'nin sınırlı genişliği nedeniyle yalnızca kaba lokalizasyon sağlanabilir. Lokalizasyon
modülünü ince detayla tamamlamak için yamalar ROI önerisi olarak kullanılır. `K` adet öneri
`x̃_k ∈ R^{h_c,w_c}` çıkarmak için **greedy (greedy) bir algoritma** tasarlanmıştır. Deneylerde
`K = 6` ve `w_c = h_c = 256` kullanılır.

**Algoritma 1 — ROI'leri çek**

```
Girdi : x ∈ R^{H,W},  A ∈ R^{h,w,|C|},  K
Çıktı : O = { x̃_k | x̃_k ∈ R^{h_c,w_c} }

 1: O = ∅
 2: her c ∈ C sınıfı için
 3:     Ã_c = min-max-normalizasyon(A_c)
 4: bitir
 5: Â = Σ_{c∈C} Ã_c
 6: l : Â üzerinde herhangi bir (h_c·h/H) × (w_c·w/W) dikdörtgen yama
 7: f_c(l, Â) = Σ_{(i,j)∈l} Â[i,j]
 8: k = 1..K için
 9:     l* = argmax_l f_c(l, Â)
10:     L = l*'ın x üzerindeki konumu
11:     O = O ∪ {L}
12:     Â[i,j] = 0,  ∀(i,j) ∈ l*        # reset kuralı
13: bitir
14: O döndür
```

12. satırdaki **reset kuralı**, çıkarılan ROI önerilerinin birbiriyle belirgin şekilde
örtüşmemesini garanti eder.

### 2.3 Çoklu Örnek Öğrenme (MIL) modülü

ROI yamaları kaba bir saliency mapndan çekildiği için her yamanın taşıdığı sınıflandırma
bilgisi büyük ölçüde değişir. Bunu ele almak üzere yamalardan gelen bilgi bir MIL çerçevesiyle
birleştirilir:

1. Her örneğe bir **tespit ağı `f_t(·)`** uygulanır ve yamalar `h̃_k ∈ R^L` özellik vektörlerine
   dönüştürülür. Tüm deneylerde `L = 128`. `f_t(·)`, **ImageNet ön-eğitimli ResNet-18** olarak
   parametrelendirilir.
2. Yamaların hepsi tahmin için ilgili olmadığından, modelin bilgilendirici yamaları seçmesi için
   **Gated Attention (Gated Attention, Ilse ve ark. 2018)** kullanılır. Seçim süreci
   attention-ağırlıklı bir temsil verir:

   `z = Σ_{k=1..K} α_k · h̃_k`,  dikkat skoru `α_k ∈ [0,1]` her yamanın ilgililiğini gösterir.
3. `z`, sigmoid aktivasyonlu tam bağlı bir katmana verilerek tahmin üretilir:
   `ŷ_mil = sigm(w_mil^T z)`,  `w_mil ∈ R^{L×|C|}` öğrenilebilir parametrelerdir.

### 2.4 Eğitim

Modeli uçtan uca eğitmek zordur: **tespit modülü türevlenebilir değildir**, dolayısıyla
`L(y, ŷ_mil)` kaybından gelen gradyan lokalizasyon modülüne akmaz. Bu problem, lokalizasyon
modülü ile MIL modülünü **eş zamanlı eğiten** bir şemayla aşılır.

Bunun için bir **aggregation fonksiyonu** `f_agg(A_c): R^{h,w} → [0,1]`, her `c`
sınıfının saliency mapnı bir `ŷ^c_loc` tahminine çevirir:

- **Global Average Pooling (GAP)** tahmini seyreltir: `A_c`'deki uzamsal konumların çoğu arka
  plana karşılık gelir ve çok az eğitim sinyali sağlar.
- **Global Max Pooling (GMP)** gradyanı tek bir uzamsal konuma geri yayar; öğrenme yavaş ve
  kararsız olur.
- Bu nedenle **GAP ile GMP arasında yumuşak bir denge** kullanılır:

  `f_agg(A_c) = (1/|H+|) · Σ_{(i,j)∈H+} A^c_{i,j}`

  burada `H+`, `A_c` içindeki **en yüksek t% değerin** konumlarını içeren kümedir; `t` bir
  hiperparametredir. `A^c_{i,j} ∈ [0,1]` olduğundan `ŷ^c_loc = f_agg(A_c)` geçerli bir
  olasılıktır.

Saliency mapnı ince ayarlamak ve lokalizasyon modülünün alakasız alanları öne
çıkarmasını engellemek için `A_c` üzerine bir **regularization (regularization)** uygulanır:

`L_reg(A_c) = Σ_{(i,j)} |A^c_{i,j}|^β`,  `β` hiperparametre.

**Toplam kayıp fonksiyonu (Denklem 1):**

```
L(y, ŷ) = Σ_{c∈C} [ BCE(y_c, ŷ^c_loc) + BCE(y_c, ŷ^c_mil) ] + λ · L_reg(A_c)
```

`BCE` ikili çapraz entropi, `λ` hiperparametredir.

**Çıkarım (inference) aşamasında** tahmin iki dalın ortalamasıdır:

`ŷ = (1/2) · (ŷ_mil + ŷ_loc)`

---

## 3. Deneyler

Model, bir mamografi tetkikinde **herhangi bir benign veya malign bulgu olup olmadığını** tahmin
etme görevinde değerlendirilir.

### Veri seti

- **229.426 tetkik (1.001.093 görüntü)**.
- Tüm veri setinde malign bulgu **985 memede**, benign bulgu **5.556 memede** mevcut.
- Her tetkik **4 gri tonlu görüntü (2944 × 1920)** içerir: iki standart görüntüleme (CC ve MLO),
  sol ve sağ meme için.
- Her memeye bir etiket `y ∈ {0,1}^2` atanır; `y_c ∈ {0,1}`, `c ∈ {benign, malign}` sınıfının o
  memede bulunup bulunmadığını gösterir. **Tüm bulgular biyopsi ile doğrulanmıştır.**
- Aynı memenin iki görüntüsü aynı etiketi paylaşır.
- Verinin **%1'den küçük bir kısmı** için piksel-seviyesi segmentasyon `M_c ∈ {0,1}^{H×W}`
  mevcuttur. **Segmentasyonlar tüm deneylerde yalnızca değerlendirme için kullanılır**, eğitimde
  kullanılmaz.

### 3.1 Deney kurulumu ve değerlendirme metrikleri

- Ön-işleme, Wu ve ark. (2019) ile **aynıdır**.
- Bölünme (birbirinden ayrık): **eğitim 186.816 / validation 28.462 / test 14.148** tetkik.
- **Her iterasyonda**, en az bir benign veya malign bulgu içeren tüm tetkikler ve **eşit sayıda
  rastgele örneklenmiş negatif tetkik** ile eğitim yapılır (dengeli örnekleme).
- Tüm görüntüler **2944 × 1920** piksele kırpılır ve normalize edilir.
- Eğitim kaybı **Adam** ile optimize edilir.
- Hiperparametreler **rastgele arama (random search)** ile, logaritmik ölçekte:
  - öğrenme oranı `η ∈ 10^[-5.5, -3.8]`
  - regularization ağırlığı `λ ∈ 10^[-5, -2.8]`
  - regularization eksponenti `β ∈ e^[-1.6, 1.6]`
  - havuzlama eşiği `t ∈ e^[-5, -1.5]`
- **100 ayrı model**, her biri **40 epoch** eğitilir.

**Sınıflandırma metriği:** ROC eğrisi altındaki alan (**AUC**), **meme seviyesinde**. Model her
görüntü için bir tahmin ürettiğinden ve her meme iki görüntüyle (CC ve MLO) ilişkili olduğundan,
**meme-seviyesi tahmin iki görüntü-seviyesi tahminin ortalaması** olarak tanımlanır.

**Lokalizasyon metriği:** sürekli (continuous) F1 skoru; precision ve recall şöyle tanımlanır:

```
P = ( Σ_{i,j ∈ M_c} A^c_{i,j} ) / ( Σ_{i,j} A^c_{i,j} )
R = ( Σ_{i,j ∈ M_c} A^c_{i,j} ) / |M_c|
```

`M_c` segmentasyon etiketi, `A_c` ise `c` sınıfının saliency mapdır. Test setinde bu
metrikler, segmentasyon etiketi bulunan görüntüler üzerinde ortalanır.

### 3.2 Sınıflandırma performansı

Hiperparametre aramasından, malign sınıflandırmada en yüksek validation AUC'yi elde eden **5
model (top-5)** seçilir ve bunların **ortalama test performansı** raporlanır.

**Tablo 1 — Temel model ve GMIC varyasyonlarının AUC değerleri**

| Model | Malign | Benign |
|-------|--------|--------|
| ResNet-22 (baseline) | 0.827 | 0.731 |
| GMIC-loc | 0.885 | 0.777 |
| GMIC-mil | 0.878 | 0.766 |
| GMIC-noattn | 0.823 | 0.726 |
| GMIC-random | 0.757 | 0.692 |
| GMIC-loc-random | 0.889 | 0.776 |
| **GMIC** | **0.900** | **0.784** |

Varyantların tanımı ve çıkarımlar:

- **GMIC-loc**: tahmin olarak yalnızca `ŷ_loc` kullanır. **GMIC-mil**: yalnızca `ŷ_mil`. Her ikisi
  de baseline'ı, özellikle malignite tahmininde, aşar.
- **Tam model GMIC** (`ŷ = (ŷ_loc + ŷ_mil)/2`) her iki varyanttan da yüksek AUC elde eder. Bu
  kazanım **lokal ve global bilginin sinerjisine** atfedilir.
- **GMIC-noattn**: her ROI yamasına eşit dikkat verir. GMIC-mil'den daha az doğrudur → **MIL
  modülündeki dikkat mekanizması sınıflandırma için gereklidir**.
- **GMIC-random**: MIL modülünü görüntüden **rastgele seçilen** yamalara uygular; GMIC-mil'den
  zayıftır.
- **GMIC-loc-random**: `ŷ = (ŷ_loc + ŷ_random)/2`; GMIC-loc üzerine hiçbir kazanım sağlamaz.

Bu gözlemler hipotezi doğrular: MIL modülünü **yüksek çözünürlüklü ROI yamalarına** uygulamak,
saliency mapnın çıkardığı global bilgiyi tamamlar ve tahminleri rafine eder.

### Okuyucu (reader) çalışması

Klinik değeri ölçmek için GMIC, Wu ve ark. (2019)'daki okuyucu çalışması verisiyle radyologlarla
karşılaştırılır. Çalışmada **14 radyolog**, **720 tarama tetkiki (1440 meme)** için malignite
olasılığı tahmini vermiştir; radyologlara yalnızca görüntüler gösterilmiş, başka veri
verilmemiştir. Tahminleri iyileştirmek için **top-5 modelin ensemble'ı** kullanılır.

| Karşılaştırma | AUC |
|---------------|-----|
| GMIC ensemble | **0.876** |
| 14 okuyucunun ortalaması | 0.778 |
| En doğru okuyucu | 0.860 |
| **İnsan-makine hibridi** (radyolog + model tahmin ortalaması) | **0.883** |

GMIC okuyucu çalışmasında test setine göre bir miktar düşük performans gösterir; çünkü okuyucu
çalışması **çok daha yüksek oranda pozitif örnek** içerir. Hibrit sonucu, modelin görevin
radyologlardan **farklı yönlerini** yakaladığını ve tarama tetkiklerinin yorumlanmasında yardımcı
araç olarak kullanılabileceğini gösterir.

### 3.3 Lokalizasyon performansı

Malignite lokalizasyonunda **en yüksek validation F1** değerine sahip model seçilir. Çıkarım
aşamasında saliency mapler, segmentasyon etiketlerinin çözünürlüğüyle eşleşmesi için **en
yakın komşu (nearest neighbour) interpolasyonu** ile büyütülür.

| Sınıf | Sürekli F1 | Precision | Recall |
|-------|-----------|-----------|--------|
| Malign | 0.207 | 0.288 | 0.254 |
| Benign | 0.133 | 0.135 | 0.224 |

En iyi lokalizasyon modeli aynı zamanda malign/benign için **0.886 / 0.78** sınıflandırma AUC'si
elde eder.

**Görsel inceleme (Şekil 4).** Test setinden seçilen üç örnekte: ilk iki örnekte belirginlik
haritaları gerçek lezyonlar üzerinde yüksek aktivasyon gösterir — yani model **piksel-seviyesi
gözetim olmadan** şüpheli lezyonları tespit edebilmektedir. Ayrıca dikkat skoru `α_k`, anotasyonlu
lezyonlarla örtüşen ROI yamalarında yoğunlaşır. Üçüncü örnekte malign saliency map büyük
bir malign lezyonun yalnızca bir kısmını işaretler; bu davranış `f_agg` tasarımıyla ilgilidir:
**sabit bir havuzlama eşiği `t` tüm ROI boyutları için optimal olamaz**. Bu gözlem ayrıca şu
noktayı gösterir: insan uzmanlardan lezyonun tamamını anotlamaları istenirken, CNN'ler yalnızca
**en bilgilendirici kısmı** öne çıkarmaya eğilimlidir.

---

## 4. Sonuç

Meme kanseri tarama tetkiki sınıflandırması için yeni bir model sunulmuştur. Yöntem girdiyi
**orijinal çözünürlüğünde** kullanırken ince detaylara odaklanabilir. Ayrıca ek yorumlanabilirlik
sağlayan saliency mapler üretir. Büyük bir mamografi veri setinde GMIC, ResNet tabanlı
baseline'ı aşar ve radyologlar kadar doğru tahminler üretir. Genel tasarımı sayesinde yöntem başka
görüntü sınıflandırma görevlerine de geniş ölçüde uygulanabilir. Gelecek araştırma, GMIC'in
lokalizasyonunu MIL modülünden gelen hata sinyalleriyle iyileştirmesini sağlayacak **ortak (joint)
eğitim mekanizmaları** tasarlamaya odaklanacaktır.

---

## Uygulama için kritik hiperparametre özeti

| Parametre | Makaledeki değer |
|-----------|------------------|
| Girdi çözünürlüğü | 2944 × 1920 (gri tonlu) |
| Global omurga `f_d` | ResNet-22 (ince/az filtreli, GAP+FC çıkarılmış) |
| Saliency map çıkışı | 1×1 conv + sigmoid, `A ∈ R^{h,w,|C|}` |
| Sınıf sayısı `\|C\|` | 2 (benign, malign) — çok-etiketli |
| Yama sayısı `K` | 6 |
| Yama boyutu | 256 × 256 |
| Yama omurgası `f_t` | ResNet-18 (ImageNet ön-eğitimli) |
| Özellik boyutu `L` | 128 |
| Dikkat | Gated Attention (Ilse ve ark. 2018) |
| `f_agg` | top-`t%` ortalama (GAP–GMP arası yumuşak denge) |
| Regularization | `L_reg = Σ\|A\|^β`, ağırlık `λ` |
| Kayıp | `BCE(y, ŷ_loc) + BCE(y, ŷ_mil) + λ·L_reg` |
| Çıkarım | `ŷ = (ŷ_loc + ŷ_mil)/2` |
| Optimizer | Adam |
| Epoch | 40 |
| Arama aralıkları | `η ∈ 10^[-5.5,-3.8]`, `λ ∈ 10^[-5,-2.8]`, `β ∈ e^[-1.6,1.6]`, `t ∈ e^[-5,-1.5]` |
| Örnekleme | her iterasyonda pozitif tetkikler + eşit sayıda rastgele negatif |
| Meme-seviyesi tahmin | CC ve MLO tahminlerinin ortalaması |

## Referanslar (orijinal numaralandırma)

1. Bergstra, Bengio — Random search for hyper-parameter optimization. JMLR 13 (2012)
2. Deng ve ark. — ImageNet. CVPR 2009
3. Diba ve ark. — Weakly supervised cascaded convolutional networks. CVPR 2017
4. Durand ve ark. — WILDCAT. CVPR 2017
5. Gao, Geras, Lewin, Moy — New frontiers: CAD for breast imaging in the age of AI. AJR 212(2) (2019)
6. Ilse, Tomczak, Welling — Attention-based deep multiple instance learning. arXiv:1802.04712
7. Kingma, Ba — Adam. ICLR 2015
8. Kopans — Organized mammographic screening reduces mortality. Cancer 94(2) (2002)
9. Ren ve ark. — Faster R-CNN. NIPS 2015
10. Ribli ve ark. — Detecting and classifying lesions in mammograms with deep learning. Sci. Rep. 8(1) (2018)
11. Simonyan, Zisserman — VGGNet. arXiv:1409.1556
12. Wang ve ark. — Densely deep supervised networks with threshold loss. MICCAI 2018
13. Wu ve ark. — Breast density classification with deep CNNs. ICASSP 2018
14. Wu ve ark. — Deep neural networks improve radiologists' performance in breast cancer screening. arXiv:1903.08297 (2019) — **baseline ve ön-işleme kaynağı**
15. Yao ve ark. — Weakly supervised medical diagnosis and localization from multiple resolutions. arXiv:1803.07703
16. Zhu ve ark. — Deep multi-instance networks with sparse label assignment. MICCAI 2017
