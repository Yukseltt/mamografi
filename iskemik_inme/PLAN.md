# ISLES'22 — Gerçek Zamanlı İskemik İnme Lezyon Segmentasyonu — Proje Planı

**Veri seti:** ISLES 2022 — *A multi-center magnetic resonance imaging stroke lesion
segmentation dataset* (Hernández Petzsche ve ark., Scientific Data 2022)
Zenodo: https://zenodo.org/records/7960856 · Lisans: CC-BY-4.0 · Challenge: https://www.isles-challenge.org/
**Tek veri seti** — ek set kullanılmıyor (kullanıcı kararı, bkz. Bölüm 2 ve Bölüm 11).
**Amaç:** DWI/ADC üzerinden akut–subakut infarkt lezyonunu segmente eden, uçtan uca gecikmesi
ölçülmüş ve optimize edilmiş bir model; çıktısı lezyon hacmi + overlay + okunabilir rapor.
**Ortam:** Kod yerelde (VSCode), yürütme Colab kernel'ine bağlanarak. Loglar/checkpointler Drive'da.

> Bu dosya `mendeley_data_density/PLAN.md` ve `real_time_segmentasyon/PLAN.md` ile aynı
> düzende tutulur: ölçülen her sayı **ölçüldü** işaretiyle plana geri yazılır, varsayımlar
> varsayım olarak kalır.

---

## 1. Klinik Arka Plan (kısa)

İskemik inme, bir serebral arterin tıkanmasıyla beslediği dokunun kan akımını kaybetmesidir.
Doku iki bölgeye ayrılır:

- **Core (çekirdek)** — akım kritik eşiğin altına düşmüş, geri dönüşsüz hasarlı doku.
- **Penumbra** — kollateral dolaşımla ayakta kalan, reperfüzyonla kurtarılabilir doku.

Görüntülemede karşılıkları:

| Bulgu | Anlamı |
|---|---|
| DWI hiperintens + ADC düşük | sitotoksik ödem → **akut infarkt (core)** |
| FLAIR henüz negatif, DWI pozitif | DWI–FLAIR mismatch → lezyon büyük olasılıkla **< 4,5 saatlik** |
| FLAIR de hiperintens | subakut evre |
| Perfüzyon (CTP/PWI) — Tmax > 6 s ile core farkı | **penumbra** |

**Kritik nokta:** penumbra ancak perfüzyon görüntülemesiyle ölçülür. ISLES'22'de perfüzyon
serisi **yok**, dolayısıyla veri setinde core/penumbra ayrımı da yok — tek bir **binary lezyon
maskesi** var (DWI temelli akut/subakut infarkt). Talepteki "core/penumbra veya lezyon maskesi"
ifadesinin bu veri setindeki karşılığı **lezyon maskesi**. Core/penumbra ayrımı ve NCCT için ek
veri setleri araştırıldı ancak **kapsam dışı bırakıldı** (kullanıcı kararı, bkz. Bölüm 2). Bu,
projenin bilinen ve raporda açıkça yazılacak bir sınırlılığıdır: **penumbra tahmini üretilmiyor**,
çalışma **yalnız MR (DWI/ADC)** üzerinde.

---

## 2. Veri Seti — Bilinenler ve Ölçülecekler

### Kaynaktan bilinenler

| Ölçüm | Değer |
|---|---|
| Yayınlanan vaka | **250** (challenge'ın eğitim kümesi) |
| Yayınlanmayan test | 150 vaka — halka açık değil, sunucuda tutuluyor |
| Seriler | **DWI (b=1000)**, **ADC**, **FLAIR** — vaka başına üçü de |
| Etiket | Uzman çizimli **binary** lezyon maskesi (NIfTI) |
| Format | NIfTI (`.nii.gz`), **BIDS** dizin düzeni, native uzay, **kayıt (registration) yapılmamış** |
| Boyut | ~1,7 GB (`ISLES-2022.zip`) + `center_ids.xlsx` |
| Merkez | Çok merkezli, çok üreticili (merkez kimlikleri `center_ids.xlsx`'te) |
| Lezyon dağılımı | Boyut, sayı ve konumda **yüksek değişkenlik**; ağırlıklı MCA sulama alanı, ~1/4 infratentoryal |

Beklenen dizin düzeni (Parça 1'de doğrulanacak):

```
ISLES-2022/
├── rawdata/sub-strokecase0001/ses-0001/
│   ├── sub-strokecase0001_ses-0001_dwi.nii.gz   (+ .json)
│   ├── sub-strokecase0001_ses-0001_adc.nii.gz
│   └── sub-strokecase0001_ses-0001_flair.nii.gz
├── derivatives/sub-strokecase0001/ses-0001/
│   └── sub-strokecase0001_ses-0001_msk.nii.gz
└── center_ids.xlsx, participants.tsv, dataset_description.json
```

### Parça 1 ölçümleri — **ölçüldü** (250 vakanın tamamı, `ozet/kesif_ozeti.xlsx`)

Gerçek dizin düzeni (beklenenden farklı — `rawdata/` yok):

```
ISLES-2022/
├── sub-strokecase0001/ses-0001/dwi/   sub-..._dwi.nii.gz  + _adc.nii.gz (+ .json)
├── sub-strokecase0001/ses-0001/anat/  sub-..._FLAIR.nii.gz (+ .json)
└── derivatives/sub-strokecase0001/ses-0001/  sub-..._msk.nii.gz + _snp.png
```

**250/250 vakada dört serinin dördü de tam** — eksik seri yok.

| # | Ölçüm | Sonuç | Plana etkisi |
|---|---|---|---|
| 1 | DWI/ADC matris & spacing | **17 farklı şekil, 7 farklı spacing.** 193 vaka 2×2×2 mm izotropik; 29 vaka 1,80×1,80×**4,8**; 23 vaka 1,15×1,15×**4,8**. Voxel hacmi 1,53–20,0 mm³ | **İzotropik resample zorunlu** (aşağıda) |
| 2 | DWI ≡ ADC grid | **250/250 aynı şekil ve aynı affine.** Maske de 250/250 DWI grid'inde | 2 kanal birleştirme sorunsuz ✔ |
| 3 | FLAIR uzayı | **0/250 uyuşuyor.** 22 farklı şekil, 21 farklı spacing; voxel hacmi 0,055–4,43 mm³ (kimi 0,71 mm izotropik 3B, kimi 5–6 mm kalın dilim) | FLAIR için **vaka başına kayıt zorunlu** → A1'in maliyeti gerçek |
| 4 | Lezyon hacmi | medyan **6,66 mL**, IQR **[1,58 – 21,17]**, min 0 / max **482,2**. **46 vaka < 1 mL**, 104 vaka < 5 mL, 30 vaka > 50 mL | Dağılım aşırı çarpık → **hacim tertili kırılımı zorunlu** ✔ |
| 5 | Bağlı lezyon sayısı | medyan **5**, **%79,6 çok odaklı**. Toplam **897 adet 5-voxel altı bileşen** | Lezyon-bazlı F1 **birincil metriklerden biri** oldu |
| 6 | Sınıf dengesizliği | lezyon/beyin voxel oranı: ortalama **%1,78**, **medyan %0,53**. Boş dilim oranı ortalama **%64,9**. Dilim ekseni 250/250 **z (aksiyel)** | A2 ablasyonu gerekçelendi + boş dilim atlama optimizasyonu büyük kazanç |
| 7 | Merkez | **yalnız 2 merkez: 198 / 52.** Medyan hacim merkez 1'de 5,37 mL, merkez 2'de 11,43 mL | Stratifikasyon gerekli **ama** merkez kırılımı zayıf kalacak |
| 8 | Yoğunluk | DWI p99 = 390 ± 133 (makul). **ADC p99 = 1163 ± 1704** — dağılım üç ayrı ölçek grubuna işaret ediyor. Beyin oranı hacmin **%18,8**'i | **ADC birim tutarsızlığı** — aşağıda ayrı başlık |
| 9 | Bütünlük | 3 vaka **boş maske**: `sub-strokecase0150`, `0151`, `0170`. NaN yok, şekil uyumsuzluğu yok | `ozet/haric_vakalar.json` yazıldı |

### Ölçümlerin doğurduğu dört karar

**1. İzotropik resample — 2×2×2 mm.** Spacing 7 farklı değerde ve bir kısmı 4,8 mm dilim
kalınlığında. Ortak grid olmadan ne hacim (mL) karşılaştırılabilir ne de 2.5D komşu dilim bağlamı
anlamlı olur (bir vakada 5 dilim = 10 mm, başkasında 24 mm). Hedef **2×2×2 mm**, çünkü zaten
193/250 vaka orada — çoğunluk hiç dokunulmadan geçiyor. Görüntü için trilinear, maske için
nearest interpolasyon. Metrikler **resample edilmiş grid üzerinde** hesaplanır; hacim (mL) ise
her zaman **orijinal spacing'den** çıkarılır (klinik büyüklük yeniden örneklemeden etkilenmemeli).

**2. ADC birim tutarsızlığı — Parça 2'nin ilk işi.** ADC max değerlerinin persentilleri:
%25 → 0,004 · %50 → 4,45 · %75 → 4095 · max → 4683. Bu tek bir birimde mümkün değil; veri
en az üç ölçekte geliyor (mm²/s ≈ 0,003 · ×10⁻³ ≈ 3 · ×10⁻⁶ ≈ 3000). Planın "ADC'ye sabit klip"
maddesi **bu haliyle geçersiz**. Yeni sıra: her vakada beyin içi ADC p99'una bakıp ölçek grubunu
belirle → ortak birime (×10⁻⁶ mm²/s) çevir → **sonra** sabit klip (0–3200) ve ölçekleme.
Dönüşüm sonrası p99 dağılımı Parça 2'de tekrar yazdırılıp tek moda düştüğü doğrulanacak.

**3. Boş maskeli 3 vaka atılmıyor, ayrıştırılıyor.** ISLES'22'de boş maske "bu hastada lezyon
yok" demek; atmak modeli yalnız lezyonlu görüntü görmüş hale getirir ve yanlış pozitifi artırır.
Karar: **3 vaka da eğitim setinde kalır** (negatif örnek), **val ve test'e alınmaz** — orada
Dice'ı tanımsız yapıp 50 vakalık test setinin ortalamasını kirletirler. `split.json` bunu
zorunlu kılacak.

**4. Küçük bileşenler — post-processing kararı şimdi verilemez.** GT'de 897 adet 5-voxel altı
bağlı bileşen var. Tahminde küçük bileşen elemek yanlış pozitifi düşürür ama **GT'nin kendisinde
bu kadar küçük bileşen olduğu için** recall'u ve lezyon-bazlı F1'i de düşürebilir. Eşik
(0 / 5 / 10 voxel) val setinde taranıp seçilecek, varsayılmayacak.

### Parça 2 sonuçları — **ölçüldü** (`ozet/hazirlik_ozeti.xlsx`)

**ADC birim tutarsızlığı doğrulandı ve çözüldü.** Veri gerçekten üç ayrı birimde geliyor:

| Birim | Vaka | Çarpan |
|---|---|---|
| mm²/s | **106** | 1e6 |
| ×10⁻⁶ mm²/s | **76** | 1 |
| ×10⁻³ mm²/s | **68** | 1e3 |

Dönüşüm öncesi beyin içi ADC p99,9 dağılımı 0,003 – 4574 (**CV ≈ 1,52**), dönüşüm sonrası
3187 – 4574, **CV = 0,088**, aykırı vaka yok — tek moda düştü. Düzeltme yapılmasaydı model
2. kanalda üç ölçeği aynı anda görecek ve ADC bilgisi kullanılamaz hale gelecekti.

**Girdi boyutu — 96 × 128 (kare değil).** Kırpma sonrası H max 87, W max 101; eksen başına
32'ye yuvarlanınca 96×128 çıkıyor ve hiçbir vakayı kırpmıyor. Kare 128×128'e göre **%33 daha az
piksel** — gecikme bütçesinden bedava kazanç. Gerçek içerik ortalaması 77×88, yani dolgu oranı
%44 (kare olsaydı %59). Dilim sayısı: medyan 78, aralık 61–90.

**Split — 161 / 39 / 50**, vaka bazlı, merkez × hacim tertili stratifiye, ikili kesişimler boş:

| | merkez 1 | merkez 2 | küçük | orta | büyük | medyan mL |
|---|---|---|---|---|---|---|
| train | 128 | 33 | 54 | 54 | 53 | 6,18 |
| val | 31 | 8 | 13 | 12 | 14 | 11,19 |
| test | 39 | 11 | 16 | 17 | 17 | 8,78 |

Boş maskeli 3 vaka üçü de train'de ✔. Sabit değerlendirme vakaları (`eval_cases.json`) hacim
aralığına yayıldı: 0,21 / 2,90 / 8,76 / 18,55 / 93,07 mL — biri merkez 2'den.

**Doğrulama:** DWI beyin içi ortalama −0,003, std 0,978 (±5 klip nedeniyle 1'in hemen altı);
ADC [0, 1]; maske {0, 1}; NaN yok. Önbellek **0,80 GB**, hazırlama 242 sn.

> **Resample'ın hacim üzerindeki etkisi.** Orijinal ↔ resample hacim farkı ortalama %0,01,
> std %2,35, uçlar −12,5% / +11,5% (yalnız birkaç voxel'lik minik lezyonlarda). Metrikler
> resample grid'inde hesaplandığı için Dice etkilenmez — GT de tahmin de aynı grid'de. Ama
> **AVD ve klinik hacim çıktısı bu ±%2,3'lük tabanı taşır**; raporda hacim sayılarının yanına
> yazılacak. İstenirse tahmin native grid'e geri taşınabilir, maliyeti Parça 8'de ölçülür.

### Merkez kırılımı hakkında uyarı

Merkez sayısı ikiden ibaret ve dağılım 198/52. %20 test ayrımında merkez 2'den yalnızca ~10 vaka
düşer. Split merkez × hacim tertiline göre stratifiye edilecek, **ama** "merkezler arası
performans farkı" üzerine kurulacak her cümle 10 vakalık bir alt gruba dayanacağı için
raporlanırken bu sayı açıkça yazılacak. Merkez 2'nin medyan lezyon hacmi merkez 1'in iki katı —
yani merkezler gerçekten farklı, stratifikasyon şart.

### Ek veri setleri — araştırıldı ve kapsam dışı bırakıldı (2026-08-19)

Penumbra ve NCCT eksiğini kapatabilecek halka açık setler tarandı. **Karar: hiçbiri
eklenmiyor, proje tek veri setiyle yürüyor** (kullanıcı kararı). Tablo, kararın gerekçesiyle
birlikte kayıt altında kalsın diye duruyor — ileride kapsam genişlerse aday listesi hazır.

| Set | İçerik | Vaka | Erişim |
|---|---|---|---|
| CPAISD | NCCT, dilim bazında core + penumbra ayrı maske | 112 | Zenodo, CC-BY-4.0, kayıtsız |
| AISD | NCCT, DWI referanslı lezyon konturu (5 mm dilim) | 397 | GitHub, akademik kullanım |
| APIS | Eşlenmiş NCCT + ADC | 60 + 36 | Kayıt gerekiyor (bivl2ab.uis.edu.co) |
| ISLES 2024 | NCCT + CTA + CTP (CBF/CBV/MTT/Tmax), final infarkt | 150 (250 toplam) | Grand-Challenge kaydı |
| ISLES 2018 | CTP + türetilmiş haritalar, infarkt core | 63 + 40 | isles-challenge.org, kayıt |

**Bu kararın doğrudan sonucu — raporda sınırlılık olarak yazılacak:**

- **Penumbra tahmini yok.** Çıktı tek binary infarkt maskesi; core/penumbra mismatch oranı
  üretilemiyor. Bu klinikte reperfüzyon kararının dayandığı büyüklük.
- **BT (NCCT) kapsam dışı.** Çalışma yalnız MR (DWI/ADC) üzerinde geçerli. Acil serviste ilk
  görüntüleme çoğu zaman NCCT olduğu için bu, klinik uygulanabilirliği daraltan bir sınır.
- **Dış doğrulama yok.** Tek veri seti kullanıldığı için sonuçlar ISLES'22 dağılımına özel;
  başka merkez/cihazda genelleme kanıtlanmıyor. Merkez bazlı kırılım (Bölüm 5) bunun yerini
  kısmen tutar ama dış doğrulama değildir.

Kazancı: kapsam daralınca bütün bütçe mimari karşılaştırması, optimizasyon merdiveni ve servise
gidiyor — talebin asıl ağırlık merkezi olan "gerçek zamanlı" tarafı derinleşiyor.

### Split kararı

Test kümesi yayınlanmadığı için **250 vaka içinde vaka bazlı ayrım** yapılacak (kullanıcı talebi):

- **%20 test = 50 vaka**, kalan 200 vaka train/val (**%80/%20 → 160/40**).
- Bölme birimi **vaka (`sub-strokecase*`)** — aynı vakanın dilimleri asla iki kümeye düşmez.
  2D dilim tabanlı eğitimde en kolay yapılan sızıntı hatası budur.
- Stratifikasyon: **merkez** × **lezyon hacmi tertili**. Test setinde küçük lezyonlu vakaların
  temsil edilmemesi tek başına Dice'ı 0,05 oynatabilir.
- `split.json` bir kez üretilir, sabit seed, tüm deneylerde aynı dosya okunur.
- 50 vakalık test seti küçük: modeller arası farklar **eşleştirilmiş** testle (vaka başına Dice
  üzerinden Wilcoxon) değerlendirilecek, çıplak ortalama farkına bakılmayacak.

---

## 3. "Gerçek Zamanlı" Ne Demek — Literatür ve Ölçüm Protokolü

Burası BUSI projesinden ayrılan yer: orada girdi bir video karesiydi, "30 FPS" doğal hedefti.
Burada girdi bir **3B hacim**; klinik akışta saniyede 30 hacim diye bir şey yok. Hedef literatüre
bakılarak belirlendi.

### Literatürdeki gerçek sayılar (araştırıldı)

| Çalışma | Ne | Gecikme | Donanım |
|---|---|---|---|
| **DeepISLES** (Nat Commun 2025) — ISLES'22 SOTA ensemble, Dice **0,82**, lezyon-F1 **0,86** | nnU-Net ensemble, Docker | **~2 dk/vaka** | RTX 3090 24 GB |
| aynı | GTX 1080 Ti | ~5 dk/vaka | 12 GB |
| aynı | web servisi | ~10 dk/vaka | — |
| **StrokeSeg** (arXiv 2510.24378) — nnU-Net, ONNX + FP16, Dice farkı < 10⁻³ | modüler dağıtım aracı | **0,9 s/hacim** (GPU) vs **149,2 s** (CPU) → 165× | GPU |
| **2.5D Transformer U-Net** (2025) — inme MR | 2.5D dilim tabanlı | **~24 ms/dilim** | GPU |
| **Jetson nnU-Net** (Sensors 2026) | TensorRT FP16, uç cihaz | — | Dice **0,78 → 0,67** düştü |

Üç sonuç çıkıyor:

1. **SOTA gerçek zamanlı değil.** ISLES'22'nin en iyi modeli klinik olarak doğrulanmış ama vaka
   başına 2 dakika sürüyor. Bu projenin katkısı burada: *aynı işi 2 mertebe daha hızlı yapmak ve
   bunun Dice bedelini ölçmek.*
2. **1 saniye altı ulaşılabilir bir hedef.** StrokeSeg ONNX+FP16 ile 0,9 s/hacim ölçmüş. Yani
   "≤ 1 s" uydurma bir hedef değil, yayınlanmış bir referans noktası.
3. **Optimizasyonun bedeli gerçek.** Jetson çalışmasında TensorRT FP16 + uç donanım Dice'ı
   0,78'den 0,67'ye düşürmüş. Bizim tablomuzda her optimizasyon adımının ΔDice'ı zorunlu kolon.

### Hedef (bu proje)

Ölçüm donanımı **yerel RTX 2060 6 GB** (kullanıcı kararı). RTX 3090'ın yaklaşık üçte biri güçte,
dolayısıyla hedef 3090 sayılarına göre değil bu karta göre konuyor:

| Hedef | Ölçüt | Eşik |
|---|---|---|
| **Birincil** | Vaka başına uçtan uca gecikme, RTX 2060 | **≤ 2 s**, hedeflenen ≤ 1 s |
| İkincil | Dilim başına ağ süresi (akış senaryosu) | **≥ 30 FPS** (≤ 33 ms) |
| Karşılaştırma | Aynı vakada DeepISLES referansı | ~120 s (literatürden, yeniden koşulmayacak) |
| **Bütçe dışı** | FLAIR hizalaması (**ölçüldü**, Bölüm 5k) | **670 ms/vaka** — en hızlı hattın 9 katı |
| Kalite tabanı | Test Dice | En hızlı konfigürasyon, taban çizgisinin **0,03**'ünden fazla gerisine düşmemeli |

**6 GB VRAM kısıtı** ayrıca model tarafını bağlıyor: 3D U-Net referansı küçük patch'le
(≈128×128×32) koşacak, eğitim Colab T4'te (15,6 GB) yapılıp ölçüm 2060'a taşınacak. Eğitim ve
ölçüm donanımının farklı olması sorun değil — **ölçüm tek ortamda, tek torch sürümünde** yapıldığı
sürece karşılaştırma geçerli.

### Ölçüm protokolü

`real_time_segmentasyon/hiz_olcumu.py` ile aynı disiplinde:

- Ölçülen şey **uçtan uca**: ön işleme (resample + normalize + pad) + ileri geçiş + son işleme
  (eşikleme + orijinal grid'e geri taşıma + bağlantılı bileşen). Diskten `.nii.gz` okuma
  **dışarıda** — servis senaryosunda hacim PACS'tan gelir, dosya okuma modele ait maliyet değil.
  Ancak ayrı bir kolonda okuma+decompress süresi de raporlanır (klinik gerçeklik için).
- Isınma koşuları atılır, batch=1, her vaka tek tek senkronlanarak ölçülür.
- **Saf ağ** ve **çerçeve yükü** ayrı ayrı verilir. BUSI'de RTMDet'in gecikmesinin yarısı
  çerçeve yüküydü; aynı tuzağa düşmemek için ayrıştırma baştan yapılır.
- Hızı ölçülen kodun Dice'ı ölçülen kodla **aynı** olduğu assert ile doğrulanır.
- Ortam: **yerel RTX 2060 6 GB** (birincil) + aynı makinenin CPU'su (dağıtım alt sınırı).
  Farklı torch sürümlerinde ölçülen sayılar karşılaştırılmaz — tüm modeller tek ortamda ölçülür.
  Colab T4 sayıları yalnızca referans kolonu olarak eklenir, ana tabloyu oluşturmaz.

---

## 4. Modeller (5 mimari + 3 ablasyon → 9 koşu)

Denge kararı (kullanıcı): mimari/ablasyon tarafı ile optimizasyon tarafı ikisi de yürüyecek,
model tarafı **hafif ağırlıklı**. Buna göre 5 mimari + 3 ablasyon, optimizasyon tarafında 8 adımlı
merdiven (Bölüm 6).

Girdi kararı: **DWI + ADC, 2 kanal**. Gerekçe — ikisi aynı grid'de (kayıt maliyeti sıfır),
akut infarktın tanımlayıcı bulgusu ikisinin birlikteliği (DWI↑ + ADC↓), FLAIR ayrı uzayda ve
kayıt adımı gerçek zamanlı bütçeye doğrudan yazılıyor. FLAIR'in katkısı **ablasyon A1 ile**
ölçülecek, baştan varsayılmayacak.

### Mimariler

| # | `MODEL_KEY` | Mimari | Girdi | Neden bu model |
|---|---|---|---|---|
| 1 | `unet_r34_2d` | U-Net + ResNet34 (smp, ImageNet ön-eğitimli) | 2 kanal, tek dilim | **Taban çizgisi.** Density projesinde 6 model arasında birinci çıktı; kalibre edilmiş, davranışı bilinen referans noktası. Diğer her şey buna karşı okunur. |
| 2 | `unet_r34_25d` | U-Net + ResNet34 | **2k+1 = 5 dilim × 2 kanal = 10 kanal** | Dilim dışı bağlam. Gecikme neredeyse değişmez (yalnız ilk conv genişler), ama infarkt 3B bir yapı — komşu dilim bilgisi ucuz kazanç. ImageNet ağırlıkları ilk katmanda kanal ortalamasıyla çoğaltılır. |
| 3 | `segformer_b0` | SegFormer-B0 (MiT-B0, `transformers`) | 2.5D, 10 kanal | Transformer tabanlı tek aday. BUSI projesinde hem en hızlı hem kalitede geride değildi; ONNX'e temiz çıkıyor. |
| 4 | `unet_mbv3` | U-Net + MobileNetV3-Large | 2.5D, 10 kanal | **Hafif uç.** Gecikme–Dice ödünleşiminin alt sınırı. INT8/pruning'e en uygun aday; 6 GB'lık kartta ve CPU'da asıl dağıtılabilir olan bu. |
| 5 | `unet3d` | 3D U-Net (taban 16), **tam hacim** 96×128×96 | 3B hacim, 2 kanal | Başlangıçta "doğruluk tavanı, gerçek zamanlı değil" diye konumlandırıldı; **ölçüm bunu çürüttü** (Bölüm 5f): 1,40 M parametreyle hem en küçük hem en hızlı hem nominal olarak en doğru. Patch gerekmedi — hacimler 96×128×≤96'ya sığıyor, vaka başına tek ileri geçiş. |

nnU-Net doğrudan kullanılmıyor: kendi ön işleme/çıkarım hattını dayatıyor ve gerçek zamanlı
bütçeyle bağdaşmıyor (varsayılanı TTA + 5-fold ensemble — DeepISLES'in 2 dk/vakasının sebebi tam
olarak bu). Onun yerine `unet3d_patch` aynı işlevi görüyor; DeepISLES'in yayınlanmış sayıları
(Dice 0,82 / lezyon-F1 0,86) karşılaştırma tablosuna **literatür satırı** olarak eklenir.

### Ablasyonlar

| # | Ablasyon | Ne ölçülüyor | Koşu |
|---|---|---|---|
| **A1** | En iyi 2.5D modelin **+FLAIR (3 modalite)** varyantı | FLAIR'in Dice katkısı **ve** kayıt adımının ms maliyeti. Katkı gecikme artışını haklı çıkarmıyorsa FLAIR kalıcı olarak dışarıda kalır. | 1 |
| **A2** | Loss: Dice+BCE → **focal-Tversky** (+ lezyonlu dilim ağırlıklı örnekleme) | Küçük lezyonlarda recall kazancı. Parça 1'de lezyon voxel oranı %1'in altı çıkarsa bu ablasyon zorunlu hale gelir. | 1 |
| **A3** | 2.5D bağlam derinliği: **k = 1 / 2 / 3** (3 / 5 / 7 dilim) | Bağlamın nerede doyduğu ve her adımın gecikme maliyeti. k=2 varsayılan; bu koşu varsayılanı doğrular ya da düzeltir. | 2 ek (k=1, k=3) |

Toplam: **5 + 4 = 9 eğitim koşusu** (A3 iki ek koşu getiriyor). Hepsi aynı `split.json`, aynı
seed, aynı epoch bütçesi — fark yalnız değişkenden gelsin diye.

### Hiperparametreler (Parça 4'te kodlandı — `MODEL_CONFIGS`)

Density projesindeki kalibre edilmiş değerlerden türetildi. Tüm koşularda ortak: AdamW
(weight decay 1e-4), 3 epoch lineer warmup → `ReduceLROnPlateau` (faktör 0,5 / patience 3 /
min_lr 1e-7), AMP, gradient clipping 1,0, `max_epochs=70`, `patience=15`, seed 42.

| `MODEL_KEY` | mimari | k | kanal | batch | LR enc/dec | loss |
|---|---|---|---|---|---|---|
| `unet_r34_2d` | U-Net + ResNet34 | 0 | 2 | 32 | 5e-5 / 2,5e-4 | Dice+BCE |
| `unet_r34_25d` | U-Net + ResNet34 | 2 | 10 | 32 | 5e-5 / 2,5e-4 | Dice+BCE |
| `segformer_b0` | SegFormer (MiT-B0) | 2 | 10 | 32 | 1e-4 (tek) | Dice+BCE |
| `unet_mbv3` | U-Net + MobileNetV3-L | 2 | 10 | 32 | 1e-4 / 5e-4 | Dice+BCE |
| `unet3d` | 3D U-Net (taban 16), tam hacim | — | 2 | 2 | 3e-4 | Dice+BCE, `tekrar=3` |
| `A1_flair` | = `unet_r34_25d` + FLAIR | 2 | **15** | 32 | 5e-5 / 2,5e-4 | Dice+BCE |
| `A2_tversky` | = `unet_r34_25d` | 2 | 10 | 32 | 5e-5 / 2,5e-4 | **BCE + focal-Tversky** (α0,3 β0,7 **γ4/3**), örnekleme taban çizgisiyle aynı |
| `A3_k1` / `A3_k3` | = `unet_r34_25d`, k=1 / k=3 | 1 / 3 | 6 / 14 | 32 | 5e-5 / 2,5e-4 | Dice+BCE |

**Kanal düzeni modalite-öncülü**: `[DWI × (2k+1)] + [ADC × (2k+1)]`. Yoğunluk jitter yalnız
DWI kanallarına uygulanıyor — ADC fizik birimli, ölçeği bozulmamalı (Bölüm 2, karar 2).

**A1 kanal sayısı 15, 11 değil.** İlk yazımda FLAIR tek kanal olarak eklenmişti (10+1);
2.5D ablasyonuyla karşılaştırılabilir olması için FLAIR de aynı 2k+1 dilimlik bağlamı almalı:
3 modalite × 5 dilim = **15 kanal**.

**3B epoch tanımı — `tekrar=3`.** 2B koşularda epoch başına ~237 optimizer adımı var
(7602 dilim / batch 32). 3B'de 161 vaka / batch 2 = 81 adım kalırdı; her vaka epoch başına
3 kez (farklı augmentasyonla) örneklenerek 241 adıma çıkarıldı. Aksi halde "3B daha kötü"
sonucunun mimariden mi yoksa üçte bir gradyan adımından mı geldiği ayrılamazdı.

**Örnekleme**: her epoch lezyonlu dilimlerin tamamı + `lezyon_orani`'na göre rastgele boş dilim.
Boş dilimler atılmıyor (Parça 1: beyin içeren dilimlerin %64,9'unda lezyon yok).

**Val protokolü**: dilim başına metrik hesaplanmıyor. Her vakanın bütün dilimleri geçirilip
hacim yeniden kuruluyor, metrikler 3B hacim üzerinde. `val_loss` aynı ileri geçişten, **train
loss ile aynı tanımla** (Dice+BCE) hesaplanıyor — böylece eğri panelindeki train/val makası
okunabilir ve `ReduceLROnPlateau` ile early stopping birbirinden bağımsız iki sinyal izliyor
(scheduler val loss, early stopping val Dice).

**`epoch_sure_sn`** checkpoint yazımını da kapsıyor: smoke testte train+val 3 sn sürerken
Drive'a checkpoint yazımı ~6 sn ekliyordu (24,4 M parametre + AdamW durumu ≈ 290 MB/epoch).
Bu maliyet dışarıda bırakılırsa epoch süresi yarıya yakın eksik görünür.

### Ön işleme (ortak hat, tüm modellerde aynı)

1. DWI ve ADC yükle, aynı grid doğrulaması (**ölçüldü: 250/250 aynı affine**).
2. **İzotropik resample 2×2×2 mm** — görüntü trilinear, maske nearest. 193/250 vaka zaten orada.
3. Beyin maskesi (eşik + morfoloji; yetmezse HD-BET) → arka plan kırpma. Beyin hacmin
   **%18,8**'i olduğu için kırpma büyük kazanç.
4. Yoğunluk normalizasyonu: DWI **maske içi z-score**; ADC önce **ölçek grubu tespiti + ortak
   birime çevirme**, sonra sabit klip (0–3200) + ölçekleme (bkz. Bölüm 2, karar 2).
5. Aksiyel dilimlere ayır (**ölçüldü: dilim ekseni 250/250 z**), **96 × 128**'e dolgu
   (**ölçüldü**, bkz. Bölüm 2 Parça 2 sonuçları). Dolgu Dataset içinde yapılır, önbellekte
   gereksiz sıfır tutulmaz.
6. `.npy` önbellek — Drive'dan tekrar tekrar `.nii.gz` açmamak için (density projesindeki
   cache deseni: ilk epoch yavaş, sonrası hızlı).

### Augmentasyon

Flip (yatay — beyin simetrik, ama lezyon lateralitesi ezberlenmesin diye dikkatli), ±10° rotasyon,
hafif ölçek, yoğunluk jitter, hafif elastic. Agresif crop yok — lezyon zaten küçük.

### Loss ve dengesizlik

Dice + BCE taban çizgisi. **Ölçüldü:** lezyon/beyin voxel oranı ortalama %1,78 ama **medyan
%0,53** — yani vakaların yarısında lezyon beyin hacminin binde beşinden küçük. Bu, A2
ablasyonunu (**focal-Tversky + lezyonlu dilim ağırlıklı örnekleme**) gerekçelendiriyor;
"gerekirse" değil, planlı koşu.

Boş dilimlerin tamamı atılmaz — **ölçüldü: beyin içeren dilimlerin %64,9'unda lezyon yok.**
Atılırsa model boş dilim hiç görmez, çıkarımda yanlış pozitif üretir ve hacim ölçümü bozulur.
Örnekleme oranı (lezyonlu : boş) A2'de taranacak.

---

## 5. Metrikler

**Birincil metrik: vaka (hacim) seviyesinde 3B Dice.** Dilim başına Dice ortalaması yanıltıcıdır —
lezyonu 2 dilimde olan vaka ile 40 dilimde olan vaka eşit ağırlık almaz.

| Metrik | Neden |
|---|---|
| **Dice (3B, vaka başına)** | Birincil |
| IoU, Precision, Recall | Hata biçimini ayırmak için (aşırı/eksik segmentasyon) |
| **HD95** | Sınır doğruluğu; Dice'ın görmediği uzak yanlış pozitifleri yakalar |
| **AVD** (mutlak hacim farkı, mL) | Klinik karar hacme bakar, örtüşmeye değil |
| **Lezyon-bazlı F1** + **lezyon sayısı farkı** | ISLES'22 resmi metrikleri. **Ölçüldü: vakaların %79,6'sı çok odaklı, medyan 5 bileşen** → bu artık ikincil değil, **Dice ile birlikte birincil** (`panoptica`) |
| Gecikme (ms/vaka, ms/dilim), FPS, parametre, FLOPs, VRAM | Bölüm 3 protokolü |

Kurallar:

- Tüm örtüşme metrikleri **manuel TP/FP/FN** üzerinden, vaka başına hesaplanıp vakalar arasında
  ortalanır. Binary'de `f1 = dice` özdeşliği korunur (density projesinde `torchmetrics`'in
  micro hesabı bunu bozmuştu, aynı hataya düşülmeyecek).
- **Boş GT maskesi**: **ölçüldü — 3 vaka** (`0150`, `0151`, `0170`). Üçü de **eğitim setinde**
  kalır (negatif örnek), val/test'e alınmaz (Bölüm 2, karar 3). Kural yine de kodda durur:
  `GT boş & tahmin boş → dice = 1`.

- **Küçük bileşen eşiği** (0 / 5 / 10 voxel) val'de taranıp seçilir. GT'de 897 adet 5-voxel altı
  bileşen olduğu için elemek recall'u da düşürebilir — varsayılmayacak, ölçülecek.
- Eşik (threshold) **val üzerinde bir kez** seçilir, test'te sabit kullanılır.
- Sonuç tablosunda **hacim tertiline göre kırılım** zorunlu — küçük lezyonlarda Dice'ın düşmesi
  bilinen bir olgu, tek ortalama bunu gizler.

---

## 5b. Parça 5 sonuçları — `unet_r34_2d` taban çizgisi **TAMAM**

52 epoch, **29,2 dakika** (Colab T4), zirve epoch 36, epoch 51'de early stopping.
24,43 M parametre · 7602 eğitim dilimi (3801 lezyonlu) · 39 val vakası.

| | val (39 vaka) | **test (50 vaka)** |
|---|---|---|
| Dice | 0,6777 | **0,6965 ± 0,1912** (SEM 0,0270) |
| IoU | 0,5418 | 0,5616 |
| Precision / Recall | — | 0,7540 / 0,6844 |
| HD95 | 17,95 mm | 16,06 ± 22,59 mm |
| AVD | 7,17 mL | **2,84 ± 4,32 mL** |
| Lezyon F1 | 0,498 | **0,5717 ± 0,2242** |
| Lezyon sayı farkı | — | 3,64 |

### Hacim tertili — planın öngördüğü etki doğrulandı

| Tertil | Dice | IoU | Recall | AVD (mL) | Lezyon F1 |
|---|---|---|---|---|---|
| küçük | **0,5638** | 0,4141 | 0,6094 | 0,48 | 0,5949 |
| orta | 0,6953 | 0,5578 | 0,6503 | 1,46 | 0,6048 |
| büyük | **0,8226** | 0,7043 | 0,7892 | 6,45 | 0,5168 |

Küçük ile büyük arasında **0,26 Dice farkı**. Tek ortalama bu tabloyu gizlerdi — Bölüm 5'teki
"hacim kırılımı zorunlu" kuralı gerekçelendi.

**Lezyon F1 ters yönde çalışıyor:** büyük lezyonlarda 0,517, küçüklerde 0,595. Çünkü büyük
infarktlar tek kütle değil, ana lezyon + uydu bileşenler; hepsini yakalamak lezyon-bazlı
sayımda daha zor. Dice'ın en iyi olduğu tertilde lezyon F1 en kötü.

### Merkez farkı — karıştırıcı değişken uyarısı

Merkez 2: Dice 0,787 (n=11) · Merkez 1: 0,671 (n=39). Bu **merkez etkisi olarak okunmamalı**:
merkez 2'nin medyan lezyon hacmi 11,43 mL, merkez 1'inki 5,37 mL (Parça 1). Yukarıdaki tertil
tablosuna göre bu fark tek başına hacimle açıklanıyor. Ayrıca n=11 çok küçük.

### Eğitim davranışı — iki gözlem

**1. Overfitting yok.** Son epoch'ta train loss 0,0823, val loss 0,0813 — val train'in altında
(train augmente edilmiş dilimlerde ölçülüyor, val temiz). Makas hiç açılmadı. Yani kapasite ya
da epoch bütçesi darboğaz değil; sınırlayan şey görevin zorluğu veya **tek dilim girdinin
bağlam eksikliği**. `unet_r34_25d` bu hipotezi doğrudan test edecek.

**2. LR, Dice hâlâ tırmanırken düştü.** `ReduceLROnPlateau` val loss'u izliyor; val loss epoch
~18'de platoya girdi ve LR sekiz kez düştü (2,5e-4 → 9,8e-7). Ama val Dice epoch 36'ya kadar
yükselmeye devam etti — yani zirveye ulaşıldığında LR zaten 1,56e-5 idi, model neredeyse
öğrenmiyordu. Density projesinde görülen "scheduler ve early stopping farklı büyüklük izliyor"
ayrışmasının aynısı.

> **Karar: protokol dokuz koşuda da sabit kalıyor.** Şimdi scheduler'ı değiştirmek bu koşuyu
> kalanlarla karşılaştırılamaz hale getirir; mimariler arası fark yalnız mimariden gelmeli.
> Daha yavaş LR düşüşünün daha iyi sonuç verip vermeyeceği **açık bir soru** olarak kayda
> geçiyor ve raporun sınırlılıklar bölümüne yazılacak.

## 5c. Parça 6 — `unet_r34_25d` (2.5D) **TAMAM** · hipotez doğrulanmadı

37 epoch, 29,2 dk, zirve epoch 21. 24,46 M parametre (2B'den yalnız 23 bin fazla — tek fark
ilk conv'un 2 yerine 10 kanal alması). Epoch süresi 30 → 40 sn.

| | `unet_r34_2d` | `unet_r34_25d` | fark |
|---|---|---|---|
| val Dice | 0,6777 (ep 36) | 0,6785 (ep 21) | +0,0008 |
| **test Dice** | **0,6965 ± 0,1912** | 0,6652 ± 0,2288 | **−0,0313** |
| test lezyon F1 | 0,5717 | 0,4820 | −0,0897 |
| test HD95 | 16,06 mm | 18,70 mm | +2,64 |
| test AVD | 2,84 mL | 3,28 mL | +0,44 |
| Dice küçük tertil | 0,5638 | 0,4929 | −0,0709 |

**Bölüm 5b'de kurduğum hipotez — "sınırlayan şey tek dilim girdinin bağlam eksikliği" —
bu koşuda desteklenmedi.** 5 komşu dilim eklemek val Dice'ı değiştirmedi (+0,0008) ve test
Dice'ı düşürdü. Kayıp en çok küçük lezyonlarda (−0,071) ve lezyon-bazlı F1'de (−0,090).

**Eşleştirilmiş test — ölçüldü:** Wilcoxon **p = 0,0249**, 50 vakanın **32'sinde taban çizgisi
kazandı, 17'sinde 2.5D** (1 beraberlik).

Bu, ilk okumamı düzeltiyor. "−0,0313 test SEM'lerinin (0,027 / 0,032) mertebesinde, yani
gürültü" demiştim; **eşleştirilmemiş SEM karşılaştırması burada yanlış araç.** Varyansın büyük
kısmı vaka zorluğundan geliyor (Dice std 0,19–0,23, hacim tertilleri arası fark 0,26); aynı 50
vaka üzerinde eşleştirince bu ortak varyans düşüyor ve testin gücü çok artıyor. İki modelin
farkı, ortalamaların belirsizliğinden küçük olmasına rağmen tutarlı yönde.

**Yine de "2.5D daha kötü" diye yazılmayacak.** Planda 8 karşılaştırma var; Bonferroni/Holm
eşiği 0,05/8 = **0,00625** ve p = 0,0249 bunu geçmiyor. Doğru ifade: *taban çizgisi lehine
tutarlı ama düzeltme sonrası kanıtlanmamış bir fark.* Nihai istatistik (bootstrap CI + Holm
düzeltmesi) Parça 7'de; ara karşılaştırma hücresi düzeltmesiz p veriyor ve bunu kendi
çıktısında yazıyor.

İki yan gözlem:

- **Val ve test ters yönde konuştu**: val +0,0008, test −0,0313. 39 vakalık val üzerinden
  checkpoint seçmenin ne kadar gürültülü olduğunu gösteriyor. Model seçimi hakkında val
  Dice'a dayanan her cümle bu belirsizliği taşıyor.
- **Erken zirve**: 2.5D epoch 21'de zirve yapıp 15 epoch iyileşmedi; 2B epoch 36'da. Fazla
  kanal daha hızlı ezberlemeye yol açmış olabilir, ama train/val makası yine açılmadı.

## 5d. Parça 6 — `segformer_b0` **TAMAM** · taban çizgisinin belirgin altında

50 epoch, 33,3 dk, zirve epoch 34. **3,73 M parametre** — taban çizgisinin 6,5'te biri.

| | `unet_r34_2d` | `segformer_b0` | fark |
|---|---|---|---|
| parametre | 24,43 M | **3,73 M** | ÷6,5 |
| val Dice | 0,6777 | 0,6345 | −0,0432 |
| **test Dice** | **0,6965** | 0,6398 ± 0,2174 | **−0,0567** |
| test lezyon F1 | 0,5717 | 0,4416 | −0,1301 |
| Dice küçük tertil | 0,5638 | 0,4539 | −0,1099 |
| test AVD | 2,84 mL | **2,67 mL** | −0,17 |

**Eşleştirilmiş Wilcoxon: p < 0,0001 · 50 vakanın 43'ünde taban çizgisi kazandı, 6'sında
SegFormer.** Bu fark 2.5D'ninkinin aksine Holm düzeltmesini (eşik 0,00625) rahatça geçiyor —
yani SegFormer-B0'ın bu görevde geride kaldığı **kanıtlanmış** sayılabilir (nihai doğrulama
Parça 7'de).

**BUSI projesiyle ters sonuç.** `real_time_segmentasyon`'da SegFormer-B0 hem en hızlı hem
kalitede geride değildi; burada net biçimde geride. Aradaki fark muhtemelen görev yapısında:
BUSI'de tek, büyük, iyi sınırlı bir lezyon vardı; burada medyan 5 bağlı bileşen ve vakaların
yarısında lezyon beyin hacminin binde beşinden küçük. SegFormer'ın 1/4 çözünürlüklü decode
head'i küçük yapıları kaybediyor olabilir — küçük tertildeki −0,110'luk kayıp bu okumayı
destekliyor.

> **Ön uyarı — gecikme sinyali.** 50 test vakasının değerlendirmesi: `unet_r34_2d` 6 sn,
> `unet_r34_25d` 10 sn, `segformer_b0` **27 sn**. Yani SegFormer parametre sayısı 6,5 kat
> küçük olmasına rağmen çıkarımda ~2,7 kat yavaş. Bu **Parça 8 protokolüyle ölçülmüş bir
> sayı değil** (çerçeve yükü dahil, ısınma yok, batch=32); yalnızca bir işaret. Ama
> "az parametre = hızlı" varsayımının bu projede de geçerli olmayabileceğini gösteriyor.

### Şu ana kadarki tablo

| model | param | test Dice | lezyon F1 | vs taban (eşleştirilmiş) |
|---|---|---|---|---|
| **`unet_r34_2d`** | 24,43 M | **0,6965** | **0,5717** | — |
| `unet_r34_25d` | 24,46 M | 0,6652 | 0,4820 | −0,0313 · p = 0,025 · 17/32 |
| `segformer_b0` | 3,73 M | 0,6398 | 0,4416 | −0,0567 · **p < 0,0001** · 6/43 |

Taban çizgisi iki alternatifi de geçti. Kalan altı koşu bunu farklı yönlerden test edecek;
özellikle `unet3d_patch` (2.5D'nin veremediği gerçek 3B bağlam) ve `A2_tversky` (küçük
lezyonlardaki kaybı hedefliyor) kritik.

## 5e. Parça 6 — `unet_mbv3` **TAMAM** · en iyi val, üçüncü test

57 epoch, 38,9 dk, zirve epoch 41. **6,69 M parametre** — taban çizgisinin 3,7'de biri.

| | `unet_r34_2d` | `unet_mbv3` | fark |
|---|---|---|---|
| parametre | 24,43 M | **6,69 M** | ÷3,7 |
| val Dice | 0,6777 | **0,6825** | **+0,0048** |
| **test Dice** | **0,6965** | 0,6744 ± 0,2096 | **−0,0221** |
| test precision | 0,7540 | **0,7630** | +0,0090 |
| test recall | 0,6844 | 0,6483 | −0,0361 |
| test HD95 | 16,06 mm | **15,08 mm** | −0,98 |
| test lezyon F1 | 0,5717 | 0,4918 | −0,0799 |

Eşleştirilmiş Wilcoxon: **p = 0,0048 · 36 kayıp / 13 kazanç**. Holm sıralamasında ikinci
sırada (eşik 0,05/7 = 0,00714) — şu anki üç karşılaştırmayla düzeltmeyi geçiyor, ama sekiz
koşu tamamlanmadan kesinleşmeyecek.

Model **muhafazakâr** davranıyor: precision taban çizgisinden yüksek, recall belirgin düşük.
Sınır kalitesi de iyi (tablodaki en düşük HD95). Kaybı lezyon-bazlı F1'de: küçük uydu
bileşenleri atlıyor.

## Val seti model sıralamasında güvenilir değil — dört koşuda üçüncü kez

| | val Dice sırası | test Dice sırası |
|---|---|---|
| 1 | `unet_mbv3` 0,6825 | **`unet_r34_2d` 0,6965** |
| 2 | `unet_r34_25d` 0,6785 | `unet_mbv3` 0,6744 |
| 3 | `unet_r34_2d` 0,6777 | `unet_r34_25d` 0,6652 |
| 4 | `segformer_b0` 0,6345 | `segformer_b0` 0,6398 |

**Val sıralaması test sıralamasını yeniden üretmiyor**; val'de birinci olan model test'te
ikinci, val'de üçüncü olan test'te birinci. Val Dice aralığı 0,6777–0,6825 (0,005) — 39 vaka
üzerinde bu fark ölçülemez. Test farkları da küçük ama **eşleştirilmiş** olarak ölçüldüğü için
ayırt edilebiliyor.

> **Sonuç:** "en iyi model" cümlesi yalnız **test setinde, eşleştirilmiş karşılaştırmayla**
> kurulacak. Val Dice'ın tek görevi checkpoint seçmek; model seçmek değil. Bu kural raporda
> açıkça yazılacak.

## Parametre sayısı hız demek değil — dört koşuluk ön sinyal

50 test vakasının değerlendirme süresi (aynı kod, aynı protokol, batch=32):

| model | parametre | k (kanal) | test değerlendirme |
|---|---|---|---|
| `unet_r34_2d` | 24,43 M | 0 (2) | **6 sn** |
| `unet_r34_25d` | 24,46 M | 2 (10) | 10 sn |
| `segformer_b0` | **3,73 M** | 2 (10) | 27 sn |
| `unet_mbv3` | 6,69 M | 2 (10) | **30 sn** |

**İki en küçük model, iki en yavaş model.** Son üç satır aynı girdiyi (10 kanal) aldığı için
bu karşılaştırma doğrudan mimariyi ölçüyor: MobileNetV3, ResNet34'ten 3,7 kat az parametreyle
3 kat yavaş. Sebebi büyük olasılıkla depthwise separable conv'ların GPU'da düşük aritmetik
yoğunlukla çalışması — FLOPs düşük, ama bellek bant genişliği ve kernel başlatma yükü baskın.

Bu **Parça 8 protokolüyle ölçülmüş değil** (çerçeve yükü dahil, ısınma yok, batch=32, T4).
Ama BUSI projesindeki dersi tekrarlıyor: mimari sıralaması hakkında parametre sayısına
bakarak yapılan her yorum yanlış çıkabiliyor. Parça 8'de ONNX/TensorRT altında yeniden
ölçülecek ve sıralamanın değişip değişmediği raporlanacak.

## 5f. Parça 6 — `unet3d` **TAMAM** · planın öncülü yanlış çıktı

51 epoch, 47,3 dk (~56 sn/epoch), zirve epoch 35.

| | `unet_r34_2d` (taban) | `unet3d` | fark |
|---|---|---|---|
| parametre | 24,43 M | **1,40 M** | **÷17,4** |
| val Dice | 0,6777 | **0,7081** | +0,0304 |
| **test Dice** | 0,6965 | **0,7023 ± 0,2310** | **+0,0058** |
| test IoU | 0,5616 | **0,5787** | +0,0171 |
| test recall | 0,6844 | **0,7562** | +0,0718 |
| test precision | **0,7540** | 0,6886 | −0,0654 |
| test AVD | **2,84 mL** | 3,76 mL | +0,92 |
| test lezyon F1 | 0,5717 | 0,5698 | −0,0019 |
| **50 vaka çıkarım** | 6 sn | **4 sn** | en hızlı |
| pik VRAM (batch 2) | 0,13 GB | 0,90 GB | — |

Eşleştirilmiş Wilcoxon: **p = 0,0844 · 30 kazanç / 19 kayıp**. Nominal olarak en iyi model,
**ama fark kanıtlanmadı** — düzeltmesiz p bile 0,05'i geçmiyor. Doğru ifade: *taban çizgisinden
kötü olmayan tek alternatif.*

### Planın "3B = doğruluk tavanı, gerçek zamanlı değil" öncülü geçersiz

Bölüm 4'te `unet3d`'yi "doğruluk tavanı, gerçek zamanlı **değil**, bilerek" diye
konumlandırmıştım. Üç ölçüm de bunun tersini söylüyor:

1. **En küçük model** — 1,40 M parametre, taban çizgisinin 17'de biri.
2. **En hızlı çıkarım** — 50 test vakası 4 sn (≈80 ms/vaka), 2B taban çizgisi 6 sn,
   `unet_mbv3` 30 sn. Sebebi yapısal: 3B model vaka başına **tek ileri geçiş** yapıyor,
   2B modeller ~78 dilimi ayrı ayrı geçiriyor. Dilim başına Python/veri hattı yükü toplanınca
   "ağır" 3B model daha ucuz çıkıyor.
3. **0,90 GB pik VRAM** — hedef donanım olan RTX 2060 6 GB'a rahat sığıyor.

Yani proje boyunca "hızlı ama daha az doğru" ile "doğru ama yavaş" arasında bir ödünleşim
beklerken, 3B model üç eksende de (doğruluk / boyut / hız) önde çıktı. Bu, Bölüm 6'daki
optimizasyon merdiveninin hangi model üzerinde koşulacağını değiştiriyor — Parça 8'de
`unet3d` de merdivene girecek.

> **Uyarı:** buradaki 4 sn / 6 sn sayıları **Parça 8 protokolüyle ölçülmedi** (T4, batch=32,
> ısınma yok, çerçeve yükü dahil, Drive'dan okuma hariç). Sıralamanın ONNX/TensorRT altında ve
> RTX 2060'ta korunup korunmadığı orada ölçülecek. BUSI projesinde sıralama ONNX'e geçince
> tersine dönmüştü.

### İki gerçek zayıflık

**Aşırı segmentasyon.** Recall 0,756 (en yüksek), precision 0,689 (üst üçlünün en düşüğü),
AVD 3,76 mL (en kötü). Klinik çıktı hacim raporladığı için bu önemli: model lezyonu buluyor
ama şişiriyor. Eşik taraması (Parça 7) bu dengeyi kaydırabilir.

**Tek 2B model gibi davranmadı — overfit etti.** Son epoch'ta train loss 0,039, val 0,063
(makas +0,024). Dört 2B koşuda val her zaman train'in altındaydı. 161 hacim, `tekrar=3` ile
epoch başına 483 örnek — çeşitlilik dilim tabanlı eğitimden çok daha az.

**LR hiç düşmedi.** 51 epoch boyunca 3,0e-4'te kaldı; val loss `ReduceLROnPlateau`'nun
patience'ını hiç dolduramadı. Koşu, yakınsadığı için değil **val Dice 15 epoch iyileşmediği
için** durdu. Yani bu model protokolün epoch/patience bütçesine sıkışmış olabilir — daha uzun
eğitimle daha iyi olabilirdi. Protokol dokuz koşuda sabit tutulduğu için değiştirilmiyor,
**açık soru olarak kaydediliyor** (Bölüm 5b'deki LR notunun ikizi, ters yönde).

### Şu ana kadarki tablo — beş koşu

| model | param | test Dice | lezyon F1 | 50 vaka | vs taban (eşleştirilmiş) |
|---|---|---|---|---|---|
| **`unet3d`** | **1,40 M** | **0,7023** | 0,5698 | **4 sn** | +0,0058 · p = 0,084 · 30/19 |
| `unet_r34_2d` | 24,43 M | 0,6965 | **0,5717** | 6 sn | — |
| `unet_mbv3` | 6,69 M | 0,6744 | 0,4918 | 30 sn | −0,0221 · p = 0,005 · 13/36 |
| `unet_r34_25d` | 24,46 M | 0,6652 | 0,4820 | 10 sn | −0,0313 · p = 0,025 · 17/32 |
| `segformer_b0` | 3,73 M | 0,6398 | 0,4416 | 27 sn | −0,0567 · p < 0,0001 · 6/43 |

Parametre sayısı ile ne doğruluk ne hız arasında ilişki var: en küçük iki model tablonun
birinci ve sonuncusu; en yavaş iki model de en küçük ikinciler.

## 5g. Parça 2b — FLAIR hazırlama (A1 için) · **TAMAM** (sonuçlar 5k'da)

A1 koşusu `hazir/` içinde FLAIR olmadığı için başladığı yerde patladı (`weight of size
[64, 11, 7, 7], got 10 channels`). Parça 2'de FLAIR'i bilerek dışarıda bırakmıştık — kayıt
gerektiriyordu — ama ablasyonun kendisi o kaydı yapmayı gerektiriyor.

**Yöntem: header tabanlı yeniden örnekleme.** İki seri aynı oturumda çekildiği için NIfTI
affine'leri ortak tarayıcı uzayını tanımlıyor; `resample_from_to` FLAIR'i DWI grid'ine bu
affine'ler üzerinden taşıyor. Sonra `_x.npy` ile **aynı** 2 mm resample ve **aynı** kırpma
kutusundan geçiriliyor (kutu aynı DWI'dan, aynı eşikle, deterministik olarak yeniden
hesaplanıyor), maske içi z-score ile normalize edilip `hazir/<vaka>_f.npy` olarak yazılıyor.

> **Sınırlılık — rapora yazılacak.** Yoğunluk tabanlı kayıt (registration) yapılmıyor. ISLES'22
> makalesi FLAIR'in referans çerçeveden "hafifçe kaymış" olduğunu söylüyor; header tabanlı
> taşıma hasta hareketini düzeltmez. Yani **A1, "FLAIR'in katkısı"nı değil "ucuz hizalamayla
> FLAIR'in katkısı"nı ölçüyor.** Sonuç negatif çıkarsa FLAIR'in bilgi taşımadığı sonucu
> çıkarılamaz. Görsel doğrulama hücresi (DWI kırmızı / FLAIR yeşil bindirme) hizalamanın
> yeterli olup olmadığını gösteriyor.

Hizalamanın **vaka başına ms maliyeti** ölçülüp `ozet/flair_hizalama.xlsx`'e yazılıyor ve
Bölüm 3 gecikme bütçesine giriyor — A1'in asıl sorusu zaten "katkısı maliyetini karşılıyor mu".

### Eklenen koruma

Kanal uyuşmazlığı artık ilk conv'da değil `egit()` başında yakalanıyor: veri setinin ürettiği
kanal sayısı konfigle karşılaştırılıp açık mesajla hata veriliyor. `npy_yukle` de FLAIR
önbelleği yoksa hangi notebook bölümünün çalıştırılması gerektiğini söylüyor.

## 5h. Parça 6 — `A2_tversky` **ÇÖKTÜ**, sebebi bulundu ve düzeltildi

İlk koşu: test Dice **0,0438**, precision 0,0237, recall 0,7746, AVD **966 mL**, HD95 101 mm.
Model beynin ~%70'ini lezyon işaretliyor. Eşleştirilmiş Wilcoxon 0/49 — 50 vakanın hepsinde
taban çizgisinin altında. Epoch 0'da bile Dice 0,0135 (taban çizgisi epoch 0'da 0,38 idi),
yani kademeli bozulma değil, **baştan bozuk**.

### Kök neden — ölçüldü, tahmin edilmedi

Loss dört senaryoda test edildi (mükemmel / çöküş / hepsi arka plan / yarım doğru):

| durum | saf focal-Tversky (koşan hali) | BCE + focal-Tversky (düzeltilmiş) |
|---|---|---|
| mükemmel | loss 0,0005 · grad 6,8e-4 | 0,0003 · 3,6e-4 |
| **çöküş (hepsi lezyon)** | loss 0,98 · **grad 1,6e-6** | loss 5,47 · **grad 0,50** |
| yarım doğru | 0,52 · 0,29 | 0,27 · 0,16 |

**Saf bölge kaybı doygunluk tuzağı yaratıyor.** Model her şeyi lezyon dediğinde loss yüksek
(0,98) ama gradyan pratikte sıfır: sigmoid doymuş ve `fn = (1−p)·hedef = 0`. Çöküş
gradyanı "yarım doğru"nun 5×10⁻⁶ katı — model bir kez çökünce çıkamıyor. Taban çizgisinin
**BCE terimi tam olarak bunu engelliyormuş**; BCE'nin gradyanı `(p − y)` ve doymuyor.

İkinci hata: **γ tersti.** Abraham & Khan (2019) γ = 4/3 kullanıyor, kodda 0,75 yazılmıştı.
γ < 1 kolay örneklerde gradyanı büyütüyor — focal'in amacının tam tersi.

### Düzeltmeler

1. `FocalTversky` artık **BCE + focal-Tversky** (taban çizgisiyle aynı yapı, `agirlik_bce=0.5`).
2. **γ = 4/3**.
3. **A2 artık tek değişken değiştiriyor.** İlk tasarımda hem bölge kaybı hem `lezyon_orani`
   (0,5 → 0,7) değişiyordu; fark çıksaydı hangisinden geldiği ayrılamazdı. Örnekleme oranı
   taban çizgisiyle aynı bırakıldı. Örnekleme kolu ayrı bir koşu olarak istenirse eklenir.

### Süreç boşluğu — kapatıldı

Metrikler yedi senaryoyla test edilmişti ama **loss'lar hiç test edilmemişti.** Notebook'a
loss testi hücresi eklendi: her loss için çöküş durumunun loss'u mükemmelden yüksek olmalı
**ve** çöküş gradyanı "yarım doğru"nun en az %1'i olmalı (`assert`). Yerelde koşuldu, ikisi de
geçiyor (`dice_bce` oranı 4,20 · `focal_tversky` 3,20).

> Bu tablo raporda kalacak: negatif sonuç değil, **düzeltilmiş bir uygulama hatası**.
> Düzeltilmiş A2 yeniden koşuldu; sonuçlar Bölüm 5i'de, karşılaştırma tablosundaki satır
> o koşuya ait.

## 5i. `A2_tversky` düzeltilmiş koşu **TAMAM** · loss çalıştı, hipotez çalışmadı

Loss testleri geçti (`dice_bce` çöküş/yarım gradyan oranı 4,20 · `focal_tversky` 3,20), sonra
70 epoch tam koşu, 54,1 dk, zirve epoch 55.

| | `unet_r34_2d` (taban) | `A2_tversky` | fark |
|---|---|---|---|
| val Dice | 0,6777 | 0,6845 | +0,0068 |
| **test Dice** | **0,6965** | 0,6806 ± 0,2088 | **−0,0159** |
| test **recall** | 0,6844 | **0,7188** | **+0,0344** |
| test precision | **0,7540** | 0,6741 | −0,0799 |
| **Dice küçük tertil** | **0,5638** | 0,5296 | **−0,0342** |
| Dice orta / büyük | 0,6953 / 0,8226 | 0,6901 / 0,8133 | −0,005 / −0,009 |
| test lezyon F1 | 0,5717 | 0,4869 | −0,0848 |

Eşleştirilmiş Wilcoxon: **p = 0,1714 · 20 kazanç / 29 kayıp** — fark anlamlı değil.

**Loss yönü doğru çalıştı, hedefi tutmadı.** β > α tasarımı beklendiği gibi dengeyi recall'a
kaydırdı (+0,034 recall, −0,080 precision). Ama A2'nin gerekçesi "küçük lezyonlarda recall
kazanmak"tı ve **küçük tertilde Dice düştü** (0,5638 → 0,5296). Yani focal-Tversky'nin ürettiği
ekstra recall küçük lezyonlara değil, zaten bulunan lezyonların çevresine gitmiş görünüyor —
AVD de 2,84 → 3,61 mL'ye çıkmış.

> **Bütçe uyarısı — bu koşu tavana dayandı.** A2 **70 epoch'un tamamını** kullandı (early
> stopping hiç tetiklenmedi, son epoch 69, zirve 55). Taban çizgisi epoch 36'da zirve yapıp
> 51'de durmuştu. focal-Tversky belirgin biçimde daha yavaş öğreniyor; `max_epochs=70` tavanı
> bu koşuda **sınırlayıcı olmuş olabilir**. Protokol dokuz koşuda sabit tutulduğu için
> uzatılmıyor, ancak "A2 daha kötü" sonucu bu uyarıyla birlikte okunmalı.

### Altı koşuluk tablo

| model | param | test Dice | recall | küçük tertil | lezyon F1 | vs taban |
|---|---|---|---|---|---|---|
| **`unet3d`** | 1,40 M | **0,7023** | 0,7562 | 0,5541 | 0,5698 | +0,006 · p = 0,084 |
| `unet_r34_2d` | 24,43 M | 0,6965 | 0,6844 | **0,5638** | **0,5717** | — |
| `A2_tversky` | 24,46 M | 0,6806 | 0,7188 | 0,5296 | 0,4869 | −0,016 · p = 0,171 |
| `unet_mbv3` | 6,69 M | 0,6744 | 0,6483 | 0,5453 | 0,4918 | −0,022 · p = 0,005 |
| `unet_r34_25d` | 24,46 M | 0,6652 | 0,6826 | 0,4929 | 0,4820 | −0,031 · p = 0,025 |
| `segformer_b0` | 3,73 M | 0,6398 | 0,6465 | 0,4539 | 0,4416 | −0,057 · p < 0,0001 |

**Holm durumu** (8 karşılaştırma planlı, 5'i yapıldı): `segformer_b0` (p < 0,0001 vs 0,00625)
ve `unet_mbv3` (p = 0,0048 vs 0,00714) düzeltmeyi geçiyor. `unet_r34_25d` (0,0249 vs 0,00833),
`unet3d` (0,084) ve `A2_tversky` (0,171) geçmiyor. Kalan üç koşudan sonra kesinleşecek.

**Küçük lezyonlar hâlâ açık sorun.** Altı koşunun hiçbiri küçük tertilde 0,564'ü geçemedi;
en iyi değer taban çizgisinin. Bunu doğrudan hedefleyen tek ablasyon (A2) durumu kötüleştirdi.

## 5j. `A3_k1` / `A3_k3` **TAMAM** · 2.5D bağlam derinliği etkisiz çıktı

A3'ün amacı 2.5D bağlamının nerede doyduğunu ölçmekti. Sonuç bundan daha faydalı oldu:
**bağlam derinliğinin etkisi koşular arası gürültünün altında.**

| k | dilim | kanal | test Dice | vs taban | p | zirve epoch |
|---|---|---|---|---|---|---|
| 0 (taban) | 1 | 2 | **0,6965** | — | — | 36 |
| **1** | 3 | 6 | 0,6896 | −0,0069 | **0,665** | 35 |
| **2** | 5 | 10 | 0,6652 | −0,0313 | 0,025 | 21 |
| **3** | 7 | 14 | 0,6796 | −0,0169 | 0,084 | 33 |

**Eğilim monoton değil.** Bağlam gerçekten işe yarasa (ya da gerçekten zarar verse) k arttıkça
tutarlı bir yön beklerdik. Onun yerine 0,6896 → 0,6652 → 0,6796 dalgalanması var; aralık 0,024,
koşuların SEM'i ~0,029. Yani **üç değer birbirinden ayırt edilemiyor**.

Bu, Bölüm 5c'deki yorumu düzeltiyor. Orada `unet_r34_25d` için p = 0,0249 ölçülüp "taban çizgisi
lehine tutarlı ama düzeltme sonrası kanıtlanmamış fark" denmişti. A3 serisi gösteriyor ki
**o p muhtemelen tek koşuluk şanssız bir çekiliş**: aynı mimarinin k=1 ve k=3 varyantları taban
çizgisine çok daha yakın (p = 0,665 ve 0,084). Doğru sonuç: *2.5D bağlam bu görevde ölçülebilir
bir katkı sağlamıyor, ama zarar verdiği de gösterilemedi.*

### Ablasyonun beklenmedik faydası — koşu varyansı tahmini

Bütçe nedeniyle hiçbir koşu tekrarlanmadı, dolayısıyla **aynı konfigürasyonun iki kez
eğitilmesi arasındaki varyansı ölçmedik.** A3 serisi bunun dolaylı bir tahminini veriyor:
k = 1/2/3 aynı mimarinin küçük varyantları ve test Dice'ları **0,024'lük bir banda** yayılıyor.

> **Bu bandın altındaki hiçbir fark yorumlanmayacak.** Tablodaki `unet3d` (+0,006),
> `A3_k1` (−0,007), `A2_tversky` (−0,016), `A3_k3` (−0,017) ve `unet_mbv3` (−0,022) farklarının
> tamamı bu bandın içinde. Ayırt edilebilir tek model `segformer_b0` (−0,057). Bu uyarı raporun
> sonuç bölümünün merkezinde duracak.

### Sekiz koşuluk tablo

| model | param | test Dice | recall | küçük tertil | lezyon F1 | vs taban |
|---|---|---|---|---|---|---|
| **`unet3d`** | **1,40 M** | **0,7023** | 0,7562 | 0,5541 | 0,5698 | +0,006 · p = 0,084 |
| `unet_r34_2d` | 24,43 M | 0,6965 | 0,6844 | **0,5638** | **0,5717** | — |
| `A3_k1` | 24,45 M | 0,6896 | 0,6796 | 0,5353 | 0,5219 | −0,007 · p = 0,665 |
| `A2_tversky` | 24,46 M | 0,6806 | 0,7188 | 0,5296 | 0,4869 | −0,016 · p = 0,171 |
| `A3_k3` | 24,47 M | 0,6796 | 0,6596 | 0,5382 | 0,4865 | −0,017 · p = 0,084 |
| `unet_mbv3` | 6,69 M | 0,6744 | 0,6483 | 0,5453 | 0,4918 | −0,022 · p = 0,005 |
| `unet_r34_25d` | 24,46 M | 0,6652 | 0,6826 | 0,4929 | 0,4820 | −0,031 · p = 0,025 |
| `segformer_b0` | 3,73 M | 0,6398 | 0,6465 | 0,4539 | 0,4416 | −0,057 · p < 0,0001 |

**Holm düzeltmesi (m = 8):** p'ler sıralı 0,0000 · 0,0048 · 0,0249 · 0,0844 · 0,0844 · 0,1714 ·
0,6652. Eşikler 0,00625 → 0,00714 → 0,00833 …
`segformer_b0` (0,0000 < 0,00625) ✔ ve `unet_mbv3` (0,0048 < 0,00714) ✔ geçiyor;
`unet_r34_25d` (0,0249 > 0,00833) ✘ ve sonrası zincirle düşüyor.
**Taban çizgisinden anlamlı biçimde ayrışan tek iki model bunlar, ikisi de daha kötü yönde.**

## 5k. Parça 2b **TAMAM** · FLAIR hizalaması gerçek zamanlı bütçeyi tek başına aşıyor

250/250 vakada FLAIR hizalandı ve `hazir/<vaka>_f.npy` yazıldı, hata yok. Şekil ve kırpma
`_x.npy` ile birebir aynı (`assert` ile doğrulandı).

**Ölçülen hizalama maliyeti — vaka başına** (250 vaka, `ozet/flair_hizalama.xlsx`):

| | süre |
|---|---|
| **medyan** | **670 ms** |
| ortalama ± std | 570 ± 533 ms |
| çeyrekler (25/75) | 220 / 741 ms |
| min / max | 44 / 4735 ms |

> **Düzeltme.** Parça 2b'de bu sayı 1193 ms olarak raporlanmıştı. Fark ölçüm sınırından
> geliyor: orada `nib.load` tembel olduğu için FLAIR hacminin **diskten okunması ve
> açılması** zamanlamanın içinde kalıyordu. Yeni ölçümde hacim önce belleğe alınıp sonra
> `resample_from_to` ölçülüyor — Bölüm 3 protokolü diskten okumayı bütçe dışında tuttuğu
> için **670 ms doğru sayı**. Sonuç değişmiyor, yalnız keskinleşiyor.

### Bu sayı A1'i gerçek zamanlı konfigürasyondan çıkarıyor

Bölüm 3'teki hedef: **vaka başına uçtan uca ≤ 2 s, hedeflenen ≤ 1 s.** Karşılaştırma:

| bileşen | süre |
|---|---|
| `unet3d` tam çıkarım (ön işleme + ağ + son işleme) | ~80 ms/vaka |
| `unet_r34_2d` tam çıkarım | ~120 ms/vaka |
| **FLAIR hizalaması tek başına** | **670 ms/vaka** |

FLAIR hizalaması, Parça 8'de ölçülen en hızlı uçtan uca hattın (**75 ms/vaka**, `unet3d` +
GPU ön işleme + TensorRT fp16) **9 katı**. Yani FLAIR eklemek, tüm çıkarım hattını dokuz kez
koşmakla aynı maliyete geliyor.

> **Karar: A1 hâlâ koşulacak, ama sonucu ne çıkarsa çıksın FLAIR gerçek zamanlı
> konfigürasyona girmeyecek.** A1'in cevaplayacağı soru artık "FLAIR eklensin mi" değil —
> o soru gecikme tarafından kapandı. Cevaplayacağı soru: *"gerçek zamanlılık kısıtı olmasaydı
> ne kadar doğruluk kazanılırdı?"* Bu, raporda kısıtın maliyetini sayısallaştıran bir veri
> noktası olarak duracak.

Hizalama `nibabel.processing.resample_from_to` ile CPU'da yapılıyor; FLAIR hacimleri büyük
(kimi 512×512×30, kimi 352×352×230). GPU'da yeniden örnekleme ya da daha düşük dereceli
interpolasyonla hızlandırılabilir — ama en iyi durumda bile modelin kendisinden pahalı kalır,
çünkü işlenen voxel sayısı bir mertebe fazla.

### Bir yan doğrulama

Split koruması çalıştı: `split.json` ve `eval_cases.json` yeniden üretildiğinde **mevcut
dosyalarla birebir aynı** çıktı. Yani Parça 2'nin bölme mantığı deterministik ve biten sekiz
koşunun test seti geçerli. (`kilitli_yaz` farklı çıksaydı hata verip mevcut dosyayı korurdu.)

## 5l. `A1_flair` **TAMAM** · Parça 6 bitti — dokuz koşu

53 epoch, 66,0 dk, zirve epoch 37. 15 kanal (3 modalite × 5 dilim), 24,47 M parametre.

| | `unet_r34_2d` (taban) | `A1_flair` | fark |
|---|---|---|---|
| **test Dice** | **0,6965** | 0,6568 ± 0,2347 | **−0,0397** |
| test precision / recall | 0,7540 / 0,6844 | 0,7245 / 0,6488 | −0,030 / −0,036 |
| test AVD | **2,84 mL** | 4,27 mL | +1,43 |
| test lezyon F1 | **0,5717** | 0,4557 | −0,1160 |
| Dice orta tertil | **0,6953** | 0,6140 | **−0,0813** |

Eşleştirilmiş Wilcoxon: **p = 0,0011 · 38 kayıp / 11 kazanç**. Holm düzeltmesini geçiyor.

**FLAIR eklemek doğruluğu anlamlı biçimde düşürdü.** Bölüm 5k'da gecikme tarafından zaten
reddedilmişti (1193 ms/vaka hizalama, çıkarımın 15 katı); şimdi doğruluk tarafında da
reddediliyor. A1 iki ayrı gerekçeyle kapanmış oluyor.

> **Ama dikkat — bu "FLAIR bilgi taşımıyor" demek değil.** Bölüm 5g'deki sınırlılık burada
> belirleyici: hizalama yalnız NIfTI header'ları üzerinden yapıldı, **yoğunluk tabanlı kayıt
> yok**. ISLES'22 makalesi FLAIR'in referans çerçeveden kaymış olduğunu söylüyor. Artık kayma
> varsa, 5 ek kanal modele bilgi değil **gürültü** taşımış olur — ve sonuç bununla tutarlı:
> kayıp en çok orta tertilde (−0,081) ve lezyon F1'de (−0,116), yani sınır hassasiyeti
> gerektiren yerlerde. Doğru ifade: **ucuz hizalamayla FLAIR zarar veriyor.** Gerçek katkısı
> ancak düzgün kayıtla ölçülebilir ve bu, gerçek zamanlı bütçeyle zaten bağdaşmıyor.

---

## 5m. Parça 6 sonucu — dokuz koşuluk tam tablo

| # | model | param | test Dice | recall | küçük | orta | büyük | lezyon F1 | AVD | vs taban |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **`unet3d`** | **1,40 M** | **0,7023** | **0,7562** | 0,5541 | **0,7111** | **0,8330** | 0,5698 | 3,76 | +0,006 |
| 2 | `unet_r34_2d` | 24,43 M | 0,6965 | 0,6844 | **0,5638** | 0,6953 | 0,8226 | **0,5717** | 2,84 | — |
| 3 | `A3_k1` | 24,45 M | 0,6896 | 0,6796 | 0,5353 | 0,6958 | 0,8286 | 0,5219 | **2,40** | −0,007 |
| 4 | `A2_tversky` | 24,46 M | 0,6806 | 0,7188 | 0,5296 | 0,6901 | 0,8133 | 0,4869 | 3,61 | −0,016 |
| 5 | `A3_k3` | 24,47 M | 0,6796 | 0,6596 | 0,5382 | 0,6720 | 0,8204 | 0,4865 | 3,09 | −0,017 |
| 6 | `unet_mbv3` | 6,69 M | 0,6744 | 0,6483 | 0,5453 | 0,6554 | 0,8148 | 0,4918 | 3,02 | −0,022 |
| 7 | `unet_r34_25d` | 24,46 M | 0,6652 | 0,6826 | 0,4929 | 0,6780 | 0,8146 | 0,4820 | 3,28 | −0,031 |
| 8 | `A1_flair` | 24,47 M | 0,6568 | 0,6488 | 0,5387 | 0,6140 | 0,8107 | 0,4557 | 4,27 | −0,040 |
| 9 | `segformer_b0` | 3,73 M | 0,6398 | 0,6465 | 0,4539 | 0,6558 | 0,7989 | 0,4416 | 2,67 | −0,057 |

### Holm düzeltmesi (m = 8) — **hesaplandı**

| model | p | Holm eşiği | sonuç |
|---|---|---|---|
| `segformer_b0` | 0,0001 | 0,00625 | **ANLAMLI** (daha kötü) |
| `A1_flair` | 0,0011 | 0,00714 | **ANLAMLI** (daha kötü) |
| `unet_mbv3` | 0,0048 | 0,00833 | **ANLAMLI** (daha kötü) |
| `unet_r34_25d` | 0,0249 | 0,01000 | anlamlı değil |
| `unet3d` | 0,0844 | 0,01250 | anlamlı değil |
| `A3_k3` | 0,0844 | 0,01667 | anlamlı değil |
| `A2_tversky` | 0,1714 | 0,02500 | anlamlı değil |
| `A3_k1` | 0,6652 | 0,05000 | anlamlı değil |

### Parça 6'nın üç sonucu

**1. Hiçbir alternatif taban çizgisini geçemedi.** Basit 2B U-Net + ResNet34, dokuz koşuluk
tablonun ikincisi ve tek nominal üstünü (`unet3d`) ayırt edilemiyor. Üç model ise anlamlı
biçimde **daha kötü**. Denenen dört fikrin (2.5D bağlam, transformer, hafif encoder, FLAIR)
hiçbiri işe yaramadı.

**2. Fark bandı 0,024.** A3 serisi (k = 1/2/3, aynı mimarinin varyantları) 0,024'lük bir
banda yayıldı (Bölüm 5j). Tablodaki ilk yedi modelin tamamı taban çizgisinden bu bandın
içinde farklı. **Kalite tarafında modeller pratikte denk.**

**3. Seçim kalitede değil, maliyette yapılacak.** Kalite ayırt edici olmadığı için karar
hız/boyut eksenine kayıyor — ki orada `unet3d` açık ara önde: **1,40 M parametre** (taban
çizgisinin 1/17'si) ve **~80 ms/vaka** çıkarım (taban çizgisi ~120 ms). Parça 8'in merdiveni
`unet3d` ve `unet_r34_2d` üzerinde koşacak.

**Küçük lezyonlar çözülmedi.** Dokuz koşunun hiçbiri küçük tertilde 0,564'ü geçemedi; en iyi
değer hâlâ taban çizgisinin. Bunu doğrudan hedefleyen ablasyon (A2) durumu kötüleştirdi.
Raporun tartışma bölümünün ana açık sorunu bu olacak.

### Bütçe

Dokuz koşu tamamlandı, toplam ~6 saat. Kalan 8 koşu (3D U-Net daha yavaş olacak) tahminen **4–6 saat**.

---

## 7b. Parça 7 sonuçları — **ölçüldü** (`degerlendirme/`)

### Eşik taraması (val) — iki model ters yöne gidiyor, kazanç ihmal edilebilir

| model | en iyi eşik | val Dice (en iyi / 0,50'de) | kazanç |
|---|---|---|---|
| `unet3d` | **0,70** | 0,7100 / 0,7081 | +0,0020 |
| `unet_r34_2d` | **0,20** | 0,6809 / 0,6777 | +0,0032 |

Yönler profillerle tutarlı: `unet3d` aşırı segmente ediyordu (recall 0,756), eşiği **yükseltmek**
istiyor; `unet_r34_2d` muhafazakârdı (precision 0,754), eşiği **düşürmek** istiyor. Ama iki
kazanç da 0,024'lük gürültü bandının bir mertebe altında.

**Test'e uygulandığında** (val'de seçilen çift, tek sefer):

| model | Dice | precision | recall | HD95 | AVD | lezyon F1 |
|---|---|---|---|---|---|---|
| `unet3d` 0,50 → **0,70** | 0,7023 → **0,7063** | 0,689 → 0,710 | 0,756 → 0,742 | 16,58 → **15,83** | 3,76 → **3,23** | 0,570 → **0,581** |
| `unet_r34_2d` 0,50 → **0,20** | 0,6965 → 0,6957 | 0,754 → 0,720 | 0,684 → 0,712 | 16,06 → 17,75 | 2,84 → 2,81 | 0,572 → 0,550 |

`unet3d`'de her metrik biraz iyileşti; `unet_r34_2d`'de Dice sabit kaldı, HD95 ve lezyon F1
kötüleşti. **Eşik ayarı anlamlı bir kaldıraç değil** — val'de seçilen değer bir modelde
transfer oldu, diğerinde olmadı.

### Küçük bileşen eleme — **hiç eleme yapılmayacak** (Bölüm 2, karar 4 kapandı)

Her iki modelde de `min_voxel = 0` seçildi; eleme Dice'ı düşürüyor.

| min_voxel | `unet3d` Dice | `unet_r34_2d` Dice | `unet_r34_2d` lezyon F1 |
|---|---|---|---|
| **0** | **0,7100** | **0,6809** | 0,4947 |
| 5 | 0,7099 | 0,6756 | **0,5552** |
| 10 | 0,7094 | 0,6748 | 0,5382 |
| 20 | 0,7002 | 0,6762 | 0,5189 |
| 50 | 0,6711 | 0,6301 | 0,4199 |

Parça 1'deki öngörü doğrulandı: GT'nin kendisinde 897 adet 5-voxel altı bileşen olduğu için
elemek gerçek lezyonları da siliyor. **Gizli bir takas var**: `unet_r34_2d`'de `min_voxel = 5`
Dice'ı 0,005 düşürürken lezyon F1'i **0,060 yükseltiyor**. Dice'a göre seçim yaptığımız için
eleme yok; lezyon tespiti öncelik olsaydı seçim farklı olurdu. Bu, raporda tek metrikle seçim
yapmanın maliyetini gösteren bir örnek olarak duracak.

### Bootstrap CI — mutlak değerler geniş, eşleştirilmiş farklar dar

| model | test Dice | %95 CI | fark | fark %95 CI | Wilcoxon p | Holm |
|---|---|---|---|---|---|---|
| `unet3d` | 0,7023 | [0,634 – 0,761] | +0,0058 | [−0,040 – +0,046] | 0,084 | değil |
| `unet_r34_2d` | 0,6965 | [0,640 – 0,745] | — | — | — | — |
| `A3_k1` | 0,6896 | [0,628 – 0,743] | −0,0069 | [−0,034 – +0,018] | 0,665 | değil |
| `A2_tversky` | 0,6806 | [0,619 – 0,734] | −0,0159 | [−0,048 – +0,014] | 0,171 | değil |
| `A3_k3` | 0,6796 | [0,621 – 0,733] | −0,0169 | [−0,038 – +0,003] | 0,084 | değil |
| `unet_mbv3` | 0,6744 | [0,614 – 0,729] | −0,0221 | [−0,051 – **+0,006**] | 0,005 | **ANLAMLI** |
| `unet_r34_25d` | 0,6652 | [0,597 – 0,724] | −0,0313 | [−0,066 – **−0,002**] | 0,025 | değil |
| `A1_flair` | 0,6568 | [0,589 – 0,718] | −0,0397 | [−0,080 – −0,004] | 0,001 | **ANLAMLI** |
| `segformer_b0` | 0,6398 | [0,577 – 0,696] | −0,0567 | [−0,084 – −0,030] | <0,001 | **ANLAMLI** |

**Mutlak CI'lar birbirini tamamen örtüyor** — dokuz modelin CI'ı [0,58 – 0,76] bandında iç içe.
Eşleştirilmiş fark CI'ları çok daha dar; eşleştirmenin gücü burada da görünüyor.

> **İki prosedür iki modelde çelişiyor, ters yönlerde.** `unet_mbv3`: Holm anlamlı diyor
> (p = 0,005) ama fark CI'ı sıfırı içeriyor. `unet_r34_25d`: fark CI'ı sıfırı içermiyor ama
> Holm anlamlı demiyor (p = 0,025 > eşik 0,010). Sebep — Wilcoxon **sıra tabanlı** (tutarlı
> işaret örüntüsüne duyarlı: 36/13), bootstrap CI ise **ortalama** üzerinde (birkaç uç vakadan
> etkileniyor); ayrıca CI çokluk için düzeltilmemiş. **Sonuç: bu iki model "sınırda" sayılacak,
> tek bir prosedüre dayanarak anlamlı ilan edilmeyecek.**

**Tek kesin sonuç `segformer_b0`**: hem Holm'u geçiyor, hem fark CI'ı sıfırı içermiyor, hem de
etkisi (−0,057) koşu varyansı bandının (0,024) iki katından fazla. `A1_flair` (−0,040) her iki
ölçütte de anlamlı ama bandın hemen üstünde.

### Lezyon F1 tanıma aşırı duyarlı — **rapor için kritik uyarı**

`unet3d` üzerinde IoU eşleştirme eşiği değiştirilince:

| IoU eşiği | lezyon F1 |
|---|---|
| > 0 (herhangi bir örtüşme) | **0,6088** |
| 0,10 (bizim varsayılan) | 0,5811 |
| 0,25 | 0,5274 |
| 0,50 | **0,3994** |

**Aralık 0,209.** Yani raporladığımız lezyon F1 değerleri (0,44–0,58) tamamen "IoU ≥ 0,1"
seçimine bağlı. Bu, DeepISLES'in yayınladığı **0,86** lezyon F1 ile karşılaştırmayı
anlamsızlaştırıyor: eşleştirme kuralları bilinmeden iki sayı yan yana konamaz. Raporda
lezyon F1 **daima eşleştirme eşiğiyle birlikte** verilecek ve literatürle doğrudan
karşılaştırılmayacak.

### `panoptica` çapraz doğrulaması — **korelasyon 1,000, değerler birebir aynı**

İlk denemede `Panoptica_Evaluator` eşleştirme algoritması istediği için hata vermişti
(`Got UnmatchedInstancePair but not InstanceMatchingAlgorithm`);
`NaiveThresholdMatching(matching_threshold=LEZYON_IOU_ESIK)` eklenip koşuldu.

10 test vakasında `panoptica`'nın RQ'su ile bizim `lezyon_f1`'imiz **ondalık basamağına
kadar aynı** çıktı:

| vaka | panoptica RQ | bizim |
|---|---|---|
| 0001 | 0,6939 | 0,6939 |
| 0004 | 0,7143 | 0,7143 |
| 0007 | 0,0000 | 0,0000 |
| 0020 | 0,8571 | 0,8571 |
| 0022 | 1,0000 | 1,0000 |
| 0031 | 0,4242 | 0,4242 |
| **ortalama** | **0,5478** | **0,5478** |

**Korelasyon 1,000.** Yani kendi hızlı uygulamamız ISLES'22'nin resmi aracıyla aynı tanımı
hesaplıyor — "benzer" değil, özdeş. `degerlendirme_ozeti.json` içinde
`panoptica_dogrulandi: true`.

> Bu, Bölüm 7b'deki IoU duyarlılığı uyarısını **ortadan kaldırmıyor, yerini değiştiriyor**:
> uygulamamız doğru, ama sonuç hâlâ eşleştirme eşiğine bağlı (0,40–0,61). Fark şu ki artık
> bu bağımlılık bizim kodumuzun değil, **metriğin kendisinin** özelliği — panoptica da aynı
> eşiği alıyor. Raporda lezyon F1 yine eşikle birlikte verilecek.

---

## 6. Optimizasyon ve Servis

Bu bölüm mimari tarafıyla **eşit ağırlıkta** yürüyecek. Sekiz basamaklı bir merdiven; her basamak
**hem gecikmeyi hem Dice'ı** raporlar. Hız kazancı ΔDice'la birlikte yazılmazsa tablo anlamsız —
Jetson çalışmasında (Bölüm 3) TensorRT FP16 Dice'ı 0,78'den 0,67'ye düşürmüştü.

Merdiven, en iyi hızlı model + `unet_mbv3` üzerinde koşulur (2 model × 8 basamak):

| # | Adım | Beklenen etki | Doğrulama |
|---|---|---|---|
| 0 | PyTorch FP32, batch=1 | Taban çizgisi | — |
| 1 | `torch.compile` | %10–30 | Çıktı özdeşliği |
| 2 | **ONNX export + onnxruntime (CUDA EP)** | Çerçeve yükü kalkar (BUSI'de 2–3×) | PyTorch çıktısına **bağıl** sapma, eşik 1e-4 |
| 3 | ONNX **FP16** | ~1,5–2× | ΔDice ≤ 0,002 beklenir, ölçülür |
| 4 | **TensorRT FP16** (2060 Turing destekliyor) | En düşük gecikme | Sayısal doğrulama aynı |
| 5 | **INT8 PTQ** (kalibrasyon: 20 train vakası) | ~2× ek | ΔDice raporlanır, gizlenmez |
| 6 | **Structured pruning** (kanal %30 / %50) + ince ayar | Model boyutu ↓ | İnce ayar sonrası Dice |
| 7 | **ONNX CPU** (dağıtım alt sınırı) | GPU'suz senaryo | StrokeSeg'te 165× fark vardı; bizde ne? |

Her satır için kolonlar: `ms/vaka`, `ms/dilim`, `VRAM`, `model_boyutu_MB`, `dice`, `Δdice`,
`hedefi_tutturdu_mu`. Sonuç grafiği: **Dice–gecikme Pareto eğrisi**, üzerinde ≤ 2 s ve ≤ 1 s
çizgileri ve literatür referansı (DeepISLES ~120 s / Dice 0,82).

Ayrıca ölçülecek iki mimari-dışı kazanç — 3B hacimde asıl maliyet bunlarda olabilir:

- **Dilim batch'leme**: hacmin tüm dilimlerini tek batch'te geçirmek vs. tek tek. Batch=1 gerçek
  zamanlı senaryonun kötü durumu, ama hacim zaten elde → batch kullanmak meşru.
- **Beyin maskesiyle boş dilim atlama**: lezyon içermesi imkânsız dilimleri ağa hiç sokmamak.
  Kazancı Parça 1'deki boş dilim oranıyla doğrudan orantılı.

### Parça 8 ön ölçümü — **merdivenin hedefi yanlış yerdeydi**

Yerel RTX 2060'ta, ham NIfTI'den maskeye tam hat ölçüldü (8 vaka, 3 ısınma atıldı,
`cuda.synchronize` ile). Ağırlıklar henüz Drive'da olduğu için rastgele — **ama gecikme
ağırlıklara değil mimariye bağlı**, dolayısıyla bu sayılar gerçek ölçümün önizlemesi:

| model | ön işleme | ağ | son işleme | **TOPLAM** | okuma (bütçe dışı) | VRAM |
|---|---|---|---|---|---|---|
| `unet3d` | 233,0 ms | 78,3 ms | 0,7 ms | **312,1 ms** | 63,5 ms | 0,46 GB |
| `unet_r34_2d` | 241,3 ms | 72,0 ms | 0,6 ms | **313,9 ms** | 32,1 ms | 0,26 GB |

**İki sonuç, ikisi de planı düzeltiyor.**

**1. Hedef zaten tutuyor — ve ağ optimizasyonu neredeyse anlamsız.** 312 ms, Bölüm 3'teki
"≤ 2 s zorunlu / ≤ 1 s hedeflenen" bandının çok altında. Ama asıl mesele dağılım: **ön işleme
uçtan uca bütçenin %75'i, ağ yalnız %23'ü.** Bölüm 6'daki sekiz basamaklı merdiven (ONNX →
FP16 → TensorRT → INT8 → pruning) tamamen ağı hedefliyor; ağ **sıfır** sürseydi bile toplam
312 → 234 ms olurdu, yani en fazla %25 kazanç. Merdiven yanlış yere kuruluydu.

Ön işlemenin kırılımı (10 vaka):

| adım | süre | pay |
|---|---|---|
| **beyin maskesi** | 76,9 ms | %33,5 |
| **resample DWI** | 72,1 ms | %31,4 |
| **resample ADC** | 71,4 ms | %31,1 |
| normalizasyon | 5,1 ms | %2,2 |
| kırpma | 4,1 ms | %1,8 |

Üç adım bütçenin %96'sı ve **üçü de CPU'da, scipy/nibabel ile** çalışıyor. Gerçek kazanç
burada: iki resample GPU'da yapılabilir (`torch.nn.functional.grid_sample`), beyin maskesindeki
`binary_closing` + `fill_holes` + bağlı bileşen de GPU'ya taşınabilir ya da daha ucuz bir
eşikleme ile değiştirilebilir. **Parça 8'in merdivenine bu iki basamak eklenecek** ve ağ
basamakları (INT8, pruning) düşük önceliğe inecek.

**2. `unet3d`'nin hız üstünlüğü ortadan kalktı.** Colab'da 3B model 2B'den hızlı görünüyordu
(4 sn vs 6 sn / 50 vaka, Bölüm 5f). Yerel tam hatta ikisi **aynı** (312 vs 314 ms) ve ağ
süreleri de yakın (78 vs 72 ms). Colab ölçümü ön işlemeyi dışarıda bırakıyordu ve `.npy`
önbellekten okuyordu; oradaki fark hattın kendisinden geliyormuş, mimariden değil.

> Bölüm 5f'deki "3B model hem en küçük hem en hızlı" cümlesi bu ölçümle **düzeltiliyor**:
> en küçük olduğu doğru (1,40 M, ÷17), en hızlı olduğu **doğru değil** — uçtan uca fark yok.
> `unet3d` yine de tercih edilebilir çünkü 17 kat küçük ve nominal Dice'ı biraz yüksek; ama
> gerekçe hız değil.

### GPU ön işleme — merdivenin yeni basamağı, **2,7× uçtan uca kazanç**

Ön ölçüm bütçenin %75'inin ön işlemede olduğunu gösterince (`resample` 143 ms + beyin
maskesi 77 ms) o adımlar `ortak/hat_gpu.py`'ye taşındı.

**`resample_gpu`** — nibabel'in `resample_to_output`'unun `grid_sample` karşılığı. Hedef grid
(şekil + affine) yine nibabel'in `vox2out_vox`'undan alınıyor, yani **grid birebir aynı**;
değişen yalnız örneklemenin nerede yapıldığı. 12 vakada bağıl sapma **5e-6 – 1,3e-5**, yani
fp32 yuvarlama düzeyi.

> Bir tuzak çıktı ve düzeltildi: `scipy.affine_transform(mode='constant', cval=0)` girdi
> sınırının dışındaki her noktaya 0 verirken `grid_sample` kenarda iç değerle sıfırı
> harmanlıyor. **Oblik affine'li** vakalarda (0003'ün affine'i belirgin rotasyonlu) bu, hacim
> sınırındaki 114 voxel'de yüzlerce birimlik fark üretiyordu — bağıl sapma 0,37. Geçerlilik
> maskesi iki davranışı eşitledi.

**`beyin_maskesi_gpu`** — eşikleme ve morfolojik kapama GPU'da (`max_pool3d` ile dilate,
`-max_pool3d(-x)` ile erode). Delik doldurma ve en büyük bileşen etiketleme GPU'da ucuz
karşılığı olmadığı için **tam çözünürlükte CPU'da** bırakıldı: 12 vakada **0 farklı voxel**,
maliyet 74 → 40 ms.

> İlk denemede bu iki adım 2 kat alt örneklenmiş maskede yapılıyordu (8 kat az voxel). Daha
> hızlıydı **ama maskeyi değiştirip kırpma kutusunu 1 voxel kaydırıyordu** ve 15 vakanın
> 8'inde çıktı şekli CPU sürümüyle uyuşmuyordu. Hız uğruna sonucu değiştirmek kazanç değil
> hata olur; alt örnekleme geri alındı.

**Kalan fark — ölçüldü ve kabul edildi.** 20 vakanın 19'unda çıktı bit düzeyinde aynı
(ortalama fark 1e-6). Bir vakada (`sub-strokecase0014`) **2 voxel** farklı (%0,0004): beyin
maskesi eşiği 5,2e-7 bağıl kayıyor ve tam eşikte duran 2 voxel taraf değiştiriyor. Kırpma
kutusu aynı. `hiz_olcumu.py`'nin Dice kilidi (tolerans 2e-3) nihai hakem olacak.

### Uçtan uca sonuç — **gerçek ağırlıklar**, RTX 2060, 47 test vakası

`dagitim/` paketiyle (eğitilmiş `best.pt`'ler + seçilen eşikler) tam ölçüm:

| model | ön işleme | ön işleme | ağ | son işleme | **TOPLAM** | ağ payı | okuma |
|---|---|---|---|---|---|---|---|
| `unet3d` | CPU | 249,1 ms | 81,6 ms | 0,7 ms | **331,4 ms** | %25 | 62,0 ms |
| `unet3d` | **GPU** | **60,1 ms** | 64,8 ms | 1,3 ms | **126,2 ms** | %51 | 29,7 ms |
| `unet_r34_2d` | CPU | 251,6 ms | 73,5 ms | 0,7 ms | **325,7 ms** | %23 | 29,7 ms |
| `unet_r34_2d` | **GPU** | **61,5 ms** | 57,0 ms | 0,8 ms | **119,3 ms** | %48 | 26,3 ms |

**Kalite kilidi geçti.** Yerel hattın ürettiği maskeler notebook'un kaydettiği vaka başına
referans Dice'larla karşılaştırıldı: **max sapma 0,0017**, tolerans 2e-3. Yani hızı ölçülen
kod, kalitesi ölçülen kodla aynı şeyi üretiyor — BUSI projesindeki disiplin burada da tutuyor.

> Sapmanın kaynağı iki kalem: (a) GPU ön işlemede eşikte duran birkaç voxel'in taraf
> değiştirmesi (Bölüm 6, 2 voxel/vaka), (b) **notebook `.npy` önbelleğini float16 tutuyordu**,
> yerel hat ise float32 hesaplıyor. CPU ön işlemede de 0,0009–0,0017 sapma görülmesi bunu
> doğruluyor — fark GPU'dan değil, önbelleğin hassasiyetinden geliyor.

Yerel ortalama Dice (0,721 / 0,709) notebook'un test değerlerinden (0,706 / 0,696) biraz
yüksek: ölçüm **47 vaka** üzerinde (ilk 3 ısınma olarak atılıyor), notebook 50 vakanın
tamamını kullanmıştı. Vaka başına sapma yukarıdaki tolerans içinde.

### Sonuçlar

**1. Hedef fazlasıyla tutuyor.** 119–126 ms, "≤ 2 s zorunlu / ≤ 1 s hedeflenen" bandının
sekizde biri. Bölüm 3'teki hedefler bu donanımda kolay çıktı.

**2. GPU ön işleme 2,6–2,7× kazandırdı**, ağa hiç dokunmadan. Ağın payı %23–25'ten %48–51'e
çıktı; artık iki bileşen dengeli.

**3. `unet3d` ile `unet_r34_2d` arasında uçtan uca anlamlı fark yok** (126 vs 119 ms).
Bölüm 5f'de Colab ölçümüne dayanarak "3B model en hızlı" denmişti; **bu doğru değil**.
O ölçüm `.npy` önbellekten okuyor ve ön işlemeyi dışarıda bırakıyordu. `unet3d`'nin gerçek
üstünlüğü **17 kat küçük olması** (1,40 M vs 24,43 M) ve VRAM'i; hız değil.

**Ne kaldı:** ön işlemenin 60 ms'sinin ~40 ms'si hâlâ CPU'daki delik doldurma + bağlı
bileşen. Ağ basamakları (ONNX/FP16/TensorRT) toplamın ~%50'sini hedefliyor.

### ONNX ve FP16 basamakları — **ölçüldü**

| model | hassasiyet | çalışma zamanı | ağ ms | PyTorch ms | hızlanma | bağıl sapma |
|---|---|---|---|---|---|---|
| `unet3d` | fp32 | ONNX CUDA | 61,1 | 63,9 | **1,05×** | 2,0e-6 |
| `unet3d` | fp32 | ONNX CPU | 723,8 | 63,9 | 0,09× | 1,4e-5 |
| `unet3d` | **fp16** | ONNX CUDA | **24,9** | 26,6 | 1,07× | 1,6e-3 |
| `unet_r34_2d` | fp32 | ONNX CUDA | 20,5 | 19,9 | **0,98×** | 5,0e-6 |
| `unet_r34_2d` | fp32 | ONNX CPU | 236,2 | 19,9 | 0,08× | 4,6e-6 |
| `unet_r34_2d` | **fp16** | ONNX CUDA | **12,0** | 11,6 | 0,97× | 5,1e-3 |

**ONNX bu projede hiç kazandırmıyor** (1,05× ve 0,98×). BUSI projesinde ONNX 2–3× hızlandırmıştı
ve sıralamayı tersine çevirmişti; oradaki kazanç **kare başına Python/çerçeve yükünü** kaldırmaktan
geliyordu. Burada öyle bir yük yok: 3B model vaka başına **tek** ileri geçiş yapıyor, 2B model
batch 32 ile ~3 geçiş. Çerçeve yükü zaten amorti edilmiş durumda. Merdivenin 2. basamağı
bu görev için ölü.

**FP16 gerçek kaldıraç.** Uçtan uca ölçüm (25 vaka, GPU ön işleme):

| model | hassasiyet | ön işleme | ağ | **TOPLAM** | Dice | Dice sapması |
|---|---|---|---|---|---|---|
| `unet3d` | fp32 | 60,3 | 63,5 | **125,0 ms** | 0,6729 | 0,00092 |
| `unet3d` | **fp16** | 60,5 | **32,3** | **94,0 ms** | 0,6728 | 0,00043 |
| `unet_r34_2d` | fp32 | 64,8 | 55,8 | **121,4 ms** | 0,6963 | 0,00111 |
| `unet_r34_2d` | **fp16** | 64,2 | **43,1** | **108,1 ms** | 0,6962 | 0,00018 |

Ağ süresi `unet3d`'de **1,97×**, `unet_r34_2d`'de 1,30× hızlandı. **Dice farkı 0,0001'in
altında** — planın öngördüğü "ΔDice ≤ 0,002" fazlasıyla tutuyor. Üstelik fp16'nın referansa
sapması fp32'ninkinden *daha küçük*; iki değer de ölçüm gürültüsü içinde.

> ONNX tablosundaki fp16 "bağıl sapma" değerleri (1,6e-3 ve 5,1e-3) fp32 toleransını aşıyor
> ama bu bir sorun değil: fp16'nın kendi hassasiyeti zaten ~1e-3. Doğru ölçüt logit sapması
> değil **Dice'a etkisi**, o da yukarıda ölçüldü. Tolerans hassasiyete göre ayrıldı.

### Ölçümü neredeyse boşa çıkaran bir tuzak

İlk ONNX koşusu "ONNX PyTorch'tan 10–50 kat **yavaş**" gibi anlamsız sayılar üretti ve
CUDA ile CPU aynı çıktı. Sebep: **`onnxruntime-gpu 1.29` CUDA 13 istiyor**, kurulu torch ise
cu126 — `cublasLt64_13.dll` eksikti ve ORT istenen sağlayıcıyı yükleyemeyince **sessizce
CPU'ya düştü**. Hiçbir hata mesajı yoktu.

İki düzeltme: `onnxruntime-gpu` 1.22'ye (CUDA 12 derlemesi) indirildi ve torch'un `lib/`
klasörü DLL arama yoluna eklendi. Ayrıca script artık **gerçekten kullanılan sağlayıcıyı**
kontrol ediyor; istenen yüklenemezse ölçümü yapmadan atlıyor ve tabloya `saglayici` kolonu
yazıyor. Sessiz geri düşüş bir daha sayı üretmeyecek.

### Merdivenin kalan basamakları — **tamamlandı**

İki basamak ilk denemede ortam eksiğinden düşmüştü; ikisi de kurulup ölçüldü.

**`torch.compile`** — `TritonMissing` hatası veriyordu (Triton Windows'ta torch ile gelmiyor).
`triton-windows` kurulunca çalıştı ve **modele göre ters yönde** sonuç verdi:

| model | fp32 | fp32 + compile | fp16 | **fp16 + compile** |
|---|---|---|---|---|
| `unet3d` | 63,3 ms | 52,1 ms (1,22×) | 26,6 ms | **16,3 ms (3,90×)** |
| `unet_r34_2d` | 20,4 ms | **37,3 ms (0,55×)** | 11,6 ms | **7,3 ms (2,78×)** |

`unet_r34_2d`'de compile **tek başına 1,8 kat yavaşlatıyor** ama fp16 ile birlikte
hızlandırıyor. Kazançlar çarpımsal değil, birbirine bağlı — basamakları ayrı ayrı ölçüp
toplamak yanlış olurdu.

**TensorRT** — üç ayrı sürüm uyuşmazlığı aşıldı. ONNX Runtime 1.29 CUDA 13 istiyordu,
pip'ten gelen `tensorrt` 11.2 de cu13'tü; kurulu torch ise cu126. Zincir
**ORT 1.22 (CUDA 12) + TensorRT 10.13 (cu12) + torch cu126** olarak hizalandı ve sağlayıcı
yüklendi.

| model | hassasiyet | CUDA EP | **TensorRT EP** | motor derlemesi |
|---|---|---|---|---|
| `unet3d` | fp32 | 59,7 ms | **45,7 ms** | 19 sn |
| `unet3d` | **fp16** | 24,7 ms | **14,3 ms** | 35 sn |
| `unet_r34_2d` | fp32 | 20,5 ms | **17,5 ms** | 2 sn |
| `unet_r34_2d` | **fp16** | 11,7 ms | **4,8 ms** | 1 sn |

**TensorRT + fp16 merdivenin en hızlısı** ve `torch.compile + fp16`'yı da geçiyor
(14,3 vs 16,3 ms · 4,8 vs 7,3 ms). Motor derlemesi tek seferlik ve önbelleğe alınıyor
(`onnx/trt_cache/`), servis açılışında ödenecek bir maliyet.

### Merdivenin tamamı

| # | basamak | `unet3d` ağ | `unet_r34_2d` ağ | durum |
|---|---|---|---|---|
| 0 | PyTorch fp32 | 63,3 ms | 20,4 ms | taban |
| — | **GPU ön işleme** | — | — | **331 → 125 ms uçtan uca (2,6×)** |
| 1 | `torch.compile` | 52,1 ms (1,22×) | 37,3 ms (**0,55×**) | modele göre değişiyor |
| 2 | ONNX (CUDA EP) | 61,1 ms (1,05×) | 20,5 ms (0,98×) | **kazanç yok** |
| 3 | fp16 | 26,6 ms (2,38×) | 11,6 ms (1,76×) | ΔDice < 0,0001 |
| 3+1 | fp16 + compile | 16,3 ms (3,90×) | 7,3 ms (2,78×) | — |
| 4 | **TensorRT fp16** | **14,3 ms (4,43×)** | **4,8 ms (4,23×)** | **en hızlı** |
| 5 | INT8 PTQ | ölçülmedi | ölçülmedi | aşağıya bkz. |
| 6 | Pruning | ölçülmedi | ölçülmedi | aşağıya bkz. |
| 7 | ONNX CPU | 723,8 ms | 236,2 ms | 12× / 6× **yavaş** |

### INT8 ve pruning neden ölçülmedi — gerekçe

Bu iki basamak bilerek kapatıldı, unutulmadı. Gerekçe ölçüme dayanıyor: TensorRT fp16'dan
sonra **ağ, uçtan uca bütçenin yalnız %19'u** (`unet3d`: 60 ms ön işleme + 14 ms ağ + 1 ms
son işleme ≈ 75 ms). Ağı tamamen sıfırlasak toplam 75 → 61 ms olurdu; INT8'in gerçekçi
kazancı bunun bir kısmı, yani **birkaç milisaniye**.

Karşı tarafta somut bir risk var: Bölüm 3'te alıntılanan Jetson çalışmasında TensorRT INT8
Dice'ı 0,78'den 0,67'ye düşürmüştü. Kalibrasyon verisi hazırlamak, Dice kaybını ölçmek ve
raporlamak gereken iş, kazanılacak birkaç ms'nin çok üstünde. **Hedef zaten 13 kat aşılmış
durumda** (75 ms vs ≤ 1 s).

Aynı gerekçe pruning için de geçerli, üstelik pruning yeniden ince ayar gerektiriyor —
yani dokuz koşuluk protokolün dışına çıkmak demek.

### En iyi konfigürasyon

| bileşen | süre |
|---|---|
| GPU ön işleme | 60,1 ms |
| **TensorRT fp16 ağ** (`unet3d`) | **14,3 ms** |
| son işleme | 1,1 ms |
| **TOPLAM** | **≈ 75 ms/vaka** |

Hedeflenen ≤ 1 s'in **on üçte biri**, zorunlu ≤ 2 s'in yirmi altıda biri. Karşılaştırma:
DeepISLES ~120.000 ms (RTX 3090, literatür).

> Bu toplam **bileşenlerden derlendi**, tek bir uçtan uca koşuda ölçülmedi: ön işleme ve son
> işleme `hiz_olcumu.py`'den (47 vaka, Dice kilidi geçmiş), ağ süresi TensorRT ölçümünden
> (30 tekrar, ısınmalı). TensorRT'yi uçtan uca hatta bağlamak servis işi (Parça 9) ve orada
> yapılacak; o zaman tek koşuda doğrulanacak.

**Ön işleme artık bütçenin %80'i.** Daha ileri gidilecekse hedef orası: kalan 60 ms'nin
~40 ms'si hâlâ CPU'daki delik doldurma + bağlı bileşen etiketleme.

## Parça 9 — Servis **TAMAM**

`servis/` altında üç modül: `motor.py` (çıkarım motoru, arka uç seçilebilir), `api.py`
(FastAPI), `rapor.py` (tek dosyalık HTML klinik rapor). Doğrulama: `servis/dogrula.py`.

### Uçtan uca doğrulama — dört arka uç, tek koşuda

Parça 8'de "75 ms" **bileşenlerden derlenmişti** (ön işleme bir ölçümden, ağ başkasından).
Burada tek hatta, tek koşuda ve **kalite kilidiyle** ölçüldü (12 vaka, 3 ısınma atıldı):

| arka uç | ön işleme | ağ | son işleme | **TOPLAM** | Dice sapması | kilit |
|---|---|---|---|---|---|---|
| `torch-fp32` | 59,4 | 64,6 | 7,0 | 131,0 ms | 0,00015 | GEÇTİ |
| **`torch-fp16`** | 59,7 | 35,0 | 7,5 | **102,1 ms** | 0,00043 | GEÇTİ |
| `onnx-cuda` | 63,8 | 68,6 | 7,1 | 139,4 ms | 0,00015 | GEÇTİ |
| `trt-fp16` | 66,0 | 34,9 | 7,3 | 108,1 ms | 0,00089 | GEÇTİ |

Dördü de kalite kilidini geçti (tolerans 2e-3), yani hızlı ama farklı bir şey üreten bir
yol dağıtıma girmiyor.

### TensorRT'nin üstünlüğü ölçüm artefaktıymış

Parça 8'in **izole** ağ ölçümünde `unet3d` fp16 için TensorRT 14,3 ms, torch 26,6 ms idi —
TensorRT 1,9 kat önde görünüyordu. Gerçek hatta ikisi de **34,9 / 35,0 ms**, yani fark yok;
üstelik TensorRT'nin oturum yükü yüzünden toplam 6 ms daha kötü.

Sebep: izole ölçümde aynı numpy dizisi 30 kez arka arkaya besleniyordu; onnxruntime giriş
bağlamalarını ve cihaz tamponlarını amorti ediyordu. Vaka başına taze dizilerle bu kazanç
kayboluyor.

> BUSI projesindeki ders burada **tersine** tekrarlandı. Orada ONNX çerçeve yükünü kaldırıp
> sıralamayı değiştirmişti; burada izole ölçüm TensorRT lehine yanıltıcı çıktı. Ortak nokta:
> **izole ağ ölçümleri dağıtılabilir sayı değildir.** Karar her iki projede de uçtan uca
> ölçümle verildi.

**Sonuç: dağıtım `torch-fp16` ile yapılacak.** Aynı hız, çok daha basit yığın — ONNX export,
TensorRT sürüm matrisi (ORT 1.22 ↔ CUDA 12 ↔ TensorRT 10.13) ve motor önbelleği gereksiz.
ONNX/TensorRT yolu `motor.py` içinde duruyor, karşılaştırma için çalışır durumda.

### Soğuk başlangıç — ilk hasta bedeli ödemesin

İlk gerçek istek **331 ms** sürüyordu (ağ 168 + ön işleme 157), kararlı hal 102 ms.
Açılışa ısıtma eklendi ve iki aşamada düzeltildi:

| ısıtma | ilk istek |
|---|---|
| yok | 331 ms |
| yalnız ağ | 194 ms |
| **tüm hat** (sentetik NIfTI ile ön işleme dahil) | **85 ms** |

Yalnız ağı ısıtmak yetmiyordu: `grid_sample` ve `max_pool3d` çekirdekleri de ilk
çağrılarında yavaş. Isıtma artık sentetik bir hacmi baştan sona geçiriyor.

Ayrıca **GPU→CPU→GPU gidiş dönüşü kaldırıldı**: ön işleme GPU'da bitip CPU'ya iniyor, ağ
için tekrar GPU'ya çıkıyordu. 3B modelde dolgu tek işlem olduğu için girdi GPU'da
kurulabiliyor — ön işleme 65,6 → 59,4 ms.

### Uç noktalar

| uç nokta | ne döner |
|---|---|
| `GET /saglik` | motor hazır mı, hangi model/arka uç/cihaz, seçilen eşik |
| `POST /segment` | klinik çıktı + **bileşen bazlı süre** (JSON) |
| `POST /rapor` | tek dosyalık HTML rapor (görseller base64 gömülü) |
| `POST /maske` | maske NIfTI (2 mm grid) |

Hepsi gerçek bir vakayla test edildi; hatalı girdi (NIfTI olmayan dosya) 400 ile
reddediliyor. Motor açılışta **bir kez** yükleniyor — her istekte model yüklemek gecikmeyi
on kat artırırdı.

### Klinik çıktı

`sub-strokecase0001` için: hacim **4,01 mL**, 20 bağlı lezyon, en büyük 0,88 mL,
hemisfer **bilateral** (sağ oran 0,295).

Rapor (`servis/ornek_rapor.html`, 127 KB) şunları içeriyor: klinik özet tablosu, vakanın
hacminin **250 vakalık dağılım içindeki konumu**, 9 dilimlik overlay mozaiği, bileşen bazlı
süre tablosu ve sınırlılıklar bölümü. Her rapor tıbbi tanı aracı olmadığı uyarısıyla
başlıyor; sınırlılıklar arasında core/penumbra ayrımının yapılmadığı, hacmin ±%2,3 resample
farkı taşıdığı ve **küçük lezyonlarda Dice'ın ~0,55'e düştüğü** açıkça yazılı.

**Servis notu:** ONNX Runtime + FastAPI. Uç nokta NIfTI (veya DICOM serisi) alır, döndürdüğü şey:
maske NIfTI + lezyon hacmi + lezyon sayısı + overlay PNG'leri + rapor. Servis içinde gecikme
bileşenlere ayrılmış olarak loglanır (okuma / ön işleme / ağ / son işleme) — hangi basamağın
darboğaz olduğu tahmin edilmeyecek, ölçülecek.

---

## 7. Klinik Çıktı

- **Lezyon hacmi (mL)** — voxel sayısı × voxel hacmi (NIfTI affine üzerinden). GT hacmiyle
  birlikte Bland-Altman ve korelasyon grafiği verilir.
- **Lezyon sayısı** ve en büyük lezyonun hacmi (bağlantılı bileşen analizi).
- **Hemisfer** (sol/sağ/bilateral) ve kabaca sulama alanı yorumu.
- **Overlay**: aksiyel mozaik — DWI arka plan, GT yeşil kontur, tahmin kırmızı dolgu; lezyon
  merkezinden geçen dilimler seçilir.
- **Vaka raporu**: tek dosyalık HTML → Chrome headless PDF (Bölüm 8'deki rapor deseni).

> **Üretilmeyen klinik büyüklük:** core/penumbra mismatch oranı. Perfüzyon verisi olmadığı
> için ISLES'22'den çıkarılamıyor; ek veri seti de kapsam dışı bırakıldı (Bölüm 2).

> Rapor tıbbi tanı aracı değildir; her çıktıya bu uyarı ve modelin test seti performans
> aralığı (CI ile) yazılır.

---

## 8. Kod Yapısı

`mendeley_data_density` ve `multiple_instance_classifier` deseni: **notebook-başına-deney**,
parametrik tek şablon. Colab'da dosya senkronizasyonu gerektirmiyor.

| Dosya | Ne |
|---|---|
| `isles_veri_kesif.ipynb` | Parça 1 — envanter, ölçümler, `ozet/` çıktıları |
| `isles_veri_hazirlik.ipynb` | Parça 2 — ön işleme, `.npy` önbellek, `split.json`, `eval_cases.json` |
| `isles_degerlendirme.ipynb` | Parça 7 — bağımsız. Gereken tanımlar `isles_egitim.ipynb`'den **programlı olarak** kopyalandı (elle taşınmadı, ikisi ayrışamaz); eğitim hücresi yok, yanlışlıkla yeniden eğitme riski yok. Baştan sona çalıştırılır |
| `isles_flair_dogrulama.ipynb` | Parça 2b'nin tamamlanmamış iki çıktısı — bindirme görseli ve hizalama maliyeti tablosu. Bağımsız |
| `isles_dagitim.ipynb` | Parça 8 hazırlığı — çıkarım-only ağırlıkları + `olcum_referansi.json` + `dagitim.zip` üretir. Eğitim notebook'undan **ayrı**: işi paketlemek, eğitimle ilgisi yok, ve o dosya zaten 67 hücre. `MODEL_CONFIGS` tekrarlandığı için state_dict'in ilk katman şekli konfigle karşılaştırılıyor — ayrışırsa hata verir |
| `isles_egitim.ipynb` | Parça 3–7 — model fabrikası, log altyapısı, eğitim döngüsü, değerlendirme; `MODEL_KEY` ile model seçilir. Parça 7 ayrı notebook'a alınmadı çünkü `konfig`/`model_yap`/`vaka_tahmin`/`vaka_metrikleri` tanımlarını olduğu gibi kullanıyor; ayırmak ~200 satır kopyalamak olurdu |
| `isles_egitim_<MODEL_KEY>.ipynb` | Biten her koşunun çıktıları gömülü arşiv kopyası |
| `ortak/modeller.py`, `ortak/hat.py`, `ortak/hat_gpu.py` | Parça 8 — yerel ölçüm için model tanımları ve çıkarım hattı (CPU + GPU ön işleme) |
| `hiz_olcumu.py`, `onnx_olcumu.py` | Parça 8 — yerel RTX 2060'ta gecikme ve ONNX ölçümü |
| `servis/` | Parça 9 — FastAPI + ONNX Runtime |
| `rapor/build_report.py`, `rapor/pdf_olustur.ps1` | Parça 10 — base64 gömülü tek HTML → PDF |

> **Senkronizasyon notu** (diğer projelerde tekrar tekrar sorun oldu): bu repodaki `.ipynb` ile
> Colab'da fiilen çalışan oturum ayrı şeyler. Buradaki değişiklikler Colab'a otomatik yansımaz.

---

## 9. Loglama Standardı — Diğer Projelerle Birebir Aynı

**Excel tabanlı epoch logu + Drive kalıcı depo + yerel mirror + resume-safe checkpoint.**

### Klasör yapısı (Drive, kalıcı)

Tek kok: **`/content/drive/MyDrive/iskemik_inme/`** — ham veri, hazırlanmış önbellek, loglar,
checkpointler ve özetler hepsi burada (kullanıcı kararı). Kod bu yolu `PROJE` değişkeninden alır,
hiçbir yere sabit yol yazılmaz.

```
/content/drive/MyDrive/iskemik_inme/
├── ISLES-2022.zip                      # kaynak arsiv (1,69 GB), bir kez indirilir
├── ISLES-2022/                         # acilmis ham veri (rawdata/ + derivatives/)
├── center_ids.xlsx
├── ozet/                               # Parca 1 ciktilari
├── hazir/                              # Parca 2: .npy onbellek
├── split.json                          # 250 vakanin train/val/test ayrimi, sabit
├── eval_cases.json                     # tum modellerde ortak, 5 sabit test vakasi
└── <MODEL_KEY>/
    ├── <MODEL_KEY>_log.xlsx            # epoch bazli canli log (SCHEMA asagida)
    ├── checkpoints/
    │   ├── last.pt                     # her epoch: model+optimizer+scheduler+scaler+epoch+best_dice+RNG
    │   └── best.pt                     # val Dice en yuksek oldugunda
    ├── <MODEL_KEY>_curves.png          # 2x3 panel
    ├── <MODEL_KEY>_cases/              # 5 sabit vaka: DWI | ADC | GT | tahmin
    └── <MODEL_KEY>_test_summary.xlsx   # final test metrikleri (tek satir)
```

Yerel mirror: `/content/outputs/<MODEL_KEY>/`, her Drive yazımından sonra `mirror_to_local()`.

> **Drive'dan okuma yavaş.** Ham veri Drive'da durduğu için ilk epoch yavaş olacak; Parça 2'nin
> `.npy` önbelleği bunu telafi eder (density projesinde aynı desen: ilk epoch yavaş, sonrası hızlı).
> Eğitim sırasında darboğaz olursa `hazir/` oturum başında `/content`'e kopyalanır — kalıcı kopya
> yine Drive'da kalır.

### Excel SCHEMA (tüm deneylerde sabit)

```
epoch, train_loss, val_loss, dice, iou, precision, recall, f1, hd95, avd_ml, lezyon_f1, lr, epoch_sure_sn
```

- `append_epoch_row(excel_path, row, resume_epoch=None)` — dosya varsa okur, `resume_epoch`
  verildiyse o epoch ve sonrasını atar, yeni satırı ekler, tekrar yazar.
- Fresh start'ta (`RESUME=False`) eski log **silinir** — yarım kalan koşunun satırları karışmasın.
- `hd95` ve `lezyon_f1` **her epoch hesaplanıyor** (`METRIK_ARALIK = 1`) — karar Parça 3'te
  ölçümle verildi, log'da NaN boşluk kalmıyor.

  Maliyet iki rejimde ayrı ölçülüyor, çünkü tek sayı yanıltıcı: **temiz tahmin** (yakınsamış
  model, dağılmış yanlış pozitif yok) ve **gürültülü tahmin** (erken epoch, binlerce bağlı
  bileşen). Yerel ölçüm 39 val vakası için: temel 0,2 sn · tam-temiz **2,0 sn** ·
  tam-gürültülü **5,4 sn**. Colab bunun ~2 katı.

  HD95 birleşim bbox'ına kırpılıp hesaplanıyor; sonuç birebir aynı (dört senaryoda fark 0,0),
  hızlanma **yalnız temiz rejimde** geçerli (2,9–4,2×). Gürültülü tahminde bbox zaten tüm
  hacmi kaplıyor, kazanç yok — bu yüzden ilk birkaç epoch pahalı, sonrası ucuz.

### Checkpoint / Resume

- `last.pt`: `model`, `optimizer`, `scheduler`, `scaler`, `epoch`, `best_val_dice`,
  `epochs_without_improve`, torch RNG state.
- `RESUME=True` → `last.pt` yüklenir, `epoch+1`'den devam, Excel'e `resume_epoch` verilir.
- `best.pt` yalnız val Dice iyileşince güncellenir; final değerlendirme ve görseller hep `best.pt`.
- "Stripped" checkpoint (optimizer/epoch eksik) sessizce kabul edilmez, hata verir.

### Tekrarlanabilirlik

Sabit `SEED=42` (`torch`/`numpy`/`random`, `cudnn.deterministic=True`). `split.json` ve
`eval_cases.json` bir kez üretilip tüm modellerce okunur — modeller arası karşılaştırma adil olsun diye.

### Real-time'a özgü ek log

`hiz_olcumu.xlsx` / `onnx_olcumu.xlsx`: model, girdi boyutu, param, FLOPs, ms/vaka, ms/dilim,
saf ağ ms, çerçeve yükü, CPU ms, VRAM, Dice. Dice–gecikme ödünleşim grafiği bu tablodan üretilir.

---

## 10. Aşamalar

Her parça tek bir somut çıktı üretir; bitmeden sonrakine geçilmez. Sıradaki parça açılırken bu
tablo güncellenir.

| # | Parça | İş | Çıktı | Durum |
|---|---|---|---|---|
| **1** | Veri keşfi | Zenodo indirme, BIDS envanteri, Bölüm 2'deki 9 ölçüm, bozuk vaka kontrolü | `isles_veri_kesif.ipynb` + `ozet/` + Bölüm 2 ölçülen sayılarla güncellendi | **TAMAM** — 250/250 vaka tam, 9 ölçümün hepsi plana işlendi, 4 karar doğdu |
| **2** | Ön işleme + split | **2×2×2 mm resample**, **ADC ölçek düzeltmesi**, beyin maskesi + kırpma, `.npy` önbellek, `split.json`, `eval_cases.json` | `isles_veri_hazirlik.ipynb` + `hazir/` (0,80 GB) | **TAMAM** — 250/250 vaka, ADC 3 birimden tek birime (CV 1,52→0,09), girdi 96×128, split 161/39/50 |
| **3** | Log/checkpoint altyapısı | SCHEMA, `append_epoch_row`, checkpoint/resume, `mirror_to_local`, `plot_training_curves`, 3B metrik fonksiyonları | `isles_egitim.ipynb` Parça 3 hücreleri | **TAMAM** — 7 metrik testi + 40 epoch resume testi + stripped checkpoint testi geçti |
| **4** | Model fabrikası + smoke test | 5 mimari + 4 ablasyon konfigi, veri hattı, loss, AMP, early stopping; şekil kontrolü + 2 epoch × küçük alt küme + resume testi | `isles_egitim.ipynb` Parça 4 hücreleri | **TAMAM** — 4 mimari şekil kontrolü, 4 epoch smoke + resume geçti; val loss ve epoch süresi düzeltildi |
| **5** | Taban çizgisi eğitimi | `unet_r34_2d` tam eğitim | log/curves/cases/test_summary/test_tahminler | **TAMAM** — test Dice 0,6965 ± 0,1912, 52 epoch / 29 dk (bkz. 5b) |
| **6** | Kalan mimariler + ablasyonlar | Model 2–5 + A1/A2/A3 → 8 koşu | 9 × tam çıktı + `karsilastirma_tablosu.xlsx` | **TAMAM** — 9/9 koşu, Holm hesaplandı (bkz. 5m) |
| **7** | Değerlendirme | Bootstrap CI, eşik taraması, küçük bileşen eşiği, `panoptica` doğrulaması | `degerlendirme/` (9 dosya) | **TAMAM** — `panoptica` doğrulandı (korelasyon 1,000), eşik ve bileşen kararları kapandı (bkz. 7b) |
| **8** | Hız + optimizasyon | Merdiven, `unet3d` + `unet_r34_2d`, **yerel RTX 2060**; GPU ön işleme + ONNX + fp16 + `torch.compile` + TensorRT; sayısal doğrulama; Dice kilidi | `hiz_olcumu.xlsx`, `onnx_olcumu.xlsx`, `tensorrt_olcumu.xlsx`, `merdiven_birlesik.xlsx`, `onnx/` | **TAMAM** — 331 → **75 ms/vaka**; INT8/pruning gerekçeyle kapatıldı |
| **9** | Servis | FastAPI + ONNX Runtime, hacim → maske+hacim+overlay, bileşen bazlı gecikme logu | `servis/` (motor + api + rapor) + `servis_dogrulama.xlsx` + `ornek_rapor.html` | **TAMAM** — 4 arka uç kalite kilidini geçti, en iyi `torch-fp16` **102 ms/vaka**; soğuk başlangıç 331 → 85 ms |
| **10** | Rapor | Vaka görselleri + base64 gömülü tek HTML → Chrome headless PDF | `rapor/isles_raporu.html` (1,79 MB) + `rapor/isles_sonuc_raporu.pdf` (1,56 MB, 18 sayfa) | **TAMAM** |

Parça 3, veriden bağımsız olduğu için 1–2 ile paralel yürütülebilir.

---

## 11. Verilen Kararlar (2026-08-19)

| # | Soru | Karar |
|---|---|---|
| 1 | Penumbra için ek set | **Hayır.** Adaylar araştırıldı (Bölüm 2) ama kapsam dışı bırakıldı. Penumbra üretilmiyor, sınırlılık olarak yazılacak. |
| 2 | NCCT kapsamda mı | **Hayır.** Çalışma yalnız MR (DWI/ADC). BT tarafı sınırlılık olarak yazılacak. |
| 3 | Gerçek zamanlı hedef | Literatüre dayandırıldı (Bölüm 3): birincil **≤ 2 s/vaka RTX 2060**, hedeflenen ≤ 1 s; ikincil ≥ 30 FPS/dilim. Referans: DeepISLES ~120 s, StrokeSeg 0,9 s. |
| 4 | Ölçüm donanımı | **Yerel RTX 2060 6 GB** + aynı makinenin CPU'su. Eğitim Colab T4'te, ölçüm tek ortamda 2060'ta. TensorRT ve INT8 bu kartta desteklendiği için merdivende. |
| 5 | Kapsam dengesi | **5 mimari + 4 ablasyon koşusu** (model tarafı hafif ağırlıklı) + **8 basamaklı optimizasyon merdiveni** × 2 model. |

Kapsam kesinleşti: **tek veri seti (ISLES'22), tek görev (binary lezyon segmentasyonu)**.
Açık soru kalmadı; Parça 1 açılabilir.

---

## 12. Bilinen Riskler

- **Küçük test seti (50 vaka).** Modeller arası küçük farklar istatistiksel olarak ayrışmayacak.
  Eşleştirilmiş test ve CI zorunlu; "X modeli en iyi" cümlesi kanıtlanmadan yazılmayacak.
- **Lezyon hacmi değişkenliği çok yüksek.** Ortalama Dice tek başına yanıltıcı; hacim kırılımı
  olmadan sonuç raporlanmayacak.
- **Merkez etkisi.** Çok merkezli veride model merkez artefaktını öğrenebilir. Split stratifiye,
  ve mümkünse merkez bazlı performans kırılımı verilecek.
- **2D dilim yaklaşımı 3B tutarlılığı garanti etmez.** Dilimler arası kopuk tahminler hacmi
  bozar; son işlemede küçük bileşen eleme ve 3B tutarlılık kontrolü gerekecek.
- **Optimizasyonun kalite maliyeti.** INT8/pruning sonrası Dice düşüşü raporda gizlenmeyecek;
  "gerçek zamanlı oldu" cümlesi kalite kaybı yazılmadan kurulmayacak.
- **Colab oturum kopması.** Resume mekanizması bu yüzden Parça 3'te, eğitimlerden önce test edilir.
- **Dış doğrulama yok.** Tek veri seti kullanılıyor; sonuçlar ISLES'22 dağılımına özel.
  Merkez bazlı kırılım bunun yerini kısmen tutar, ama başka bir kohortta genelleme kanıtlanmıyor.
- **Kapsam daralması.** Penumbra ve NCCT kapsam dışı (Bölüm 2). Talebin bu iki kalemi
  karşılanmıyor ve rapor bunu açıkça yazacak.
- **6 GB VRAM.** 3D U-Net referansı küçük patch'e sıkışıyor; "3B daha iyi" sonucu çıkmazsa bunun
  mimariden mi bellek kısıtından mı geldiği ayrılamayabilir. Bu sınırlılık rapora yazılacak.

---

## 12b. Açık kalan işler — **tam liste**

Parça 8 kapanırken geriye kalan her şey burada. Bunlar dışında bekleyen iş yok.

### Colab'da koşulacak — **kapandı**

İkisi için de bağımsız notebook yazıldı ve koşuldu.

| notebook | üretti | sonuç |
|---|---|---|
| `isles_degerlendirme.ipynb` | Parça 7'nin tamamı + `panoptica` | **`panoptica_dogrulandi: true`** — korelasyon 1,000, değerler birebir aynı (Bölüm 7b) |
| `isles_flair_dogrulama.ipynb` | `flair_hizalama.png` + `flair_hizalama.xlsx` + `flair_ozeti.json` | Hizalama medyanı **670 ms** (önceki 1193 ms okuma süresini de içeriyordu, düzeltildi) |

`flair_hizalama.xlsx` daha önce **hiç yazılmamıştı** — Parça 2b'de görsel hücresi `np.ptp`
hatasıyla durunca sonraki hücre çalışmamıştı. Artık Drive'da.

Bu iki iş dışında Colab'da bekleyen bir şey yok.

### Bilerek yapılmayanlar (gerekçeleri kayıtlı)

| iş | gerekçe |
|---|---|
| INT8 PTQ, pruning | Ağ, bütçenin %19'u; kazanç birkaç ms, risk Dice kaybı (Bölüm 6) |
| Koşuların tekrarı (seed varyansı) | Bütçe. A3 serisi dolaylı tahmin veriyor: **±0,024 bant** (Bölüm 5j) |
| `unet3d` için daha uzun eğitim | Protokol dokuz koşuda sabit tutuldu; LR hiç düşmedi, model bütçeye sıkışmış olabilir (Bölüm 5f) |
| A2 için daha uzun eğitim | Aynı gerekçe; A2 70 epoch tavanına dayandı (Bölüm 5i) |
| Yoğunluk tabanlı FLAIR kaydı | Gerçek zamanlı bütçeyle bağdaşmıyor (1193 ms/vaka header kaydı bile) (Bölüm 5k) |
| Dış doğrulama seti | Kapsam kararı (Bölüm 2) |

### Sıradaki parçalar

Parça 9 (servis) ve Parça 10 (rapor).

---

## 13. Referanslar (plandaki sayıların kaynağı)

- ISLES'22 veri seti — Hernández Petzsche ve ark., *Sci Data* 2022, https://www.nature.com/articles/s41597-022-01875-5
- **DeepISLES** — *Nat Commun* 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12335569/ (Dice 0,82 / lezyon-F1 0,86 / ~2 dk-vaka RTX 3090)
- **StrokeSeg** — arXiv 2510.24378 (ONNX+FP16, 0,9 s/hacim GPU vs 149,2 s CPU)
- **Jetson nnU-Net** — *Sensors* 2026, https://www.mdpi.com/1424-8220/26/11/3322 (TensorRT FP16, Dice 0,78 → 0,67)
- Kapsam dışı bırakılan aday setler (kayıt için): CPAISD arXiv 2404.02518 / Zenodo 10892316 ·
  AISD https://github.com/GriffinLiang/AISD · APIS https://bivl2ab.uis.edu.co/challenges/apis ·
  ISLES'24 https://isles-24.grand-challenge.org/

---

## 14. Durum

**On parcanin tamami bitti.** Proje uctan uca tamamlandi: veri kesfi, hazirlik, dokuz egitim
kosusu, degerlendirme, gecikme optimizasyonu, calisir servis ve final rapor.

### Ciktilar

| Ne | Nerede |
|---|---|
| Final rapor | `rapor/isles_raporu.html`, `rapor/isles_sonuc_raporu.pdf` (18 sayfa, 12 figur, 8 tablo) |
| Cikarim servisi | `servis/` (motor, FastAPI, vaka raporu) |
| Olcum araclari | `hiz_olcumu.py`, `onnx_olcumu.py`, `ortak/` |
| Notebook'lar | veri kesfi, hazirlik, egitim, degerlendirme, FLAIR dogrulama, dagitim, rapor paketi |
| Egitim ciktilari | Drive: dokuz model klasoru, `degerlendirme/`, `ozet/` |

### Sonuclar

Kalite tarafinda dokuz kosunun hicbiri 2B U-Net taban cizgisini gecemedi; uc alternatif Holm
duzeltmesinden sonra anlamli bicimde daha kotu cikti. Hiz tarafinda kazanc modelden degil on
islemeden geldi: GPU'ya tasima ve yarim hassasiyet vaka basina sureyi 385 ms'den 138 ms'ye
indirdi. ONNX ve TensorRT bu hatta kazanc saglamadi.

Cozulmemis ana problem kucuk lezyonlar: en kucuk hacim tertilinde Dice 0,55 civarinda ve bunu
hedefleyen ablasyon durumu kotulestirdi. Dis dogrulama yapilmadi.
