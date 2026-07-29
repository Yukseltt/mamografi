# Meme Yoğunluğu (Breast Density) Segmentasyonu — Deney Planı

## Veri Özeti (doğrulandı)

- Kaynak: `mydrive/Mendeley_Data_Density_Entegrasyonu`
- Format: NIfTI (`.nii.gz`), tek kanal, `512x512`, `float64`, piksel değer aralığı `0-238`
- Maske: **binary** (`0` / `1`) — dens doku vs arkaplan
- `train_density.json` / `val_density.json`: `case_id -> yoğunluk yüzdesi` (float, örn. `40.0`) — segmentasyon hedefi değil, **metadata**; rapordaki vaka görsellerinde bağlam olarak kullanılacak (örn. "Vaka 3 — Density: %24")
- Split: `imagesTr+labelsTr` (447 vaka) = train, `imagesVal+labelsVal` (89 vaka) = validation. Ek bölünme yok.
- Ortam: Google Colab Pro, **T4 GPU**

## Modeller (6 deney)

1. U-Net + ResNet34
2. U-Net + EfficientNet-B0
3. U-Net++ + ResNet34
4. DeepLabV3+ + ResNet50
5. DeepLabV3+ + EfficientNet-B3
6. SegFormer-B2

Encoder'lar `segmentation_models_pytorch` (1-6 hariç SegFormer), SegFormer-B2 `transformers` (`nvidia/segformer-b2-finetuned-ade-512-512` ön-eğitimli ağırlıklardan fine-tune).

## Kod Yapısı — Gerçek Uygulama

İlk taslakta ayrı `.py` modülleri + `yaml` config'ler planlanmıştı; gerçek uygulamada **tek bir parametrik notebook** kullanıldı (`mass_lezyon_projesi`'ndeki notebook-başına-deney desenine daha sadık, Colab'da dosya senkronizasyonu gerektirmiyor):

- **`mendeley_data_density/density_segmentation_train.ipynb`** — 6 modelin hepsi için ortak şablon; her zaman **sırada eğitilecek modele** ayarlı tutulur, çıktıları temiz.
- **`mendeley_data_density/density_segmentation_<MODEL_KEY>.ipynb`** — biten her modelin çalışmış kopyası, çıktıları (loglar, grafikler, vaka görselleri) gömülü halde arşivlenir. `mass_lezyon_projesi`'ndeki notebook-başına-model deseninin aynısı. Şu an: `density_segmentation_unet_resnet34.ipynb`.

Şablonun içeriği: Tüm mimariler, hiperparametreler, veri pipeline'ı, loss/metrik, log/checkpoint yardımcıları tek notebook içinde hücrelere bölünmüş halde tanımlı; `MODEL_KEY` değişkeni hangi modelin eğitileceğini seçiyor (`MODEL_CONFIGS` dict'i içinde 6 modelin de hiperparametreleri tanımlı).
- Farklı model eğitmek için: `MODEL_KEY` değiştirilip Colab'da "Save a copy as" ile yeni isimle kaydedilecek (yolo11s / yolo11s_kirpilmis ayrımındaki desenin aynısı).
- **Önemli senkronizasyon notu**: bu repodaki `.ipynb` dosyası ile Colab'da fiilen çalışan oturum ayrı şeyler — buradaki değişiklikler Colab'a otomatik yansımıyor, Drive'daki dosyanın yeniden yüklenmesi/açılması veya ilgili hücrelerin elle güncellenmesi gerekiyor.

## Kayıt / Log Standardı — `mass_lezyon_projesi` ile aynı desen

Önceki mass lezyon çalışmasında (`mamosis_mass_lezyon_yolo11s_kirpilmis.ipynb`) kurulan desen burada da birebir uygulandı: **Excel tabanlı epoch logu + Drive kalıcı depo + yerel mirror + resume-safe checkpoint**. Sebep: tekrarlanabilirlik (her run'ın tam geçmişi tek dosyada, Drive'da kalıcı) ve T4'te oturum kopmalarına karşı dayanıklılık.

### Klasör yapısı (Drive, kalıcı)

```
/content/drive/MyDrive/density_segmentation/
├── eval_cases.json                    # tum modellerde ortak, 5 sabit val vakasi
└── <MODEL_NAME>/                      # örn. unet_resnet34
    ├── <MODEL_NAME>_log.xlsx          # epoch bazlı canlı log (SCHEMA aşağıda)
    ├── checkpoints/
    │   ├── last.pt                    # her epoch üzerine yazılır: model+optimizer+scheduler+scaler+epoch+best_dice+RNG state
    │   └── best.pt                    # val Dice en yüksek olduğunda güncellenir
    ├── <MODEL_NAME>_curves.png         # 6 grafik (train/val loss, dice, iou, precision, recall, f1)
    ├── <MODEL_NAME>_cases/             # 5 sabit-seed val vakası: orijinal | GT | tahmin
    └── <MODEL_NAME>_test_summary.xlsx  # final val metrikleri (tek satır)
```

Yerel mirror: `/content/outputs/<MODEL_NAME>/` — her Drive yazımından sonra `mirror_to_local()` ile kopyalanır (mass lezyon projesindeki `mirror_to_local` fonksiyonuyla aynı).

### Excel SCHEMA (tüm 6 model için sabit, karşılaştırılabilirlik için)

```
epoch, train_loss, val_loss, dice, iou, precision, recall, f1, lr
```

- `append_epoch_row(excel_path, row, resume_epoch=None)`: mass lezyon projesindeki fonksiyonun aynısı — dosya varsa okur, `resume_epoch` verildiyse o epoch ve sonrasını atıp resume sırasında tekrar satır oluşmasını engeller, yeni satırı ekler, tekrar yazar.
- Fresh start'ta (RESUME=False) önce eski `<MODEL_NAME>_log.xlsx` silinir — aynı MODEL_NAME ile önceki yarım/başarısız run'ın satırlarının üstüne binmesin diye (mass lezyon projesindeki gerekçeyle aynı).

### Checkpoint / Resume Mekanizması

- Her epoch sonunda `last.pt` şunları içerir: `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `epoch`, `best_val_dice`, `torch` RNG state (tekrarlanabilirlik için).
- `RESUME` bayrağı: `True` yapılırsa `last.pt` yüklenir, `epoch+1`'den devam edilir, Excel logunda `resume_epoch=last_epoch+1` verilerek olası çakışan satırlar temizlenir.
- `best.pt` sadece val Dice iyileştiğinde güncellenir; final değerlendirme ve görsel üretimi hep `best.pt` ile yapılır.
- Mass lezyon projesindeki `find_resumable_checkpoint()` uyarısı burada da geçerli: eğer `last.pt` "stripped" (optimizer/epoch bilgisi eksik) ise resume edilemeyeceği kontrol edilip hata verilecek, sessizce yanlış varsayılana düşülmeyecek.

### Tekrarlanabilirlik (reproducibility)

- Her model config'inde sabit `SEED` (örn. 42): `torch.manual_seed`, `numpy.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic=True`.
- Val setinden seçilen **5 vaka** tüm 6 model için aynı sabit seed ile seçilecek (`case_ids` listesi bir kere üretilip `configs/eval_cases.json` içine yazılacak, her model aynı listeyi okuyacak) — modeller arası görsel karşılaştırma adil olsun diye.
- Train/val split zaten sabit (kullanıcı talebiyle, klasör bazlı), ek rastgelelik yok.

### Final Karşılaştırma Tablosu

Tüm modellerin `<MODEL_NAME>_test_summary.xlsx` dosyaları tek bir `comparison_table.xlsx` içinde birleştirilecek (mass lezyon projesindeki `summary.to_excel(...)` deseninin çoklu-model versiyonu). Bu tablo PDF raporun son bölümüne aynen aktarılacak.

## Hiperparametreler (literatür taraması sonrası netleşen, kullanıcıyla birlikte karara bağlandı)

Literatür kaynakları: [U-Net/U-Net++ training params analysis](https://portal.amelica.org/ameli/journal/602/6024323010/html/), [DeepLabV3+ mammography](https://www.uniwriter.ai/computing/deeplabv3-model-for-mammography-segmentation/), [DeepLabV3+ EfficientNet-b0 breast tumor segmentation](https://www.tandfonline.com/doi/full/10.1080/21681163.2024.2373996), [SegFormer fine-tuning](https://learnopencv.com/tag/fine-tune-segformer-pytorch/), [SegFormer3D](https://arxiv.org/html/2404.10156v1)

| Model | Batch | LR (encoder/decoder) | Epoch (max) | Image Size |
|---|---|---|---|---|
| U-Net + ResNet34 | 16 | 5e-5 / 2.5e-4 | 70 | 512 |
| U-Net + EffNet-B0 | 16 | 5e-5 / 2.5e-4 | 70 | 512 |
| U-Net++ + ResNet34 | 12 | 5e-5 / 2.5e-4 | 70 | 512 |
| DeepLabV3+ + ResNet50 | 8 | 1e-4 / 5e-4 | 70 | 512 |
| DeepLabV3+ + EffNet-B3 | 8 | 1e-4 / 5e-4 | 70 | 512 |
| SegFormer-B2 | 8 | 1e-4 (tek LR, ayrım yok) | 70 | 512 |

> **Epoch bütçesi eşitlendi (model 4 başlamadan önce).** İlk planda modeller 4-6 için `max_epochs=60, patience=12` öngörülmüştü (ağır mimarilere daha az epoch, literatür temelli). Ancak ilk üç koşu zirvelerini **epoch 54 ve 50**'de yaptı; 60 tavanıyla epoch ~50'de zirve yapan bir model patience dolmadan kesilirdi. Altı modelin farkının yalnızca mimariden gelmesi için bütçe hepsinde `max_epochs=70, patience=15` yapıldı (kullanıcı onayıyla).

- Optimizer: AdamW, weight decay 1e-4
- **Scheduler: `ReduceLROnPlateau`** (val loss 3 epoch iyileşmezse LR×0.5, min_lr 1e-7) — küçük medikal veri setlerinde cosine annealing'den daha yaygın kullanıldığı için tercih edildi (kullanıcı onayı ile, cosine+warmup fikri terk edildi)
- **+ 3 epoch linear warmup**: eğitimin ilk 3 epoch'unda LR hedef değere kademeli çıkar (`WARMUP_EPOCHS=3`, `TARGET_LRS` üzerinden manuel ramp), `ReduceLROnPlateau.step()` sadece warmup bittikten sonra çağrılır — erken eğitimdeki gürültünün plateau'nun patience sayacını kirletmesini önler. **Neden eklendi**: ilk `unet_resnet34` denemesinde warmup'sız haliyle val loss/dice çok sert sıçrıyordu (bkz. Şu Anki Durum).
- Loss: Dice + BCE (binary)
- Early stopping: patience 12-15, val Dice izlenerek
- AMP (mixed precision): açık (`torch.amp.autocast('cuda')` / `torch.amp.GradScaler('cuda')` — eski `torch.cuda.amp.*` API'si deprecated olduğu için güncellendi)
- **Gradient clipping**: `scaler.unscale_(optimizer)` sonrası `clip_grad_norm_(model.parameters(), max_norm=1.0)` — AMP altında ani gradyan sıçramalarını sınırlamak için eklendi (yine ilk denemedeki dengesizliğe karşı)
- **Girdi normalizasyonu**: görüntü Dataset içinde per-görüntü min-max ile `[0,1]`'e çekiliyor, ardından ImageNet mean/std uygulanıyor. `A.Normalize` **`max_pixel_value=1.0` ile çağrılmalı** — varsayılan `255.0` ile görüntü ikinci kez 255'e bölünür ve dinamik aralık 1/255'e iner (bkz. Şu Anki Durum, çözülen ana bug).
- **BatchNorm momentum varsayılan (0.1)**: bir ara 0.01'e çekilmişti (küçük batch'te running istatistikler gürültülü diye), ancak asıl dengesizliğin kaynağı normalizasyon bug'ıydı. Epoch başına yalnızca 27 adım varken 0.01 running istatistiklerin oturmasını ~4 epoch geciktirdiği için varsayılana geri alındı.
- Augmentation: horizontal/vertical flip, ±10-15° rotation, brightness/contrast jitter, hafif elastic deformation, hafif CoarseDropout — agresif crop/zoom yok (global doku oranı korunmalı). MICCAI 2024'te küçük organ segmentasyonunda CutMix'in daha iyi sonuç verdiği bulgusu değerlendirildi ancak kullanıcı tercihiyle mevcut set korundu, CutMix eklenmedi.
- Gerekirse gradient accumulation ile efektif batch sabitlenecek
- **Tüm metrikler manuel hesaplanıyor** (TP/FP/FN üzerinden, görüntü başına hesaplanıp batch içinde ortalanarak). Başlangıçta sadece Dice manuel, IoU/Precision/Recall/F1 `torchmetrics.functional` ile hesaplanıyordu; torchmetrics batch-geneli (micro) hesapladığı için binary'de eşit olması gereken Dice ve F1 farklı çıkıyordu, modeller arası karşılaştırma tablosunu bozacaktı. Şimdi hepsi aynı per-görüntü tanımını kullanıyor (`f1 = dice`).

## Şu Anki Durum (güncel ilerleme)

- **Model 1/6 (`unet_resnet34`) tamamlandı — val Dice 0.8059.** Diğer 5 model henüz başlatılmadı. Detaylar aşağıda "Model 1/6 sonucu" bölümünde.
- Sırasıyla şu sorunlar tespit edilip düzeltildi: (1) case eşleştirme doğrulandı (447 train/89 val, dosya adları uyumlu), (2) `torchmetrics.dice` `AttributeError`'ı → manuel Dice hesaplamasına geçildi, (3) `torch.cuda.amp.*` deprecation uyarısı → `torch.amp.*` API'sine geçildi, (4) val loss/dice epoch'tan epoch'a çok sert sıçrıyordu (train_loss düzgün azalırken) → 3 epoch warmup + gradient clipping + BatchNorm momentum düşürme eklendi.

### Volatilitenin kök nedeni bulundu (çözüldü)

14 epoch'luk gözlem, (4)'teki üç düzeltmenin volatiliteyi **çözmediğini** gösterdi — train loss 1.62→0.47 düzgün inerken val Dice 0.0002 ile 0.56 arasında rastgele zıplamaya devam etti. Semptomun kaynağı optimizasyon değil, **girdi normalizasyonuydu**:

Görüntü Dataset içinde zaten min-max ile `[0,1]`'e çekiliyordu, ardından `A.Normalize` varsayılan `max_pixel_value=255.0` ile çağrıldığı için ikinci kez 255'e bölünüyordu. Sonuç: `(img/255 - 0.485)/0.229` → tüm görüntü `[-2.118, -2.101]` aralığına sıkışıyor, dinamik aralık 0.017 (olması gereken ~4.37, tam **255x** fark — yerelde doğrulandı). Encoder'a giden şey neredeyse sabit bir düzlem oluyordu.

Bu tam olarak gözlenen semptomu üretir: BatchNorm train-mode'da batch istatistikleriyle minik sinyali geri büyütüp öğrenmeyi mümkün kılıyor (train loss düşüyor), ama eval-mode'da running mean/var ile batch istatistikleri arasındaki en ufak fark aynı oranda büyütüldüğü için val çıktısı her epoch farklı bir dağılıma kayıyordu. Warmup/clipping/BN-momentum semptomu tedavi etmeye çalıştığı için etkisiz kaldı.

### Uygulanan düzeltmeler (Colab'da doğrulandı — aşağıdaki sonuçlara bkz.)

1. **`A.Normalize(..., max_pixel_value=1.0)`** — ana düzeltme. Tek bir `imagenet_normalize()` fabrika fonksiyonuna toplandı, üç yerde de (train/val transform + görselleştirmedeki `predict_mask`) aynı fonksiyon kullanılıyor.
2. **BatchNorm momentum varsayılana (0.1) döndürüldü** — 0.01 override'ı kaldırıldı.
3. **`A.ElasticTransform` alpha 1 → 30** — alpha deformasyon genliğini ölçeklediği için alpha=1 pratikte no-op'tu, elastic augmentation hiç çalışmıyordu.
4. **Tüm metrikler per-görüntü tanımına geçirildi** — Dice/IoU/Precision/Recall/F1 aynı TP/FP/FN'den, `torchmetrics` bağımlılığı metrik hesabından çıktı (bkz. Hiperparametreler).
5. **Yerel `.npy` cache** (`/content/npy_cache`) — her epoch 447 `.nii.gz` dosyasını Drive'dan okumak eğitimin en yavaş kısmıydı; ilk okumada yerel diske yazılıp sonraki epoch'larda oradan okunuyor (worker'lar arası yarım dosya olmasın diye atomik `os.replace`).
6. **`epochs_without_improve` checkpoint'e eklendi** — resume'da early stopping sayacı sıfırlanıyordu.
7. **Loss fp32'de hesaplanıyor (`logits.float()`)** — AMP altında logits fp16 geliyordu ve soft-Dice'ın `probs.sum()` terimi 262.144 piksel üzerinden alındığı için ortalama olasılık ~0.25'i geçtiğinde fp16 tavanını (65504) aşıp `inf`'e dönüyordu. Bu durumda `dice=0`, `dice_loss=1` sabitleniyor ve **hiç gradyan üretmiyordu** — yani loss fiilen salt BCE'ye düşüyordu. En çok yüksek yoğunluklu vakalarda (dens alan geniş) tetiklendiği için Dice terimi tam gerektiği yerde sessizce kapanıyordu. Eski logdaki epoch 0 `train_loss=1.6177` değeri bununla tutarlı (BCE ~0.62 + sabitlenmiş dice_loss 1.0).
8. Küçük: train loss `drop_last=True` yüzünden görülmeyen 15 örneği de bölene katıyordu (~%3 düşük raporlanıyordu); `find_resumable_checkpoint` checkpoint'i iki kez yüklüyordu (artık yüklenen dict doğrudan `restore_checkpoint`'e veriliyor).

Dataset hücresinin sonuna bir **doğrulama print'i** eklendi: girdi tensörünün aralığını ve maske değerlerini basıyor. Aralık `~[-2.1, 2.6]` görünmeli; hepsi -2.1 civarında sıkışıksa `max_pixel_value` hâlâ yanlış demektir.

### Doğrulama sonucu — volatilite çözüldü

`unet_resnet34` sıfırdan (`RESUME=False`) yeniden eğitildi. İlk 12 epoch, düzeltmelerin çalıştığını doğruluyor:

| Epoch | Eski koşu (bozuk) | Yeni koşu |
|---|---|---|
| 0 | dice 0.1071 | 0.2749 |
| 1 | dice 0.0082 | 0.5064 |
| 3 | dice 0.0013 | 0.5992 |
| 9 | dice 0.5593 | 0.7425 |
| 11 | dice 0.0312 | 0.7690 |

Val loss 1.4410 → 0.4274 düzgün iniyor, val Dice 12 epoch boyunca ciddi geri adım yapmadan tırmanıyor (epoch 3 ve 7'de küçük dalgalanma). Eskiden aynı aralıkta 0.0002 ile 0.56 arasında rastgele zıplıyordu.

Not: epoch 0 `train_loss` 1.5660 geldi; bu bölümde önceden "~1.3-1.4 beklenir" yazıyordu, tahmin dens alan büyüklüğüne dair kaba bir varsayıma dayanıyordu. Dice teriminin canlı olduğunun asıl kanıtı epoch 0 Dice'ının 0.1071'den 0.2749'a çıkması ve sonrasındaki düzgün tırmanış.

Gözlenen sağlıklı davranışlar:
- **Val loss sürekli train loss'un altında** — train augmentation'lı (flip/rotate/elastic/dropout), val temiz görüntü üzerinden hesaplandığı için beklenen fark. İkisi kesiştiğinde (train < val) overfitting başlangıcı demektir, izlenecek sinyal bu.
- **LR henüz düşürülmedi** — warmup epoch 3'te bitti, `ReduceLROnPlateau` aktif ama val loss her epoch iyileştiği için patience sayacı dolmadı.

### Model 1/6 sonucu — `unet_resnet34` TAMAMLANDI (tüm çıktılar üretildi)

| | |
|---|---|
| En iyi val Dice | **0.8059** (epoch 54) |
| IoU | 0.6922 |
| Precision | 0.8153 |
| Recall | 0.8392 |
| F1 | 0.8059 (= Dice) |
| val_loss | 0.2624 |
| Toplam epoch | 70 (0-69) |
| Bitiş | early stopping, epoch 69 |

Final değerlendirme hücresi `best.pt` ile eğitim logundaki epoch 54 satırını birebir doğruladı. Recall > Precision (+0.024): model GT'den biraz fazla dens alan işaretliyor, hafif over-segmentation — 5 vaka görselinde de görünüyor (GT'nin ayrı tuttuğu adacıkları model birleştiriyor). Yoğunluk yüzdesini bir miktar fazla tahmin etme yönünde bir sapma, klinik olarak güvenli taraf.

Üretilen çıktılar: `unet_resnet34_log.xlsx`, `unet_resnet34_curves.png`, `unet_resnet34_cases/unet_resnet34_5cases.png`, `unet_resnet34_test_summary.xlsx`, `checkpoints/best.pt` + `last.pt`.

LR paneli (düzeltilmiş grafikte) çıkarılan takvimi loglanmış veriyle doğruladı; Dice'ın epoch 29-32 ve 36-37'deki sıçramaları ilk iki LR düşüşünün hemen ardından geliyor — "plato göründü ama LR düşünce model tekrar öğrendi" tespitinin görsel kanıtı.

Early stopping epoch 69'da tetiklendi (54 + patience 15). `max_epochs=70` zaten epoch 69'dan sonra bitecekti, yani bu koşu bütçesini tam kullandı — early stopping fiilen bağlayıcı olmadı.

**LR takvimi**: `ReduceLROnPlateau` 7 kez tetiklendi, decoder LR 2.5e-4 → 1.95e-6:

```
epoch 29 -> 1.25e-4     epoch 51 -> 1.56e-5     epoch 66 -> 1.95e-6
epoch 36 -> 6.25e-5     epoch 58 -> 7.81e-6
epoch 47 -> 3.13e-5     epoch 62 -> 3.91e-6
```

İlk LR düşüşünden sonra Dice 0.7877 → 0.8059 (+0.018) kazandı, yani plato göründüğü anda müdahale etmemek doğru karardı.

**Overfitting makası açıldı sonra dondu** — LR düşüşleri işini yaptı:

| Epoch aralığı | Ortalama (val_loss − train_loss) |
|---|---|
| 20-29 | +0.028 |
| 30-44 | +0.039 |
| 45-54 | +0.049 |
| 60-69 | +0.049 |

Makas epoch 45'ten sonra genişlemeyi bıraktı; kontrolsüz overfitting yok.

**Tam yakınsama**: son 10 epoch'ta val_loss 0.2633-0.2644 bandında (genişlik 0.0011), Dice 0.8042-0.8054. LR ~2e-6'da model fiilen donmuş. Daha fazla epoch bütçesi bu mimariye bir şey katmazdı.

Dice 0.8059, literatürdeki 0.80-0.90 bandının alt ucu — 6 mimarinin en hafifi (24.4M parametre) için beklenen yer. Karşılaştırmanın taban çizgisi bu.

### Model 2/6 sonucu — `unet_effnetb0` TAMAMLANDI

| | resnet34 | effnetb0 |
|---|---|---|
| En iyi val Dice | **0.8059** | 0.7419 |
| IoU | **0.6922** | 0.6120 |
| Precision | **0.8153** | 0.7665 |
| Recall | **0.8392** | 0.7915 |
| En iyi epoch | 54 | 15 |
| Toplam epoch | 70 | 31 (early stopping) |
| LR düşüşü | 7 | 2 |
| Ortalama train/val makası | +0.028 | **+0.081** |

**Ana bulgu: EfficientNet-B0 encoder'ı 4 kat küçük (5.3M / 21M) olmasına rağmen belirgin şekilde daha çok overfit ediyor.** Makas 3 katı ve epoch 28-30'da +0.12'ye çıkıyor. Kapasite sorunu değil; ImageNet üzerinde öğrenilmiş depthwise-separable özniteliklerin mamografi dokusuna ResNet'inkiler kadar iyi transfer olmadığı hipotezi (raporda hipotez olarak sunulmalı, kanıtlanmış değil). Gözlem tarafı net: val Dice eğrisi model 1'in düzgün tırmanışına karşılık sürekli zıplıyor.

**"Erken mi kesildi?" — hayır.** Şüphe meşru: model 2 epoch 30'da, ikinci LR düşüşünden sadece 1 epoch sonra durdu; model 1 ise epoch 30'da 0.7877'deydi ve sonraki 24 epoch'ta LR düşüşleriyle 0.8059'a çıkmıştı. Ama (1) Dice 15 epoch boyunca 0.7419'a bir daha dokunmadı (bant 0.6711-0.7328), (2) makas büyüyor — az eğitilmiş değil ezberleyen model imzası, (3) epoch 20'deki ilk LR düşüşünden sonra 10 epoch geçti, toparlanma olmadı (model 1 aynı müdahaleye anında karşılık vermişti). Yeniden koşturulmadı: protokolü tek model için değiştirmek (patience artırmak) 6 modelin karşılaştırılabilirliğini bozar.

**Metodoloji notu (rapora girmeli)**: erken durdurma **val Dice**'ı, LR scheduler **val loss**'u izliyor. Model 2'de ayrıştılar — Dice epoch 15'te platoya girdi, val_loss epoch 25'e kadar iyileşti (0.3848). Protokol 6 modelde birebir aynı uygulandığı için karşılaştırma adil, ama **gürültülü Dice eğrisine sahip modeller sistematik olarak daha erken kesiliyor**. Kalan modellerde de epoch ~30 civarı duruşlar görülürse bu desen raporda açıkça belirtilmeli.

**Görsel doğrulama**: Vaka 3'te model 2'nin tahmini belirgin şekilde daha kaba — GT'nin dağınık küçük adacıklarını model 1 yakalarken model 2 birleştirip yuvarlıyor. İnce yapı kaybı, Dice farkının görsel karşılığı.

### Model 3/6 sonucu — `unetpp_resnet34` TAMAMLANDI

| Model | Encoder | Decoder | Dice | IoU | Precision | Recall | En iyi epoch |
|---|---|---|---|---|---|---|---|
| unet_resnet34 | ResNet34 | U-Net | **0.8059** | **0.6922** | 0.8153 | **0.8392** | 54 |
| unet_effnetb0 | EffNet-B0 | U-Net | 0.7419 | 0.6120 | 0.7665 | 0.7915 | 15 |
| unetpp_resnet34 | ResNet34 | U-Net++ | 0.7980 | 0.6818 | **0.8162** | 0.8310 | 50 |

**ANA BULGU — iki faktörlü ayrıştırma.** İlk üç model kazara temiz bir 2-faktörlü tasarım oluşturdu:

```
decoder sabit (U-Net), encoder ResNet34 -> EffNet-B0 : Dice -0.0640
encoder sabit (ResNet34), decoder U-Net -> U-Net++   : Dice -0.0079
                                 -> encoder etkisi decoder etkisinin 8.1 katı
```

Bu veri setinde **encoder seçimi decoder seçiminden çok daha belirleyici**. Raporun ana bulgularından biri olmalı; 6 modeli tek tek sıralamaktan daha değerli bir çıkarım.

**U-Net vs U-Net++ farkı istatistiksel olarak anlamsız.** 89 validasyon vakasıyla, per-görüntü Dice std'si için hangi makul değer alınırsa alınsın 0.0079'luk fark 1 SEM'in altında kalıyor (std 0.10 → 0.75 SEM; std 0.15 → 0.50 SEM; std 0.20 → 0.37 SEM). Raporda "U-Net daha iyi" **yazılmamalı**; doğru ifade "bu veride ayırt edilemez". Üstelik U-Net++ bunu daha pahalıya yapıyor — nested skip connection'lar yüzünden batch 16'dan 12'ye inmek zorunda kaldı, karşılığında kazanç yok.

**Açık iş**: final değerlendirme hücresine per-görüntü Dice standart sapması eklenirse karşılaştırma tablosunda hata payı olur ve hangi farkların gerçek olduğu netleşir. Biten modellere geriye dönük uygulamak ucuz (`best.pt` yükleyip val'i bir kez koşturmak, yeniden eğitim yok).

### Model 4/6 sonucu — `deeplabv3p_resnet50` TAMAMLANDI

| # | Model | Encoder | Decoder | Dice | IoU | Precision | Recall | R−P | En iyi epoch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | unet_resnet34 | ResNet34 | U-Net | **0.8059** | **0.6922** | 0.8153 | **0.8392** | +0.024 | 54 |
| 2 | unetpp_resnet34 | ResNet34 | U-Net++ | 0.7980 | 0.6818 | **0.8162** | 0.8310 | +0.015 | 50 |
| 3 | deeplabv3p_resnet50 | ResNet50 | DeepLabV3+ | 0.7811 | 0.6591 | 0.7773 | 0.8293 | **+0.052** | 39 |
| 4 | unet_effnetb0 | EffNet-B0 | U-Net | 0.7419 | 0.6120 | 0.7665 | 0.7915 | +0.025 | 15 |

**Kaybı precision'dan veriyor.** Model 4'ün recall'ı model 1'e çok yakın (0.8293 / 0.8392) ama precision'ı belirgin düşük (0.7773 / 0.8153); recall−precision makası +0.052, diğer üç modelin iki katı. DeepLabV3+ dens dokuyu buluyor ama sınırlarından taşıyor.

**Hipotez — skip connection yoğunluğu ↔ precision:**

| Decoder | Skip yapısı | Precision |
|---|---|---|
| U-Net++ | nested dense (en yoğun) | 0.8162 |
| U-Net | her ölçekte standart | 0.8153 |
| DeepLabV3+ | yok; ASPP + kaba 4× upsample | 0.7773 |

Mekanizma: DeepLabV3+ Cityscapes/PASCAL gibi büyük bütün nesneler için tasarlandı, sondaki kaba upsample orada maliyetsiz. Meme yoğunluğu dağınık ve girintili sınırlara sahip olduğu için kaba decoder taşıyor ve yanlış pozitif üretiyor. **Confound uyarısı**: model 4'ün encoder'ı da farklı (ResNet50), bu ilişki temiz değil — raporda hipotez olarak sunulmalı. Yine de etki büyüklüğü dikkat çekici: precision farkı 0.038, U-Net/U-Net++ arasındaki fark 0.001.

**Not — tahmin tutmadı**: "encoder baskınsa ResNet50 model 1'e yakın/üstünde çıkar" beklentisi 0.0248 farkla gerçekleşmedi. Temiz bir çürütme değil (decoder de değişti, iki faktörlü tasarımda bu hücre bilgi vermiyor) ama kayda geçirildi.

**Bütçe değişikliği bağlayıcı olmadı**: zirve epoch 39, duruş epoch 54. Eski bütçeyle (patience 12) 51'de dururdu, zirve yine 39'da olurdu. 70/15'e çıkarmak bu koşuda sonucu değiştirmedi ama kalan modellerde confound'u kaldırıyor.

**Test edilebilir tahmin (model 5)**: `deeplabv3p_effnetb3` en zayıf encoder ailesini en zayıf decoder'la birleştiriyor → **0.7419'un altı bekleniyor**. Çıkarsa hem encoder bulgusu hem decoder-precision hipotezi güçlenir; çıkmazsa ikisi de gözden geçirilmeli.

### Model 5/6 sonucu — `deeplabv3p_effnetb3` TAMAMLANDI

| # | Model | Encoder | Decoder | Dice | IoU | Precision | Recall | R−P | En iyi epoch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | unet_resnet34 | ResNet34 | U-Net | **0.8059** | **0.6922** | 0.8153 | 0.8392 | +0.024 | 54 |
| 2 | unetpp_resnet34 | ResNet34 | U-Net++ | 0.7980 | 0.6818 | **0.8162** | 0.8310 | +0.015 | 50 |
| 3 | deeplabv3p_resnet50 | ResNet50 | DeepLabV3+ | 0.7811 | 0.6591 | 0.7773 | 0.8293 | +0.052 | 39 |
| 4 | deeplabv3p_effnetb3 | EffNet-B3 | DeepLabV3+ | 0.7751 | 0.6497 | 0.7517 | **0.8420** | **+0.090** | 31 |
| 5 | unet_effnetb0 | EffNet-B0 | U-Net | 0.7419 | 0.6120 | 0.7665 | 0.7915 | +0.025 | 15 |

#### Hipotez 1 (encoder baskın) — ÇÜRÜDÜ

Model 4 sonrası yazılan "encoder etkisi decoder'ın 8 katı" ifadesi üç modelden yapılmış aşırı genellemeydi; dördüncü hücre gelince düştü. 2×2 tablo bir **etkileşim** gösteriyor:

```
                ResNet     EffNet    encoder etkisi
U-Net           0.8059     0.7419        -0.0640
DeepLabV3+      0.7811     0.7751        -0.0060
decoder etkisi -0.0248    +0.0332
```

ResNet→EffNet cezası U-Net ile 0.064, DeepLabV3+ ile 0.006 (10 kat fark). Decoder etkisinin **işareti** encoder'a göre değişiyor. Doğru ifade: encoder ile decoder etkileşiyor, tek başına hiçbiri baskın değil. Raporda "encoder daha önemli" **yazılmamalı**.

**Rakip açıklama (daha basit, elenemedi)**: EffNet-B3 (12M), EffNet-B0'ın (5.3M) iki katından fazla. Model 5'in model 2'yi geçmesi decoder'dan değil sadece daha büyük encoder'dan kaynaklanıyor olabilir. Ayırt edici eksik hücre: **U-Net + EffNet-B3**. ~0.80 çıkarsa sorun B0'ın küçüklüğüydü; ~0.74 çıkarsa etkileşim gerçek. Opsiyonel 7. deney (~1 saat).

#### Hipotez 2 (DeepLabV3+ aşırı segmentasyon) — GÜÇLENDİ

İki farklı encoder'da doğrulandı, temiz ayrım — U-Net ailesi hepsi R−P < 0.03, DeepLabV3+ hepsi > 0.05:

| Decoder | Encoder | R−P |
|---|---|---|
| U-Net++ | ResNet34 | +0.015 |
| U-Net | ResNet34 | +0.024 |
| U-Net | EffNet-B0 | +0.025 |
| DeepLabV3+ | ResNet50 | +0.052 |
| DeepLabV3+ | EffNet-B3 | +0.090 |

Model 5 tablonun en yüksek recall'una (0.8420) ve en düşük precision'ına (0.7517) sahip: dens dokuyu en iyi buluyor, sınırlarından en çok taşıyor. Kaba 4× upsample + ince ölçekli skip connection yokluğunun imzası.

**Model 6 için hipotez 2'den türeyen tahmin**: SegFormer'ın decoder'ı da all-MLP başlık + 4× upsample, ince skip connection yok — yapısal olarak DeepLabV3+'a benziyor. Hipotez 2 doğruysa Dice'ından bağımsız olarak **R−P > 0.05** çıkmalı.

**Eğitim notu**: model 5 epoch 2'de Dice 0.18 / recall 0.12'ye çöküp epoch 7'de toparladı (warmup dönemi), kalıcı etkisi olmadı.

### Model 6/6 sonucu — `segformer_b2` TAMAMLANDI · ALTI MODEL BİTTİ

| # | Model | Encoder | Decoder | Decoder tipi | Dice | IoU | Precision | Recall | R−P | En iyi epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | unet_resnet34 | ResNet34 | U-Net | ince skip | **0.8059** | **0.6922** | 0.8153 | 0.8392 | +0.024 | 54 |
| 2 | unetpp_resnet34 | ResNet34 | U-Net++ | ince skip | 0.7980 | 0.6818 | **0.8162** | 0.8310 | +0.015 | 50 |
| 3 | segformer_b2 | MiT-B2 | SegFormer | kaba 4× | 0.7932 | 0.6742 | 0.7883 | 0.8396 | +0.051 | 33 |
| 4 | deeplabv3p_resnet50 | ResNet50 | DeepLabV3+ | kaba 4× | 0.7811 | 0.6591 | 0.7773 | 0.8293 | +0.052 | 39 |
| 5 | deeplabv3p_effnetb3 | EffNet-B3 | DeepLabV3+ | kaba 4× | 0.7751 | 0.6497 | 0.7517 | **0.8420** | +0.090 | 31 |
| 6 | unet_effnetb0 | EffNet-B0 | U-Net | ince skip | 0.7419 | 0.6120 | 0.7665 | 0.7915 | +0.025 | 15 |

#### ANA BULGU — decoder tipi aşırı segmentasyonu belirliyor (hipotez 2 doğrulandı)

Model 5'ten sonra yapılan tahmin — "SegFormer'ın decoder'ı da kaba 4× upsample, R−P > 0.05 çıkmalı" — **tuttu (+0.0513)**. Altı model ve üç farklı decoder mimarisi boyunca **örtüşme yok**:

```
ince skip connection (U-Net, U-Net++)      : +0.015, +0.024, +0.025   (max 0.025)
kaba 4x upsample (DeepLabV3+, SegFormer)   : +0.051, +0.052, +0.090   (min 0.051)
                                              aradaki bosluk: 0.026
```

Encoder'dan bağımsız, tamamen decoder mimarisiyle belirlenen bir imza. Mekanizma: ince ölçekli skip connection'a sahip decoder'lar sınırları koruyor; sondan 4× upsample yapanlar dens dokuyu buluyor (recall yüksek) ama sınırlarından taşıyor (precision düşük). Meme yoğunluğu dağınık ve girintili çıkıntılı sınırlara sahip olduğu için bu ceza burada belirgin. Raporun ana teknik bulgusu bu olmalı.

#### Ölçülen hata payları (varsayım değil)

Karşılaştırma hücresi çalıştırıldı, per-görüntü Dice std'leri ölçüldü (n=89):

| Model | Dice | std | SEM |
|---|---|---|---|
| unet_resnet34 | 0.8059 | 0.1353 | 0.0143 |
| unetpp_resnet34 | 0.7980 | 0.1374 | 0.0146 |
| deeplabv3p_resnet50 | 0.7811 | 0.1409 | 0.0149 |
| deeplabv3p_effnetb3 | 0.7751 | 0.1349 | 0.0143 |
| unet_effnetb0 | 0.7419 | 0.1645 | 0.0174 |

Önceki bölümlerde varsayılan 0.10-0.20 aralığı doğrulandı; gerçek değerler 0.135-0.165.

**Eşleşmemiş (unpaired) karşılaştırma, en iyiye göre**: `unetpp_resnet34` 0.39 SEM, `segformer_b2` ~0.62 SEM, `deeplabv3p_resnet50` 1.20 SEM, `deeplabv3p_effnetb3` 1.52 SEM, `unet_effnetb0` **2.84 SEM (tek anlamlı fark)**.

#### Metodoloji düzeltmesi — eşleşmiş test kullanılmalı

İlk yazılan karşılaştırma hücresi eşleşmemiş test yapıyordu; **bu veride fazla muhafazakâr**. Ölçülen std ~0.14'ün büyük kısmı modeller arası farkı değil **vakalar arası zorluk farkını** yansıtıyor (zor vaka her model için zor). Altı model aynı 89 vakada değerlendirildiği ve `val_loader` `shuffle=False` olduğu için i'inci skor her modelde aynı vakaya karşılık geliyor — eşleştirme geçerli ve ortak varyans sadeleşiyor.

Hücre güncellendi: vaka başına fark, **paired SEM**, t oranı ve **Wilcoxon signed-rank p** hesaplanıyor; sonuç `comparison_table.xlsx` içinde `paired_vs_best` sayfasına yazılıyor. "Hepsi ayırt edilemez" sonucu testin zayıflığından kaynaklanıyor olabilir — kesin cevap hücre yeniden çalıştırılınca gelecek.

#### İlk üç ayırt edilemez (eşleşmemiş teste göre — eşleşmiş sonuç beklendikten sonra güncellenecek)

0.8059 / 0.7980 / 0.7932 — yayılım sadece 0.0127. 89 validasyon vakasıyla, per-görüntü Dice std'si için hangi makul değer alınırsa alınsın 1. ile 3. arasındaki fark 1 SEM'in altında (std 0.10 → 0.85 SEM; std 0.15 → 0.56; std 0.20 → 0.42). **Raporda "U-Net+ResNet34 en iyi model" denmemeli**; doğru ifade "ilk üç bu veride ayırt edilemez, U-Net+EffNet-B0 açık farkla sonuncu". Kesin rakam için karşılaştırma hücresi artık per-görüntü std/SEM hesaplıyor (aşağıya bkz.).

#### Karşılaştırma tablosu hücresi eklendi

Notebook'un sonuna iki hücre eklendi (`comparison_table.xlsx` üretir):
- 6 modelin `test_summary.xlsx` dosyalarını birleştirir
- Her modelin `best.pt`'siyle val setini bir kez geçip **per-görüntü Dice dağılımını** çıkarır → `dice_std`, `dice_sem`
- En iyi modele göre farkları birleşik SEM cinsinden raporlar (|fark| < 2 SEM → "ayırt edilemez")
- Decoder imzası tablosunu (recall − precision) basar

Herhangi bir model çalıştırılmadan, notebook baştan sona koşturulup son hücreye gelinerek üretilebilir (`build_model` + `best.pt` yeterli, yeniden eğitim yok).

### Görselleştirme düzeltmeleri (model 1'in çıktıları incelendikten sonra)

İlk `curves.png` ve `5cases.png` üretildikten sonra dört kusur tespit edilip düzeltildi. Hiçbiri yeniden eğitim gerektirmiyor — mevcut Excel logu ve `best.pt` ile grafikler saniyeler içinde yeniden üretiliyor. **Kalan 5 model başlamadan düzeltildi ki 6 figür seti tutarlı olsun.**

1. **`cmap='Reds'` tüm paneli beyazlatıyordu.** `Reds` colormap'inde değer 0 → `(1.0, 0.961, 0.941)`, yani beyaz ve tam opak (yerelde doğrulandı). `alpha=0.4` ile maskenin sıfır olduğu her piksele %40 beyaz peçe çekiliyordu; bu yüzden orijinal panelin arka planı siyah, GT/tahmin panellerininki gri çıkıyordu ve mamografi detayı tüm görüntüde soluyordu. Yerine `overlay_mask()`: sadece `mask==1` pikselleri renkli, gerisi tamamen şeffaf (`alpha=0`).
2. **`suptitle` ilk satırın panel başlıklarını eziyordu** → `tight_layout(rect=[0,0,1,0.985])`.
3. **Metrik panelleri otomatik ölçekleniyordu.** Recall paneli 0.725-0.925 aralığına zoomlanıp ±0.1'lik dalgalanmayı kararsızlık gibi gösteriyordu. Daha önemlisi 6 modelin figürleri farklı ölçeklerde çıkacağı için görsel olarak karşılaştırılamazdı. Dice/IoU/Precision/Recall panelleri artık sabit `METRIC_YLIM = (0, 1)`.
4. **F1 paneli Dice panelinin birebir kopyasıydı** (metrik düzeltmesinden sonra `f1 = dice`). O panel **LR takvimine** ayrıldı (log ölçek) — Dice'ın epoch 29/36/47/51'deki sıçramalarının LR düşüşleriyle hizalandığı orada görülüyor. Başlığa `F1 = Dice` notu eklendi.

### Gözlem: görüntülerin yarısından fazlası boş dolgu (şimdilik işlem yok)

5 vaka görselinde her 512×512 görüntünün üst ~%40-45'inde meme dokusu, altında keskin yatay bir sınır ve saf siyah dolgu var. Kaynak görüntüler enine olup alttan padding'lenmiş görünüyor. Sonuç: model 512×512 bütçesinin yarısından fazlasını boş tuvale harcıyor, meme fiilen ~512×230 çözünürlükte işleniyor.

İlk refleks meme bounding box'ına kırpmak (etkin çözünürlük ~2 katına çıkar). **Ama `mass_lezyon_projesi` bu ablasyonu zaten yapmış ve kırpma işe yaramamış**: `mamosis_mass_lezyon_yolo11s.ipynb` (kırpmasız) mAP@50 **0.6928**, `_kirpilmis` **0.6901** — pratikte eşit (bkz. `ILERLEME_RAPORU.md`). Mekanizma birebir aynı değil (nesne tespitinde kutu lokal, burada doku yaygın ve çözünürlük daha çok önemli olabilir) ama aynı görüntü alanında yapılmış doğrudan bir karşı-kanıt.

**Karar: dokunulmuyor.** Şimdi kırpma eklemek `unet_resnet34`'ün baştan eğitilmesini ve 6 modelin karşılaştırılabilirliğinin bozulmasını gerektirir; beklenen kazanç ise kendi verilerine göre belirsiz. Rapordaki "gelecek çalışma" bölümüne not edilecek.

### Sıradaki adım

### Rapor üretildi

`mass_lezyon_projesi/mass_lezyon_sonuc_raporu.pdf` örnek alınarak aynı yapı ve üretim yöntemiyle hazırlandı (HTML → headless Chrome ile PDF).

| Dosya | Açıklama |
|---|---|
| `rapor/density_segmentation_raporu.html` | Kaynak, 4,43 MB, figürler base64 gömülü — tek dosya, bağımlılık yok |
| `rapor/density_segmentation_sonuc_raporu.pdf` | **21 sayfa, 3,59 MB** |
| `rapor/figurler/*.png` | Arşiv notebook'larından çıkarılmış 12 figür (yeniden üretim için) |

Bölümler: 1. Amaç ve Kapsam · 2. Modellerde Kullanılan Yöntemler (+ literatür tablosu) · 3. Kendi Veri Setimizdeki Uygulama (+ tespit edilen iki hata) · 4. Yöntem Bazlı Sonuçlar (6 alt bölüm, her biri metrik tablosu + eğri + 5 vaka) · 5. Altı Yöntemin Karşılaştırılması · 6. Sonuç ve Değerlendirme (+ 7 maddelik sınırlamalar) · Kaynaklar.

Yeniden üretmek için: `scratchpad/build_report.py` HTML'i yeniden yazar, ardından
`chrome --headless --no-pdf-header-footer --print-to-pdf=... file:///...raporu.html`.

#### Araştırmadan çıkan iki kritik bulgu

1. **Veri setinin kaynağı bulundu** — Behravan vd. (2024), *Data in Brief*. Orijinal: VinDr-Mammo'dan seçilmiş 745 görüntü (596/149), JPG 2800×3518, uzman radyolog maskesi, **hem meme alanı hem dens doku**. Elimizdeki 536 vakalık NIfTI 512×512 sürüm bunun bir alt kümesi; meme alanı maskesi bu sürümde yok, bu yüzden "dens alan / meme alanı" oranı doğrudan hesaplanamıyor. Kitle projesiyle aynı kaynak koleksiyon (VinDr-Mammo).
2. **Radyologlar arası uyum Dice = 0,76 ± 0,17** (Larroza vd., 2022). Bu çalışmadaki en iyi sonuç (0,8059) bu insan-insan uyum düzeyinin **üzerinde**. Yorumu değiştiriyor: modeller arası 0,01-0,03'lük farklar referans etiketin kendi belirsizliğinin çok altında, dolayısıyla bu farklara mimari üstünlüğü atfetmek savunulabilir değil. Ayrıca ölçülen per-görüntü std'lerimiz (0,135-0,165) literatürdeki ±0,13-0,14 bandıyla örtüşüyor — değerlendirme bu görev için olağan aralıkta.

#### Eşleşmiş test sonuçları — ÖNCEKİ İSTATİSTİK ÇIKARIMI TERSİNE ÇEVİRDİ

Karşılaştırma hücresi eşleşmiş testle çalıştırıldı. Sonuç, eşleşmemiş teste dayanan "ilk beş model ayırt edilemez" çıkarımını **geçersiz kıldı**:

| Model (referans: unet_resnet34) | Fark | Eşleşmemiş SEM | Eşleşmiş SEM | Güç | Wilcoxon p | Sonuç |
|---|---|---|---|---|---|---|
| unetpp_resnet34 | 0,0078 | 0,0204 | 0,0032 | 6,4× | 0,054 | ayırt edilemez (sınırda) |
| segformer_b2 | 0,0127 | 0,0201 | 0,0042 | 4,8× | 4,7e-4 | **anlamlı** |
| deeplabv3p_resnet50 | 0,0247 | 0,0206 | 0,0038 | 5,4× | 2,2e-8 | **anlamlı** |
| deeplabv3p_effnetb3 | 0,0309 | 0,0202 | 0,0046 | 4,4× | 1,1e-8 | **anlamlı** |
| unet_effnetb0 | 0,0642 | 0,0225 | 0,0091 | 2,5× | 4,5e-13 | **anlamlı** |

Eşleşmiş standart hatalar 2,5-6,4 kat küçük. **Eşleşmemiş: 1/5 anlamlı → Eşleşmiş: 4/5 anlamlı.** U-Net + ResNet34 artık istatistiksel olarak da savunulabilir bir seçim; yalnızca U-Net++ ile arasındaki fark sınırda (p=0,054).

Eklenen çekinceler (rapor 5.3 ve 6.6'da): (a) her mimari tek çekirdekle bir kez eğitildi, eşleşmiş test eğitim rastgeleliğini kapsamıyor; (b) farklar 0,008-0,031 aralığında, radyolog uyumunun (0,76 ± 0,17) bir mertebe altında — sıralama gerçek, pratik karşılığı mütevazı.

SegFormer-B2 std/SEM: 0,1340 / 0,0142. Raporda bekleyen alan kalmadı.

### Kalan iş

1. **`comparison_table.xlsx` üretilecek** — notebook'un sonundaki karşılaştırma hücresi çalıştırılarak. (`MODEL_KEY` hangi modelde olursa olsun fark etmez; hücre 6 modeli de kendi `best.pt`'lerinden okuyor.) Çıktı: ilk üçün gerçekten ayrışıp ayrışmadığının SEM cinsinden cevabı.
2. **Final PDF raporu** — her model bir bölüm (hiperparametreler + metrik tablosu + 6 grafik + 5 vaka) + karşılaştırma tablosu + en başarılı model teknik değerlendirmesi + referanslar.

Rapora mutlaka girmesi gereken üç nokta: (a) decoder tipi ↔ aşırı segmentasyon bulgusu, (b) ilk üçün istatistiksel olarak ayırt edilemez olduğu, (c) erken durdurmanın gürültülü Dice eğrisine sahip modelleri sistematik olarak daha erken kestiği (model 2).

**Opsiyonel 7. deney**: `U-Net + EfficientNet-B3`. Model 5'in model 2'yi geçmesinin decoder'dan mı yoksa sadece daha büyük EffNet'ten mi kaynaklandığını ayırt eden tek hücre. ~1 saat.

> **Colab iş akışı**: biten modelin notebook'unu indirirken `density_segmentation_train.ipynb`'nin **üzerine yazmayın** — doğrudan `density_segmentation_<MODEL_KEY>.ipynb` adıyla kaydedin. Üzerine yazılırsa şablon önceki modelin çıktılarıyla dolu ve yanlış etiketli kalıyor (model 2 ve 3'te oldu, elle ayrıştırıldı).

Model değiştirmek için **tek satır** yeterli — doğrulandı, model adı notebook'ta başka hiçbir yerde elle geçmiyor (`MODEL_CONFIGS` sadece arama tablosu):

```python
MODEL_KEY = 'unet_effnetb0'
```

`CFG`, `DRIVE_DIR`, `CKPT_DIR`, `CASES_DIR`, `EXCEL_LOG`, `CURVES_PNG`, `TEST_SUMMARY_XLSX`, batch size, LR'ler, `max_epochs`, `patience` — hepsi bundan türüyor. `RESUME = False` kalmalı.

Kalan sıra: `unetpp_resnet34`, `deeplabv3p_resnet50`, `deeplabv3p_effnetb3`, `segformer_b2`.

**Her model bitince**: 3 çıktı hücresi çalıştırılır (final değerlendirme + curves + 5 vaka), sonra notebook `density_segmentation_<MODEL_KEY>.ipynb` olarak arşivlenir, şablon bir sonraki modele ayarlanır.

**Sağlamalar**:
- Final değerlendirme hücresi, eğitim logundaki en iyi epoch'un Dice'ını birebir basmalı (val'de augmentation yok, model eval-mode'da). Farklıysa `best.pt` yüklemesinde sorun var.
- Epoch 0 val Dice ≥ 0.25 olmalı (model 1'de 0.2749).
- `eval_cases.json` zaten yazılmış durumda; her model aynı 5 vakayı okuyacak, görseller karşılaştırılabilir.

**`.npy` cache** yeni Colab oturumunda boş başlar; ilk epoch Drive okumasıyla yavaş, sonrası hızlı. Aynı oturumda kalınırsa modeller arası da korunur.

**Dikkat**: DeepLabV3+ ve SegFormer için `max_epochs=60`. `unet_resnet34` epoch 54'te en iyisini bulduğuna göre bu üç model kapağa dayanabilir; koşu sonunda Dice hâlâ tırmanıyorsa `max_epochs` yetersiz kalmış demektir, logdan kontrol edilmeli.

> **Not**: bu repodaki `.ipynb` güncellendi ama Colab oturumu ayrı — düzeltmelerin etkili olması için Drive'daki notebook'un yeniden yüklenmesi ya da değişen 7 hücrenin elle kopyalanması gerekiyor (bkz. Kod Yapısı bölümündeki senkronizasyon notu). Değişen hücreler: Dataset+Augmentation, Model Factory, Loss/Metrikler, Checkpoint/Resume, Eğitim Döngüsü (2 hücre), 5 Vaka Görseli.

### Her yeni model koşusunda ilk epoch'larda kontrol edilecekler

Kalan 5 model için de aynı kontroller geçerli:

- Dataset hücresinin doğrulama print'i: `Girdi tensor araligi: [-2.1xx, 2.2xx]` benzeri geniş bir aralık, `maske degerleri: [0.0, 1.0]`. Aralık `[-2.118, -2.101]` gibi dar gelirse `max_pixel_value` düzeltmesi uygulanmamış demektir.
- Epoch 0 val Dice **0.25 civarı ve üstü** olmalı (`unet_resnet34`'te 0.2749). 0.10 civarı gelip sonra çökerse fp16 Dice taşması ya da normalizasyon düzeltmesi Colab oturumuna geçmemiştir.
- Val Dice ilk birkaç epoch'tan sonra monoton yükselmeli; 0.0002 ↔ 0.5 arası sıçrama tekrarlarsa düzeltmeler uygulanmamıştır.
- İlk epoch Drive'dan okuma yüzünden yavaş, ikinci epoch'tan itibaren `.npy` cache devreye girip belirgin hızlanmalı (cache tüm modellerde ortak, oturum kopmadıysa 2. modelden itibaren ilk epoch da hızlı).
- Train/val loss makasının kapanması: val loss train loss'un altındayken sağlıklı, kesişip train altta kalmaya başlayınca overfitting başlıyor demektir.

## Uygulama Sırası

1. ~~Ortak `data.py`/`augment.py`/`metrics.py`/`train.py` yazılır~~ → tek notebook olarak uygulandı (yukarıya bkz.)
2. `eval_cases.json` (5 sabit val vakası) — notebook ilk çalıştırıldığında otomatik üretiliyor, sonraki modeller aynı dosyayı okuyor
3. 6 model sırayla (hafiften ağıra) eğitilir, her biri kendi `<MODEL_NAME>/` klasörüne yazar — **şu an 1/6 aşamasında**
4. Her model bitince: curves.png, 5 vaka görseli, test_summary.xlsx üretilir
5. Tüm modeller bitince `comparison_table.xlsx` birleştirilir
6. Final PDF derlenir: her model bir bölüm (hiperparametreler + metrik tablosu + 6 grafik + 5 vaka) + karşılaştırma tablosu + en başarılı model teknik değerlendirmesi + referanslar
