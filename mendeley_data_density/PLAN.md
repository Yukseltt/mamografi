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

- **`mendeley_data_density/density_segmentation_train.ipynb`** — 6 modelin hepsi için ortak şablon. Tüm mimariler, hiperparametreler, veri pipeline'ı, loss/metrik, log/checkpoint yardımcıları tek notebook içinde hücrelere bölünmüş halde tanımlı; `MODEL_KEY` değişkeni hangi modelin eğitileceğini seçiyor (`MODEL_CONFIGS` dict'i içinde 6 modelin de hiperparametreleri tanımlı).
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
| DeepLabV3+ + ResNet50 | 8 | 1e-4 / 5e-4 | 60 | 512 |
| DeepLabV3+ + EffNet-B3 | 8 | 1e-4 / 5e-4 | 60 | 512 |
| SegFormer-B2 | 8 | 1e-4 (tek LR, ayrım yok) | 60 | 512 |

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

- **Model 1/6 (`unet_resnet34`) hâlâ ilk başarılı eğitim koşusunu bekliyor.** Diğer 5 model henüz başlatılmadı.
- Sırasıyla şu sorunlar tespit edilip düzeltildi: (1) case eşleştirme doğrulandı (447 train/89 val, dosya adları uyumlu), (2) `torchmetrics.dice` `AttributeError`'ı → manuel Dice hesaplamasına geçildi, (3) `torch.cuda.amp.*` deprecation uyarısı → `torch.amp.*` API'sine geçildi, (4) val loss/dice epoch'tan epoch'a çok sert sıçrıyordu (train_loss düzgün azalırken) → 3 epoch warmup + gradient clipping + BatchNorm momentum düşürme eklendi.

### Volatilitenin kök nedeni bulundu (çözüldü)

14 epoch'luk gözlem, (4)'teki üç düzeltmenin volatiliteyi **çözmediğini** gösterdi — train loss 1.62→0.47 düzgün inerken val Dice 0.0002 ile 0.56 arasında rastgele zıplamaya devam etti. Semptomun kaynağı optimizasyon değil, **girdi normalizasyonuydu**:

Görüntü Dataset içinde zaten min-max ile `[0,1]`'e çekiliyordu, ardından `A.Normalize` varsayılan `max_pixel_value=255.0` ile çağrıldığı için ikinci kez 255'e bölünüyordu. Sonuç: `(img/255 - 0.485)/0.229` → tüm görüntü `[-2.118, -2.101]` aralığına sıkışıyor, dinamik aralık 0.017 (olması gereken ~4.37, tam **255x** fark — yerelde doğrulandı). Encoder'a giden şey neredeyse sabit bir düzlem oluyordu.

Bu tam olarak gözlenen semptomu üretir: BatchNorm train-mode'da batch istatistikleriyle minik sinyali geri büyütüp öğrenmeyi mümkün kılıyor (train loss düşüyor), ama eval-mode'da running mean/var ile batch istatistikleri arasındaki en ufak fark aynı oranda büyütüldüğü için val çıktısı her epoch farklı bir dağılıma kayıyordu. Warmup/clipping/BN-momentum semptomu tedavi etmeye çalıştığı için etkisiz kaldı.

### Uygulanan düzeltmeler (notebook güncellendi, Colab'da henüz çalıştırılmadı)

1. **`A.Normalize(..., max_pixel_value=1.0)`** — ana düzeltme. Tek bir `imagenet_normalize()` fabrika fonksiyonuna toplandı, üç yerde de (train/val transform + görselleştirmedeki `predict_mask`) aynı fonksiyon kullanılıyor.
2. **BatchNorm momentum varsayılana (0.1) döndürüldü** — 0.01 override'ı kaldırıldı.
3. **`A.ElasticTransform` alpha 1 → 30** — alpha deformasyon genliğini ölçeklediği için alpha=1 pratikte no-op'tu, elastic augmentation hiç çalışmıyordu.
4. **Tüm metrikler per-görüntü tanımına geçirildi** — Dice/IoU/Precision/Recall/F1 aynı TP/FP/FN'den, `torchmetrics` bağımlılığı metrik hesabından çıktı (bkz. Hiperparametreler).
5. **Yerel `.npy` cache** (`/content/npy_cache`) — her epoch 447 `.nii.gz` dosyasını Drive'dan okumak eğitimin en yavaş kısmıydı; ilk okumada yerel diske yazılıp sonraki epoch'larda oradan okunuyor (worker'lar arası yarım dosya olmasın diye atomik `os.replace`).
6. **`epochs_without_improve` checkpoint'e eklendi** — resume'da early stopping sayacı sıfırlanıyordu.
7. **Loss fp32'de hesaplanıyor (`logits.float()`)** — AMP altında logits fp16 geliyordu ve soft-Dice'ın `probs.sum()` terimi 262.144 piksel üzerinden alındığı için ortalama olasılık ~0.25'i geçtiğinde fp16 tavanını (65504) aşıp `inf`'e dönüyordu. Bu durumda `dice=0`, `dice_loss=1` sabitleniyor ve **hiç gradyan üretmiyordu** — yani loss fiilen salt BCE'ye düşüyordu. En çok yüksek yoğunluklu vakalarda (dens alan geniş) tetiklendiği için Dice terimi tam gerektiği yerde sessizce kapanıyordu. Eski logdaki epoch 0 `train_loss=1.6177` değeri bununla tutarlı (BCE ~0.62 + sabitlenmiş dice_loss 1.0).
8. Küçük: train loss `drop_last=True` yüzünden görülmeyen 15 örneği de bölene katıyordu (~%3 düşük raporlanıyordu); `find_resumable_checkpoint` checkpoint'i iki kez yüklüyordu (artık yüklenen dict doğrudan `restore_checkpoint`'e veriliyor).

Dataset hücresinin sonuna bir **doğrulama print'i** eklendi: girdi tensörünün aralığını ve maske değerlerini basıyor. Aralık `~[-2.1, 2.6]` görünmeli; hepsi -2.1 civarında sıkışıksa `max_pixel_value` hâlâ yanlış demektir.

### Sıradaki adım

`unet_resnet34`'ü **sıfırdan** (`RESUME=False`) yeniden eğitmek — mevcut `last.pt`/`best.pt` ve Excel logu bozuk girdiyle eğitildiği için kullanılamaz. Val Dice'ın artık monoton yükselmesi bekleniyor. Stabilize olduktan sonra SegFormer-B2 dahil diğer 5 mimari sırayla denenecek (notebook kodu hazır, hiç çalıştırılmadı).

> **Not**: bu repodaki `.ipynb` güncellendi ama Colab oturumu ayrı — düzeltmelerin etkili olması için Drive'daki notebook'un yeniden yüklenmesi ya da değişen 7 hücrenin elle kopyalanması gerekiyor (bkz. Kod Yapısı bölümündeki senkronizasyon notu). Değişen hücreler: Dataset+Augmentation, Model Factory, Loss/Metrikler, Checkpoint/Resume, Eğitim Döngüsü (2 hücre), 5 Vaka Görseli.

### İlk epoch'ta kontrol edilecekler

- Dataset hücresinin doğrulama print'i: `Girdi tensor araligi: [-2.1xx, 2.2xx]` benzeri geniş bir aralık, `maske degerleri: [0.0, 1.0]`. Aralık `[-2.118, -2.101]` gibi dar gelirse `max_pixel_value` düzeltmesi uygulanmamış demektir.
- Epoch 0 `train_loss` **1.6 değil ~1.3-1.4 civarı** olmalı — Dice terimi artık `inf`'e düşmediği için gerçek bir değer üretiyor.
- Val Dice ilk birkaç epoch'tan sonra monoton yükselmeli; 0.0002 ↔ 0.5 arası sıçrama tekrarlarsa normalizasyon düzeltmesi Colab oturumuna geçmemiştir.
- İlk epoch Drive'dan okuma yüzünden yavaş, ikinci epoch'tan itibaren `.npy` cache devreye girip belirgin hızlanmalı.

## Uygulama Sırası

1. ~~Ortak `data.py`/`augment.py`/`metrics.py`/`train.py` yazılır~~ → tek notebook olarak uygulandı (yukarıya bkz.)
2. `eval_cases.json` (5 sabit val vakası) — notebook ilk çalıştırıldığında otomatik üretiliyor, sonraki modeller aynı dosyayı okuyor
3. 6 model sırayla (hafiften ağıra) eğitilir, her biri kendi `<MODEL_NAME>/` klasörüne yazar — **şu an 1/6 aşamasında**
4. Her model bitince: curves.png, 5 vaka görseli, test_summary.xlsx üretilir
5. Tüm modeller bitince `comparison_table.xlsx` birleştirilir
6. Final PDF derlenir: her model bir bölüm (hiperparametreler + metrik tablosu + 6 grafik + 5 vaka) + karşılaştırma tablosu + en başarılı model teknik değerlendirmesi + referanslar
