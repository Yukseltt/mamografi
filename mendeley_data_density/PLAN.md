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
- **BatchNorm momentum düşürüldü** (varsayılan 0.1 → 0.01, model oluşturulduktan hemen sonra tüm `BatchNorm2d` katmanlarına uygulanıyor): küçük batch (8-16) ve görüntüden görüntüye değişen doku oranı yüzünden running mean/var gürültülü kalıyordu, bu da eval-mode (validasyon) çıktısının epoch'tan epoch'a çok farklı bir dağılıma kaymasına (bazen tamamen "arkaplan" tahmini) yol açıyordu — momentum düşürülerek running istatistikler daha kararlı hale getirildi.
- Augmentation: horizontal/vertical flip, ±10-15° rotation, brightness/contrast jitter, hafif elastic deformation, hafif CoarseDropout — agresif crop/zoom yok (global doku oranı korunmalı). MICCAI 2024'te küçük organ segmentasyonunda CutMix'in daha iyi sonuç verdiği bulgusu değerlendirildi ancak kullanıcı tercihiyle mevcut set korundu, CutMix eklenmedi.
- Gerekirse gradient accumulation ile efektif batch sabitlenecek
- **Dice metriği manuel hesaplanıyor** (TP/FP/FN üzerinden), `torchmetrics.functional.dice` bazı sürümlerde mevcut olmadığı için (`AttributeError`) sürümden bağımsız hale getirildi; IoU/Precision/Recall/F1 hâlâ `torchmetrics.functional` (`task='binary'`) ile hesaplanıyor.

## Şu Anki Durum (güncel ilerleme)

- **Model 1/6 (`unet_resnet34`) Colab'da aktif olarak eğitiliyor/hata ayıklanıyor.** Diğer 5 model henüz başlatılmadı.
- Sırasıyla şu sorunlar tespit edilip düzeltildi: (1) case eşleştirme doğrulandı (447 train/89 val, dosya adları uyumlu), (2) `torchmetrics.dice` `AttributeError`'ı → manuel Dice hesaplamasına geçildi, (3) `torch.cuda.amp.*` deprecation uyarısı → `torch.amp.*` API'sine geçildi, (4) ilk eğitim denemelerinde val loss/dice epoch'tan epoch'a çok sert sıçrıyordu (train_loss düzgün azalırken) → 3 epoch warmup + gradient clipping + BatchNorm momentum düşürme eklendi.
- **Henüz doğrulanmadı**: bu üç düzeltmenin (warmup+clipping+BN momentum) volatiliteyi gerçekten çözüp çözmediği — bir sonraki adım bunu en az 10-15 epoch boyunca gözlemlemek.
- SegFormer-B2 dahil diğer 5 mimari için notebook kodu hazır ama hiç çalıştırılmadı; `unet_resnet34` stabilize olduktan sonra sırayla denenecek.

## Uygulama Sırası

1. ~~Ortak `data.py`/`augment.py`/`metrics.py`/`train.py` yazılır~~ → tek notebook olarak uygulandı (yukarıya bkz.)
2. `eval_cases.json` (5 sabit val vakası) — notebook ilk çalıştırıldığında otomatik üretiliyor, sonraki modeller aynı dosyayı okuyor
3. 6 model sırayla (hafiften ağıra) eğitilir, her biri kendi `<MODEL_NAME>/` klasörüne yazar — **şu an 1/6 aşamasında**
4. Her model bitince: curves.png, 5 vaka görseli, test_summary.xlsx üretilir
5. Tüm modeller bitince `comparison_table.xlsx` birleştirilir
6. Final PDF derlenir: her model bir bölüm (hiperparametreler + metrik tablosu + 6 grafik + 5 vaka) + karşılaştırma tablosu + en başarılı model teknik değerlendirmesi + referanslar
