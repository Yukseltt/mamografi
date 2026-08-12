"""RTMDet-Ins / BUSI lezyon segmentasyonu - mmdet config.

Girdi boyutu GIRDI degiskeniyle secilir (256 ve 512 icin iki ayri kosu planlandi,
bkz. PLAN.md "Cikti-stride tavani").

Augmentasyon SegFormer ve YOLO ile birebir ayni albumentations zinciri (Albu
sarmalayicisi uzerinden). RTMDet'in kendi Mosaic/MixUp recipe'i bilerek kapatildi:
uc model arasindaki farkin augmentasyondan degil mimariden gelmesi icin.
"""
_base_ = 'mmdet::rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'

# ---- deney basina degisen ----
# 256: diger iki modelle esit girdi butcesi, maske izgarasi 32x32, tavan 0.9814
# 512: esit maske tavani (izgara 64x64, tavan 0.9922) -- bkz. PLAN.md
GIRDI = 256
BATCH = 8                   # 256'da tepe VRAM 606 MB; 512'de ~2.4 GB bekleniyor.
                            # OOM olursa 4'e dusur (lr otomatik olceklenir).
MAX_EPOCH = 100             # 70 denendi ve geri alindi. "100 epoch'luk kosunun ilk
                            # N epoch'u" analizi butce secmek icin gecerli degil:
                            # MAX_EPOCH'u kisaltmak takvimi kirpmiyor, cosine'i
                            # yeniden olcekliyor. 70'te zirve ep28'e dusup segm_mAP
                            # 0.642 -> 0.621 oldu. Kirilim kaybin maske kalitesinden
                            # degil skor kalibrasyonundan geldigini gosterdi:
                            # oracle instance Dice 70'lik kosuda daha iyiydi (0.892
                            # vs 0.885), ama tespitlerin guveni dusuk kaldigi icin
                            # esik 0.4'te Dice 0.771'e iniyordu (100'lukte 0.808).
AUG_SEVIYE = 'orta'         # 'hafif' | 'orta'
TOHUM = 44                  # Faz 2: 42 / 43 / 44 ile uc kosu

KOK = 'd:/mamografi/real_time_segmentasyon'
data_root = f'{KOK}/Dataset_BUSI_with_GT/'
ANN = f'{KOK}/veri/coco'

metainfo = dict(classes=('lezyon',), palette=[(220, 20, 60)])
work_dir = f'{KOK}/rtmdet/calisma/rtmdet_ins_tiny_{GIRDI}_{AUG_SEVIYE}_s{TOHUM}'

# COCO on-egitimli agirliktan fine-tune
load_from = ('https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
             'rtmdet-ins_tiny_8xb32-300e_coco/'
             'rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth')

model = dict(bbox_head=dict(num_classes=1))

# ---- augmentasyon: rt_seg_augmentasyon.ipynb ile birebir ayni ----
ALBU = {
    'hafif': [
        dict(type='HorizontalFlip', p=0.5),
        dict(type='Affine', translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
             rotate=(-10, 10), p=0.5),
        dict(type='RandomBrightnessContrast', brightness_limit=0.15,
             contrast_limit=0.15, p=0.3),
    ],
    'orta': [
        dict(type='HorizontalFlip', p=0.5),
        dict(type='Affine', translate_percent=(-0.1, 0.1), scale=(0.85, 1.15),
             rotate=(-15, 15), p=0.7),
        dict(type='ElasticTransform', alpha=30, sigma=6, p=0.25),
        dict(type='RandomBrightnessContrast', brightness_limit=0.2,
             contrast_limit=0.2, p=0.5),
        dict(type='RandomGamma', gamma_limit=(80, 120), p=0.3),
        dict(type='MultiplicativeNoise', multiplier=(0.85, 1.15), elementwise=True, p=0.3),
        dict(type='MotionBlur', blur_limit=7, p=0.2),
    ],
}[AUG_SEVIYE]

# Albu yerine AlbuBosGuvenli: mmdet'in Albu'su bos anotasyonlu orneklerde cokuyor
# (transforms.py:1770), BUSI'de bunlar 92 negatif ornek. bkz. ozel_transformlar.py
custom_imports = dict(imports=['ozel_transformlar', 'ozel_metrikler'],
                      allow_failed_imports=False)

albu_sarmalayici = dict(
    type='AlbuBosGuvenli',
    transforms=ALBU,
    bbox_params=dict(type='BboxParams', format='pascal_voc',
                     label_fields=['gt_bboxes_labels'],
                     min_visibility=0.0, filter_lost_elements=True),
    keymap={'img': 'image', 'gt_masks': 'masks', 'gt_bboxes': 'bboxes'},
)

# Albu, Resize'dan once: augmentasyon notebook'unda da aug orijinal cozunurlukte
# uygulanip Resize en sona konuluyor. Ayni sira korunuyor.
# keep_ratio=True + kareye Pad (letterbox). keep_ratio=False ile x/y olcek
# katsayilari farkli oluyordu ve RTMDet-Ins'in maske geri-olcekleme kodu tek
# katsayi varsaydigi icin maske goruntuyle hizalanmiyordu: olculen sonuc, maskenin
# yalnizca %59'u kendi tahmin kutusunun icine dusuyordu (bazi goruntulerde %0) ve
# segm_mAP hep 0 cikiyordu. Kutuyu maske olarak kullanmak Dice 0.804 verirken
# gercek maskeler 0.49'da kaliyordu -- yani kayip tamamen tasima hatasindandi.
# Tek olcek katsayisi ile bu ortadan kalkiyor. YOLOv11-Seg de letterbox kullandigi
# icin modeller arasi tutarlilik da artiyor.
#
# Maske olceklemesi bilerek mmdet'in varsayilani (cv2.INTER_NEAREST) ile birakildi.
# INTER_NEAREST maskeyi kucultturken +0.5 px kaydiriyor ve bunu "duzeltmek" icin
# nearest_exact kullanan bir Resize denendi -- OLCUM REDDETTI. mmdet'in cikarim
# tarafindaki geri-olcekleme ters yonde ~-2 px kayma tasiyor; nearest'in bias'i
# onu telafi ediyormus. Ayrintili sayilar PLAN.md hata 9'da.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    albu_sarmalayici,
    dict(type='Resize', scale=(GIRDI, GIRDI), keep_ratio=True),
    dict(type='Pad', size=(GIRDI, GIRDI), pad_val=dict(img=(114, 114, 114), masks=0)),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(GIRDI, GIRDI), keep_ratio=True),
    dict(type='Pad', size=(GIRDI, GIRDI), pad_val=dict(img=(114, 114, 114), masks=0)),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# ---- veri ----
train_dataloader = dict(
    batch_size=BATCH,
    num_workers=2,                 # Windows'ta spawn maliyeti yuksek, dusuk tutuldu
    persistent_workers=True,
    dataset=dict(
        _delete_=True,             # base'deki MultiImageMixDataset (Mosaic) sarmalayicisi kaldirilir
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'{ANN}/busi_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),   # bos maskeli normal goruntuler kalsin
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'{ANN}/busi_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'{ANN}/busi_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

# CocoMetric yerine CocoMetricHizali: RTMDet-Ins maskeyi ori_shape'ten 1 px kisa
# uretebiliyor, COCOeval boyut uyusmayinca o goruntuyu sifir sayiyor. 256'da val
# goruntulerinin %21.6'si, 512'de %1.7'si etkileniyordu -- iki kosunun segm_mAP
# farkinin tamami buydu. bkz. ozel_transformlar.py
# Dice metrigi ikinci sirada: checkpoint secimi artik `busi/dice_tum` uzerinden,
# yani raporlanan metrigin kendisiyle. Onceki kosularda secim `segm_mAP`
# uzerindendi ve o metrik bozuk cikinca yanlis epoch secilmisti (hata 8).
val_evaluator = [
    dict(type='CocoMetricHizali', ann_file=f'{ANN}/busi_val.json',
         metric=['bbox', 'segm'], format_only=False),
    dict(type='BusiDiceMetric', manifest=f'{KOK}/veri/veri_manifest.csv'),
]
test_evaluator = dict(
    type='CocoMetricHizali',
    ann_file=f'{ANN}/busi_test.json',
    metric=['bbox', 'segm'],
    format_only=False,
)

# Val loss icin ayri hat: anotasyonlar Resize'dan ONCE yukleniyor ki hedefler
# model girdisiyle ayni uzayda olsun. Augmentasyon yok -- olcum deterministik olmali.
val_loss_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='Resize', scale=(GIRDI, GIRDI), keep_ratio=True),
    dict(type='Pad', size=(GIRDI, GIRDI), pad_val=dict(img=(114, 114, 114), masks=0)),
    dict(type='PackDetInputs'),
]

val_loss_dataloader = dict(
    batch_size=BATCH,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'{ANN}/busi_val.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=val_loss_pipeline,
    ),
)

# ---- egitim takvimi ----
# base lr 0.004, toplam batch 256 (8 GPU x 32) icin. Lineer olcekleme: 8/256 * 0.004
base_lr = 0.004 * BATCH / 256
train_cfg = dict(max_epochs=MAX_EPOCH, val_interval=1, dynamic_intervals=None)

# AMP bilerek KAPALI. AmpOptimWrapper ile grad_norm surekli nan/inf geliyordu;
# FP32'de 60/60 iterasyonda kayip ve gradyan sonlu. AMP'nin faydasi bellek ve hiz,
# ikisine de ihtiyacimiz yok: FP32'de tepe VRAM 606 MB / 6144 MB, adim suresi
# 0.206 sn (AMP 0.175) -- 100 epoch icin 23 dk yerine 20 dk farki.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=1.0),            # AMP altinda gradyan sicramalarina karsi
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=MAX_EPOCH // 2, end=MAX_EPOCH, T_max=MAX_EPOCH // 2,
         by_epoch=True, convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best='busi/dice_tum', rule='greater'),
    logger=dict(type='LoggerHook', interval=10),
)

# base'deki PipelineSwitchHook Mosaic'i kapatmak icindi; Mosaic zaten yok
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='ValLossHook', dataloader=val_loss_dataloader),
]

randomness = dict(seed=TOHUM, deterministic=False)
