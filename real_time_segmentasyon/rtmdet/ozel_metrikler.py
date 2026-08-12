"""RTMDet icin val Dice metrigi ve val loss hook'u.

Iki bosluk kapatiliyor (PLAN.md Faz 2):

1. **Checkpoint secim olcutu.** mmdet `segm_mAP` ile seciyordu, SegFormer ise
   raporlanan metrikle (orijinal cozunurlukte Dice). Uc modelin ayni olcutle
   secmesi icin Dice bir mmengine metrigi haline getirildi; `CheckpointHook`
   `save_best='busi/dice_tum'` ile dogrudan onu kullanabiliyor.

2. **Val loss.** mmdet validation'i tahmin modunda calistiriyor, kayip
   uretmiyor. `ValLossHook` ayri bir dataloader uzerinden `mode='loss'` ile
   ileri gecis yapip kaybi message_hub'a yaziyor, boylece scalars.json'a ve
   egitim egrisine giriyor.

Oncelik iki kisiti birden karsiliyor (9):
  - `RuntimeInfoHook` (10) metrikleri message_hub'a itiyor; kayiplarin loglanmasi
    icin `metrics` sozlugu ondan **once** guncellenmeli. 48'de denendi, eklenen
    anahtarlar hicbir yere ulasmadi.
  - `EMAHook` (49) `after_val_epoch`'ta EMA parametrelerini geri aliyor; kaybin
    val metrikleriyle ayni agirliklarda olculmesi icin ondan da once calismali.
"""
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmdet.registry import METRICS


@METRICS.register_module()
class BusiDiceMetric(BaseMetric):
    """Orijinal cozunurlukte Dice -- ortak/degerlendirme.py ile ayni tanim.

    GT olarak COCO poligonlari degil **orijinal ikili maske birlesimi** kullanilir;
    raporlanan metrik de o. Maskeler manifest'ten okunup onbellege aliniyor.
    """

    default_prefix = 'busi'

    def __init__(self, manifest, esikler=(0.1, 0.2, 0.3, 0.4, 0.5), veri_koku=None,
                 collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        import pandas as pd
        # Esik taranip en iyi dice_tum aliniyor -- SegFormer'in secim olcutuyle
        # ayni tanim. Sabit esikte olcmek modeller arasi kiyasi bozardi: birinin
        # skor dagilimi digerinden farkli ve optimum esik ayni yerde degil.
        self.esikler = tuple(esikler)
        self.veri_koku = veri_koku
        df = pd.read_csv(manifest)
        self.satir = {r.case_id: r for r in df.itertuples()}
        self._gt = {}

    def _case_id(self, img_path):
        from pathlib import Path
        p = Path(str(img_path))
        return f'{p.parent.name}/{p.stem}'

    def _gt_maske(self, case_id):
        if case_id not in self._gt:
            import sys
            from pathlib import Path
            ortak = Path(__file__).parent.parent / 'ortak'
            if str(ortak) not in sys.path:
                sys.path.insert(0, str(ortak))
            import degerlendirme as dg
            self._gt[case_id] = dg.gt_maskesi(self.satir[case_id], self.veri_koku)
        return self._gt[case_id]

    def process(self, data_batch, data_samples):
        import cv2
        for ornek in data_samples:
            case_id = self._case_id(ornek['img_path'])
            gt = self._gt_maske(case_id)
            h, w = gt.shape

            pred = ornek['pred_instances']
            skor = pred['scores'].cpu().numpy()
            maskeler = None
            if 'masks' in pred and len(skor):
                maskeler = pred['masks'].cpu().numpy()

            kayit = {'bos': bool(gt.sum() == 0)}
            for esik in self.esikler:
                sec = skor >= esik
                birlesim = np.zeros((h, w), np.uint8)
                if maskeler is not None and sec.any():
                    for m in maskeler[sec]:
                        m = m.astype(np.uint8)
                        if m.shape != (h, w):
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        np.maximum(birlesim, m, out=birlesim)
                tp = float(np.logical_and(birlesim, gt).sum())
                fp = float(np.logical_and(birlesim, 1 - gt).sum())
                fn = float(np.logical_and(1 - birlesim, gt).sum())
                kayit[esik] = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
            self.results.append(kayit)

    def compute_metrics(self, results):
        bos = np.array([r['bos'] for r in results])
        en_iyi = None
        for esik in self.esikler:
            d = np.array([r[esik] for r in results])
            kayit = {
                'dice_tum': float(d.mean()),
                'dice_lezyonlu': float(d[~bos].mean()) if (~bos).any() else float('nan'),
                'bos_dogru': int((d[bos] == 1.0).sum()) if bos.any() else 0,
                'esik': esik,
            }
            if en_iyi is None or kayit['dice_tum'] > en_iyi['dice_tum']:
                en_iyi = kayit
        return en_iyi


@HOOKS.register_module()
class ValLossHook(Hook):
    """Val bolumunde `mode='loss'` ile ileri gecis yapip kaybi loglar.

    Kendi dataloader'ini kuruyor cunku val_dataloader'in pipeline'i anotasyonlari
    orijinal cozunurlukte yukluyor (LoadAnnotations, Resize'dan sonra); kayip
    hesabi icin hedeflerin model girdisiyle ayni uzayda olmasi gerekiyor.
    """

    priority = 9        # RuntimeInfoHook (10) ve EMAHook (49) ikisinden de once

    def __init__(self, dataloader):
        self.dl_cfg = dataloader
        self.dl = None

    def _dataloader(self, runner):
        if self.dl is None:
            self.dl = runner.build_dataloader(self.dl_cfg)
        return self.dl

    def after_val_epoch(self, runner, metrics=None):
        model = runner.model
        cekirdek = model.module if hasattr(model, 'module') else model
        egitimde = cekirdek.training
        cekirdek.eval()

        toplam, sayac = {}, 0
        with torch.no_grad():
            for veri in self._dataloader(runner):
                veri = cekirdek.data_preprocessor(veri, False)
                kayip = cekirdek(**veri, mode='loss')
                for k, v in kayip.items():
                    if isinstance(v, (list, tuple)):
                        v = sum(x.sum() for x in v)
                    toplam[k] = toplam.get(k, 0.0) + float(v.detach())
                sayac += 1

        if egitimde:
            cekirdek.train()
        if not sayac:
            return

        # Anahtarlarda "loss" gecmiyor: mmengine LogProcessor adinda loss olan
        # skalerleri pencere ortalamasiyla raporluyor (train loss'un yumusak
        # gorunmesinin sebebi bu) ve ayni islem scalars.json'a da yansiyordu --
        # epoch 2'de gercek 1.8666 iken kayda 2.1161 dusuyordu. `kayip` adiyla
        # epoch degeri oldugu gibi kaydediliyor.
        kayitlar = {f"val_kayip_{k.replace('loss_', '')}": v / sayac
                    for k, v in toplam.items()}
        kayitlar['val_kayip'] = sum(kayitlar.values())

        # message_hub'a yazmak yerine `metrics` sozlugu guncelleniyor: message_hub
        # HistoryBuffer tutuyor ve LoggerHook pencere ortalamasi aliyor, yani
        # scalars.json'a epoch degeri degil kosu boyunca yumusatilmis deger
        # giriyordu (epoch 3'te gercek 1.04 iken kayda 1.32 dusuyordu).
        # `metrics` epoch basina bir kez yaziliyor ve LoggerHook (priority 60)
        # bu hook'tan (48) sonra calistigi icin guncellemeyi goruyor.
        if metrics is not None:
            metrics.update(kayitlar)
        runner.logger.info('val kayip: ' + '  '.join(
            f'{k} {v:.4f}' for k, v in kayitlar.items()))
