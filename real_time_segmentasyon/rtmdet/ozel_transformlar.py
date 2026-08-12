"""Projeye ozel mmdet transformlari ve metrikleri."""
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.datasets.transforms import Albu
from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS, TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class AlbuBosGuvenli(Albu):
    """Bos anotasyonlu orneklerde cokmeyen Albu.

    Ust sinifin `_postprocess_results` metodu maske boyutunu
    `results['masks'][0].shape` uzerinden okuyor; hic maske kalmadiginda
    IndexError veriyor (mmdet 3.3.0, transforms.py:1770). Hemen altindaki
    `elif` dalinda dogru davranis zaten var, sadece bu dalda kullanilmamis.

    BUSI'de 538 egitim goruntusunun 92'si bos maskeli `normal` vakalar ve
    negatif ornek olarak bilerek veri setinde -- yani bu, verinin %17'sini
    kullanilamaz hale getiren bir hata.

    Tek fark: maske boyutu augmentasyon sonrasi goruntuden aliniyor.
    """

    def _postprocess_results(self, results, ori_masks=None):
        if 'gt_bboxes_labels' in results and isinstance(results['gt_bboxes_labels'], list):
            results['gt_bboxes_labels'] = np.array(results['gt_bboxes_labels'], dtype=np.int64)
        if 'gt_ignore_flags' in results and isinstance(results['gt_ignore_flags'], list):
            results['gt_ignore_flags'] = np.array(results['gt_ignore_flags'], dtype=bool)

        if 'bboxes' not in results:
            return results

        if isinstance(results['bboxes'], list):
            results['bboxes'] = np.array(results['bboxes'], dtype=np.float32)
        results['bboxes'] = HorizontalBoxes(results['bboxes'].reshape(-1, 4))

        if self.filter_lost_elements:
            for label in self.origin_label_fields:
                results[label] = np.array([results[label][i] for i in results['idx_mapper']])
            # gt_ignore_flags label_fields'te olmadigi icin ust sinif onu filtrelemiyor.
            # Augmentasyon bir orneğin tek instance'ini dusurdugunde (Affine goruntuden
            # cikariyor / min_visibility eliyor) kutular sifira inip bayraklar eski
            # uzunlugunda kaliyor; PackDetInputs bunlardan valid_idx uretip bos diziyi
            # indeksliyor ve IndexError veriyor (formatting.py:102). Egitimi rastgele
            # bir epoch'ta olduren buydu.
            if 'gt_ignore_flags' in results:
                bayrak = np.asarray(results['gt_ignore_flags'])
                if len(bayrak) != len(results['idx_mapper']):
                    results['gt_ignore_flags'] = bayrak[np.asarray(results['idx_mapper'], dtype=int)]
            if 'masks' in results:
                assert ori_masks is not None
                # bu noktada keymap henuz geri cevrilmedi, goruntu 'image' anahtarinda
                goruntu = results.get('image', results.get('img'))
                h, w = goruntu.shape[:2]
                secili = [results['masks'][i] for i in results['idx_mapper']]
                dizi = np.array(secili, dtype=np.uint8).reshape(-1, h, w)
                results['masks'] = ori_masks.__class__(dizi, h, w)
            if not len(results['idx_mapper']) and self.skip_img_without_anno:
                return None
        elif 'masks' in results:
            results['masks'] = ori_masks.__class__(results['masks'],
                                                   ori_masks.height, ori_masks.width)
        return results


@METRICS.register_module()
class CocoMetricHizali(CocoMetric):
    """Tahmin maskesini encode'dan once ori_shape'e getiren CocoMetric.

    RTMDet-Ins'in maske geri-olceklemesi maskeyi ori_shape'ten 1 px kisa
    uretebiliyor (mmdet 3.3.0, rtmdet_ins_head.py:500-507): tersine cevrilmis
    scale_factor'un indisleri takas edilmis, yukseklige w_scale, genislige
    h_scale uygulaniyor. keep_ratio=True ile bile Resize'in tamsayi
    yuvarlamasi yuzunden w_scale != h_scale, dolayisiyla buyutulen maske
    hedeften kisa kaliyor ve `[..., :ori_h, :ori_w]` kirpmasi bunu telafi
    edemiyor.

    COCOeval iki RLE'nin boyutu uyusmayinca IoU'yu gecersiz sayiyor, yani o
    goruntu segm_mAP'e sifir katkiyla giriyor. Sessiz ve girdi boyutuna gore
    degisen bir kayip: val'de 256 kosusunda 116 goruntunun 25'i (%21.6),
    512'de 2'si (%1.7) etkileniyor -- 256'nin segm_mAP'i 0.412 gorunurken
    hizalanmis olcumde 0.626. PLAN.md'deki "iki metrik ayrisiyor" sorusunun
    tamami bu artefakttan geliyordu.

    Dice degerlendirmemiz bundan etkilenmiyordu; orada maske zaten orijinal
    boyuta olcekleniyor (dice_degerlendirme.py).
    """

    def process(self, data_batch, data_samples):
        for ornek in data_samples:
            maske = ornek.get('pred_instances', {}).get('masks')
            if maske is None or len(maske) == 0:
                continue
            h, w = ornek['ori_shape'][:2]
            if tuple(maske.shape[-2:]) == (h, w):
                continue
            duz = F.interpolate(maske[None].to(torch.uint8), size=(h, w),
                                mode='nearest')[0]
            ornek['pred_instances']['masks'] = duz.to(torch.bool)
        super().process(data_batch, data_samples)
