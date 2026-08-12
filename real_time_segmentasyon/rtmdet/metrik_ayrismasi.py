"""segm_mAP ile orijinal cozunurlukteki Dice neden ayrisiyor?

512 kosusu segm_mAP'te 256'yi acik ara geciyor (0.623 vs 0.412) ama birlesim
Dice'inda fark gormuyor -- hatta 256 hafif onde (0.803 vs 0.784). Uc mimarinin
karsilastirmasinda ana metrik bu karara bagli oldugu icin fark olculmeden
ilerlenmiyor (PLAN.md "Acik soru: iki metrik ayrisiyor").

Bulunan asil sebep artefakt: RTMDet-Ins maskeyi ori_shape'ten 1 px kisa
uretebiliyor, COCOeval boyut uyusmayinca o goruntuyu sifir sayiyor. 256'da val
goruntulerinin %21.6'si, 512'de %1.7'si etkileniyordu. Ayrinti ve duzeltme:
ozel_transformlar.CocoMetricHizali. Bu modul `ham` (egitimdeki bozuk olcum) ve
`baz` (hizalanmis) mAP'i yan yana basarak farki gosteriyor.

Olculen ayrismalar:
  1. Birlesim Dice vs instans basina Dice  -> maske kalitesi goruntu duzeyinde yikaniyor mu
  2. Oracle instans Dice (skor esigi yok)  -> modelin uretebildigi en iyi maske ne kadar iyi
  3. Sinir bandi Dice                      -> ince sinir kalitesindeki fark
  4. Lezyon boyutu kirilimi                -> tavan analizi farki kucuk lezyonda bekliyordu
  5. Bos goruntulerde yanlis pozitif       -> mAP'i cezalandirir, lezyonlu Dice hic gormez
  6. mAP varyantlari (ham / oracle maske / oracle skor)
  7. 256 vs 512 eslestirilmis test         -> 0.803 vs 0.784 tek split'te gurultu mu

GT tanimi iki yerde farkli ve bu bilerek:
  - birlesim Dice orijinal ikili maskeyi kullanir (PLAN.md degerlendirme protokolu)
  - instans ve mAP hesaplari COCO JSON poligonlarini kullanir (modelin gordugu GT)

Kullanim:
    python metrik_ayrismasi.py                 # her iki kosu, val
    python metrik_ayrismasi.py --split test
"""
import argparse
import copy
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI))

import torch_uyum        # noqa: F401  torch 2.6 checkpoint uyumu
import ozel_transformlar  # noqa: F401  transform kaydi
from dice_degerlendirme import MANIFEST, gt_maskesi, metrikler, read_gray, yerel_yol

KOK = BURASI.parent
CALISMA = BURASI / 'calisma'
ANN = KOK / 'veri' / 'coco'

ESIK = 0.2          # PLAN.md'deki tablo bu esikte; sweep yine de basiliyor
MIN_SKOR = 0.005    # RLE'ye cevrilen tahminlerin alt siniri; AP kuyrugu ihmal edilebilir
BANT = 5            # sinir bandi yaricapi (orijinal cozunurlukte px)
ESLESME_IOU = 0.1   # instans eslemesi icin gevsek esik; 0.5 kirilimi ayrica raporlanir
BOYUT_SINIR = [0, 80, 140, 220, np.inf]
BOYUT_AD = ['<80', '80-140', '140-220', '>220']


def kosu_bul(ad):
    """work_dir'den en iyi checkpoint + dump edilmis config.

    Faz 2'de secim olcutu segm_mAP'ten orijinal cozunurlukteki Dice'a gecti, dosya
    adi da onunla degisti. Once yeni adi ariyoruz; ikisi de varsa Faz 2 kazanir,
    yoksa eski kosular (Faz 1 karsilastirmalari) calismaya devam eder.
    """
    work = CALISMA / ad
    for kalip in ('best_busi_dice_tum_epoch_*.pth', 'best_coco_segm_mAP_epoch_*.pth'):
        adaylar = list(work.glob(kalip))
        if adaylar:
            ckpt = max(adaylar, key=lambda p: int(re.search(r'epoch_(\d+)', p.name).group(1)))
            return work / 'rtmdet_ins_busi.py', ckpt
    raise FileNotFoundError(f'best_* checkpointi yok: {work}')


def dice(a, b):
    kesisim = float(np.logical_and(a, b).sum())
    toplam = float(a.sum() + b.sum())
    return 2 * kesisim / toplam if toplam else 1.0


def iou(a, b):
    birlesim = float(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / birlesim if birlesim else 1.0


def sinir_bandi(gt, b=BANT):
    """GT sinirinin +-b px komsulugu. Genis ic bolge sayilmadan sinir kalitesi olculur."""
    k = np.ones((2 * b + 1, 2 * b + 1), np.uint8)
    return (cv2.dilate(gt, k) - cv2.erode(gt, k)).astype(bool)


def bant_dice(tahmin, gt, bant):
    t, g = tahmin[bant], gt[bant]
    tp = float(np.logical_and(t, g).sum())
    payda = float(t.sum() + g.sum())
    return 2 * tp / payda if payda else 1.0


def cikarim_topla(config, ckpt, alt, cihaz):
    """Her goruntude bir kez cikarim; maskeler RLE olarak tutulur.

    RLE hem bellek hem COCOeval icin dogru format: 100 instans x 116 goruntuluk
    bool dizi yaklasik 1.7 GB tutardi.

    `rle` hizalanmis (ori_shape'e olceklenmis), `rle_ham` modelin dondurdugu
    boyutta. Ikisi ancak model maskeyi kisa urettiginde ayrisiyor; `rle_ham`
    egitim sirasinda loglanan bozuk segm_mAP'i yeniden uretmek icin duruyor.
    """
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(str(config), str(ckpt), device=cihaz)
    kayit = []
    for r in alt.itertuples():
        yol = yerel_yol(r.image_path)
        h, w = read_gray(yol).shape
        pred = inference_detector(model, str(yol)).pred_instances
        skor = pred.scores.cpu().numpy()
        sec = skor >= MIN_SKOR
        rleler, hamlar, kacik = [], [], 0
        if sec.any():
            for tek in pred.masks.cpu().numpy()[sec]:
                tek = tek.astype(np.uint8)
                hamlar.append(maskUtils.encode(np.asfortranarray(tek)))
                if tek.shape != (h, w):
                    kacik = 1
                    tek = cv2.resize(tek, (w, h), interpolation=cv2.INTER_NEAREST)
                rleler.append(maskUtils.encode(np.asfortranarray(tek)))
        kayit.append(dict(case_id=r.case_id, shape=(h, w), rle=rleler,
                          rle_ham=hamlar, skor=skor[sec], kacik=kacik))
    del model
    torch.cuda.empty_cache()
    return kayit


def birlesim(rleler, skorlar, shape, esik):
    out = np.zeros(shape, np.uint8)
    for rle, s in zip(rleler, skorlar):
        if s >= esik:
            out = np.maximum(out, maskUtils.decode(rle))
    return out


def esle(pred_m, pred_s, gt_m, esik):
    """Skora gore acgozlu esleme. Donen: (eslesme listesi, bos GT sayisi, fazla tahmin)."""
    aday = [i for i, s in enumerate(pred_s) if s >= esik]
    aday.sort(key=lambda i: -pred_s[i])
    kalan = list(range(len(gt_m)))
    eslesme = []
    for i in aday:
        if not kalan:
            break
        j = max(kalan, key=lambda j: iou(pred_m[i], gt_m[j]))
        v = iou(pred_m[i], gt_m[j])
        if v >= ESLESME_IOU:
            eslesme.append((i, j, v, dice(pred_m[i], gt_m[j])))
            kalan.remove(j)
    return eslesme, len(kalan), len(aday) - len(eslesme)


def goruntu_olcumleri(kayit, alt, coco, id_ile, esik):
    """Model basina goruntu duzeyinde tum olcumler tek gecise sigar."""
    satir = []
    for r, k in zip(alt.itertuples(), kayit):
        gt_bin = gt_maskesi(r.mask_paths, k['shape'])
        gt_inst = [coco.annToMask(a).astype(bool)
                   for a in coco.loadAnns(coco.getAnnIds(imgIds=id_ile[r.case_id]))]
        pred_m = [maskUtils.decode(x).astype(bool) for x in k['rle']]

        u = birlesim(k['rle'], k['skor'], k['shape'], esik)
        m = metrikler(u, gt_bin)
        d = dict(case_id=r.case_id, cls=r.cls, eq_diam=r.eq_diam,
                 bos_gt=r.lesion_px == 0, n_gt=len(gt_inst),
                 n_pred=int((k['skor'] >= esik).sum()), **m)

        if gt_inst:
            d['bant_dice'] = bant_dice(u.astype(bool), gt_bin.astype(bool),
                                       sinir_bandi(gt_bin))
            eslesme, kacan, fazla = esle(pred_m, k['skor'], gt_inst, esik)
            d['inst_dice'] = float(np.mean([e[3] for e in eslesme])) if eslesme else np.nan
            d['inst_iou'] = float(np.mean([e[2] for e in eslesme])) if eslesme else np.nan
            d['eslesen'] = len(eslesme)
            d['eslesen50'] = sum(1 for e in eslesme if e[2] >= 0.5)
            d['kacan_gt'] = kacan
            d['fazla_pred'] = fazla
            # oracle: skor esigi yok, GT basina en iyi tahmin maskesi
            d['oracle_dice'] = float(np.mean(
                [max((dice(p, g) for p in pred_m), default=0.0) for g in gt_inst]))
        else:
            d['fazla_pred'] = d['n_pred']
        satir.append(d)
    return pd.DataFrame(satir)


def sweep(kayit, alt, esikler=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Birlesim Dice'in esige duyarliligi -- 0.2 secimini sabitlemeden once."""
    gtler = [gt_maskesi(r.mask_paths, k['shape']) for r, k in zip(alt.itertuples(), kayit)]
    out = []
    for e in esikler:
        kayitlar = []
        for r, k, gt in zip(alt.itertuples(), kayit, gtler):
            m = metrikler(birlesim(k['rle'], k['skor'], k['shape'], e), gt)
            m['bos_gt'] = r.lesion_px == 0
            kayitlar.append(m)
        df = pd.DataFrame(kayitlar)
        lez = df[~df['bos_gt']]
        out.append(dict(esik=e, dice_lezyonlu=round(float(lez['dice'].mean()), 4),
                        dice_tum=round(float(df['dice'].mean()), 4)))
    return pd.DataFrame(out)


def coco_sonuclari(kayit, id_ile, coco, varyant):
    """COCO sonuc formati. Varyantlar mAP farkinin kaynagini ayristirir:

    ham          : maske modelin dondurdugu boyutta -> egitimde loglanan bozuk olcum
    baz          : maske ori_shape'e hizalanmis -> dogru olcum
    oracle_maske : eslesen tahminin maskesi GT ile degistirilir -> maske kalitesi devre disi,
                   geriye tespit + siralama kalir
    oracle_skor  : maske korunur, skor yerine GT ile gercek IoU -> skor kalibrasyonu devre disi
    """
    sonuc = []
    for k in kayit:
        img_id = id_ile[k['case_id']]
        gt_rle = [coco.annToRLE(a) for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))]
        for rle, s in zip(k['rle_ham'] if varyant == 'ham' else k['rle'], k['skor']):
            r, skor = rle, float(s)
            if gt_rle and varyant.startswith('oracle'):
                iouler = maskUtils.iou([rle], gt_rle, [0] * len(gt_rle))[0]
                en = int(np.argmax(iouler))
                if varyant == 'oracle_maske' and iouler[en] > 0:
                    r = gt_rle[en]
                elif varyant == 'oracle_skor':
                    skor = float(iouler[en])
            elif varyant == 'oracle_skor' and not gt_rle:
                skor = 0.0
            sonuc.append(dict(image_id=img_id, category_id=1, score=skor,
                              segmentation=dict(size=list(r['size']),
                                                counts=r['counts'].decode('ascii'))))
    return sonuc


def map_hesapla(coco, sonuc, img_ids=None):
    if not sonuc:
        return dict(segm_mAP=0.0, segm_mAP_50=0.0, segm_mAP_75=0.0)
    dt = coco.loadRes(copy.deepcopy(sonuc))   # loadRes girdi listesini degistiriyor
    ev = COCOeval(coco, dt, 'segm')
    if img_ids is not None:
        ev.params.imgIds = list(img_ids)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(segm_mAP=round(float(ev.stats[0]), 4),
                segm_mAP_50=round(float(ev.stats[1]), 4),
                segm_mAP_75=round(float(ev.stats[2]), 4))


def eslestirilmis_test(a, b, ad_a, ad_b, tohum=42, n=10000):
    """Ayni goruntulerde farkin guven araligi + Wilcoxon. 116 goruntu tek split."""
    from scipy.stats import wilcoxon

    fark = a - b
    rng = np.random.default_rng(tohum)
    idx = rng.integers(0, len(fark), size=(n, len(fark)))
    boot = fark[idx].mean(axis=1)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:      # tum farklar sifir
        p = 1.0
    return dict(
        n=len(fark), ort_a=round(float(a.mean()), 4), ort_b=round(float(b.mean()), 4),
        fark=round(float(fark.mean()), 4),
        ag_alt=round(float(np.percentile(boot, 2.5)), 4),
        ag_ust=round(float(np.percentile(boot, 97.5)), 4),
        wilcoxon_p=round(p, 4),
        a_daha_iyi=int((fark > 0).sum()), b_daha_iyi=int((fark < 0).sum()),
        etiket=f'{ad_a} - {ad_b}',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--kosular', nargs='+',
                    default=['rtmdet_ins_tiny_256_orta', 'rtmdet_ins_tiny_512_orta'])
    # Her kosu kendi val esigiyle karsilastirilir: tek esik dayatmak, esigi baska
    # yerde optimal olan modeli haksiz yere geride gosterir (256 -> 0.4, 512 -> 0.2).
    ap.add_argument('--esik', type=float, nargs='+', default=[ESIK])
    ap.add_argument('--cihaz', default='cuda:0')
    ap.add_argument('--cikti', default=str(CALISMA / 'metrik_ayrismasi.xlsx'))
    args = ap.parse_args()

    alt = pd.read_csv(MANIFEST).query('split == @args.split').reset_index(drop=True)
    coco = COCO(str(ANN / f'busi_{args.split}.json'))
    id_ile = {im['case_id']: im['id'] for im in coco.dataset['images']}
    lezyonlu_id = [id_ile[c] for c in alt.loc[alt.lesion_px > 0, 'case_id']]

    if len(args.esik) == 1:
        args.esik = args.esik * len(args.kosular)
    if len(args.esik) != len(args.kosular):
        raise SystemExit('--esik ya tek deger ya da kosu sayisi kadar olmali')

    goruntu, ozet, map_satir, sweepler = {}, [], [], []
    for ad, kosu_esigi in zip(args.kosular, args.esik):
        config, ckpt = kosu_bul(ad)
        etiket = ad.replace('rtmdet_ins_tiny_', '').replace('_orta', '')
        print(f'[{etiket}] esik {kosu_esigi}', flush=True)
        print(f'\n[{etiket}] {ckpt.name}', flush=True)

        kayit = cikarim_topla(config, ckpt, alt, args.cihaz)
        g = goruntu_olcumleri(kayit, alt, coco, id_ile, kosu_esigi)
        goruntu[etiket] = g
        sweepler.append(sweep(kayit, alt).assign(model=etiket))

        lez, bos = g[~g.bos_gt], g[g.bos_gt]
        ozet.append(dict(
            model=etiket,
            dice_birlesim=round(float(lez.dice.mean()), 4),
            dice_instans=round(float(lez.inst_dice.mean()), 4),
            dice_oracle=round(float(lez.oracle_dice.mean()), 4),
            dice_bant=round(float(lez.bant_dice.mean()), 4),
            iou_instans=round(float(lez.inst_iou.mean()), 4),
            gt_toplam=int(lez.n_gt.sum()),
            eslesen=int(lez.eslesen.sum()),
            eslesen_iou50=int(lez.eslesen50.sum()),
            kacan_gt=int(lez.kacan_gt.sum()),
            fazla_lezyonlu=int(lez.fazla_pred.sum()),
            fazla_bos=int(bos.fazla_pred.sum()),
            bos_temiz=int((bos.n_pred == 0).sum()),
            bos_toplam=len(bos),
            kisa_maske=int(sum(k['kacik'] for k in kayit)),
        ))

        for varyant in ['ham', 'baz', 'oracle_maske', 'oracle_skor']:
            sonuc = coco_sonuclari(kayit, id_ile, coco, varyant)
            map_satir.append(dict(model=etiket, varyant=varyant, kapsam='tum',
                                  **map_hesapla(coco, sonuc)))
            if varyant == 'baz':
                map_satir.append(dict(model=etiket, varyant=varyant, kapsam='lezyonlu',
                                      **map_hesapla(coco, sonuc, lezyonlu_id)))

    ozet = pd.DataFrame(ozet)
    map_tablo = pd.DataFrame(map_satir)
    sweep_tablo = pd.concat(sweepler, ignore_index=True)

    # boyut kirilimi: tavan analizi farki kucuk lezyonlarda bekliyordu
    boyut = []
    for etiket, g in goruntu.items():
        lez = g[~g.bos_gt].copy()
        lez['kova'] = pd.cut(lez.eq_diam, BOYUT_SINIR, labels=BOYUT_AD)
        for kova, alt_g in lez.groupby('kova', observed=True):
            boyut.append(dict(model=etiket, kova=str(kova), n=len(alt_g),
                              dice_birlesim=round(float(alt_g.dice.mean()), 4),
                              dice_oracle=round(float(alt_g.oracle_dice.mean()), 4),
                              dice_bant=round(float(alt_g.bant_dice.mean()), 4)))
    boyut = pd.DataFrame(boyut)

    # eslestirilmis test: ayni goruntuler, iki model
    testler = []
    etiketler = list(goruntu)
    for i in range(len(etiketler)):
        for j in range(i + 1, len(etiketler)):
            a, b = goruntu[etiketler[i]], goruntu[etiketler[j]]
            assert (a.case_id.values == b.case_id.values).all()
            lez = ~a.bos_gt.values
            for kolon in ['dice', 'oracle_dice', 'bant_dice']:
                testler.append(dict(metrik=kolon, **eslestirilmis_test(
                    a.loc[lez, kolon].values, b.loc[lez, kolon].values,
                    etiketler[i], etiketler[j])))
    testler = pd.DataFrame(testler)

    print(f'\n=== {args.split} | esik {args.esik} | ozet ===')
    print(ozet.to_string(index=False))
    print('\n=== birlesim Dice esik taramasi ===')
    print(sweep_tablo.pivot(index='esik', columns='model', values='dice_lezyonlu').to_string())
    print('\n=== segm mAP varyantlari ===')
    print(map_tablo.to_string(index=False))
    print('\n=== lezyon boyutu kirilimi (eq_diam, px) ===')
    print(boyut.to_string(index=False))
    print('\n=== eslestirilmis fark (bootstrap %95 GA) ===')
    print(testler.to_string(index=False))

    cikti = Path(args.cikti)
    with pd.ExcelWriter(cikti) as w:
        ozet.to_excel(w, sheet_name='ozet', index=False)
        map_tablo.to_excel(w, sheet_name='map_varyant', index=False)
        boyut.to_excel(w, sheet_name='boyut', index=False)
        testler.to_excel(w, sheet_name='eslestirilmis', index=False)
        sweep_tablo.to_excel(w, sheet_name='esik_taramasi', index=False)
        for etiket, g in goruntu.items():
            g.to_excel(w, sheet_name=f'goruntu_{etiket}', index=False)
    print('\nkaydedildi:', cikti)


if __name__ == '__main__':
    main()
