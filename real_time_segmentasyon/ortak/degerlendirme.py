"""Uc mimarinin ortak degerlendirme protokolu.

SegFormer-B0 semantic, YOLOv11-Seg ve RTMDet-Ins instance segmentation yapiyor.
Karsilastirmanin mimariyi olcmesi icin metrik tanimi tek yerden gelmeli; bu modul
o tek yer. Colab'da da yerelde de import edilebilir (mmdet/ultralytics/torch
bagimliligi yok).

PLAN.md kararlari:
  - Degerlendirme **orijinal cozunurlukte**, orijinal ikili maskeye karsi. 256'da
    karsilastirmak SegFormer'a yapay avantaj verirdi (onun GT'si zaten 256'ya
    indirilmis maske).
  - Bos maskeli `normal` goruntulerde GT bos ve tahmin de bossa Dice 1 sayilir
    (0/0 tanimsiz). Rapor lezyonlu / bos kirilimini ayri verir.
  - Tum metrikler manuel TP/FP/FN uzerinden, goruntu basina hesaplanip ortalanir
    (ikili durumda f1 = dice, ozdes).

Ortak soyutlama: her model icin `harita_fn(satir) -> (H, W) float32 skor haritasi`
yazilir, esik taramasi bu harita uzerinde yapilir.
  - SegFormer : lezyon sinifinin piksel olasiligi
  - instance  : her pikselde, o pikseli kaplayan instance'larin maksimum skoru
Instance modellerde bir haritayi t'de esiklemek, "skoru >= t olan instance'larin
birlesimi"ne birebir esit -- boylece uc modelde de tek bir esik taramasi calisiyor.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLAB_VERI = '/content/drive/MyDrive/real_time_segmentasyon/Dataset_BUSI_with_GT'

# Alt uc 0.01'e kadar iniyor: YOLO ilk taramada 0.05'i, yani izgaranin sinirini
# secmisti -- sinirdaki optimum "gercek optimum disarida kalmis olabilir" demektir.
# Ust uc 0.9'a kadar cikiyor cunku SegFormer'da esik piksel olasiligi, dagilimi
# instance skorlarindan farkli.
ESIKLER = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _ilk_var_olan(adaylar, ne):
    for c in adaylar:
        if c and Path(c).is_dir():
            return Path(c)
    raise FileNotFoundError(f'{ne} bulunamadi. Denenenler: {[str(c) for c in adaylar if c]}')


def proje_koku(kok=None):
    """veri_manifest.csv'nin bulundugu proje koku. RT_SEG_KOK ile ezilebilir."""
    return _ilk_var_olan([
        kok, os.environ.get('RT_SEG_KOK'),
        '/content/drive/MyDrive/real_time_segmentasyon',
        r'd:\mamografi\real_time_segmentasyon',
    ], 'proje koku')


def veri_koku(kok=None):
    """Goruntulerin bulundugu klasor. Colab'da veri hiza icin /content'e kopyalaniyor."""
    return _ilk_var_olan([
        kok, os.environ.get('RT_SEG_VERI'),
        '/content/dataset', COLAB_VERI,
        r'd:\mamografi\real_time_segmentasyon\Dataset_BUSI_with_GT',
    ], 'veri koku')


def manifest_oku(split=None, kok=None):
    df = pd.read_csv(proje_koku(kok) / 'veri' / 'veri_manifest.csv')
    return df if split is None else df[df['split'] == split].reset_index(drop=True)


def gri_oku(p):
    """Her kanal duzenini kabul eder: (H,W), (H,W,1), (H,W,3), (H,W,4).

    `ndim == 3` varsayimi kirilgandi: `ultralytics` import edildiginde cv2.imread'i
    global olarak degistiriyor (ultralytics.utils.patches) ve gri goruntulere
    (H,W,1) seklinde kanal ekseni ekliyor. Piksel verisi ayni kaliyor ama kanal
    sayisina gore dallanan kod cokuyor -- ve bu, import sirasina bagli oldugu icin
    ancak bazi kosularda ortaya cikiyordu.
    """
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.ndim == 2:
        return im
    if im.shape[2] == 1:
        return im[:, :, 0]
    return cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)


def yol_coz(manifest_yolu, veri=None):
    """Manifest Colab yollarini tutuyor; hangi makinede calisiyorsak ona cevir."""
    return veri_koku(veri) / str(manifest_yolu).replace(COLAB_VERI, '').lstrip('/\\')


def gt_maskesi(satir, veri=None):
    """EDA / augmentasyon / COCO donusumu ile ayni birlesim tanimi."""
    shape = (int(satir.height), int(satir.width))
    out = np.zeros(shape, np.uint8)
    for p in str(satir.mask_paths).split(';'):
        m = gri_oku(yol_coz(p, veri))
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, (m > 127).astype(np.uint8))
    return out


def metrikler(tahmin, gt):
    """Goruntu basina TP/FP/FN. GT bos ve tahmin de bossa hepsi 1 (0/0 tanimsiz)."""
    tp = float(np.logical_and(tahmin, gt).sum())
    fp = float(np.logical_and(tahmin, 1 - gt).sum())
    fn = float(np.logical_and(1 - tahmin, gt).sum())
    if tp + fp + fn == 0:
        return dict(dice=1.0, iou=1.0, precision=1.0, recall=1.0)
    return dict(
        dice=2 * tp / (2 * tp + fp + fn),
        iou=tp / (tp + fp + fn),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
    )


def instans_haritasi(maskeler, skorlar, shape):
    """Instance modelleri icin skor haritasi: her pikselde onu kaplayan en yuksek skor.

    Maske modelin kendi uzayindan geliyorsa orijinal boyuta nearest ile getirilir --
    RTMDet-Ins maskeyi ori_shape'ten 1 px kisa uretebiliyor (PLAN.md hata 8).
    """
    h, w = shape
    out = np.zeros((h, w), np.float32)
    for m, s in zip(maskeler, skorlar):
        m = np.asarray(m, np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        np.maximum(out, m.astype(np.float32) * float(s), out=out)
    return out


def goruntu_basina(harita_fn, split='val', esikler=ESIKLER, kok=None, veri=None):
    """Her goruntu x her esik icin metrik satiri. Cikarim goruntu basina bir kez."""
    alt = manifest_oku(split, kok)
    kayit = []
    for r in alt.itertuples():
        gt = gt_maskesi(r, veri)
        harita = np.asarray(harita_fn(r), np.float32)
        if harita.shape != gt.shape:
            raise ValueError(f'{r.case_id}: harita {harita.shape}, GT {gt.shape} '
                             '-- skor haritasi orijinal cozunurlukte olmali')
        for e in esikler:
            kayit.append(dict(case_id=r.case_id, cls=r.cls, split=split, esik=e,
                              eq_diam=r.eq_diam, lesion_px=r.lesion_px,
                              bos_gt=r.lesion_px == 0,
                              **metrikler(harita >= e, gt)))
    return pd.DataFrame(kayit)


def esik_tablosu(gb):
    """goruntu_basina ciktisi -> esik basina ozet. Lezyonlu / bos kirilimi ayri."""
    satir = []
    for e, g in gb.groupby('esik'):
        lez, bos = g[~g.bos_gt], g[g.bos_gt]
        satir.append(dict(
            esik=e,
            dice_tum=round(float(g.dice.mean()), 4),
            dice_lezyonlu=round(float(lez.dice.mean()), 4),
            iou_lezyonlu=round(float(lez.iou.mean()), 4),
            precision_lezyonlu=round(float(lez.precision.mean()), 4),
            recall_lezyonlu=round(float(lez.recall.mean()), 4),
            bos_dogru=int((bos.dice == 1.0).sum()),
            bos_toplam=len(bos),
        ))
    return pd.DataFrame(satir)


def en_iyi_esik(tablo, olcut='dice_tum'):
    """Esigi val'de secer. Varsayilan olcut **tum goruntuler** uzerinden Dice.

    `dice_lezyonlu` ile secmek bos maskeli goruntuleri hesaba katmiyor ve yanlis
    pozitif ureten esikleri odullendiriyor: YOLO'da lezyonlu Dice 0.05'te zirve
    yapiyordu ama o esikte 20 normal goruntunun 8'inde yanlis pozitif vardi
    (0.20'de 1'inde). Projenin `normal` sinifini dahil etme gerekcesi tam olarak
    false-positive baskilamayi ogretmekti (PLAN.md), dolayisiyla secim olcutu de
    onlari saymali.

    RTMDet'te iki olcut ayni esigi veriyor (0.4), yani degisiklik yalnizca
    farkin onemli oldugu yerde etkili oluyor.

    Optimum izgaranin ucunda cikarsa hata verilir: gercek optimum aralik disinda
    kalmis olabilir ve bunu sessizce raporlamak yanlis olur.
    """
    esikler = tablo['esik'].to_numpy()
    secilen = float(tablo.loc[tablo[olcut].idxmax(), 'esik'])
    if secilen in (esikler.min(), esikler.max()):
        raise ValueError(
            f'{olcut} icin en iyi esik ({secilen}) tarama araliginin ucunda '
            f'[{esikler.min()}, {esikler.max()}] -- gercek optimum disarida '
            'kalmis olabilir. ESIKLER genisletilmeli.')
    return secilen


def kaydirma_taramasi(harita_fn, split='val', esik=0.5, kaymalar=(-2, -1, 0, 1, 2),
                      kok=None, veri=None):
    """Tahmini kaydirinca Dice artiyor mu? Hizalama hatasinin model-bagimsiz testi.

    Bir modelin girdi hazirligi ile cikarim geri-olceklemesi ayni piksel-merkezi
    konvansiyonunu kullanmiyorsa tahmin sistematik olarak kayar ve Dice (0,0)
    disinda bir noktada zirve yapar. Uc mimaride de calisir, kurulumu bilmeyi
    gerektirmez.

    RTMDet'te bu test pahali bir varsayimi cururtmustu: mmdet'in maske kucultme
    konvansiyonunu "duzeltmek" hizalamayi bozdu, cunku cikarim tarafi ayni bias'i
    ters yonde tasiyordu ve ikisi birbirini goturuyordu (PLAN.md hata 9).
    Zirve (0,0)'da degilse ON ISLEME BOZUK demektir -- egitimi tekrarlamadan once
    burasi duzeltilmeli.
    """
    alt = manifest_oku(split, kok)
    lezyonlu = [(r, gt_maskesi(r, veri)) for r in alt.itertuples() if r.lesion_px > 0]
    haritalar = [np.asarray(harita_fn(r), np.float32) >= esik for r, _ in lezyonlu]

    satir = []
    for dy in kaymalar:
        for dx in kaymalar:
            skor = []
            for (r, gt), m in zip(lezyonlu, haritalar):
                if dx or dy:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    m = cv2.warpAffine(m.astype(np.uint8), M, (gt.shape[1], gt.shape[0]),
                                       flags=cv2.INTER_NEAREST)
                skor.append(metrikler(m, gt)['dice'])
            satir.append(dict(dx=dx, dy=dy, dice=round(float(np.mean(skor)), 4)))
    return pd.DataFrame(satir).pivot(index='dy', columns='dx', values='dice')


def hizalama_ozeti(tablo, kazanc_esigi=0.003):
    """Kaydirma taramasini yorumlar. `argmax` tek basina yetmez.

    Yuzey duz oldugunda zirvenin yeri gurultudur; anlamli olan kaydirmayla
    kazanilan Dice. Olculen iki ornek:
      RTMDet yanlis hizalanmis kosu (v4) : kazanc 0.0043  -> gercek sorun
                                           (merkez farki da −2 px, dogrulandi)
      YOLOv11n-seg                       : kazanc 0.0012  -> gurultu
    Varsayilan esik ikisinin arasinda. **Iki gozleme dayaniyor, gecici**: uc
    model de olculdukten sonra gozden gecirilmeli. Sinirda kalan bir sonucu
    `kayma_testi.py` (tahmin-GT merkez farki) ile teyit etmek gerekir.

    Zirve izgaranin ucunda ve kazanc anlamliysa ayrica uyarilir: optimum aralik
    disinda kalmis olabilir.
    """
    dy, dx = tablo.stack().idxmax()
    en_iyi, merkez = float(tablo.stack().max()), float(tablo.loc[0, 0])
    kazanc = en_iyi - merkez
    kenar = (dx in (tablo.columns.min(), tablo.columns.max())
             or dy in (tablo.index.min(), tablo.index.max()))

    if kazanc < kazanc_esigi:
        durum = f'hizalama tamam (kaydirma kazanci {kazanc:.4f} < {kazanc_esigi})'
    elif kenar:
        durum = (f'DIKKAT: kaydirma {kazanc:.4f} Dice kazandiriyor ve zirve '
                 f'({dy}, {dx}) izgaranin ucunda -- izgara genisletilmeli')
    else:
        durum = (f'DIKKAT: tahmin ({dy}, {dx}) px kaymis, kaydirma {kazanc:.4f} '
                 'Dice kazandiriyor -- on isleme gozden gecirilmeli')
    return dict(dy=int(dy), dx=int(dx), merkez=round(merkez, 4),
                en_iyi=round(en_iyi, 4), kazanc=round(kazanc, 4),
                kenar=bool(kenar), durum=durum)


def boyut_kirilimi(gb, esik, sinir=(0, 80, 140, 220, np.inf),
                   ad=('<80', '80-140', '140-220', '>220')):
    """Lezyon boyutuna gore Dice. Cikti-stride tavani farki burada gorunmeli."""
    lez = gb[(gb.esik == esik) & (~gb.bos_gt)].copy()
    lez['kova'] = pd.cut(lez.eq_diam, sinir, labels=list(ad))
    out = lez.groupby('kova', observed=True).agg(
        n=('dice', 'size'), dice=('dice', 'mean'), iou=('iou', 'mean')).reset_index()
    return out.round({'dice': 4, 'iou': 4})
