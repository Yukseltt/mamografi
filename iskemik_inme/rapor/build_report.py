# -*- coding: utf-8 -*-
"""ISLES'22 sonuc raporu -> tek dosyalik HTML (Chrome ile PDF'e basilir).

Figurler base64 gomulu; rapor tek dosya olarak tasinabiliyor. Drive'dan gelen figurler
rapor/girdi/ altinda (isles_rapor_paketi.ipynb uretiyor), gecikme figurleri
rapor/figurler/ altinda yerel olcumlerden uretiliyor (figurler.py).

Eksik girdi varsa rapor yine uretilir; eksik figurun yerine gorunur bir not konur.
Sessizce bos bolum birakilmaz.
"""
import base64
import json
from pathlib import Path

import pandas as pd

BURASI = Path(__file__).resolve().parent
KOK = BURASI.parent
GIRDI = BURASI / 'girdi'
FIG = BURASI / 'figurler'
CIKTI = BURASI / 'isles_raporu.html'


def _oku_json(ad, girdi_dizini=True):
    y = (GIRDI if girdi_dizini else KOK) / ad
    return json.load(open(y, encoding='utf-8')) if y.exists() else None


def _oku_xlsx(ad, girdi_dizini=True, **kw):
    y = (GIRDI if girdi_dizini else KOK) / ad
    return pd.read_excel(y, **kw) if y.exists() else None


def img(dosya, caption, dizin=None, sinif='fig'):
    p = (dizin or FIG) / dosya
    if not p.exists():
        return (f'<p class="eksik">[figur uretilemedi: {dosya} - '
                f'girdi dosyasi yok]</p>')
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure class="{sinif}"><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def tablo(df, vurgu=None, ondalik=4):
    if df is None:
        return '<p class="eksik">[tablo uretilemedi - girdi dosyasi yok]</p>'
    bas = ''.join(f'<th>{c}</th>' for c in df.columns)
    satir = []
    for i, r in df.iterrows():
        hucre = ''.join(
            f'<td class="num">{v:.{ondalik}f}</td>' if isinstance(v, float)
            else f'<td class="num">{v}</td>' if isinstance(v, (int,))
            else f'<td>{v}</td>' for v in r)
        satir.append(f'<tr class="{"best" if i == vurgu else ""}">{hucre}</tr>')
    return f'<table><tr>{bas}</tr>{"".join(satir)}</table>'


CSS = '''
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
       color: #1a1a1a; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 16pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 7pt; padding-bottom: 3pt; border-bottom: 1.5px solid #2c5aa0;
     color: #2c5aa0; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 6pt; color: #1a3d6b; page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9pt;
        page-break-inside: avoid; }
th { background: #eef2f8; text-align: left; font-weight: 600; }
th, td { border: 1px solid #c3ccd9; padding: 3.5pt 6pt; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { background: #f2f8f2; }
figure { margin: 9pt 0 12pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 1px solid #d5dae2; }
.fig-tall img { max-height: 215mm; width: auto; }
figcaption { font-size: 8.5pt; color: #4a4a4a; margin-top: 4pt; text-align: left; line-height: 1.35; }
.callout { border-left: 3px solid #2c5aa0; background: #f6f9fd; padding: 7pt 11pt; margin: 9pt 0;
           page-break-inside: avoid; }
.callout p:last-child { margin-bottom: 0; }
.uyari { border-left: 3px solid #c0392b; background: #fdf6f5; padding: 7pt 11pt; margin: 9pt 0; }
.tblnote { font-size: 8.5pt; color: #555; margin: -5pt 0 10pt; line-height: 1.35; }
.eksik { color: #c0392b; font-size: 9pt; }
ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
li { margin-bottom: 3pt; }
a { color: #2c5aa0; text-decoration: none; }
ol.refs { font-size: 9.5pt; }
ol.refs li { margin-bottom: 6pt; }
.brk { page-break-before: always; }
'''

# ------------------------------------------------------------------ veriler
kars = _oku_xlsx('karsilastirma_tablosu.xlsx')
boot = _oku_xlsx('bootstrap_ci.xlsx')
esik_etki = _oku_xlsx('esik_bilesen_etkisi.xlsx')
pan = _oku_xlsx('panoptica_karsilastirma.xlsx')
kesif = _oku_json('kesif_ozeti.json') or {}
hazirlik = _oku_json('hazirlik_ozeti.json') or {}
flair = _oku_json('flair_ozeti.json') or {}
deg = _oku_json('degerlendirme_ozeti.json') or {}
srv = _oku_xlsx('servis/servis_dogrulama.xlsx', girdi_dizini=False)
hiz = _oku_xlsx('hiz_olcumu.xlsx', girdi_dizini=False, sheet_name='ozet')
onnx = _oku_xlsx('onnx_olcumu.xlsx', girdi_dizini=False)
trt = _oku_xlsx('tensorrt_olcumu.xlsx', girdi_dizini=False)

AD = {'unet3d': '3D U-Net', 'unet_r34_2d': 'U-Net + ResNet34 (2D)',
      'unet_r34_25d': 'U-Net + ResNet34 (2.5D)', 'segformer_b0': 'SegFormer-B0',
      'unet_mbv3': 'U-Net + MobileNetV3', 'A1_flair': 'A1: +FLAIR',
      'A2_tversky': 'A2: focal-Tversky', 'A3_k1': 'A3: k=1', 'A3_k3': 'A3: k=3'}

# --- tablo 1: mimari karsilastirmasi
t_kars = tablo(pd.DataFrame({
    'Model': [AD.get(k, k) for k in kars.model_key],
    'Parametre (M)': [f'{v / 1e6:.2f}' for v in kars.parametre],
    'Zirve epoch': kars.en_iyi_epoch,
    'Val Dice': kars.val_dice,
    'Test Dice': kars.test_dice,
    'SEM': kars.test_dice_sem,
    'IoU': kars.test_iou,
    'Precision': kars.test_precision,
    'Recall': kars.test_recall,
    'HD95 (mm)': kars.test_hd95,
    'AVD (mL)': kars.test_avd_ml,
}).reset_index(drop=True), vurgu=0)

# --- tablo 2: bootstrap + Holm
bk = boot[boot.model != 'unet_r34_2d'].copy()
t_boot = tablo(pd.DataFrame({
    'Model': [AD.get(k, k) for k in boot.model],
    'Test Dice': boot.dice,
    '%95 GA': [f'{a:.3f} - {b:.3f}' for a, b in zip(boot.ci_alt, boot.ci_ust)],
    'Taban çizgisine göre fark': [('-' if pd.isna(v) else f'{v:+.4f}') for v in boot.fark],
    'Farkın %95 GA': [('-' if pd.isna(a) else f'{a:+.3f} - {b:+.3f}')
                      for a, b in zip(boot.fark_ci_alt, boot.fark_ci_ust)],
    'Wilcoxon p': [('-' if pd.isna(v) else (f'{v:.4f}' if v >= 1e-4 else '<0,0001'))
                   for v in boot.wilcoxon_p],
    'Holm sonrası': [('referans' if pd.isna(v) else v) for v in boot.holm],
}).reset_index(drop=True))

# --- tablo 3: hacim tertili kirilimi
t_tertil = tablo(pd.DataFrame({
    'Model': [AD.get(k, k) for k in kars.model_key],
    'Küçük tertil': kars.dice_kucuk,
    'Orta tertil': kars.dice_orta,
    'Büyük tertil': kars.dice_buyuk,
    'Lezyon F1': kars.test_lezyon_f1,
    'Lezyon sayı farkı': kars.get('test_lezyon_sayi_fark', pd.Series([float('nan')] * len(kars))),
}).reset_index(drop=True), vurgu=0)

# --- tablo 4: esik ve bilesen secimi
t_esik = tablo(pd.DataFrame({
    'Model': [AD.get(k, k) for k in esik_etki.model],
    'Ayar': esik_etki.ayar,
    'Dice': esik_etki.dice,
    'Precision': esik_etki.precision,
    'Recall': esik_etki.recall,
    'HD95 (mm)': esik_etki.hd95,
    'AVD (mL)': esik_etki.avd_ml,
    'Lezyon F1': esik_etki.lezyon_f1,
}).reset_index(drop=True)) if esik_etki is not None else tablo(None)

# --- tablo 5: panoptica
t_pan = tablo(pd.DataFrame({
    'Vaka': [v.replace('sub-strokecase', '') for v in pan.vaka],
    'panoptica RQ': pan.panoptica_rq,
    'Bu çalışmanın uygulaması': pan.bizim,
    'Fark': (pan.panoptica_rq - pan.bizim),
}).reset_index(drop=True)) if pan is not None else tablo(None)

# --- tablo 6: gecikme merdiveni
h3 = hiz[hiz.model == 'unet3d'] if hiz is not None else None
t_merdiven = tablo(pd.DataFrame({
    'Ön işleme': ['CPU', 'CPU', 'GPU', 'GPU'],
    'Hassasiyet': ['fp32', 'fp16', 'fp32', 'fp16'],
    'Ön işleme (ms)': [float(h3[(h3.onisleme == o) & (h3.hassasiyet == p)].on_isleme_ms.iloc[0])
                       for o, p in (('cpu', 'fp32'), ('cpu', 'fp16'), ('gpu', 'fp32'), ('gpu', 'fp16'))],
    'Ağ (ms)': [float(h3[(h3.onisleme == o) & (h3.hassasiyet == p)].ag_ms.iloc[0])
                for o, p in (('cpu', 'fp32'), ('cpu', 'fp16'), ('gpu', 'fp32'), ('gpu', 'fp16'))],
    'Toplam (ms)': [float(h3[(h3.onisleme == o) & (h3.hassasiyet == p)].toplam_ms.iloc[0])
                    for o, p in (('cpu', 'fp32'), ('cpu', 'fp16'), ('gpu', 'fp32'), ('gpu', 'fp16'))],
}).reset_index(drop=True), vurgu=3, ondalik=1) if h3 is not None else tablo(None)

# --- tablo 7: servis arka uclari
t_srv = tablo(pd.DataFrame({
    'Arka uç': srv.arka_uc,
    'Ön işleme (ms)': srv.on_isleme_ms,
    'Ağ (ms)': srv.ag_ms,
    'Son işleme (ms)': srv.son_isleme_ms,
    'Toplam (ms)': srv.toplam_ms,
    'Std (ms)': srv.toplam_std_ms,
    'Dice': srv.dice,
    'Referanstan en büyük sapma': srv.max_sapma,
    'Kalite kilidi': srv.kalite_kilidi,
}).reset_index(drop=True), ondalik=2) if srv is not None else tablo(None)

# --- tablo 8: izole olcum vs servis hatti
if onnx is not None and trt is not None and srv is not None:
    o3 = onnx[onnx.model == 'unet3d']
    t3 = trt[(trt.model == 'unet3d') & (trt.durum == 'OK')]
    izole = {
        'torch-fp32': float(o3[o3.hassasiyet == 'fp32'].torch_ms.iloc[0]),
        'torch-fp16': float(o3[o3.hassasiyet == 'fp16'].torch_ms.iloc[0]),
        'onnx-cuda': float(o3[(o3.hassasiyet == 'fp32') &
                              (o3.calisma_zamani == 'ONNX CUDA')].ms.iloc[0]),
        'trt-fp16': float(t3[(t3.hassasiyet == 'fp16') &
                             (t3.calisma_zamani == 'TensorRT')].ms.iloc[0]),
    }
    t_izole = tablo(pd.DataFrame({
        'Arka uç': list(srv.arka_uc),
        'İzole ölçüm (ms)': [izole[k] for k in srv.arka_uc],
        'Servis hattı (ms)': list(srv.ag_ms),
        'Oran': [srv.ag_ms.iloc[i] / izole[k] for i, k in enumerate(srv.arka_uc)],
    }).reset_index(drop=True), ondalik=2)
else:
    t_izole = tablo(None)

en_iyi = srv.loc[srv.toplam_ms.idxmin()] if srv is not None else None

# ------------------------------------------------------------------ cikti
T = {'kars': t_kars, 'boot': t_boot, 'tertil': t_tertil, 'esik': t_esik,
     'pan': t_pan, 'merdiven': t_merdiven, 'srv': t_srv, 'izole': t_izole}

F = {
    'veri_kesfi': img('veri_kesfi.png',
                      'Şekil 1. Veri setinin dağılımları. Sırasıyla lezyon hacmi, vaka başına '
                      'bağlı lezyon sayısı, lezyon voxel oranı, lezyonsuz dilim oranı, merkeze '
                      'göre hacim ve DWI voxel aralıkları. Hacim dağılımının çarpıklığı ve çok '
                      'odaklılığın yaygınlığı, sonuçların tek bir ortalamayla verilemeyeceğini '
                      'gösteriyor.', dizin=GIRDI),
    'adc': img('adc_olcek_duzeltme.png',
               'Şekil 2. ADC üst değerlerinin dağılımı, ölçek düzeltmesi öncesi ve sonrası. Solda '
               'logaritmik eksende üç ayrı kümelenme görülüyor. Düzeltmeden sonra dağılım tek '
               'modda toplanıyor.', dizin=GIRDI),
    'onisleme': img('onisleme_dogrulama.png',
                    'Şekil 3. Ön işleme çıktısının doğrulanması. Üç örnek vakada normalize DWI, '
                    'ölçeklenmiş ADC, referans maske ve overlay gösteriliyor. DWI beyin içi '
                    'ortalaması sıfıra, standart sapması bire yakın. ADC sıfır ile bir arasında.',
                    dizin=GIRDI, sinif='fig-tall'),
    'dice_boyut': img('dice_boyut.png',
                      'Şekil 4. Solda dokuz koşunun test Dice değerleri, sağda aynı modellerin '
                      'parametre sayısı. Gri bant, aynı mimarinin küçük varyantları arasında '
                      'ölçülen koşu varyansını gösteriyor. Bu bandın içinde kalan farklar ayırt '
                      'edilebilir değil ve dokuz modelin yedisi bandın içinde. Kırmızı işaretli '
                      'iki model gecikme ölçümlerine taşınan adaylar. Kalite sıralaması ile model '
                      'boyutu arasında bir ilişki görünmüyor. En küçük model listenin başında, '
                      'ikinci en küçüğü sonunda.'),
    'forest': img('forest_dice.png',
                  'Şekil 5. Taban çizgisine göre eşleştirilmiş fark ve yüzde 95 güven aralığı. '
                  'Yıldızlı satırlar Holm düzeltmesinden sonra anlamlı kalan farklar. Gri bant '
                  'koşu varyansı aralığını gösteriyor.', dizin=GIRDI),
    'curves_taban': img('unet_r34_2d_curves.png',
                        'Şekil 6. Taban çizgisinin eğitim eğrileri. Train ve validation kayıpları '
                        'hiç ayrışmıyor. Learning rate sekiz kez düşürüldüğü için val Dice '
                        'zirvesine ulaştığında öğrenme oranı başlangıç değerinin on altıda birine '
                        'inmiş durumda.', dizin=GIRDI),
    'curves_3b': img('unet3d_curves.png',
                     'Şekil 7. 3B U-Net eğitim eğrileri. Train ve validation kayıpları arasındaki '
                     'makas açılıyor ve bu, çalışmadaki tek aşırı öğrenme örüntüsü. Learning rate '
                     'hiç düşmedi. Koşu yakınsadığı için değil, val Dice on beş epoch iyileşmediği '
                     'için durdu.', dizin=GIRDI),
    'cases': img('unet3d_cases.png',
                 'Şekil 8. Sabit değerlendirme vakaları. Sütunlar sırasıyla DWI, referans maske, '
                 'tahmin ve overlay. Vakalar hacim aralığına yayılmış olarak seçildi. En üstteki '
                 'vaka 0,21 mL, en alttaki 93 mL. Küçük lezyonlardaki düşük doğruluk görsel olarak '
                 'da izlenebiliyor.', dizin=GIRDI, sinif='fig-tall'),
    'merdiven': img('gecikme_merdiveni.png',
                    'Şekil 9. Vaka başına sürenin bileşenlere dağılımı. Ön işlemenin GPU\'ya '
                    'taşınması toplam süreyi üçte birine indiriyor. Yarım hassasiyet ağ süresini '
                    'yarıya düşürüyor. Kesikli çizgi hedeflenen üst sınırı gösteriyor.'),
    'arka_uc': img('arka_uc.png',
                   'Şekil 10. Solda dört arka ucun uçtan uca süresi. Sağda aynı ağların izole '
                   'ölçümdeki ve gerçek hattaki süreleri. TensorRT\'nin izole ölçümdeki üstünlüğü '
                   'vaka başına çalışmada kayboluyor.'),
    'esik': img('esik_taramasi.png',
                'Şekil 11. Validation kümesinde eşik taraması. İki model ters yönlere gidiyor. '
                'Aşırı segmentasyon eğilimindeki 3B model daha yüksek, muhafazakâr 2B taban '
                'çizgisi daha düşük eşik seçiyor. Kazanç iki modelde de binde üç mertebesinde.',
                dizin=GIRDI),
    'flair': img('flair_hizalama.png',
                 'Şekil 12. FLAIR hizalamasının görsel doğrulaması. Son panelde DWI kırmızı, '
                 'hizalanmış FLAIR yeşil kanalda. Beyin sınırlarının örtüşmesi header tabanlı '
                 'taşımanın kabaca doğru çalıştığını gösteriyor.',
                 dizin=GIRDI, sinif='fig-tall'),
}

import metin  # noqa: E402

HTML = ('<!DOCTYPE html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        '<title>İskemik İnme Lezyon Segmentasyonu — Sonuç Raporu</title>\n'
        f'<style>{CSS}</style></head><body>\n'
        + metin.govde(T, F) + '\n</body></html>')

CIKTI.write_text(HTML, encoding='utf-8')
eksik = HTML.count('uretilemedi')
print('yazildi:', CIKTI, f'{CIKTI.stat().st_size/1e6:.2f} MB')
print('gomulu figur:', HTML.count('data:image/png;base64'), '| eksik oge:', eksik)
