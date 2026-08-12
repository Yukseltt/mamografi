# -*- coding: utf-8 -*-
"""Faz 2 raporu -> tek dosyalik HTML (Chrome ile PDF'e basilir).

Faz 1 raporundan fark: her sayi uc tohumun dagilimiyla veriliyor. Faz 1 tek kosuya
dayaniyordu ve "SegFormer en iyi" diyordu; uc tohumla o fark gurultunun icinde kaldi.
Rapor artik siralama degil, ayirt edilebilirlik ve kirilim uzerine kurulu.
"""
import base64
from pathlib import Path

import pandas as pd

BURASI = Path(__file__).parent
KOK = BURASI.parent
FIG = BURASI / 'figurler_faz2'
OUT = BURASI / 'rt_seg_raporu_faz2.html'
XLSX = KOK / 'faz2_karsilastirma.xlsx'

KOVA_SIRA = ['<80', '80-140', '140-220', '>220']


def img(dosya, caption, sinif='fig-curves'):
    p = FIG / dosya
    if not p.exists():
        print(f'  UYARI: figur yok, atlaniyor -> {dosya}')
        return (f'<p class="eksik">[figür üretilemedi: {dosya}]</p>')
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure class="{sinif}"><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def tablo(df, vurgu=None, ondalik=4):
    bas = ''.join(f'<th>{c}</th>' for c in df.columns)
    satir = []
    for i, r in df.iterrows():
        hucre = ''.join(
            f'<td class="num">{v:.{ondalik}f}</td>' if isinstance(v, float)
            else f'<td class="num">{v}</td>' if isinstance(v, int)
            else f'<td>{v}</td>' for v in r)
        satir.append(f'<tr class="{"best" if i == vurgu else ""}">{hucre}</tr>')
    return f'<table><tr>{bas}</tr>{"".join(satir)}</table>'


# ---- veriler ----
val = pd.read_excel(XLSX, sheet_name='val_esik')
test = pd.read_excel(XLSX, sheet_name='test_kosu')
var = pd.read_excel(XLSX, sheet_name='tohum_varyansi')
esles = pd.read_excel(XLSX, sheet_name='eslestirilmis_ort')
capraz = pd.read_excel(XLSX, sheet_name='capraz_tohum')
kirilim = pd.read_excel(XLSX, sheet_name='boyut_kirilimi')
hiz = pd.read_excel(KOK / 'hiz_olcumu.xlsx')
onnx = pd.read_excel(KOK / 'onnx_olcumu.xlsx')

t_varyans = tablo(pd.DataFrame({
    'Mimari': var.mimari,
    'Test Dice (lezyonlu)': [f'{m:.4f} ± {s:.4f}'
                             for m, s in zip(var.dice_lezyonlu_mean, var.dice_lezyonlu_std)],
    'Test Dice (tüm)': [f'{m:.4f} ± {s:.4f}'
                        for m, s in zip(var.dice_tum_mean, var.dice_tum_std)],
    'Koşular arası aralık': [
        f'{test[test.mimari == m].dice_lezyonlu.max() - test[test.mimari == m].dice_lezyonlu.min():.4f}'
        for m in var.mimari],
}).reset_index(drop=True))

t_kosu = tablo(pd.DataFrame({
    'Mimari': test.mimari, 'Koşu': [t - 41 for t in test.tohum], 'Eşik': test.esik,
    'Test Dice (lezyonlu)': test.dice_lezyonlu, 'Test Dice (tüm)': test.dice_tum,
    'IoU': test.iou_lezyonlu, 'Precision': test.precision, 'Recall': test.recall,
    'Boş görüntü': [f'{a}/{b}' for a, b in zip(test.bos_dogru, test.bos_n)],
}).reset_index(drop=True))

t_esles = tablo(pd.DataFrame({
    'A': esles.a, 'B': esles.b, 'Fark': esles.fark,
    '%95 GA': [f'{a:+.3f} … {b:+.3f}' for a, b in zip(esles.ga_alt, esles.ga_ust)],
    'Wilcoxon p': esles.wilcoxon_p,
    'A iyi / B iyi': [f'{a} / {b}' for a, b in zip(esles.a_iyi, esles.b_iyi)],
    'Ayrışıyor': esles.anlamli,
}).reset_index(drop=True), ondalik=3)

t_capraz = tablo(pd.DataFrame({
    'A': capraz.a, 'B': capraz.b,
    'Fark (min / ort / maks)': [f'{a:+.3f} / {b:+.3f} / {c:+.3f}'
                                for a, b, c in zip(capraz.fark_min, capraz.fark_ort,
                                                   capraz.fark_max)],
    'Tek başına anlamlı': [f'{v}/9' for v in capraz.anlamli],
    'İşaret hep aynı': capraz.a_hep_ustte,
}).reset_index(drop=True), ondalik=3)

kp = kirilim.pivot_table(index='kova', columns='mimari', values='dice',
                         aggfunc=['mean', 'std'], observed=True).reindex(KOVA_SIRA)
kp.columns = ['_'.join(c) for c in kp.columns]
n_kova = kirilim[kirilim.tohum == 42].groupby('kova', observed=True).n.first().reindex(KOVA_SIRA)
t_kirilim = tablo(pd.DataFrame({
    'Lezyon çapı (px)': [f'{k} (n={int(v)})' for k, v in zip(KOVA_SIRA, n_kova)],
    'RTMDet-Ins': [f'{m:.3f} ± {s:.3f}' for m, s in zip(kp['mean_RTMDet-Ins'],
                                                       kp['std_RTMDet-Ins'])],
    'SegFormer-B0': [f'{m:.3f} ± {s:.3f}' for m, s in zip(kp['mean_SegFormer-B0'],
                                                         kp['std_SegFormer-B0'])],
    'YOLOv11n-Seg': [f'{m:.3f} ± {s:.3f}' for m, s in zip(kp['mean_YOLOv11n-Seg'],
                                                         kp['std_YOLOv11n-Seg'])],
}).reset_index(drop=True))

ad = {'RTMDet-Ins 256': 'RTMDet-Ins', 'SegFormer-B0': 'SegFormer-B0',
      'YOLOv11n-Seg': 'YOLOv11n-Seg'}
h = hiz.assign(mimari=hiz.model.map(ad)).merge(
    onnx.assign(mimari=onnx.model.map(ad))[['mimari', 'onnx_gpu_ms', 'onnx_cpu_ms']],
    on='mimari')
t_hiz = tablo(pd.DataFrame({
    'Mimari': h.mimari, 'Param (M)': h.parametre_M,
    'PyTorch uçtan uca (ms)': h.gpu_ms_medyan, 'FPS': h.gpu_fps,
    'Çerçeve yükü': [f'%{v:.0f}' for v in h.cerceve_yuku_pct],
    'ONNX GPU (ms)': h.onnx_gpu_ms, 'ONNX CPU (ms)': h.onnx_cpu_ms,
    'VRAM (MB)': h.tepe_vram_MB,
}).reset_index(drop=True), ondalik=2)

CSS = '''
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
       color: #1a1a1a; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 4pt; }
h1 + .sub { font-size: 11.5pt; color: #555; margin: 0 0 18pt; font-weight: 400; }
h2 { font-size: 13.5pt; margin: 18pt 0 7pt; padding-bottom: 3pt; border-bottom: 1.5px solid #2c5aa0;
     color: #2c5aa0; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 6pt; color: #1a3d6b; page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9pt;
        page-break-inside: avoid; }
th { background: #eef2f8; text-align: left; font-weight: 600; }
th, td { border: 1px solid #c3ccd9; padding: 3.5pt 6pt; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { background: #f2f8f2; font-weight: 600; }
figure { margin: 9pt 0 12pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 1px solid #d5dae2; }
.fig-cases img { max-height: 215mm; width: auto; }
figcaption { font-size: 8.5pt; color: #4a4a4a; margin-top: 4pt; text-align: left; line-height: 1.35; }
.callout { border-left: 3px solid #2c5aa0; background: #f6f9fd; padding: 7pt 11pt; margin: 9pt 0;
           page-break-inside: avoid; }
.callout p:last-child { margin-bottom: 0; }
.uyari { border-left: 3px solid #c0392b; background: #fdf6f5; padding: 7pt 11pt; margin: 9pt 0; }
.tblnote { font-size: 8.5pt; color: #555; margin: -5pt 0 10pt; line-height: 1.35; }
.eksik { color: #c0392b; font-size: 9pt; font-style: italic; }
ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
li { margin-bottom: 3pt; }
.brk { page-break-before: always; }
'''

HTML = f'''<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>Gerçek Zamanlı Meme Ultrason Segmentasyonu — Sonuç Raporu</title>
<style>{CSS}</style></head><body>

<h1>Gerçek Zamanlı Meme Ultrason Lezyon Segmentasyonu</h1>
<p class="sub">Üç mimarinin kalite ve hız karşılaştırması — RTMDet-Ins-tiny, SegFormer-B0,
YOLOv11n-Seg · her mimari, başlangıç değeri değiştirilerek üç kez eğitildi</p>

<p>Özet. Üç mimari 256×256 girdide, aynı veri bölünmesi ve aynı değerlendirme
protokolüyle, her biri farklı rastgele başlangıçla üç kez eğitildi — toplam dokuz koşu. Sonuç: RTMDet-Ins
ile SegFormer-B0 arasında ayırt edilebilir bir kalite farkı yok. SegFormer-B0,
YOLOv11n-Seg'e göre istatistiksel olarak anlamlı bir üstünlük gösteriyor; RTMDet-Ins'in ortalama
Dice değeri de YOLOv11n-Seg'den yüksek olmakla birlikte bu fark anlamlı bulunmadı. Buna karşılık modeller
nerede iyi oldukları bakımından belirgin şekilde ayrılıyor: küçük lezyonlarda SegFormer,
büyük lezyonlarda RTMDet önde. Hız sıralaması ise çalışma ortamına göre tümüyle değişiyor.</p>

<h2>1. Amaç, veri ve protokol</h2>

<p>Hedef, meme ultrasonunda lezyonun canlı görüntü akışı üzerinde segmentasyonu. İki soru var:
hangi mimari daha iyi segmentliyor ve hangisi gerçek zamanlı çalışabiliyor. Üç
mimari seçildi — biri semantic (SegFormer-B0), ikisi instance segmentation (RTMDet-Ins-tiny,
YOLOv11n-Seg). Karşılaştırmanın mimariyi ölçmesi için veri bölünmesi, girdi hazırlığı, augmentasyon
zinciri, girdi boyutu ve değerlendirme protokolü üçünde de sabit tutuldu.</p>

<p>Görev tanımları aynı olmadığı için instance modellerinin çıktıları tek bir ikili lezyon
maskesine indirgendi: her piksel, kendisini kapsayan örneklerin en yüksek skorunu alıyor, yani
eşikleme skoru eşiğin üstündeki örneklerin birleşimine karşılık geliyor. Üç modelin çıktısı da bu
biçimde orijinal çözünürlükte Dice üzerinden karşılaştırıldı. Çalışmanın sorusu lezyonların ayrı
ayrı sayılması değil sınırlarının doğru çizilmesi olduğundan indirgeme bilgi kaybına yol
açmıyor.</p>

<p>Veri kaynağı BUSI (Breast Ultrasound Images): 780 görüntü, benign 437 / malignant 210 /
normal 133. Veri setinde 115 teyitli kopya çifti var; 179 görüntü (%23) kopya
gruplarının içinde ve 11'i "aynı görüntü, çelişkili maske". Tespit iki aşamalı: önce algısal
karma (dHash, 64 bit, Hamming uzaklığı ≤ 6) ile aday çiftler üretildi, sonra her aday 128×128'e
küçültülüp z-normalize edilerek Pearson korelasyonu ve maske IoU'su ile teyit edildi; korelasyonu
0.95'in üzerinde olan çiftler kopya sayıldı. Rastgele bölünmede bu görüntüler eğitim
ve teste dağılsaydı model ezberlediği kareyi tekrar görür, Dice yapay olarak şişerdi. Çelişkili
çiftlerde deterministik bir kuralla tek temsilci bırakıldı (780 görüntüden 769'a) ve bölme kopya grubu
düzeyinde yapıldı: eğitim 538, validation 116, test 115; val ve test tamamen kopyasız.</p>

<h3>1.1 Ölçüm protokolü</h3>

<p>Ortak metrik orijinal çözünürlükte Dice; küçültülmüş uzayda ölçmek semantic modele yapay
avantaj verirdi. Lezyonsuz normal görüntüler veri setinde tutuldu — canlı akışta karelerin çoğunda
lezyon yok ve boş maskeler yanlış pozitif baskılamayı öğretiyor. Bu görüntülerde gerçek maske ve
tahmin birlikte boşsa Dice 1 sayılıyor; lezyonlu ve boş kırılımı ayrı veriliyor.</p>

<p>Karar eşiği validation'da seçilip teste sabit uygulandı, test üzerinde eşik taraması yapılmadı.
Seçim ölçütü tüm görüntüler üzerinden Dice; yalnızca lezyonlu görüntülere bakan bir ölçüt, boş
görüntülerde yanlış pozitif üreten düşük eşikleri ödüllendiriyor.</p>

<h3>1.2 Neden her mimari üç kez eğitildi</h3>

<p>İlk turda her mimari bir kez eğitilmiş ve sonuç "SegFormer en iyi" çıkmıştı. Rastgele
başlangıç değiştirilerek yapılan tekrarlarda tek bir mimarinin kendi koşuları arasındaki fark,
mimariler arasında ölçtüğümüz farktan büyük çıktı; bu da ilk turun sıralamasını geçersiz kılıyor.
Aşağıdaki bütün sayılar üç koşunun dağılımıyla veriliyor.</p>

<h2>2. Kalite sonuçları</h2>

{t_varyans}
<p class="tblnote">Test seti, 115 görüntü (96 lezyonlu, 19 boş). Her hücre üç koşunun ortalaması ±
standart sapması.</p>

{img('kosu_dagilimi.png', 'Şekil 1. Dokuz koşunun test Dice değerleri. Noktalar tek koşular, '
     'yatay çizgi ortalama, bant ±1 standart sapma. RTMDet ve SegFormer bantları büyük ölçüde '
     'üst üste biniyor — aralarındaki fark koşudan koşuya değişen gürültünün içinde kalıyor.')}

<h3>2.1 Modeller ayrışıyor mu</h3>

<p>Her görüntü için üç koşunun Dice ortalaması alınıp mimariler görüntü bazında eşleştirilerek
karşılaştırıldı (bootstrap %95 güven aralığı ve Wilcoxon işaretli sıra testi, n=96).</p>

{t_esles}

{img('eslestirilmis.png', 'Şekil 2. Eşleştirilmiş fark ve %95 güven aralığı. Yalnızca '
     'SegFormer–YOLO aralığı sıfırı dışarıda bırakıyor.')}

<p>Sonuç belirli bir başlangıç değerine bağlı mı diye, koşuların dokuz çapraz eşleşmesi de ayrı ayrı test edildi:</p>

{t_capraz}

<div class="callout">
<p>Ne söylenebilir: SegFormer-B0, YOLOv11n-Seg'i lezyonlu Dice'ta geçiyor
(+0.035, p=0.003). Dokuz çapraz koşu eşleşmesinin dokuzunda da fark aynı yönde — bu sonuç belirli bir
başlangıç değerine bağlı değil.</p>
<p>Ne söylenemez: RTMDet-Ins ile SegFormer-B0 arasında fark yok (p=0.85, güven
aralığı sıfırı kapsıyor, 96 görüntünün 46'sında biri 45'inde diğeri önde). Bu iki mimari eldeki
veriyle ayırt edilemiyor.</p>
</div>

<p>RTMDet–YOLO farkı (+0.023) SegFormer–YOLO farkından (+0.035) çok geride olmadığı hâlde anlamlı
çıkmıyor (p=0.058). Sebep RTMDet'in koşular arası değişkenliğinin (±0.016) SegFormer'ınkinin
(±0.008) iki katı olması. Model seçiminde ortalama performans kadar eğitim kararlılığı da bir
ölçüt; SegFormer'ın encoder'ı ImageNet ön eğitimli olduğu için başlangıç noktası sabit, RTMDet'in
tespit başı sıfırdan öğreniyor.</p>

<h3 class="brk">2.2 Hangi model nerede iyi</h3>

<p>Toplam ortalamalar ayırt edilemiyor olsa da, lezyon boyutuna göre kırdığımızda tablo değişiyor.</p>

{t_kirilim}
<p class="tblnote">Test Dice, üç koşunun ortalaması ± standart sapması. Kova sınırları lezyonun
eşdeğer çapına göre.</p>

{img('boyut_kirilimi.png', 'Şekil 3. Lezyon boyutuna göre test Dice. İki bağımsız örüntü var: '
     'orta boy lezyonlardaki düşüş üç mimaride de aynı yerde, buna karşılık uç kovalarda '
     'mimariler ters yönde ayrışıyor.')}

<p>Bu veri setindeki sonuçlar, küçük lezyonlarda SegFormer-B0'ın, büyük lezyonlarda ise
RTMDet-Ins'in daha yüksek Dice değeri elde etme eğiliminde olduğunu gösteriyor. 80 pikselin
altındaki lezyonlarda SegFormer RTMDet'i 0.13 geçiyor (0.776 / 0.648); 220 pikselin üstünde
RTMDet SegFormer'ı 0.067 geçiyor (0.750 / 0.683). Kova başına görüntü sayısı 16–29 arasında
olduğundan bunlar gözlenen eğilimler, kesin genellemeler değil. SegFormer'ın stride 4'teki yoğun
decoder'ı ile RTMDet'in bölge tabanlı maske başı düşünüldüğünde beklenen bir davranış. İki modelin
toplam ortalamalarının neden ayırt edilemediğini de açıklıyor: farklı yerlerde iyiler ve ortalamada
birbirlerini götürüyorlar.</p>

<p>İki yönün kanıt gücü eşit değil. Küçük lezyonlardaki fark koşular arası saçılmanın çok üstünde
(0.13'e karşı ±0.027 ve ±0.009). Büyük lezyonlardaki ters yöndeki fark ise RTMDet'in bu kovadaki
saçılması yüksek olduğu için (±0.046) daha zayıf.</p>

<p>140–220 piksel kovasındaki düşüş mimariden bağımsız: üç mimaride de aynı yerde ve birbirine
çok yakın (0.655–0.671), dokuz koşunun hepsinde. Örüntü monotonik olmadığından — daha büyük
lezyonlarda Dice tekrar yükseliyor — lezyon büyüklüğüyle açıklanmıyor. Bu kovadaki 29 görüntünün
incelenmesi bu raporun kapsamı dışında bırakıldı.</p>

{img('vakalar.png', 'Şekil 4. Sabit değerlendirme vakalarında üç modelin tahminleri (her mimarinin ilk koşusu). '
     'Boş maskeli normal görüntüler, boyut kovalarının her birinden birer örnek.', 'fig-cases')}

<h2 class="brk">3. Hız</h2>

<p>Üç model de aynı makinede ölçüldü (RTX 2060, batch=1). Ölçülen büyüklük kare başına uçtan uca
süre: ön işleme, ileri geçiş ve maskeyi orijinal çözünürlüğe taşıma. Diskten görüntü okuma hariç,
çünkü canlı akışta kare cihazdan gelir.</p>

{t_hiz}
<p class="tblnote">"Çerçeve yükü", uçtan uca sürenin saf ağ ileri geçişi dışında kalan kısmı.
ONNX sütunları yalnızca ağ ileri geçişini ölçüyor; üç ağ da ONNX'e çevrilip aynı çalışma zamanında
(onnxruntime) koşuldu, bağıl sapma üçünde de 1e-5'in altında.</p>

{img('kalite_hiz.png', 'Şekil 5. Kalite–hız dengesi üç farklı çalışma ortamında. Sıralama üç '
     'panelde üç farklı çıkıyor. Kesikli çizgi 30 FPS.')}

<div class="callout">
<p>Ölçüm koşulları altında üç model de 30 FPS eşiğinin üzerinde; başka bir donanımda veya farklı
bir görüntü alma hattında sonuç değişebilir. Sıralama ortama göre tersine dönüyor: PyTorch'ta
RTMDet en yavaş (43 FPS), ancak gecikmesinin %48'i mmdet çerçevesinin kare başına Python yükü.
Aynı çalışma zamanında ONNX olarak ölçüldüğünde en düşük süreyi RTMDet veriyor (4.07 ms), CPU'da
ise YOLO açık ara önde (6.72 ms). "Şu model gerçek zamanlı çalışır" cümlesi, çalışma zamanı ve
dağıtım yolu belirtilmeden anlamlı değil.</p>
</div>

<h2>4. Eğitim davranışı</h2>

{img('egriler.png', 'Şekil 6. Üstte eğitim (düz) ve validation (kesik) kaybı, altta validation '
     'kalitesi; her mimari için üç koşu. Kayıp fonksiyonları farklı olduğundan üst satırdaki '
     'üç panelin ölçekleri karşılaştırılabilir değil; alt satırda RTMDet ve SegFormer Dice, '
     'YOLO ise Ultralytics’in ürettiği maske mAP değerini gösteriyor. YOLO panelinde ölçek '
     'ilk 20 epoch dışına göre kuruldu, erken sıçramalar grafiği eziyordu.')}

<p>Üç modelde de eğitim kaybı sona kadar düşerken validation kaybı 25.–40. epoch civarında
düzleşiyor, buna rağmen validation kalitesi düşmüyor. Kayba bakarak erken durdurma yapılsaydı daha
kötü bir checkpoint seçilirdi; bu yüzden seçim üç mimaride de orijinal çözünürlükteki Dice'a göre
yapıldı. Seçilen epoch'lar 56–93 aralığına dağılıyor, yani 100 epoch'luk eğitim uzunluğu dar
değil.</p>

<p>Kayıp fonksiyonu ve çerçeveye özgü validation metriği mimariler arasında farklı olduğundan bu
grafikler modeller arası karşılaştırma için kullanılmadı; nihai karşılaştırma ortak Dice metriği
üzerinden yapıldı.</p>

<p>Koşular arası saçılma bu eğrilerde de görünüyor ve model karşılaştırmasındaki tablo ile
tutarlı: SegFormer'ın üç eğrisi neredeyse üst üste biniyor, RTMDet'in validation kaybı ise
koşudan koşuya belirgin şekilde ayrışıyor.</p>

<h2>5. Nasıl doğrulandı</h2>

<ul>
<li>Colab'da eğitilen altı modelin validation sonucu yerel makinede yeniden hesaplandı; en büyük
sapma 0.0001, yani çıkarım yolu ayrışmamış.</li>
<li>Her koşuda tahmin maskesi ±2 piksel kaydırılarak sistematik hizalama kayması arandı; dokuz
koşuda da kazanç 0.003 eşiğinin altında.</li>
<li>Eşik yalnızca validation'da seçildi; optimum tarama aralığının ucunda çıkarsa değerlendirme
betiği hata veriyor.</li>
<li>Hız ölçümünde kullanılan çıkarım fonksiyonunun kalite değerlendirmesindekiyle aynı çıktıyı
ürettiği her modelde kontrol edildi (fark tam sıfır).</li>
</ul>

<h2>6. Sınırlılıklar</h2>

<ul>
<li>Test seti küçük: 115 görüntü, 96'sı lezyonlu. Bu boyutta ±0.03 mertebesindeki
farklar güvenilir şekilde ayrıştırılamıyor; raporun sıralama değil ayırt edilebilirlik üzerine
kurulmasının sebebi de bu.</li>
<li>Tek veri kaynağı. BUSI tek merkezli; başka cihaz ve popülasyona genelleme
bu çalışmayla gösterilmiş değil.</li>
<li>Mimari başına üç koşu azdır. Standart sapma tahminleri üç ölçümden geliyor, kendileri de
gürültülü. Yön doğru, büyüklük kaba.</li>
<li>Augmentasyon tam eşlenemedi: RTMDet ve SegFormer zincirindeki elastik
deformasyon YOLO tarafında karşılıksız — Ultralytics'in sarmalayıcısı geometrik dönüşüme maske
taşımıyor. YOLO'nun aleyhine küçük bir fark yaratmış olabilir.</li>
<li>140–220 piksel kovasındaki düşüş açıklanmadı. Örüntünün gerçek olduğu
gösterildi, sebebi araştırılmadı.</li>
<li>Girdi boyutunun ve augmentasyon şiddetinin etkisi bu çalışmada incelenmedi.</li>
</ul>

<h2>7. Sonuç</h2>

<p>Kalite. RTMDet-Ins ile SegFormer-B0 bu veri setinde ayırt edilemiyor. SegFormer-B0'ın
YOLOv11n-Seg'e üstünlüğü istatistiksel olarak anlamlı; RTMDet-Ins de YOLOv11n-Seg'den yüksek
ortalama veriyor, ancak bu fark anlamlılık düzeyine ulaşmıyor. Tek bir "en iyi model" ilan etmek
eldeki ölçümlerle desteklenmiyor.</p>
<p>Seçim ölçütü olarak kırılım. Modeller ortalamada değil, lezyon boyutuna göre
ayrışma eğilimi gösteriyor: küçük lezyonlarda SegFormer, büyük lezyonlarda RTMDet daha yüksek Dice
veriyor. Uygulama senaryosuna bağlı olarak lezyon boyutu ve çalışma ortamı dikkate alınarak model
seçimi yapılabilir.</p>
<p>Hız. Ölçüm koşulları altında üç model de gerçek zamanlı eşiğin üzerinde. Dağıtım PyTorch
üzerinden olacaksa SegFormer-B0 en hızlısı (90.7 FPS); ONNX Runtime üzerinde üç model de birbirine
yakın çıkarım sürelerine ulaşıyor ve en düşük ileri geçiş süresini RTMDet-Ins veriyor (4.07 ms,
SegFormer 4.23 ms, YOLO 4.95 ms); CPU'da ise YOLOv11n-Seg belirgin şekilde önde.</p>
<p>Öneri. PyTorch dağıtımı ve karışık lezyon boyutları için SegFormer-B0:
kalitede en kötü ihtimalle diğerleriyle aynı düzeyde, küçük lezyonlarda belirgin şekilde önde,
koşular arası en kararlı ve PyTorch üzerinde en hızlı model.</p>

<p>Sonraki aşama. Farklı girdi boyutlarının ve augmentasyon şiddetlerinin model performansı
üzerindeki etkisinin incelenmesi ve 140–220 piksel kovasındaki düşüşün kaynağının araştırılması
planlanmaktadır.</p>

</body></html>
'''

OUT.write_text(HTML, encoding='utf-8')
print('yazildi:', OUT)
