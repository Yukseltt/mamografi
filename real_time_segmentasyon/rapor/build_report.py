# -*- coding: utf-8 -*-
"""real_time_segmentasyon raporu -> tek dosyalik HTML (Chrome ile PDF'e basilir).

mendeley_data_density/rapor/build_report.py deseniyle: figurler base64 gomulu,
tek HTML, pdf_olustur.ps1 ile PDF.
"""
import base64
from pathlib import Path

import pandas as pd

BURASI = Path(__file__).parent
KOK = BURASI.parent
FIG = BURASI / 'figurler'
OUT = BURASI / 'rt_seg_raporu.html'


def img(dosya, caption, sinif='fig-curves'):
    p = FIG / dosya
    if not p.exists():
        # Sessizce atlamak yerine derlemede goze batsin; rapora kirmizi bir hata
        # notu koymak da istenmiyor, o yuzden metinde ayrica aciklaniyor.
        print(f'  UYARI: figur yok, atlaniyor -> {dosya}')
        return ''
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure class="{sinif}"><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def tablo(df, vurgu=None, ondalik=4):
    """DataFrame -> HTML tablo. `vurgu` verilen satir indeksini isaretler."""
    bas = ''.join(f'<th>{c}</th>' for c in df.columns)
    satir = []
    for i, r in df.iterrows():
        hucre = ''.join(
            f'<td class="num">{v:.{ondalik}f}</td>' if isinstance(v, float)
            else f'<td class="num">{v}</td>' if isinstance(v, (int,))
            else f'<td>{v}</td>' for v in r)
        satir.append(f'<tr class="{"best" if i == vurgu else ""}">{hucre}</tr>')
    return f'<table><tr>{bas}</tr>{"".join(satir)}</table>'


# ---- veriler ----
ozet = pd.read_excel(KOK / 'uc_model_karsilastirma.xlsx', sheet_name='ozet')
esles = pd.read_excel(KOK / 'uc_model_karsilastirma.xlsx', sheet_name='eslestirilmis')
hiz = pd.read_excel(KOK / 'hiz_olcumu.xlsx')
onnx = pd.read_excel(KOK / 'onnx_olcumu.xlsx')

t_kalite = tablo(pd.DataFrame({
    'Model': ozet.model, 'Eşik': ozet.esik,
    'Test Dice (lezyonlu)': ozet.dice_lezyonlu, 'Test Dice (tüm)': ozet.dice_tum,
    'IoU': ozet.iou_lezyonlu, 'Precision': ozet.precision, 'Recall': ozet.recall,
    'Boş görüntü': [f'{a}/{b}' for a, b in zip(ozet.bos_dogru, ozet.bos_n)],
}).reset_index(drop=True), vurgu=0)

t_esles = tablo(pd.DataFrame({
    'A': esles.a, 'B': esles.b, 'Fark': esles.fark,
    '%95 GA': [f'{a:+.3f} … {b:+.3f}' for a, b in zip(esles.ga_alt, esles.ga_ust)],
    'Wilcoxon p': esles.wilcoxon_p,
    'A iyi / B iyi': [f'{a} / {b}' for a, b in zip(esles.a_iyi, esles.b_iyi)],
    'Ayrışıyor': esles.anlamli,
}).reset_index(drop=True), ondalik=3)

h = hiz.merge(onnx[['model', 'onnx_gpu_ms', 'onnx_cpu_ms', 'hizlanma']], on='model')
t_hiz = tablo(pd.DataFrame({
    'Model': h.model, 'Param (M)': h.parametre_M,
    'PyTorch ms': h.gpu_ms_medyan, 'PyTorch FPS': h.gpu_fps,
    'Saf ağ ms': h.saf_ileri_ms, 'Çerçeve yükü': [f'%{v:.0f}' for v in h.cerceve_yuku_pct],
    'ONNX GPU ms': h.onnx_gpu_ms, 'ONNX CPU ms': h.onnx_cpu_ms,
    'VRAM (MB)': h.tepe_vram_MB,
}).reset_index(drop=True), ondalik=2)

HTML = f'''<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>Gerçek Zamanlı Meme Ultrason Segmentasyonu — Sonuç Raporu</title>
<style>
@page {{ size: A4; margin: 18mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
        color: #1a1a1a; max-width: 190mm; margin: 0 auto; }}
h1 {{ font-size: 18pt; line-height: 1.25; margin: 0 0 4pt; }}
h1 + .sub {{ font-size: 11.5pt; color: #555; margin: 0 0 18pt; font-weight: 400; }}
h2 {{ font-size: 13.5pt; margin: 18pt 0 7pt; padding-bottom: 3pt; border-bottom: 1.5px solid #2c5aa0;
      color: #2c5aa0; page-break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 14pt 0 6pt; color: #1a3d6b; page-break-after: avoid; }}
p {{ margin: 0 0 7pt; text-align: justify; }}
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9pt;
         page-break-inside: avoid; }}
th {{ background: #eef2f8; text-align: left; font-weight: 600; }}
th, td {{ border: 1px solid #c3ccd9; padding: 3.5pt 6pt; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.best td {{ background: #f2f8f2; font-weight: 600; }}
figure {{ margin: 9pt 0 12pt; page-break-inside: avoid; text-align: center; }}
figure img {{ max-width: 100%; border: 1px solid #d5dae2; }}
.fig-cases img {{ max-height: 215mm; width: auto; }}
figcaption {{ font-size: 8.5pt; color: #4a4a4a; margin-top: 4pt; text-align: left; line-height: 1.35; }}
code {{ font-family: Consolas, monospace; font-size: 9pt; background: #f4f6f9;
        padding: 1px 4px; border-radius: 2px; }}
.callout {{ border-left: 3px solid #2c5aa0; background: #f6f9fd; padding: 7pt 11pt; margin: 9pt 0;
            page-break-inside: avoid; }}
.callout p:last-child {{ margin-bottom: 0; }}
.verdict {{ border: 1.5px solid #2c5aa0; background: #f6f9fd; padding: 9pt 13pt; margin: 10pt 0;
            page-break-inside: avoid; }}
.uyari {{ border-left: 3px solid #c0392b; background: #fdf6f5; padding: 7pt 11pt; margin: 9pt 0; }}
.tblnote {{ font-size: 8.5pt; color: #555; margin: -5pt 0 10pt; line-height: 1.35; }}
.eksik {{ color: #c0392b; font-size: 9pt; font-style: italic; }}
ul, ol {{ margin: 0 0 8pt; padding-left: 17pt; }}
li {{ margin-bottom: 3pt; }}
.brk {{ page-break-before: always; }}
</style></head><body>

<h1>Gerçek Zamanlı Meme Ultrason Lezyon Segmentasyonu</h1>
<p class="sub">Üç mimarinin kalite ve hız karşılaştırması — SegFormer-B0, RTMDet-Ins-tiny, YOLOv11n-Seg</p>

<h2>1. Amaç ve veri</h2>

<p>Meme ultrasonunda lezyonun canlı görüntü akışı üzerinde segmentasyonu hedeflendi. Soru iki
parçalı: <em>hangi mimari daha iyi segmentliyor</em> ve <em>hangisi gerçek zamanlı çalışabiliyor</em>.
Üç mimari seçildi — biri semantic (SegFormer-B0), ikisi instance segmentation (RTMDet-Ins-tiny,
YOLOv11n-Seg) — ve karşılaştırmanın mimariyi ölçmesi için veri bölünmesi, girdi hazırlığı,
augmentasyon zinciri ve değerlendirme protokolü üçünde de sabit tutuldu.</p>

<p>Veri kaynağı BUSI (Breast Ultrasound Images): 780 görüntü, benign 437 / malignant 210 /
normal 133. Çözünürlük 190×310 ile 1048×719 arasında değişiyor (780 görüntüde 639 farklı boyut),
maskeler ikili ve <code>normal</code> sınıfının maskesi tamamen boş.</p>

<h3>1.1 Kopya görüntüler</h3>

<p>Veri setinde <strong>115 teyitli kopya çifti</strong> bulundu; 179 görüntü (%23) çok görüntülü
kopya gruplarının içinde. Bunların 11'i "aynı görüntü, çelişkili maske" — ikisinde aynı kare
biri "lezyon var" diğeri "lezyon yok" diye etiketlenmiş. Rastgele bölünmede bu görüntüler eğitim
ve test'e dağılsa model ezberlediği kareyi tekrar görür ve Dice yapay olarak şişer.</p>

<p>Çözüm iki adımlı: çelişkili çiftlerde deterministik bir kuralla tek temsilci bırakıldı
(780 → 769 görüntü) ve bölme <strong>kopya grubu düzeyinde</strong> yapıldı. Sonuç: eğitim 538,
validation 116, test 115; <strong>val ve test tamamen kopyasız</strong>, her değerlendirme vakası
ayrı bir sahne. Sınıf ve lezyon boyutu dağılımı üç bölümde de dengeli.</p>

<h3>1.2 Değerlendirme protokolü</h3>

<p>Üç modelin ortak metriği <strong>orijinal çözünürlükte Dice</strong>. Küçültülmüş uzayda
karşılaştırmak semantic modele yapay avantaj verirdi, çünkü onun hedefi zaten küçültülmüş maske.
Her modelin tahmini orijinal görüntü boyutuna taşınıp orijinal ikili maskeyle karşılaştırılıyor.</p>

<p><code>normal</code> sınıfı bilerek veri setinde: canlı akışta karelerin çoğunda lezyon yok ve
boş maskeler modele yanlış pozitif baskılamayı öğretiyor. Bu görüntülerde GT ve tahmin ikisi de
boşsa Dice 1 sayılıyor; raporda lezyonlu ve boş kırılımı ayrı veriliyor.</p>

<p>Karar eşiği <strong>validation'da</strong> seçilip test'e sabit uygulandı. Test üzerinde eşik
taraması yapılmadı — değerlendirme betiği test için eşiğin dışarıdan verilmesini zorunlu kılıyor
ki bu yanlışlıkla ihlal edilemesin.</p>

<h2>2. Kurulum</h2>

<p>Girdi hazırlığı üçünde de <strong>letterbox</strong>: en-boy korunarak 256×256'ya küçültme,
sağ-alta doldurma. Kare resize ilk denemede kullanılmıştı ve RTMDet'in maske geri-ölçekleme kodu
tek ölçek katsayısı varsaydığı için maskeleri görüntüden kaydırıyordu; ölçüldüğünde maskenin
yalnızca %59'u kendi tahmin kutusunun içine düşüyordu. Letterbox'a geçince sorun kaynağında kalktı.</p>

<p>Augmentasyon zinciri (yatay çevirme, affine, elastik deformasyon, parlaklık/kontrast, gamma,
çarpımsal gürültü, hareket bulanıklığı) üç modelde ortak. Dikey çevirme bilerek yok: ultrasonda
derinlik ekseni sabittir, dikey çevirme fiziksel olarak imkânsız görüntü üretir. CLAHE de yok —
vakaların %75'i hipoekoik ve kontrast standardizasyonu bu sinyali bozar. RTMDet ve YOLO'nun kendi
Mosaic/MixUp reçeteleri kapatıldı.</p>

<p>Anotasyon tarafında RTMDet ve YOLO aynı COCO JSON'undan besleniyor; YOLO etiketleri o JSON'dan
türetildi ve geri-rasterize edildiğinde Dice <strong>1.0</strong> çıkıyor, yani dönüşüm kayıpsız.
Böylece iki instance modeli birebir aynı anotasyonu görüyor.</p>

<h2 class="brk">3. Segmentasyon kalitesi</h2>

<p>Aşağıdaki sayılar <strong>test bölümünde</strong>, her model kendi validation eşiğiyle. Dört
konfigürasyon da yerelde, tek ortamda yeniden çalıştırıldı ve eğitim ortamındaki sonuçlar
birebir yeniden üretildi (sapma 0.0000) — yani çıkarım yolu ile eğitim ortamı ayrışmıyor.</p>

{t_kalite}
<p class="tblnote">96 lezyonlu + 19 boş maskeli test görüntüsü. Eşikler validation'da
<code>dice_tum</code> ölçütüyle seçildi.</p>

<h3>3.1 Fark istatistiksel olarak anlamlı değil</h3>

<p>Ortalamalar SegFormer-B0'ı öne koyuyor, ama aynı görüntüler üzerinde eşleştirilmiş
karşılaştırma bunu desteklemiyor: <strong>altı çiftin hiçbirinde güven aralığı sıfırı
dışlamıyor.</strong></p>

{t_esles}

{img('eslestirilmis.png', 'Test bölümünde ikili farklar ve %95 bootstrap güven aralıkları. '
     'Tüm aralıklar sıfır çizgisini kesiyor.')}

<div class="verdict">
<p><strong>Kalite tarafında dört konfigürasyon ayırt edilemiyor.</strong> Tek koşuluk sıralama
SegFormer-B0'ı öne koyuyor ancak fark ölçülen gürültü bandının (±0.03 Dice) içinde. Sıralamayı
kesinleştirmek tohum tekrarı gerektirir.</p>
</div>

<h3>3.2 Modeller başarısızlık biçiminde ayrışıyor</h3>

<p>Ortalama ile medyan ters yönü gösteriyor ve bu tek başına bilgi taşıyor: RTMDet tipik vakada
biraz daha iyi (medyan her iki bölümde de önde), SegFormer kuyrukta daha sağlam. Test bölümünde
Dice 0.3'ün altında kalan görüntü sayısı SegFormer'da 14, RTMDet 512'de 16, YOLO'da 17,
RTMDet 256'da 18.</p>

<p>SegFormer'ın ortalama öndeliği buradan geliyor: çoğu görüntüde biraz geride, birkaç görüntüde
çok önde. Klinik olarak bir lezyonu tamamen kaçırmak, sınırını birkaç piksel kaba çizmekten
pahalı olduğu için bu ölçüt SegFormer'ı tercih ettirir — ama fark ortalamada anlamlı olmadığından
bu bir eğilim, kanıt değil.</p>

{img('boyut_kirilimi.png', 'Lezyon büyüklüğüne göre test Dice. Dört model de 140–220 px '
     'bandında belirgin şekilde zayıflıyor; bu, veri setine özgü ortak bir zorluk.')}

<h2 class="brk">4. Gerçek zamanlı ölçüm</h2>

<p>Üç model üç farklı ortamda eğitildi, dolayısıyla hız ölçümü <strong>tek makinede</strong>
tekrarlandı: RTX 2060, batch=1, ısınma sonrası, kare başına uçtan uca süre (ön işleme + ileri
geçiş + maskeyi orijinal çözünürlüğe taşıma). Diskten görüntü okuma dışarıda — canlı akışta kare
cihazdan gelir. Ölçüm betiği, hızı ölçülen kodun kaliteyi ölçen kodla aynı olduğunu doğruluyor.</p>

{t_hiz}
<p class="tblnote">Tekrarlanan koşularda GPU süreleri %5–10 oynuyor; modeller arası farklar bunun
üzerinde ama küçük farklar anlamlı okunmamalı.</p>

{img('kalite_hiz.png', 'Kalite–hız ödünleşimi. Solda PyTorch uçtan uca, sağda ONNX (yalnızca ağ). '
     'Kesikli çizgi 30 FPS. Dört konfigürasyon da her iki ölçümde gerçek zamanlı bandın üzerinde.')}

<h3>4.1 Girdi boyutu gecikmeyi neredeyse hiç etkilemiyor</h3>

<p>RTMDet 256 ile 512 aynı süreyi veriyor (27.2 vs 26.0 ms), saf ağ ölçümünde de öyle — dört kat
piksele rağmen. Sebep, bu boyutlarda batch=1'de GPU'nun kernel başlatma yükünde takılı olması,
hesap yükünde değil. CPU ölçümü bunu doğruluyor: orada girdi boyutu fark yaratıyor (94.6 vs
48.2 ms), çünkü CPU hesap yüküne bağlı.</p>

<p>Pratik sonucu: 256 ile 512 arasında ne Dice farkı var ne FPS farkı. 512'nin tek somut maliyeti
eğitimdeki bellek (2102 MB vs 606 MB).</p>

<h3>4.2 Çerçeve yükü kaldırılınca sıralama değişiyor</h3>

<p>RTMDet'in gecikmesinin yarısı mmdet'in kare başına Python yükü (saf ağ 13.3 ms, uçtan uca
27.2 ms). SegFormer'da bu oran %12, YOLO'da %33. Üç ağ da ONNX'e aktarılıp aynı çalışma zamanında
ölçüldüğünde <strong>RTMDet-Ins-tiny 256 en hızlısı çıkıyor</strong> (4.40 ms). Export'ların
sayısal doğruluğu bağıl hata ile kontrol edildi: 5e-7 … 1.3e-5, fp32 yuvarlama düzeyinde.</p>

<p>Yani PyTorch ölçümüne bakarak mimari sıralaması yapmak yanıltıcı olurdu; "RTMDet 2.6 kat yavaş"
cümlesi mimariyi değil çerçeveyi ölçüyor.</p>

<p><strong>CPU'da YOLOv11n-Seg açık ara önde</strong>: ONNX ile 7.4 ms, yani GPU'suz bir cihazda
bile 135 FPS. Gerçek zamanlı hedef için ayrık GPU'nun zorunlu olmadığını gösteriyor.</p>

<div class="uyari">
<p>ONNX sayıları <strong>dağıtılabilir FPS değil</strong>: grafik yalnızca ağı içeriyor. Instance
modellerde son işleme (NMS + maske birleştirme) dışarıda ve ayrıca ölçüldü (RTMDet'te 2.8 ms /
1.6 ms), SegFormer'ınki eşikleme + yeniden boyutlandırmadan ibaret. Uçtan uca dağıtımda
SegFormer'ın avantajı bu tablodan büyük, RTMDet'inki küçük.</p>
</div>

<h2 class="brk">5. Model bazında eğitim davranışı</h2>

{img('egri_segformer_b0.png', 'SegFormer-B0 — 100 epoch. Validation Dice epoch 74’te zirve '
     '(0.842), sonrasında düz. Seçim ölçütü raporlanan metriğin kendisi: eşik taranarak '
     'hesaplanan dice_tum.')}

{img('egri_rtmdet_256.png', 'RTMDet-Ins 256 — 100 epoch. Zirve epoch 64 (segm mAP 0.642). '
     'Maske mAP kutu mAP’inin üzerinde; bu veri setinde beklenen sıralama.')}

{img('egri_rtmdet_512.png', 'RTMDet-Ins 512 — 100 epoch. Zirve epoch 76 (0.638). 256 ile '
     'pratikte aynı yere yakınsıyor.')}

<div class="uyari">
<p><strong>Eğrilerde validation kaybı bilerek yok.</strong> SegFormer ve RTMDet'te validation adımı
kayıp değil kalite metriği üretiyor (Dice / mAP) — mmdet validation'ı tahmin modunda çalıştırıyor,
SegFormer döngüsü de aynı yapıyı izliyor. Ultralytics validation kaybını hesaplıyor ama tek modelde
gösterip diğer ikisinde göstermemek eğrileri karşılaştırılamaz hale getirirdi.</p>
<p>Kayıp bilgisi zaten modeller arası okunamaz: üç mimarinin kayıp fonksiyonu farklı
(Dice+BCE / cls+bbox+mask / box+seg+cls+dfl). Val kaybının asıl işi aşırı öğrenmeyi göstermek ve
bunu validation kalite eğrisi daha doğrudan yapıyor — üç eğride de orta panelde.</p>
</div>

{img('egri_yolo11n_seg.png', 'YOLOv11n-Seg — 100 epoch.')}

<p class="tblnote">YOLOv11n-Seg'in eğitim eğrisi bu sürümde yok: Ultralytics'in
<code>results.csv</code> log dosyası rapor üretilirken elde değildi. Modelin validation mAP
seyri epoch 60 civarında plato yapıyor (eğitim çıktısından). Dosya eklendiğinde figür
<code>figurleri_uret.py</code> ile otomatik üretiliyor.</p>

<h2 class="brk">6. Sabit değerlendirme vakaları</h2>

<p>Aynı 11 vaka üç modelde de gösteriliyor: sınıf × lezyon boyutu kovalarından seçilmiş 8 lezyonlu
ve 3 boş maskeli görüntü. Her model kendi iyi örneğini seçseydi görsel karşılaştırma hiçbir şey
söylemezdi.</p>

{img('vakalar_segformer_b0.png', 'SegFormer-B0 — sabit değerlendirme vakaları.', 'fig-cases')}

{img('vakalar_rtmdet_256.png', 'RTMDet-Ins 256 — aynı vakalar. Üç normal görüntünün hepsinde '
     'temiz; buna karşılık çok büyük lezyonlu benign (236) vakasını tamamen kaçırıyor '
     '(SegFormer aynı vakada 0.605 alıyor).', 'fig-cases')}

{img('vakalar_yolo11n_seg.png', 'YOLOv11n-Seg — aynı vakalar.', 'fig-cases')}

<h2 class="brk">7. Ölçüm sırasında düzeltilen üç hata</h2>

<p>Bu bölüm rapora bilerek konuluyor: üçü de sessizce yanlış sonuç üretiyordu ve ikisi bir süre
yanlış yorumlanmıştı.</p>

<h3>7.1 segm_mAP'i girdi boyutuna bağlı olarak düşüren maske hatası</h3>

<p>RTMDet-Ins maskeyi orijinal boyuttan 1 piksel kısa üretebiliyor (mmdet'te ters çevrilmiş ölçek
katsayılarının indisleri takas edilmiş). COCO değerlendirmesi iki maskenin boyutu uyuşmayınca
o görüntüyü sıfır sayıyor. Etkilenen görüntü oranı girdi boyutuna bağlı: 256'da val görüntülerinin
%21.6'sı, 512'de %1.7'si. Sonuç, 256'nın <code>segm_mAP</code>'i 0.412 görünürken gerçekte 0.626
olması.</p>

<p>Bu, "512 mimari olarak çok daha iyi" gibi okunuyordu ve o okuma yanlış olurdu. Düzeltildikten
sonra iki koşu arasında anlamlı fark kalmadı. İkinci bedeli checkpoint seçimiydi: bozuk metrikle
seçilen epoch 40 yerine düzeltilmiş metrikle epoch 64 seçildi ve Dice 0.803'ten 0.808'e çıktı.</p>

<h3>7.2 Ölçümle reddedilen bir "düzeltme"</h3>

<p><code>cv2.INTER_NEAREST</code> maskeyi küçültürken yarım piksel kaydırıyor. Bunu düzeltmenin
~0.01 Dice kazandıracağı düşünülüp mmdet yamalandı. <strong>Ölçüm reddetti</strong>: aynı config,
100 epoch, tek değişken yama — yamalı koşu her metrikte daha kötü çıktı. Sebep, mmdet'in çıkarım
tarafının ters yönde bir kayma taşıması ve nearest'in bias'ının onu telafi ediyor olmasıydı; tek
ucu düzeltmek tutarlılığı bozdu. Yama geri alındı.</p>

<p>Buradan kalıcı bir kontrol çıktı: tahmini ±2 piksel kaydırıp Dice'ın nerede zirve yaptığına
bakan, modelden bağımsız bir hizalama testi. Dört konfigürasyonda da kaydırma kazancı ihmal
edilebilir (0.0000–0.0012); reddedilen yamalı koşuda 0.0043 ile açıkça işaretleniyor.</p>

<h3>7.3 Boş görüntüleri saymayan eşik seçimi</h3>

<p>Eşik ilk olarak yalnızca lezyonlu görüntülerdeki Dice'a bakılarak seçiliyordu. YOLO'da bu,
20 normal görüntünün 8'inde yanlış pozitif üreten bir eşiği seçti. Ölçüt boş görüntüleri de sayan
<code>dice_tum</code>'e çevrildi; YOLO'nun eşiği 0.05'ten 0.20'ye taşındı ve yanlış pozitif 1'e
düştü. RTMDet'te iki ölçüt de aynı eşiği veriyor, yani değişiklik yalnızca farkın önemli olduğu
yerde etkili oldu.</p>

<h2>8. Sonuç</h2>

<div class="verdict">
<p><strong>"En iyi model" sorusunun cevabı dağıtım hedefine bağlı.</strong></p>
<ul>
<li><strong>Kalite:</strong> dört konfigürasyon istatistiksel olarak ayrışmıyor. SegFormer-B0
ortalamada önde ve ağır başarısızlıkları en az; RTMDet tipik vakada biraz daha iyi.</li>
<li><strong>PyTorch, olduğu gibi dağıtım:</strong> SegFormer-B0 (96 FPS) — mmdet'in çerçeve yükü
RTMDet'i geride bırakıyor.</li>
<li><strong>ONNX + GPU:</strong> RTMDet-Ins 256 ile SegFormer-B0 ayrışmıyor (4.4 ms).</li>
<li><strong>GPU'suz cihaz:</strong> YOLOv11n-Seg açık ara (ONNX CPU, 135 FPS).</li>
</ul>
<p>Tek bir öneri gerekiyorsa <strong>SegFormer-B0</strong>: her ölçümde ilk ikide, en az ağır
başarısızlık, boş görüntülerde en temiz ve dağıtımı en basit (instance son işlemesi yok).</p>
</div>

<h3>8.1 Bu raporun sınırları</h3>

<ul>
<li><strong>Her model tek kez eğitildi.</strong> Kalite sıralaması bu yüzden kapatılamıyor;
kapatmanın yolu tohum tekrarı.</li>
<li><strong>Checkpoint seçim ölçütü üç modelde farklı</strong> (SegFormer'da Dice, RTMDet'te mAP,
YOLO'da Ultralytics fitness). SegFormer lehine küçük bir avantaj olabilir; ara epoch ağırlıkları
saklanmadığı için geriye dönük ölçülemiyor.</li>
<li><strong>Augmentasyon paritesi tam değil:</strong> elastik deformasyon YOLO'da yok, çünkü
Ultralytics'in albumentations sarmalayıcısı maskeleri taşımıyor.</li>
<li><strong>Test bölümü küçük:</strong> 96 lezyonlu + 19 boş görüntü. Boş görüntülerdeki
"19/19 vs 17/19" farkı iki görüntü demek.</li>
<li><strong>ONNX ölçümü ağ ile sınırlı;</strong> uçtan uca dağıtım hattı ayrıca ölçülmeli.</li>
</ul>

</body></html>'''

OUT.write_text(HTML, encoding='utf-8')
print(f'yazildi: {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)')
