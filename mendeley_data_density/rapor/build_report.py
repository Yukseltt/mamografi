# -*- coding: utf-8 -*-
"""density_segmentation raporu -> tek dosyalik HTML (Chrome ile PDF'e basilir)."""
import base64, os

os.chdir(r'd:\mamografi\mendeley_data_density')
FIG = 'rapor/figurler'
OUT = 'rapor/density_segmentation_raporu.html'

KEYS = ['unet_resnet34', 'unetpp_resnet34', 'segformer_b2',
        'deeplabv3p_resnet50', 'deeplabv3p_effnetb3', 'unet_effnetb0']


def img(key, kind, caption):
    p = f'{FIG}/{key}_{kind}.png'
    b64 = base64.b64encode(open(p, 'rb').read()).decode()
    cls = 'fig-cases' if kind == 'cases' else 'fig-curves'
    return (f'<figure class="{cls}"><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


R = {
 'unet_resnet34':       dict(enc='ResNet34', dec='U-Net', dice=0.8059, iou=0.6922, p=0.8153, r=0.8392,
                             vl=0.2624, ep=54, tot=70, lr=7, bs=16, std=0.1352, sem=0.0143, par=24436369),
 'unetpp_resnet34':     dict(enc='ResNet34', dec='U-Net++', dice=0.7980, iou=0.6818, p=0.8162, r=0.8310,
                             vl=0.2717, ep=50, tot=66, lr=7, bs=12, std=0.1373, sem=0.0146, par=26078609),
 'segformer_b2':        dict(enc='MiT-B2', dec='SegFormer', dice=0.7932, iou=0.6742, p=0.7883, r=0.8396,
                             vl=0.2756, ep=33, tot=49, lr=7, bs=8, std=0.1340, sem=0.0142, par=27347393),
 'deeplabv3p_resnet50': dict(enc='ResNet50', dec='DeepLabV3+', dice=0.7811, iou=0.6591, p=0.7773, r=0.8293,
                             vl=0.2906, ep=39, tot=55, lr=6, bs=8, std=0.1409, sem=0.0149, par=26677585),
 'deeplabv3p_effnetb3': dict(enc='EffNet-B3', dec='DeepLabV3+', dice=0.7751, iou=0.6497, p=0.7517, r=0.8420,
                             vl=0.3008, ep=31, tot=47, lr=6, bs=8, std=0.1349, sem=0.0143, par=11680185),
 'unet_effnetb0':       dict(enc='EffNet-B0', dec='U-Net', dice=0.7419, iou=0.6120, p=0.7665, r=0.7915,
                             vl=0.4158, ep=15, tot=31, lr=2, bs=16, std=0.1641, sem=0.0174, par=6251469),
}
LABEL = {'unet_resnet34': 'U-Net + ResNet34', 'unetpp_resnet34': 'U-Net++ + ResNet34',
         'segformer_b2': 'SegFormer-B2', 'deeplabv3p_resnet50': 'DeepLabV3+ + ResNet50',
         'deeplabv3p_effnetb3': 'DeepLabV3+ + EfficientNet-B3', 'unet_effnetb0': 'U-Net + EfficientNet-B0'}


def metric_table(k):
    d = R[k]
    return f'''<table class="metrics">
<tr><th>Val Loss</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th><th>F1</th>
<th>std</th><th>SEM</th><th>Parametre</th><th>En iyi epoch</th></tr>
<tr><td class="num">{d['vl']:.4f}</td><td class="num strong">{d['dice']:.4f}</td><td class="num">{d['iou']:.4f}</td>
<td class="num">{d['p']:.4f}</td><td class="num">{d['r']:.4f}</td><td class="num">{d['dice']:.4f}</td>
<td class="num">{d['std']:.4f}</td><td class="num">{d['sem']:.4f}</td>
<td class="num">{d['par']:,}</td><td class="num">{d['ep']}</td></tr></table>'''


NOTES = {
 'unet_resnet34': '''<p>Dice, IoU ve val loss bakımından birinci. Eğitim 70 epoch bütçesinin tamamını
kullandı; <code>ReduceLROnPlateau</code> yedi kez tetiklendi (2,5&times;10<sup>-4</sup> &rarr;
2,0&times;10<sup>-6</sup>). Dice'ın epoch 29-32 ve 36-37'deki sıçramaları ilk iki LR düşüşünü takip
ediyor; epoch 25'teki 0,7877 platosu LR düşüşüyle aşıldı. Son 10 epoch'ta val loss 0,2633-0,2644
bandında. Train/val loss makası epoch 19-20'de açıldı, epoch 45'ten sonra +0,049'da sabitlendi.</p>''',

 'unetpp_resnet34': '''<p>Encoder 3.1 ile aynı; tek değişken decoder. Aradaki 0,0079'luk fark doğrudan
U-Net &rarr; U-Net++ geçişine atfedilebilir ve eşleşmiş validation karşılaştırmasında %5 eşiğini geçmiyor (p = 0,054, bkz. 4.3).
Tablodaki en yüksek precision (0,8162) ve en düşük R&minus;P makası (+0,0148). Nested skip connection'lar
nedeniyle batch 16&rarr;12 düşürüldü.</p>''',

 'segformer_b2': '''<p>Tek transformer tabanlı model; <code>nvidia/segformer-b2-finetuned-ade-512-512</code>
ağırlıklarından ince ayar, tek LR (1&times;10<sup>-4</sup>), encoder/decoder ayrımı yok. Dice bakımından
üçüncü, ancak R&minus;P makası +0,0513 ile U-Net ailesinin iki katından fazla (bkz. 4.2).</p>''',

 'deeplabv3p_resnet50': '''<p>Kayıp precision tarafında: recall 0,8293 ile 3.1'e yakın, precision 0,7773
ile belirgin geride. R&minus;P +0,0520. Zirve epoch 39, duruş epoch 54 &mdash; epoch bütçesi eşitlemesi
(bkz. 2.2) bu koşuda sonucu değiştirmedi.</p>''',

 'deeplabv3p_effnetb3': '''<p>Tablonun en yüksek recall'u (0,8420) ve en düşük precision'ı (0,7517);
R&minus;P +0,0903 ile altı modelin en yükseği. Epoch 2'de Dice 0,18 / recall 0,12'ye düşüp epoch 7'de
toparladı (warmup dönemi), kalıcı etkisi olmadı.</p>''',

 'unet_effnetb0': '''<p>Eşleşmiş validation karşılaştırmasında en iyi modelden anlamlı biçimde geride kalan en büyük fark
(0,0642; p = 4,5&times;10<sup>-13</sup>). Zirve epoch 15, duruş epoch 30 &mdash; bütçesinin yarısını
kullanmadan erken durdurmayla kesildi. Encoder dört kat küçük olmasına rağmen ortalama train/val makası
+0,081 (3.1'de +0,028).</p>
<p>Bu koşu bir protokol etkisini görünür kılıyor: erken durdurma val Dice'ı, LR scheduler'ı val loss'u
izliyor. Bu modelde ikisi ayrıştı &mdash; Dice epoch 15'te platoya girdi, val loss epoch 25'e kadar
iyileşti. Protokol altı modelde aynı uygulandığından karşılaştırma etkilenmiyor, ancak gürültülü Dice
eğrisine sahip modeller sistematik olarak daha erken kesiliyor.</p>''',
}

sections3 = []
for i, k in enumerate(KEYS, 1):
    d = R[k]
    sections3.append(f'''
<section class="model-section">
<h3>3.{i} {LABEL[k]}</h3>
{metric_table(k)}
{NOTES[k]}
{img(k, 'curves', f'Şekil {i*2-1}. {LABEL[k]} eğitim eğrileri, {d["tot"]} epoch. Sağ alt panel LR takvimi; F1 ikili segmentasyonda Dice ile özdeş olduğundan ayrı panel verilmemiştir.')}
{img(k, 'cases', f'Şekil {i*2}. {LABEL[k]} — beş sabit doğrulama vakası. Orijinal | uzman maskesi | model tahmini. Vakalar tüm modellerde sabit seed ile aynı.')}
</section>''')

rows4 = '\n'.join(
    f'<tr{" class=\"best\"" if i == 1 else ""}><td>{i}</td><td>{LABEL[k]}</td>'
    f'<td class="num">{R[k]["vl"]:.4f}</td>'
    f'<td class="num strong">{R[k]["dice"]:.4f}</td><td class="num">{R[k]["iou"]:.4f}</td>'
    f'<td class="num">{R[k]["p"]:.4f}</td><td class="num">{R[k]["r"]:.4f}</td>'
    f'<td class="num">{R[k]["dice"]:.4f}</td>'
    f'<td class="num">{R[k]["r"]-R[k]["p"]:+.4f}</td>'
    f'<td class="num">{R[k]["par"]:,}</td><td class="num">{R[k]["ep"]}</td></tr>'
    for i, k in enumerate(KEYS, 1))

HTML = f'''<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>Mamografide Meme Yoğunluğu Segmentasyonu — Sonuç Raporu</title>
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
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9.2pt;
         page-break-inside: avoid; }}
th {{ background: #eef2f8; text-align: left; font-weight: 600; }}
th, td {{ border: 1px solid #c3ccd9; padding: 3.5pt 6pt; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.strong {{ font-weight: 700; }}
tr.best td {{ background: #f2f8f2; }}
figure {{ margin: 9pt 0 12pt; page-break-inside: avoid; text-align: center; }}
figure img {{ max-width: 100%; border: 1px solid #d5dae2; }}
.fig-cases img {{ max-height: 205mm; width: auto; }}
figcaption {{ font-size: 8.5pt; color: #4a4a4a; margin-top: 4pt; text-align: left; line-height: 1.35; }}
code {{ font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f4f6f9;
        padding: 1px 4px; border-radius: 2px; }}
.callout {{ border-left: 3px solid #2c5aa0; background: #f6f9fd; padding: 7pt 11pt; margin: 9pt 0;
            page-break-inside: avoid; }}
.callout p:last-child {{ margin-bottom: 0; }}
.tblnote {{ font-size: 8.5pt; color: #555; margin: -5pt 0 10pt; line-height: 1.35; }}
.model-section {{ page-break-before: always; }}
.refs {{ font-size: 9.2pt; }}
.refs li {{ margin-bottom: 5pt; }}
ul, ol {{ margin: 0 0 8pt; padding-left: 17pt; }}
li {{ margin-bottom: 3pt; }}
.mono-block {{ font-family: Consolas, monospace; font-size: 8.8pt; background: #f4f6f9;
               border: 1px solid #dde3ec; padding: 7pt 10pt; margin: 8pt 0; white-space: pre;
               page-break-inside: avoid; }}
.verdict {{ border: 1.5px solid #2c5aa0; background: #f6f9fd; padding: 9pt 13pt; margin: 10pt 0;
            page-break-inside: avoid; }}
</style></head><body>

<h1>Mamografide Meme Yoğunluğu (Breast Density) Segmentasyonu</h1>
<p class="sub">Altı derin öğrenme modelinin karşılaştırmalı sonuç raporu</p>

<h2>1. Amaç ve Veri</h2>
<p>Mamografi görüntülerinde yoğun (dens) meme dokusunun ikili segmentasyonu için dört mimari
(U-Net, U-Net++, DeepLabV3+, SegFormer) ve üç encoder ailesi (ResNet, EfficientNet, Mix Transformer)
birleştirilerek altı model eğitildi. Veri bölünmesi, ön işleme, augmentation stratejisi, kayıp
fonksiyonu ve genel optimizasyon yaklaşımı tüm modellerde sabit tutulmuş; mimarilerin bellek ve
optimizasyon gereksinimleri doğrultusunda batch size ve öğrenme oranı gibi model-spesifik
hiperparametreler kontrollü biçimde belirlenmiştir (bkz. 2.2).</p>

<p>Veri kaynağı Behravan vd. (2024) tarafından yayımlanan <em>Mammogram Density Assessment Dataset</em>'tir.
Orijinal set VinDr-Mammo'dan seçilmiş 745 görüntüden oluşur (596 eğitim / 149 test), JPG, 2800&times;3518;
maskeler uzman radyolog tarafından işaretlenmiş, hem meme alanını hem dens dokuyu kapsamaktadır. Bu
çalışmada kullanılan sürüm, bu setin NIfTI'ye dönüştürülmüş ve 512&times;512'ye yeniden boyutlandırılmış
536 vakalık alt kümesidir.</p>

<table>
<tr><th>Bölme</th><th>Vaka</th><th>Klasör</th></tr>
<tr><td>Eğitim</td><td class="num">447</td><td><code>imagesTr</code> / <code>labelsTr</code></td></tr>
<tr><td>Doğrulama</td><td class="num">89</td><td><code>imagesVal</code> / <code>labelsVal</code></td></tr>
</table>

<p>Görüntüler tek kanallı <code>float64</code>, piksel aralığı 0-238; maskeler ikili. Bölme klasör
bazlıdır ve değiştirilmemiştir. Yoğunluk yüzdesi JSON dosyaları segmentasyon hedefi olarak
kullanılmamış, yalnızca vaka görsellerinde bağlam bilgisi olarak gösterilmiştir. Kullanılan sürümde
meme alanı maskesi bulunmadığından "dens alan / meme alanı" oranı doğrudan hesaplanamamıştır.</p>

<h2>2. Yöntem</h2>

<h3>2.1 Veri hattı</h3>
<p>Her görüntü kendi içinde min-maks normalizasyonuyla [0, 1] aralığına çekildi (ham piksel aralığı
dosyadan dosyaya değiştiğinden sabit global maksimum varsayılmadı), üç kanala kopyalandı ve ImageNet
istatistikleriyle normalize edildi.</p>
<p>Eğitim augmentasyonu: yatay çevirme (p=0,5), dikey çevirme (p=0,3), &plusmn;15&deg; döndürme (p=0,5),
parlaklık/kontrast (&plusmn;0,15, p=0,5), elastik deformasyon (p=0,2), CoarseDropout (p=0,2). Görev doku
oranı ölçümüne dayandığından global kompozisyonu bozacak agresif kırpma/yakınlaştırma kullanılmadı.
Doğrulamada yalnızca yeniden boyutlandırma ve normalizasyon uygulandı.</p>

<h3>2.2 Eğitim yapılandırması</h3>
<p>Kayıp: Dice + BCE. Metrikler görüntü başına hesaplanıp ortalandı; ikili segmentasyonda F1 tanım
gereği Dice'a eşittir. AMP, gradyan kırpma (<code>max_norm=1,0</code>), 3 epoch doğrusal warmup ve
<code>ReduceLROnPlateau</code> (val loss 3 epoch iyileşmezse &times;0,5, min 1&times;10<sup>-7</sup>)
kullanıldı. Optimizer AdamW (weight decay 1&times;10<sup>-4</sup>), görüntü 512&times;512, seed 42.</p>

<table>
<tr><th>Model</th><th>Batch</th><th>LR (encoder / decoder)</th><th>Maks. epoch</th><th>Patience</th></tr>
<tr><td>U-Net + ResNet34</td><td class="num">16</td><td>5&times;10<sup>-5</sup> / 2,5&times;10<sup>-4</sup></td><td class="num">70</td><td class="num">15</td></tr>
<tr><td>U-Net + EfficientNet-B0</td><td class="num">16</td><td>5&times;10<sup>-5</sup> / 2,5&times;10<sup>-4</sup></td><td class="num">70</td><td class="num">15</td></tr>
<tr><td>U-Net++ + ResNet34</td><td class="num">12</td><td>5&times;10<sup>-5</sup> / 2,5&times;10<sup>-4</sup></td><td class="num">70</td><td class="num">15</td></tr>
<tr><td>DeepLabV3+ + ResNet50</td><td class="num">8</td><td>1&times;10<sup>-4</sup> / 5&times;10<sup>-4</sup></td><td class="num">70</td><td class="num">15</td></tr>
<tr><td>DeepLabV3+ + EfficientNet-B3</td><td class="num">8</td><td>1&times;10<sup>-4</sup> / 5&times;10<sup>-4</sup></td><td class="num">70</td><td class="num">15</td></tr>
<tr><td>SegFormer-B2</td><td class="num">8</td><td>1&times;10<sup>-4</sup> (tek LR)</td><td class="num">70</td><td class="num">15</td></tr>
</table>

<p>Epoch bütçesi başlangıçta ağır mimariler için 60/12 olarak planlanmıştı. İlk üç modelin zirvelerini
epoch 50 ve 54'te yapması üzerine, 60 tavanının bazı modelleri erken kesme riski taşıdığı görüldü ve
bütçe altı modelde 70/15'e eşitlendi. Değişiklik dördüncü model eğitilmeye başlanmadan uygulandı.</p>

<p>Doğrulama setinden sabit seed ile seçilen beş görsel karşılaştırma vakası tüm modellerde aynıdır.</p>

<h3>2.3 Tespit edilen iki hata</h3>
<p>İlk denemelerde train loss düzgün azalırken doğrulama Dice'ı epoch'lar arasında 0,0002-0,56 bandında
sıçruyordu. İki ayrı hata tespit edildi; her ikisi de hata vermeden çalışıp sonucu bozduğu için
belgelenmiştir.</p>

<p><strong>(a) Çift normalizasyon.</strong> Görüntü veri kümesi içinde [0, 1] aralığına çekildikten
sonra <code>Normalize</code> dönüşümü varsayılan <code>max_pixel_value=255</code> ile bir kez daha
bölüyordu. Dinamik aralık [&minus;2,118, &minus;2,101]'e sıkışıyor, encoder'a neredeyse sabit bir düzlem
gidiyordu. BatchNorm eğitim kipinde batch istatistikleriyle bu sinyali geri büyüttüğünden train loss
düşüyor, değerlendirme kipinde running istatistikleriyle batch istatistikleri arasındaki fark aynı oranda
büyütüldüğünden doğrulama çıktısı her epoch farklı dağılıma kayıyordu.</p>

<p><strong>(b) AMP altında Dice teriminin taşması.</strong> Yumuşak Dice paydasındaki olasılık toplamı
262.144 piksel üzerinden alındığından, ortalama tahmin olasılığı ~0,25'i geçtiğinde 16-bit üst sınırını
(65.504) aşıp sonsuza dönüyordu. Dice sıfır, Dice kaybı sabit 1 oluyor ve gradyan üretmiyordu; kayıp
fiilen salt BCE'ye düşüyordu. Eğitimin başında ve yüksek yoğunluklu vakalarda tetikleniyordu. Kayıp
hesabı 32-bit'e alınarak düzeltildi.</p>

<p>Düzeltmelerin ardından ilk epoch doğrulama Dice'ı 0,1071'den 0,2749'a çıktı. Raporlanan tüm sonuçlar
düzeltme sonrası koşulara aittir.</p>

<h2>3. Model Sonuçları</h2>
<p>Tüm metrikler doğrulama setinin tamamı (89 vaka) üzerinde, en iyi doğrulama Dice'ını veren kontrol
noktasıyla hesaplanmıştır. <code>std</code> ve <code>SEM</code> görüntü başına Dice dağılımından
türetilmiştir.</p>
{''.join(sections3)}

<section class="model-section">
<h2>4. Karşılaştırma</h2>

<h3>4.1 Genel tablo</h3>
<table>
<tr><th>#</th><th>Model</th><th>Val Loss</th><th>Dice</th><th>IoU</th>
<th>Precision</th><th>Recall</th><th>F1</th><th>R&minus;P</th><th>Parametre</th><th>En iyi epoch</th></tr>
{rows4}
</table>
<p class="tblnote">R&minus;P: recall &minus; precision farkı (bkz. 4.2). İkili segmentasyonda F1 = Dice.</p>

<h3>4.2 Decoder tipi ve aşırı segmentasyon</h3>
<p>Modeller arasındaki en düzenli örüntü toplam Dice'ta değil, recall ile precision arasındaki farkta
görülmektedir. Decoder'ın ayrıntıyı nasıl geri getirdiğine göre iki grup oluşmakta ve gruplar
arasında örtüşme bulunmamaktadır:</p>

<div class="mono-block">ince skip connection (U-Net, U-Net++)     : +0,0148  +0,0239  +0,0250    max 0,0250
kaba 4x büyütme (DeepLabV3+, SegFormer)  : +0,0513  +0,0520  +0,0903    min 0,0513
                                                                boşluk 0,0263</div>

<p>Ayrım encoder'dan bağımsızdır: SegFormer'ın MiT-B2 encoder'ı diğerlerinin konvolüsyonel encoder'larıyla
akraba olmadığı hâlde kaba büyütme grubuyla aynı davranışı göstermektedir; EfficientNet encoder'ı ise hem
U-Net (+0,0250) hem DeepLabV3+ (+0,0903) modellerinde yer almasına rağmen fark decoder'a göre
değişmektedir.</p>

<p>Her ölçekte skip connection taşıyan decoder'lar konumsal ayrıntıyı koruyup sınırları takip
etmektedir; sondan tek adımda 4&times; büyütenler dens dokuyu bulmakta (grubun recall değerleri yüksek)
ancak sınırlardan taşarak yanlış pozitif üretmektedir. Hedef bölgenin sınırları dağınık ve süreksiz
olduğundan bu ceza belirginleşmektedir. İlişki gözlemseldir ve kontrollü değildir &mdash; DeepLabV3+
modellerinin encoder'ları da farklıdır; iddia "decoder tipi tek başına bu farkı yaratır" değil,
"recall&minus;precision farkı encoder'a değil decoder tipine göre gruplanmaktadır" biçimindedir.</p>

<h3>4.3 Farkların istatistiksel değerlendirmesi</h3>
<p>Görüntü başına Dice standart sapması 0,134-0,164 aralığındadır. Bu sapmanın büyük kısmı modeller arası
farkı değil vakalar arası zorluk farkını yansıttığından, eşleşmemiş karşılaştırma bu tasarımda
gücü düşürmektedir. Altı model aynı 89 vakada aynı sırayla değerlendirildiğinden vaka başına fark
alınarak ortak varyans sadeleştirilebilir. Aşağıda iki yaklaşım karşılaştırılmakta olup referans
U-Net + ResNet34'tür.</p>

<p>Wilcoxon işaretli sıra testi, modellerin aynı doğrulama vakaları üzerindeki performans farkını
değerlendirmek amacıyla uygulanmıştır; ancak her modelin tek seed ile eğitilmiş olması nedeniyle
eğitim kaynaklı varyans istatistiksel analize dâhil edilmemiştir.</p>

<table>
<tr><th>Model</th><th>Fark</th><th>Eşleşmemiş SEM</th><th>Eşleşmiş SEM</th><th>SEM oranı</th><th>t</th><th>Wilcoxon p</th><th>Sonuç</th></tr>
<tr><td>U-Net++ + ResNet34</td><td class="num">0,0078</td><td class="num">0,0204</td><td class="num">0,0032</td><td class="num">6,4&times;</td><td class="num">2,40</td><td class="num">0,054</td><td>ayırt edilemez</td></tr>
<tr><td>SegFormer-B2</td><td class="num">0,0127</td><td class="num">0,0201</td><td class="num">0,0042</td><td class="num">4,8&times;</td><td class="num">3,05</td><td class="num">4,7&times;10<sup>-4</sup></td><td><strong>anlamlı</strong></td></tr>
<tr><td>DeepLabV3+ + ResNet50</td><td class="num">0,0247</td><td class="num">0,0206</td><td class="num">0,0038</td><td class="num">5,4&times;</td><td class="num">6,43</td><td class="num">2,2&times;10<sup>-8</sup></td><td><strong>anlamlı</strong></td></tr>
<tr><td>DeepLabV3+ + EfficientNet-B3</td><td class="num">0,0309</td><td class="num">0,0202</td><td class="num">0,0046</td><td class="num">4,4&times;</td><td class="num">6,67</td><td class="num">1,1&times;10<sup>-8</sup></td><td><strong>anlamlı</strong></td></tr>
<tr><td>U-Net + EfficientNet-B0</td><td class="num">0,0642</td><td class="num">0,0225</td><td class="num">0,0091</td><td class="num">2,5&times;</td><td class="num">7,05</td><td class="num">4,5&times;10<sup>-13</sup></td><td><strong>anlamlı</strong></td></tr>
</table>

<p>Eşleşmiş standart hatalar 2,5-6,4 kat küçüktür. Eşleşmemiş karşılaştırmaya göre yalnızca bir model anlamlı
biçimde geride kalırken, eşleşmiş validation karşılaştırmasına göre beş modelin dördü U-Net + ResNet34'ten anlamlı biçimde
geridedir; yalnızca U-Net++ sınırda kalmaktadır (p = 0,054).</p>

<div class="callout">
<p>Test, sıralamanın vaka örnekleme gürültüsünden kaynaklanmadığını göstermektedir. Kapsamadığı iki
nokta: (i) her mimari tek seed ile bir kez eğitildiğinden eğitim rastgeleliği hesaba katılmamıştır;
(ii) farkların büyüklüğü 0,008-0,031 aralığındadır ve literatürde radyologlar arası uyum
Dice = 0,76 &plusmn; 0,17 olarak bildirilmektedir (Larroza vd., 2022) &mdash; sıralama gerçek, pratik
karşılığı sınırlıdır.</p>
</div>

<h3>4.4 Encoder ve decoder seçimlerinin birlikte etkisi</h3>
<p>İlk üç model tamamlandığında iki faktörlü bir tasarım oluşmuştu: decoder sabitken encoder değişimi
0,0640, encoder sabitken decoder değişimi 0,0079 fark üretiyordu. Buradan encoder'ın belirleyici olduğu
sonucuna varılmıştı. Dördüncü model eklendiğinde bu genelleme geçerliliğini yitirmektedir; encoder ve
decoder etkileri bağımsız değil, birbirine bağlıdır:</p>

<div class="mono-block">                    ResNet      EffNet    encoder etkisi
U-Net               0,8059      0,7419        -0,0640
DeepLabV3+          0,7811      0,7751        -0,0060
decoder etkisi     -0,0248     +0,0332</div>

<p>ResNet &rarr; EfficientNet geçişinin bedeli U-Net decoder'ıyla 0,0640, DeepLabV3+ decoder'ıyla
0,0060'tır; decoder etkisinin işareti encoder'a göre değişmektedir. Daha basit bir açıklama da
elenememiştir: EfficientNet-B3 (11,7M), EfficientNet-B0'ın (6,3M) iki katına yakındır ve fark
decoder'dan değil encoder boyutundan kaynaklanıyor olabilir. İki açıklamayı ayıracak eksik model
U-Net + EfficientNet-B3'tür ve bu çalışmada eğitilmemiştir.</p>

<h3>4.5 Model boyutu ve checkpoint'ler</h3>
<p>Kontrol noktaları model ağırlıklarının yanı sıra optimizer, scheduler, scaler durumlarını,
epoch sayacını ve RNG durumlarını içermekte; Drive üzerinde
<code>density_segmentation/&lt;MODEL_KEY&gt;/checkpoints/best.pt</code> düzeninde saklanmaktadır.</p>

<table>
<tr><th>Model</th><th>Parametre</th><th>best.pt epoch</th><th>Kaydedilen val Dice</th><th>Toplam epoch</th></tr>
<tr class="best"><td>U-Net + ResNet34</td><td class="num">24.436.369</td><td class="num">54</td><td class="num">0,80585975</td><td class="num">70</td></tr>
<tr><td>U-Net++ + ResNet34</td><td class="num">26.078.609</td><td class="num">50</td><td class="num">0,79801751</td><td class="num">66</td></tr>
<tr><td>SegFormer-B2</td><td class="num">27.347.393</td><td class="num">33</td><td class="num">0,79320747</td><td class="num">49</td></tr>
<tr><td>DeepLabV3+ + ResNet50</td><td class="num">26.677.585</td><td class="num">39</td><td class="num">0,78111506</td><td class="num">55</td></tr>
<tr><td>DeepLabV3+ + EfficientNet-B3</td><td class="num">11.680.185</td><td class="num">31</td><td class="num">0,77506189</td><td class="num">47</td></tr>
<tr><td>U-Net + EfficientNet-B0</td><td class="num">6.251.469</td><td class="num">15</td><td class="num">0,74193355</td><td class="num">31</td></tr>
</table>

<p>Parametre sayısı sıralamayı açıklamamaktadır. İlk dört model 24,4M-27,3M bandında toplanmakta,
Dice değerleri 0,7811-0,8059 arasında dağılmaktadır; band içinde ilişki ters yöndedir &mdash; bandın en çok
parametreli modeli üçüncü, en az parametreli modeli birincidir. DeepLabV3+ + EfficientNet-B3, ResNet50
muadilinin parametresinin %44'üyle 0,0060 geride kalmaktadır. 447 eğitim görüntüsüyle ek kapasite
karşılık bulmamaktadır.</p>

<h2>5. En Başarılı Model</h2>

<div class="verdict">
<p><strong>U-Net + ResNet34</strong> &mdash; Dice 0,8059 &middot; IoU 0,6922 &middot; Precision 0,8153
&middot; Recall 0,8392 &middot; F1 0,8059 &middot; Val Loss 0,2624 &middot; 24.436.369 parametre
&middot; en iyi epoch 54</p>
<p style="margin-bottom:0">Dice, IoU ve val loss bakımından birinci; eşleşmiş validation karşılaştırmasında beş alternatifin
dördünden anlamlı biçimde üstün (p aralığı 4,7&times;10<sup>-4</sup> &ndash; 4,5&times;10<sup>-13</sup>),
U-Net++ ile sınırda (p = 0,054).</p>
</div>

<p><strong>Decoder mimarisi.</strong> Her ölçekte skip connection taşıdığından sınır ayrıntısını
korumaktadır; R&minus;P farkı +0,0239 ile ince skip grubunun içindedir. Hedef dokunun dağınık ve süreksiz
sınırlara sahip olduğu bu görevde belirleyici özellik budur (bkz. 4.2).</p>

<p><strong>Encoder transferi.</strong> Decoder sabitken ResNet34 &rarr; EfficientNet-B0 geçişi 0,0640
kaybettirmektedir. Ölçülebilir yansıması overfitting davranışıdır: EfficientNet-B0 modeli dört kat
küçük olmasına rağmen ortalama train/val makası +0,081, seçilen modelde +0,028'dir.</p>

<p><strong>Kapasite-veri dengesi.</strong> İlk dördün en küçüğü olmasına rağmen birincidir; 447 eğitim
görüntüsü ölçeğinde ek kapasitenin karşılık bulmadığı 4.5'te görülmektedir.</p>

<p><strong>Eğitim kararlılığı.</strong> 70 epoch bütçesini tam kullanan tek modeldir; yedi LR düşüşünün
her birine karşılık vermiş, epoch 25'teki platoyu aşarak epoch 54'te 0,8059'a ulaşmıştır. Son 10 epoch'ta
val loss bandı 0,0011 genişliğindedir.</p>

<p><strong>Hata profili.</strong> Precision (0,8153) ve recall (0,8392) birbirine yakındır. Görevin klinik
karşılığı alan tabanlı yoğunluk yüzdesi tahmini olduğundan dengeli profil, sistematik olarak taşan
(DeepLabV3+ + EfficientNet-B3, +0,0903) veya eksik segmente eden bir modele göre avantajlıdır.</p>

<p>Seçim, U-Net++ ile arasındaki farkın sınırda olması ve tek eğitim koşusu kısıtı dikkate alınarak
"tüm toplam metriklerde önde ve U-Net++ dışındaki alternatiflere karşı daha yüksek toplam
segmentasyon performansı sağlayan model" olarak okunmalıdır.</p>

<h2>6. Sonuç</h2>
<p>Altı model aynı veri işleme, augmentation, loss ve genel eğitim protokolü altında değerlendirilmiş; mimari gereksinimlere göre batch size ve learning rate gibi model-spesifik hiperparametreler ayarlanmıştır. En yüksek Dice'ı U-Net + ResNet34 verdi (0,8059); U-Net++ +
ResNet34 ikinci sırada yer aldı (0,7980) ve aradaki fark eşleşmiş validation karşılaştırmasında %5 eşiğini geçmedi. En geride
U-Net + EfficientNet-B0 kaldı (0,7419).</p>

<p>Çalışmanın en düzenli bulgusu decoder mimarisi ile aşırı segmentasyon eğilimi arasındaki ilişkidir:
ince ölçekli skip connection taşıyan decoder'ların R&minus;P farkı 0,025'in altında, sondan tek adımda
büyütenlerin 0,051'in üzerinde kalmış, altı model ve dört farklı decoder mimarisi boyunca gruplar arasında örtüşme
gözlenmemiştir. Bu, tek bir toplam skorda görünmeyen ancak model seçiminde karşılığı olan bir davranış
farkıdır: yüksek recall'un öncelikli olduğu senaryolarda DeepLabV3+ ailesinin profili, sınır doğruluğunun
önemli olduğu alan ölçümü senaryolarında U-Net ailesi tercih edilebilir.</p>

<p>Parametre sayısı sıralamayı açıklamamaktadır; ilk dört model dar bir parametre bandında toplanmasına
rağmen band içindeki ilişki ters yöndedir. Ayrıca ilk üç modelden çıkarılan "encoder decoder'dan daha
belirleyicidir" genellemesi dördüncü model eklendiğinde geçerliliğini yitirmiş; encoder ve decoder
etkilerinin bağımsız olmadığı görülmüştür (bkz. 4.4).</p>

<h3>6.1 Sınırlamalar</h3>
<ul>
<li>Her mimari tek seed ile (seed 42) bir kez eğitilmiştir; eşleşmiş validation vakaları tekrar birimi
olarak almakta, eğitim koşusu varyansını kapsamamaktadır. Çalışmanın en önemli istatistiksel
sınırlamasıdır.</li>
<li>Ayrı bir test seti yoktur; tüm sonuçlar doğrulama seti üzerindedir ve checkpoint seçimi de aynı
set üzerinde yapıldığından bir miktar iyimserlik beklenmelidir. Çapraz doğrulama uygulanmamıştır.</li>
<li>Referans etiketin kendisi belirsizdir (radyologlar arası uyum Dice = 0,76 &plusmn; 0,17); modeller
arası 0,008-0,031'lik farklar bu belirsizliğin bir mertebe altındadır.</li>
<li>Mimari başına hiperparametre araması yapılmamıştır. Özellikle EfficientNet encoder'larının farklı bir
öğrenme oranı rejimine ihtiyaç duyup duymadığı sınanmamıştır.</li>
<li>512&times;512 karelerin yaklaşık %55'i boş dolgudur. Meme bounding box çevresine kırpma etkin
çözünürlüğü artırabilirdi; aynı görüntü alanında yürütülen kitle tespiti çalışmasında kırpma ablasyonu
kayda değer kazanç sağlamadığı için (mAP@50: 0,6928 kırpmasız / 0,6901 kırpılmış) ve karşılaştırılabilirliği
korumak adına uygulanmamıştır.</li>
<li>Erken durdurma val Dice'ı, LR scheduler'ı val loss'u izlemektedir. İki ölçüt U-Net +
EfficientNet-B0 modelinde ayrışmıştır; protokol tüm modellerde aynı olduğundan karşılaştırma
etkilenmemekle birlikte gürültülü Dice eğrisine sahip modeller daha erken kesilmektedir.</li>
<li>Kullanılan veri sürümünde meme alanı maskesi bulunmadığından, tahmin maskesinden alan tabanlı
yoğunluk yüzdesi hesaplanamamış ve klinik olarak Dice'tan daha yorumlanabilir olan bu ölçüt
raporlanamamıştır.</li>
</ul>

<h2>Kaynaklar</h2>
<ol class="refs">
<li>Behravan, H., Gudhe, N. R., Okuma, H., Sudah, M., &amp; Mannermaa, A. (2024). A dataset of
mammography images with area-based breast density values, breast area, and dense tissue segmentation
masks. <em>Data in Brief</em>.
<a href="https://www.sciencedirect.com/science/article/pii/S2352340924009429">S2352340924009429</a>
&mdash; <a href="https://data.mendeley.com/datasets/tdx3h2fn9v/2">data.mendeley.com/datasets/tdx3h2fn9v</a></li>
<li>Larroza, A. vd. (2022). Breast Dense Tissue Segmentation with Noisy Labels: A Hybrid Threshold-Based
and Mask-Based Approach. <em>Diagnostics</em>.
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9406546/">PMC9406546</a></li>
<li>Saffari, N. vd. (2020). Fully Automated Breast Density Segmentation and Classification Using Deep
Learning. <em>Diagnostics</em>, 10(11), 988.
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7700286/">PMC7700286</a></li>
<li>He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep Residual Learning for Image Recognition.
<em>CVPR</em>. &mdash; ResNet34 ve ResNet50 encoder'ları</li>
<li>Tan, M., &amp; Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural
Networks. <em>ICML</em>. &mdash; EfficientNet-B0 ve EfficientNet-B3 encoder'ları</li>
<li>Ronneberger, O., Fischer, P., &amp; Brox, T. (2015). U-Net: Convolutional Networks for Biomedical
Image Segmentation. <em>MICCAI</em>.</li>
<li>Zhou, Z. vd. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
<em>DLMIA</em>.</li>
<li>Chen, L.-C. vd. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation. <em>ECCV</em>.</li>
<li>Xie, E. vd. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers. <em>NeurIPS</em>.</li>
<li>Nguyen, H. T. vd. (2023). VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis
in full-field digital mammography. <em>Scientific Data</em>.</li>
</ol>
</section>
</body></html>'''

os.makedirs('rapor', exist_ok=True)
open(OUT, 'w', encoding='utf-8').write(HTML)
print(f'{OUT}  ->  {os.path.getsize(OUT)/1024/1024:.2f} MB')
