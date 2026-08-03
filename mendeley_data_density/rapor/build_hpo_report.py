# -*- coding: utf-8 -*-
"""Bagimsiz HPO raporu -> tek dosyalik HTML. Uslup/terminoloji ana raporla ayni."""
import base64, os

os.chdir(r'd:\mamografi\mendeley_data_density')
FIG = 'rapor/figurler'
OUT = 'rapor/hpo_raporu.html'

CAP1 = ('Şekil 1. Karar eşiği taraması. Eğitim eğrisi 0,45-0,60 aralığında düzdür; eğitim ve '
        'doğrulama optimumları farklı eşiklerde oluşmaktadır.')
CAP2 = ('Şekil 2. Altı deneyin doğrulama Dice değerleri. Kesikli çizgi baseline, hata çubukları '
        'standart hatadır. Kırmızı: Bonferroni düzeltmesinden sonra anlamlı düşüş. Gri: baseline '
        'ile ayırt edilemez.')


def img(name, caption):
    b64 = base64.b64encode(open(f'{FIG}/{name}', 'rb').read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


STYLE = '''@page { size: A4; margin: 18mm 16mm; }
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
.strong { font-weight: 700; }
tr.best td { background: #f2f8f2; }
figure { margin: 9pt 0 12pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 1px solid #d5dae2; }
figcaption { font-size: 8.5pt; color: #4a4a4a; margin-top: 4pt; text-align: left; line-height: 1.35; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f4f6f9;
       padding: 1px 4px; border-radius: 2px; }
.callout { border-left: 3px solid #2c5aa0; background: #f6f9fd; padding: 7pt 11pt; margin: 9pt 0;
           page-break-inside: avoid; }
.callout.warn { border-left-color: #b8860b; background: #fdfaf2; }
.callout p:last-child { margin-bottom: 0; }
.tblnote { font-size: 8.5pt; color: #555; margin: -5pt 0 10pt; line-height: 1.35; }
ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
li { margin-bottom: 3pt; }
.mono-block { font-family: Consolas, monospace; font-size: 8.8pt; background: #f4f6f9;
              border: 1px solid #dde3ec; padding: 7pt 10pt; margin: 8pt 0; white-space: pre;
              page-break-inside: avoid; }'''

BODY = '''
<h1>Hiperparametre Optimizasyonu — Deney Raporu</h1>
<p class="sub">U-Net + ResNet34 · Meme yoğunluğu segmentasyonu</p>

<h2>1. Amaç ve Kapsam</h2>
<p>Ana çalışmada altı model karşılaştırılmış, en yüksek Dice'ı U-Net + ResNet34 vermiştir (0,8059).
Söz konusu çalışmada hiperparametreler literatür taramasıyla belirlenmiş, mimari başına ayrı bir
arama yürütülmemişti. Bu eksiği kapatmak üzere aynı model üzerinde yedi deney yapılmıştır: eğitim
gerektirmeyen bir karar eşiği taraması ve altı yeniden eğitim.</p>

<p>Amaç, literatürden alınan ayarların bu veri seti için uygun olup olmadığını ve hedefli
değişikliklerle daha yüksek Dice elde edilip edilemeyeceğini sınamaktır.</p>

<h2>2. Yöntem</h2>
<h3>2.1 Deney tasarımı</h3>
<p>Her deney baseline'dan <strong>yalnızca tek bir ayarı</strong> değiştirmiştir. Kademeli (greedy)
arama, yani bir aşamanın kazananını sabitleyip sonrakine geçme yaklaşımı kullanılmamıştır; böylece
her sonuç tek başına yorumlanabilmekte ve seçim yanlılığı aşamalar boyunca birikmemektedir.</p>

<p>Veri bölünmesi (447 eğitim / 89 doğrulama), ön işleme, seed (42), değerlendirme kodu ve en iyi
checkpoint seçim ölçütü ana çalışmayla aynı tutulmuştur. Baseline bu deney setinde yeniden ölçülmüş
ve ana çalışmayla aynı sonucu vermiştir (Dice 0,805860 &middot; IoU 0,692159 &middot; Precision
0,815256 &middot; Recall 0,839154 &middot; en iyi epoch 54); bu, iki notebook arasındaki tutarlılığı
doğrulamaktadır.</p>

<h3>2.2 Model seçimi</h3>
<p>Ana çalışmada ilk iki sırayı U-Net + ResNet34 (0,8059) ve U-Net++ + ResNet34 (0,7980) almış,
aralarındaki fark eşleşmiş validation karşılaştırmasında %5 eşiğini geçmemişti (p = 0,054). Her ikisi
de aynı encoder'ı kullanmaktadır. Buna karşın arama yalnızca birinci model üzerinde yürütülmüştür.
Gerekçeler aşağıdadır.</p>

<p><strong>Doğrulama olanağı.</strong> İki modelde paralel arama yürütmek yerine birinde arayıp
kazanan ayarı diğerinde tekrarlamak, elde edilen kazancın gerçek mi yoksa doğrulama gürültüsü mü
olduğunu ayırt etmeyi sağlamaktadır. İki model aynı encoder'ı paylaştığından, özellikle öğrenme oranı
bulgularının transfer olması beklenir; transfer gerçekleşmezse kazancın veriye özgü olduğu anlaşılır.
Paralel arama bu ayrımı yapma imkânını ortadan kaldırırdı.</p>

<p><strong>Çoklu karşılaştırma yükü.</strong> Aramanın iki modelde yürütülmesi karşılaştırma sayısını
altıdan on ikiye çıkarır; Bonferroni eşiği 0,0083'ten 0,0042'ye iner ve gerçek bir farkın
saptanabilmesi zorlaşır.</p>

<p><strong>Seçim yanlılığı.</strong> Denenen konfigürasyon sayısı arttıkça en iyisinin seçilmesinden
doğan yapay kazanç büyümektedir. Ölçülen SEM (0,0143) üzerinden beklenen şişme altı konfigürasyon
için +0,018 Dice iken on iki konfigürasyon için +0,023'e çıkmaktadır &mdash; ana çalışmadaki ilk üç
modelin toplam farkının (0,0127) neredeyse iki katı.</p>

<p>Bu tercih ayrıca hesaplama maliyetini yarıya indirmektedir; koşu başına süre 15-30 dakikadır.</p>

<h3>2.3 İstatistiksel karşılaştırma</h3>
<p>Karşılaştırmalar aynı 89 doğrulama vakasında, vaka başına fark üzerinden (eşleşmiş validation
karşılaştırması) yapılmıştır. Ölçülen görüntü başına Dice standart sapması ~0,14'tür ve bunun büyük
kısmı modeller arası farkı değil vakalar arası zorluk farkını yansıtmaktadır; eşleşmiş karşılaştırma
bu ortak bileşeni sadeleştirerek küçük gerçek farkların görülmesini sağlamaktadır. p değerleri
Wilcoxon işaretli sıra testinden gelmektedir.</p>

<p>Altı karşılaştırma birden yürütüldüğünden çoklu karşılaştırma düzeltmesi uygulanmış, anlamlılık
eşiği Bonferroni ile <strong>0,05 / 6 = 0,0083</strong> alınmıştır. Her deney tek seed ile
eğitildiğinden eğitim kaynaklı varyans analize dâhil edilmemiştir.</p>

<h2>3. Karar Eşiği Taraması</h2>
<p>Ana çalışmanın tamamında karar eşiği 0,50'de sabittir. Optimal eşik <strong>eğitim seti
üzerinde</strong> aranmıştır; doğrulama setinde aramak, sonucun yine aynı sette raporlanması
nedeniyle yapay kazanç üretirdi. Eğitim seti beş kat büyüktür ve bu tür bir sızıntı
oluşturmamaktadır. Bu deney yeniden eğitim gerektirmemekte, mevcut <code>best.pt</code> ile yalnızca
çıkarım yapılmaktadır.</p>

<table>
<tr><th>Eşik</th><td class="num">0,30</td><td class="num">0,35</td><td class="num">0,40</td>
<td class="num">0,45</td><td class="num">0,50</td><td class="num">0,55</td><td class="num">0,60</td>
<td class="num">0,65</td><td class="num">0,70</td></tr>
<tr><th>Eğitim Dice</th><td class="num">0,8558</td><td class="num">0,8573</td><td class="num">0,8584</td>
<td class="num">0,8591</td><td class="num">0,8595</td><td class="num strong">0,8596</td>
<td class="num">0,8595</td><td class="num">0,8588</td><td class="num">0,8576</td></tr>
<tr><th>Doğrulama Dice</th><td class="num">0,8061</td><td class="num strong">0,8064</td>
<td class="num strong">0,8064</td><td class="num">0,8063</td><td class="num">0,8059</td>
<td class="num">0,8051</td><td class="num">0,8042</td><td class="num">0,8029</td>
<td class="num">0,8010</td></tr>
</table>
<p class="tblnote">Koyu değerler satır maksimumudur. Seçim eğitim satırına göre yapılmıştır.</p>

@FIG1@

<p>Kazanç elde edilmemiş, eşik 0,50'de bırakılmıştır. Üç gözlem öne çıkmaktadır.</p>
<ul>
<li><strong>Eğri düzdür.</strong> 0,45-0,60 arasındaki dört noktada eğitim Dice yayılımı yalnızca
0,0005'tir. Bu bir optimum değil sayısal gürültüdür; <code>argmax</code>'ın 0,55'i seçmesi anlamlı
bir tercih sayılamaz. Düz bir eğride varsayılan değer ancak belirli bir marjı aşan iyileşme için
değiştirilmelidir, burada öyle bir marj bulunmamaktadır.</li>
<li><strong>Eğitim ve doğrulama optimumları ayrışmaktadır</strong> (0,55 / 0,35). Model eğitim
görüntülerinde daha güvenli tahmin ürettiğinden orada daha yüksek eşik uygun düşmektedir. Eğitimde
seçip doğrulamada uygulama yaklaşımı yöntemsel olarak temiz olmakla birlikte bu parametrede transfer
sağlamamıştır.</li>
<li><strong>Doğrulamanın kendi optimumu da kazanç vermemektedir.</strong> Eşik 0,35-0,40 ile
doğrulama Dice 0,8064'tür; baseline'a göre +0,0005 (0,03 SEM). Doğrulamada arama yapılıp yanlılık
göze alınsaydı bile sonuç değişmezdi.</li>
</ul>

<p>Yan bulgu: eşik 0,50'de eğitim Dice 0,8595, doğrulama 0,8059'dur; aradaki fark +0,0536. Ana
çalışmada bildirilen train/val loss makasıyla (+0,049) tutarlıdır ve hafif overfitting'i bağımsız
bir ölçütle doğrulamaktadır.</p>

<h2>4. Denenen Ayarlar</h2>
<table>
<tr><th>Deney</th><th>Değişen ayar</th><th>Baseline</th><th>Denenen</th><th>Gerekçe</th></tr>
<tr><td><code>no_vflip</code></td><td>Dikey çevirme olasılığı</td><td class="num">0,3</td>
<td class="num">0,0</td>
<td>Mamografide dens doku dağılımı anatomik olarak yapılıdır; dikey çevirmenin bu ön bilgiyi bozduğu
ve literatürde sık tercih edilmediği değerlendirilmiştir.</td></tr>
<tr><td><code>enc_lr_3e5</code></td><td>Encoder LR</td><td>5&times;10<sup>-5</sup></td>
<td>3&times;10<sup>-5</sup></td>
<td rowspan="2">ImageNet ön-eğitimli encoder'ın küçük veri setinde fazla güncellenmesinin overfitting
ürettiği hipotezi. Ana çalışmada train/val makasının +0,049'da sabitlenmesi bu yönde işaret
sayılmıştır. Literatürde encoder için 1&times;10<sup>-5</sup>-3&times;10<sup>-5</sup> bandı
yaygındır.</td></tr>
<tr><td><code>enc_lr_1e5</code></td><td>Encoder LR</td><td>5&times;10<sup>-5</sup></td>
<td>1&times;10<sup>-5</sup></td></tr>
<tr><td><code>cosine</code></td><td>Scheduler</td><td>ReduceLROnPlateau</td>
<td>CosineAnnealingLR</td>
<td>Ön-eğitimli encoder'ın daha yumuşak güncellenmesi ve son epoch'larda daha iyi yerel minimum
bulunması beklentisi; son yıllarda segmentasyon çalışmalarında kullanımı artmıştır.</td></tr>
<tr><td><code>dice_focal</code></td><td>Loss</td><td>Dice + BCE</td>
<td>Dice + Focal (α=0,25, γ=2)</td>
<td>Kolay piksellerin ağırlığı azaltılarak zor sınır bölgelerine odaklanma.</td></tr>
<tr><td><code>dice_tversky</code></td><td>Loss</td><td>Dice + BCE</td>
<td>Dice + Tversky (α=0,7, β=0,3)</td>
<td>Baseline'da recall (0,8392) precision'dan (0,8153) yüksektir; yanlış pozitifin daha ağır
cezalandırılarak dengenin precision lehine kaydırılması.</td></tr>
</table>

<p>Değiştirilmeyen ayarlar: batch 16, decoder LR 2,5&times;10<sup>-4</sup>, AdamW, weight decay
1&times;10<sup>-4</sup>, 3 epoch doğrusal warmup, gradyan kırpma (max_norm 1,0), AMP, görüntü boyutu
512&times;512, maksimum 70 epoch, patience 15.</p>

<div class="callout">
<p><code>cosine</code> deneyinde erken durdurma devre dışı bırakılmıştır (patience = 70). Kosinüs
çevrimi kesildiğinde model düşük öğrenme oranındaki ince ayar fazına girememekte ve yöntem haksız
biçimde dezavantajlı duruma düşmektedir. Bu, bir deneyde bilinçli olarak değiştirilen tek ek
ayardır.</p>
</div>

<h2>5. Sonuçlar</h2>
<table>
<tr><th>Deney</th><th>Dice</th><th>Fark</th><th>Eşleşmiş SEM</th><th>Wilcoxon p</th><th>Karar</th>
<th>IoU</th><th>Precision</th><th>Recall</th><th>R&minus;P</th><th>En iyi<br>epoch</th>
<th>Toplam<br>epoch</th><th>Süre<br>(dk)</th></tr>
<tr class="best"><td>BASELINE</td><td class="num strong">0,8059</td><td class="num">&mdash;</td>
<td class="num">&mdash;</td><td class="num">&mdash;</td><td>referans</td>
<td class="num">0,6922</td><td class="num">0,8153</td><td class="num">0,8392</td>
<td class="num">+0,0239</td><td class="num">54</td><td class="num">70</td><td class="num">&mdash;</td></tr>
<tr><td><code>dice_focal</code></td><td class="num">0,8015</td><td class="num">&minus;0,0043</td>
<td class="num">0,0026</td><td class="num">0,0074</td><td><strong>anlamlı düşüş</strong></td>
<td class="num">0,6859</td><td class="num">0,7979</td><td class="num">0,8482</td>
<td class="num">+0,0503</td><td class="num">46</td><td class="num">62</td><td class="num">20,2</td></tr>
<tr><td><code>no_vflip</code></td><td class="num">0,7992</td><td class="num">&minus;0,0067</td>
<td class="num">0,0033</td><td class="num">0,0221</td><td>ayırt edilemez</td>
<td class="num">0,6822</td><td class="num">0,8083</td><td class="num">0,8397</td>
<td class="num">+0,0313</td><td class="num">59</td><td class="num">70</td><td class="num">30,0</td></tr>
<tr><td><code>enc_lr_3e5</code></td><td class="num">0,7978</td><td class="num">&minus;0,0081</td>
<td class="num">0,0038</td><td class="num">0,0094</td><td>ayırt edilemez</td>
<td class="num">0,6804</td><td class="num">0,8077</td><td class="num">0,8345</td>
<td class="num">+0,0268</td><td class="num">48</td><td class="num">64</td><td class="num">22,3</td></tr>
<tr><td><code>cosine</code></td><td class="num">0,7956</td><td class="num">&minus;0,0102</td>
<td class="num">0,0030</td><td class="num">0,0017</td><td><strong>anlamlı düşüş</strong></td>
<td class="num">0,6794</td><td class="num">0,8206</td><td class="num">0,8199</td>
<td class="num">&minus;0,0007</td><td class="num">50</td><td class="num">70</td><td class="num">22,5</td></tr>
<tr><td><code>dice_tversky</code></td><td class="num">0,7953</td><td class="num">&minus;0,0106</td>
<td class="num">0,0031</td><td class="num">0,0004</td><td><strong>anlamlı düşüş</strong></td>
<td class="num">0,6789</td><td class="num">0,7917</td><td class="num">0,8465</td>
<td class="num">+0,0548</td><td class="num">28</td><td class="num">44</td><td class="num">14,6</td></tr>
<tr><td><code>enc_lr_1e5</code></td><td class="num">0,7847</td><td class="num">&minus;0,0211</td>
<td class="num">0,0068</td><td class="num">0,0004</td><td><strong>anlamlı düşüş</strong></td>
<td class="num">0,6671</td><td class="num">0,8092</td><td class="num">0,8145</td>
<td class="num">+0,0053</td><td class="num">63</td><td class="num">70</td><td class="num">27,7</td></tr>
</table>
<p class="tblnote">Anlamlılık eşiği Bonferroni düzeltmesiyle 0,0083. R&minus;P: recall &minus;
precision farkı. Süreler NVIDIA T4 üzerinde ölçülmüştür; <code>no_vflip</code> sıradaki ilk deney
olduğundan veri önbelleği soğuktur ve epoch başına süresi diğerlerinden yüksektir.</p>

@FIG2@

<h2>6. Değerlendirme</h2>
<p>Denenen hiçbir değişiklik baseline'ı geçmemiştir. Altı deneyin dördü Bonferroni düzeltmesinden
sonra anlamlı düşüş göstermiş, ikisi baseline'dan ayırt edilememiştir. Literatür taramasıyla
belirlenen ayarlar bu veri seti için yerel bir optimumda görünmektedir.</p>

<h3>6.1 Öğrenme oranı</h3>
<p>Encoder öğrenme oranı düşürüldükçe Dice monoton biçimde azalmaktadır:</p>
<div class="mono-block">encoder LR   5x10^-5 (baseline)  ->  Dice 0,8059
             3x10^-5             ->  Dice 0,7978   (-0,0081)
             1x10^-5             ->  Dice 0,7847   (-0,0211)</div>

<p>Bu sonuç, encoder'ın fazla güncellendiği ve overfitting ürettiği hipotezini çürütmektedir.
ImageNet öznitelikleri mamografi dokusuna uyum sağlamak için belirgin bir güncelleme
gerektirmektedir; doğal görüntülerle mamografi arasındaki alan farkı, düşük öğrenme oranının yeterli
olamayacağı kadar büyüktür. Nitekim <code>enc_lr_1e5</code> doğrulama Dice'ını epoch 63'e kadar
iyileştirmeye devam etmiş, yani overfitting değil yetersiz öğrenme göstermiştir.</p>

<p>Eğilim tek yönlü olduğundan doğal devam adımı, encoder öğrenme oranının 5&times;10<sup>-5</sup>'in
üzerine çıkarılmasını sınamaktır (örneğin 7&times;10<sup>-5</sup> ve 1&times;10<sup>-4</sup>). Bu
çalışmada yapılmamıştır.</p>

<h3>6.2 Dikey çevirme</h3>
<p>Anatomik tutarlılık gerekçesiyle kapatıldığında Dice 0,0067 düşmüştür. Fark Bonferroni sonrası
anlamlı değildir (p = 0,0221), ancak yön beklenenin tersidir. 447 eğitim görüntülük bir sette
augmentation çeşitliliğinin katkısı anatomik sadakatten daha belirleyici görünmektedir; dikey çevirme
bir bozulma kaynağı değil, düzenleyici (regularizer) işlevi görmektedir.</p>

<h3>6.3 Cosine scheduler</h3>
<p>Bu deney tablonun en yüksek precision'ını (0,8206) ve tek negatif R&minus;P değerini
(&minus;0,0007) üretmiştir; ana çalışmada U-Net ailesinde gözlenen hafif aşırı segmentasyon eğilimini
ortadan kaldırmıştır. Buna karşılık toplam Dice 0,0102 düşmüştür. Sınır doğruluğunun toplam
örtüşmeden öncelikli olduğu bir kullanımda bu ayar tercih edilebilir.</p>

<div class="callout warn">
<p><strong>Açıklanamayan sonuç.</strong> Tversky kaybında α=0,7 seçilerek yanlış pozitifin daha ağır
cezalandırılması ve precision'ın yükselmesi hedeflenmişti. Gerçekleşen bunun tersidir: precision
0,8153'ten 0,7917'ye düşmüş, recall 0,8392'den 0,8465'e yükselmiştir. Loss fonksiyonunun TP/FP/FN
tanımları doğrulanmıştır. Olası açıklama, bu deneyin en erken sonlanan koşu olmasıdır (en iyi epoch
28, erken durdurma epoch 43); asimetrik loss doğrulama kaybı eğrisini değiştirerek durdurma ölçütünü
erken tetiklemiş olabilir. Sonuç, loss fonksiyonunun kalitesinden çok erken sonlanmayla karışmış
durumdadır ve bu çalışmada ayrıştırılamamıştır.</p>
</div>

<h2>7. Sınırlamalar</h2>
<p>Ayrı bir test seti bulunmadığından, doğrulama setinde yürütülen her hiperparametre araması
raporlanan sonucu iyimser yönde şişirme riski taşımaktadır. Altı konfigürasyon denenip en iyisi
seçildiğinde beklenen yapay kazanç, ölçülen SEM (0,0143) üzerinden yaklaşık +0,018 Dice'tır; bu değer
ana çalışmadaki ilk üç modelin toplam farkından (0,0127) büyüktür. Arama bu nedenle dar tutulmuş,
konfigürasyonlar önceden belirlenmiş ve tümü raporlanmıştır; yalnızca en iyisinin bildirilmesi bu
yanlılığı gizlerdi. Söz konusu risk somut olarak gerçekleşmemiştir: hiçbir deney baseline'ı
geçmediğinden seçilecek bir kazanan oluşmamış, dolayısıyla seçim yanlılığı da doğmamıştır.</p>

<ul>
<li>Her deney tek seed (42) ile bir kez eğitilmiştir; eşleşmiş karşılaştırma vakaları tekrar birimi
olarak almakta, eğitim koşusu varyansını kapsamamaktadır. Gözlenen 0,004-0,021 aralığındaki farkların
ne kadarının eğitim rastgeleliğinden geldiği ölçülememiştir.</li>
<li>Tek faktör değiştirme tasarımı ayarlar arasındaki etkileşimleri görememektedir. Örneğin daha
düşük encoder öğrenme oranı, daha uzun epoch bütçesiyle birlikte farklı sonuç verebilirdi.</li>
<li>Sonuçlar yalnızca U-Net + ResNet34 için geçerlidir. Bir ayarın gerçekten etkili olduğunun
doğrulanması için kazanan konfigürasyonun ikinci bir model üzerinde (U-Net++ + ResNet34)
tekrarlanması gerekir (bkz. 2.2); kazanan çıkmadığından bu adım uygulanmamıştır.</li>
<li>Eşik taramasında yalnızca tek bir global eşik denenmiş, vaka bazında uyarlanan eşik stratejileri
sınanmamıştır.</li>
</ul>

<h2>8. Sonuç</h2>
<p>Yedi deneyin hiçbiri U-Net + ResNet34 baseline'ını (Dice 0,8059) geçememiştir. Karar eşiği zaten
optimaldir; encoder öğrenme oranının düşürülmesi, dikey çevirmenin kapatılması, cosine scheduler'a
geçilmesi ve loss fonksiyonunun Focal veya Tversky ile değiştirilmesi sonucu iyileştirmemiştir. Bu
bulgu, ana çalışmadaki hiperparametre seçimlerinin bu veri seti için isabetli olduğunu göstermekte ve
raporun mimari başına hiperparametre araması yapılmadığına ilişkin sınırlamasını kapatmaktadır.</p>

<p>Elde edilen tek yönlü sinyal encoder öğrenme oranındadır: eğilim, mevcut değerin altına değil
üstüne çıkılması gerektiğini düşündürmektedir. Bu, sonraki deney turunun doğal başlangıç
noktasıdır.</p>
'''

body = BODY.replace('@FIG1@', img('hpo_threshold.png', CAP1)).replace('@FIG2@', img('hpo_comparison.png', CAP2))
html = ('<!DOCTYPE html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        '<title>Hiperparametre Optimizasyonu — Deney Raporu</title>\n<style>\n'
        + STYLE + '\n</style></head><body>\n' + body + '\n</body></html>')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'{OUT}  ->  {os.path.getsize(OUT)/1024:.0f} KB')
