# -*- coding: utf-8 -*-
"""Rapor metni. Tablolar build_report.py'den geliyor.

Metin burada ayri tutuluyor cunku build_report.py bicimlendirme isi yapiyor; ikisini ayni
dosyada tutmak her ikisini de okunmaz hale getiriyor.

Metindeki koseli parantezli numaralar sondaki kaynak listesine karsilik geliyor. Sayisal
her iddia, kaynagin kendi ozeti ya da tam metni uzerinden dogrulandi; dogrulanamayan
iddialar metinden cikarildi ya da acikca isaretlendi.
"""

KAYNAK_SAYISI = 35


def govde(T):
    """T: tablo sözlüğü (hazır HTML parçaları)."""
    return f"""
<h1>Mamografi ile Meme Kanseri Tahmini</h1>
<p class="sub">Derin öğrenme literatürünün iki kollu değerlendirmesi: tespit ve
sınıflandırma, risk tahmini &middot; 2020&ndash;2026</p>

<h2>1. Amaç ve kapsam</h2>

<p>Bu rapor mamografi üzerinde derin öğrenme ile meme kanseri tahmini alanını 2020&ndash;2026
döneminde inceliyor. Alan tek bir problem gibi görünüyor ama değil. İki ayrı kol var ve
bunlar farklı veri, farklı metrik ve farklı klinik senaryo taşıyor. Raporun kurgusu bu
ayrım üzerine oturuyor, çünkü iki kolun sayılarını yan yana koymak yanıltıcı sonuç veriyor.</p>

<p>Birinci kol tanısal tespit ve sınıflandırma. Elde bir mamogram var, içinde kitle ya da
kalsifikasyon aranıyor ve bulunan lezyonun iyi mi kötü mü huylu olduğu soruluyor. Tipik
metrikler sınıflandırmada AUC, duyarlılık ve özgüllük, tespitte mAP, IoU ve FROC. İkinci kol
prognostik risk tahmini. Şu anda kanser bulgusu olmayan bir kadında önümüzdeki bir ile beş
yıl içindeki kanser olasılığı hesaplanıyor. Metrikler 1&ndash;5 yıllık AUC ve C-indeksi;
gerekli veri ise uzun takipli tarama kohortları, çünkü etiket görüntünün kendisinde değil
takip kaydında.</p>

<p>Kısa sonuç şu. Birinci kolda mimari CNN'lerden transformer ve vizyon-dil temel modellerine
kaymış durumda, ama küçük ve dengesiz veri setlerinde bildirilen yüzde 99&ndash;100 doğruluk
değerleri gerçek tarama koşullarını yansıtmıyor. Gerçekçi ölçekte, 2023 RSNA yarışmasında
değerlendirilen 1.537 algoritmanın en iyisi yüzde 48,6 duyarlılık ve yüzde 99,5 özgüllük
verdi; medyan algoritma yüzde 27,6 duyarlılıkta kaldı [3]. İkinci kolda referans model Mirai;
geliştiricinin kendi test kümelerinde C-indeksi 0,76&ndash;0,81, bağımsız popülasyonlarda
0,63'e kadar düşüyor [17, 27]. İki kolun ortak darboğazı aynı: dış doğrulama. Lowry ve
arkadaşlarının 41 çalışmalık risk derlemesinde çalışmaların tamamı retrospektif ve yalnızca
altısı kalibrasyon değerlendirmiş [30].</p>

<h2>2. Yöntem</h2>

<p>Tarama PubMed, Scopus, Web of Science, arXiv, Papers with Code ve Semantic Scholar
üzerinden yürütüldü. İki başlangıç noktası kullanıldı. Birincisi her kol için bir PRISMA
sistematik derlemesi: Kol 1 için Amin ve arkadaşları [1], Kol 2 için Lowry ve arkadaşları
[30]. İkincisi atıf kartopu: Mirai [17] ve AsymMirai [20] düğümlerinden ileri ve geri atıf
grafiği izlendi, VinDr-Mammo ve EMBED veri seti makalelerinin atıfları taranarak bu kümeleri
kullanan yeni modeller bulundu.</p>

<p>Kullanılan sorgular aşağıda. Bunlar raporun tekrar üretilebilmesi için veriliyor.</p>

<div class="mono-block">PubMed, Kol 1:
("Mammography"[Mesh]) AND ("Deep Learning"[Mesh] OR "Neural Networks, Computer"[Mesh])
AND ("Early Detection of Cancer"[Mesh] OR "diagnosis"[Subheading])

PubMed, Kol 2:
("Mammography"[Mesh]) AND ("Risk Assessment"[Mesh]) AND ("Deep Learning"[Mesh])
AND ("Breast Neoplasms"[Mesh])

Derleme filtresi: AND (systematic[sb] OR "systematic review"[Publication Type])

Scopus / Web of Science:
TITLE-ABS-KEY(mammography AND ("deep learning" OR "convolutional neural network"
OR transformer) AND (classification OR detection OR segmentation)
AND ("systematic review" OR PRISMA))

TITLE-ABS-KEY(mammography AND "deep learning" AND ("risk prediction"
OR "future breast cancer") AND ("external validation" OR PRISMA))

arXiv (cat:eess.IV OR cat:cs.CV):
mammography transformer classification / mammogram multi-view Swin
longitudinal mammogram risk prediction / mammography foundation model self-supervised</div>

<p>Rapora giren her sayısal değer kaynağın kendi özeti ya da tam metni üzerinden ayrıca
doğrulandı. Doğrulanamayan değerler metne alınmadı. İkincil kaynaklarda geçen ama başka bir
çalışmaya ait olan sonuçlar, ikincil kaynağa değil asıl çalışmaya atfedildi. Rapor
dolayısıyla taranan tüm literatürü değil, doğrulanmış kaynakları temsil ediyor; sistematik
derlemelerin kendisi birkaç yüz makalelik daha geniş bir havuzu kapsıyor.</p>

<h2>3. Kol 1: tespit ve sınıflandırma</h2>

<h3>3.1 Alanın görünümü ve mimari evrim</h3>

<p>Amin ve arkadaşlarının PRISMA derlemesi 1.051 kaydı tarayıp 287 makaleyi dahil ederek
alanı haritalıyor [1]. Çalışmaların çoğunluğu lezyon sınıflandırmasına odaklanmış; tespit,
segmentasyon ve meme yoğunluğu değerlendirmesi görece daha az çalışılmış. Yıllık üretim
2018'de 15 makaleden 2023'te 86'ya çıkmış, yaklaşık altı kat artış. Fajrin ve Min'in
2018&ndash;2025 arasını kapsayan ikinci derlemesi de benzer bir tablo veriyor ve
hibrit/ensemble yaklaşımların payının arttığını gösteriyor [2].</p>

<p>Mimari evrim şu hattı izliyor: CNN aileleri (ResNet, VGG, DenseNet, EfficientNet), sonra
dikkat mekanizmaları, sonra vizyon transformerları (ViT, Swin), sonra hibrit
CNN&ndash;self-attention, en son vizyon-dil temel modelleri. Amin derlemesinin
2024&ndash;Haziran 2025 ek taramasında dört yükselen eğilim belirlenmiş: transformer
mimarileri, çok görünümlü analiz, çok modlu adaptasyonlar ve yeni veri setleri [1].</p>

<h3>3.2 Lezyon tespiti</h3>

<p>Tespit tarafında YOLO ailesi baskın. Shia ve Ku, YOLOv8-m ile mikrokalsifikasyon
tespitinde 11.303 görüntü ve 5.712 kadınlık bir kümede mAP50 0,921, mAP50-95 0,709, F1 0,82
ve recall 0,796 bildiriyor [5]. Ho ve arkadaşları YOLOv9'u OPTIMAM üzerinde ön eğitip özel
veriyle ince ayar yaptığında mAP yüzde 73,3 (&plusmn;16,7) ve F1 yüzde 76,0 (&plusmn;13,4)
elde ediyor; aynı kurulumda YOLOv7'ye göre mAP'te 8,1 puanlık üstünlük var [6]. İki sonuç
arasındaki büyük fark veri setlerinin zorluğundan geliyor, mimariden değil.</p>

<p>Faster R-CNN tarafında sık alıntılanan sonuç, INbreast üzerinde görüntü başına 0,3 yanlış
pozitifte yüzde 92 duyarlılık. Bu sonuç Agarwal ve arkadaşlarına ait [7]. Literatürde bu
değer zaman zaman Famouri ve arkadaşlarının çalışmasına atfediliyor, ancak o çalışma sonucu
yalnızca alıntılıyor; kendi deneyleri CBIS-DDSM üzerinde anotasyon gürültüsünün etkisini
ölçüyor ve gürültülü kutularla FROC eğrisi altındaki alanın yüzde 9'a kadar düştüğünü
gösteriyor [8].</p>

<h3>3.3 Sınıflandırma ve temel modeller</h3>

<p>Çok görünümlü transformerlar öne çıkıyor. MV-Swin-T, kaydırmalı pencere tabanlı dinamik
dikkat bloğuyla CC ve MLO görünümlerini birlikte işliyor ve görünümler arası korelasyonu
uzamsal öznitelik haritası seviyesinde taşıyor; CBIS-DDSM ile VinDr-Mammo üzerinde
değerlendirilmiş [9]. Mammo-Clustering ise context clustering tabanlı üç seviyeli füzyonla
lezyon lokalizasyonu ve sınıflandırmayı birleştiriyor [11].</p>

<p>Vizyon-dil temel modeli Mammo-CLIP, mamogram-rapor çiftleriyle ön eğitiliyor ve
Mammo-FActOR modülüyle cümle seviyesinde öznitelik atfı sağlıyor; bu hem yorumlanabilirlik
hem de veri verimliliği getiriyor [10]. MammoDINO, DINOv2 çerçevesi üzerine kurulu
self-supervised bir alternatif. 1,4 milyon mamogram üzerinde ön eğitilmiş, meme dokusuna
kısıtlı augmentasyon örnekleyici ve DBT hacminde komşu kesitler arasında tutarlılık zorlayan
bir kayıp kullanıyor [12]. İki modelin ortak vaadi aynı: etiketli veri ihtiyacını düşürmek.</p>

<h3>3.4 Meme yoğunluğu değerlendirmesi</h3>

<p>Dört sınıflı BI-RADS yoğunluk sınıflandırması, alanın dış doğrulama kırılganlığını en net
gösteren görev. Khara ve arkadaşları 69.697 çalışma, 451.642 görüntü ve 23.057 kadınlık
çeşitli bir kümede, yalnızca FFDM ile eğitilen modelin FFDM üzerinde yüzde 80,5, hiç
görülmemiş sentetik 2D verisinde yüzde 79,4 doğruluk verdiğini bildiriyor; her iki modaliteyle
eğitildiğinde bu değer ikisinde de yüzde 82,3'e çıkıyor [13]. Aynı çalışma ırk alt
gruplarında yanlılık bulmuyor, ancak görüntüden ırkın yüzde 86,7 doğrulukla tahmin
edilebildiğini de gösteriyor.</p>

<p>Karşı örnek dış doğrulamadan geliyor. Ticari bir yoğunluk modeli iki sınıflı yoğun ve
yoğun değil ayrımında iyi çalışırken, dört sınıflı ayrımda ortalama yüzde 56,7 doğruluk ve
Cohen kappa 0,325 seviyesine düşüyor [15]. Farklı mimarilerin kafa kafaya
karşılaştırmasında da ResNet-50, EfficientNet-B0 ve DenseNet-121 birbirine yakın sonuç
veriyor; asıl fark mimariden değil modaliteden geliyor ve dijital mamogramlar sentetik
olanları geçiyor [14].</p>

<h3>3.5 Sentez matrisi</h3>

{T['kol1']}

<p class="tblnote">Dış doğrulama sütunu, modelin eğitildiği dağılımdan farklı bir kaynakta
sınanıp sınanmadığını gösteriyor. "Kısmi", public kümelerde değerlendirme yapıldığını ama
bunun bağımsız bir klinik kohort olmadığını ifade ediyor.</p>

<h2>4. Kol 2: risk tahmini</h2>

<h3>4.1 Referans model: Mirai</h3>

<p>Mirai alanın referans modeli [17]. Dört modülden oluşuyor: görüntü kodlayıcı (ResNet-18),
görüntü toplayıcı (transformer), risk faktörü öngörücü ve additif hazard katmanı. CC ve MLO
görünümlerini birlikte kullanıyor ve risk faktörü bilgisi eksik olduğunda da çalışacak
şekilde tasarlanmış. MGH üzerinde eğitilip üç test kümesinde sınanmış; C-indeksi MGH'de 0,76
(yüzde 95 GA 0,74&ndash;0,80), Karolinska'da 0,81 (0,79&ndash;0,82), CGMH'de 0,79
(0,79&ndash;0,83).</p>

<p>Sonraki çok kurumlu doğrulama beş ülkedeki yedi hastaneden 62.185 hastaya ait 128.793
mamogramı kapsıyor ve C-indeksini 0,75 ile 0,84 arasında buluyor; en düşük değer MGH ve
Novant'ta, en yüksek değer Barretos'ta [18]. Aynı çalışmada Mirai, Tyrer-Cuzick modelini
geçiyor. Mirai'nin referans kabul edilmesinin sebebi tek bir yüksek skor değil, bu
kurumlar arası tutarlılık.</p>

<h3>4.2 Açıklanabilirlik: AsymMirai</h3>

<p>AsymMirai, Mirai'nin ön uç CNN'ini koruyup geri kalanını yorumlanabilir bir yerel iki
taraflı benzemezlik modülüyle değiştiriyor [20]. EMBED üzerinde 81.824 hastaya ait 210.067
tarama mamogramında 1, 3 ve 5 yıl AUC değerleri sırasıyla 0,79, 0,68 ve 0,66; aynı kümede
Mirai 0,84, 0,72 ve 0,71 veriyor. Yani basitleştirilmiş ve açıklanabilir model, referans
modelin öngörü gücünün büyük kısmını koruyor. Buradaki asıl bulgu skor değil, sinyalin
kaynağı: iki meme arasındaki asimetri tek başına Mirai'nin tahmin gücüne yaklaşıyor.</p>

<h3>4.3 Longitudinal ve temporal modeller</h3>

<p>Mirai'nin bilinen sınırı tek zaman noktalı mamogram kullanması ve meme dokusundaki
zamansal değişimi yakalayamaması. Bu boşluğu dolduran yönelim longitudinal modeller.</p>

<ul>
<li>LRP-NET, dört ardışık muayenede iki taraflı doku değişimini yakalıyor ve eğitim öncesi
affine kayıt uyguluyor. Ancak 200 hastalık dengeli, küçük bir case-control kohortunda
eğitilmiş ve VGG16 gövdesi dondurulmuş [22]. Bu iki kısıt sonucun genellenebilirliğini
sınırlıyor.</li>
<li>LoMaR, Mirai'yi genişleterek keyfi sayıda longitudinal mamogramı transformer ile
işliyor. Büyük ölçekli bir kümede, tarama-tespitli kanserler hariç tutulduğunda beşinci
yılda Mirai'ye göre yaklaşık yüzde 10 ROCAUC üstünlük gösteriyor [23]. Kazanç kısa ufuklarda
küçük, uzun ufuklarda büyüyor.</li>
<li>TRINet, zaman-azalımlı dikkat, radyomik öznitelikler ve attention-based multiple instance
learning birleşimi kullanıyor. EMBED test kümesinde 1&ndash;5 yıl AUC değerleri 0,851, 0,811,
0,796, 0,793 ve 0,789; ayrıca CSAW ile de değerlendirilmiş [24].</li>
<li>VMRA-MaR, Vision-Mamba RNN ile iki taraflı asimetriyi birleştiriyor. CSAW-CC üzerinde 4
ve 5 yıllık ufuklarda ROCAUC 0,84 ve aynı noktalarda C-indeksi 0,82 bildiriyor; LoMaR'a göre
farkın p değeri 0,061, yani sınırda [25]. Yüksek yoğunluklu alt grupta üstünlük daha
belirgin.</li>
<li>LongiMam, güncel muayeneye dört öncekini ekleyerek benzer bir yönü izliyor ve kazancın
yoğun memeli, 55 yaş üstü ve yoğunluğu zamanla değişen kadınlarda yoğunlaştığını
gösteriyor [26].</li>
</ul>

<h3>4.4 Dış doğrulama sınırları</h3>

<p>Dış doğrulama bu kolun en kritik zayıflığı. Mirai, Meksikalı kadınlardan oluşan 3.110
hastalık bir kohortta (76 kanser) yalnızca C-indeksi 0,63 (yüzde 95 GA 0,6&ndash;0,7) elde
ediyor; aynı çalışmada performans cihaz markasına göre de değişiyor ve IMS sistemlerinde 0,55'e
iniyor [27]. Bu, geliştiricinin kendi kümelerindeki 0,76&ndash;0,81 aralığına göre belirgin
bir düşüş.</p>

<p>Farklı bir modelin dış doğrulaması da benzer bir tablo veriyor. İsveç kohortlarında
geliştirilen ProFound AI Risk 1.0, ABD'de 176 kanser ve 4.963 kontrolden oluşan bir kohortta
AUC 0,68 (0,64&ndash;0,72) veriyor; beyaz kadınlarda 0,67, siyah kadınlarda 0,70 ve aradaki
fark anlamlı değil [28]. Bu çalışma Mirai'yi değil iCAD'ın modelini doğruluyor ve literatürde
zaman zaman karıştırılıyor.</p>

<div class="uyari">
<p>Kol 2'nin gerçekçi tavanı, tek tek çalışmaların bildirdiği en yüksek değerlerden değil
derlemelerden okunmalı. Schopf ve arkadaşları 16 çalışmada görüntü tabanlı modellerde medyan
AUC 0,72, yoğunluk ve klinik risk araçlarında 0,61 buluyor [29]. Lowry ve arkadaşları 41
çalışmada medyan AUC'yi iki yıla kadar 0,71, üç ile dört yılda 0,72, beş yıl ve üzerinde 0,71
buluyor; indeks kanserler dahil edildiğinde 0,75, hariç tutulduğunda 0,68 [30]. Yani tekil
çalışmalarda görülen 0,85 mertebesindeki değerler alanın ortalaması değil, üst ucu.</p>
</div>

<h3>4.5 Sentez matrisi</h3>

{T['kol2']}

<h2>5. Veri setleri</h2>

<p>İki kolun veri ihtiyacı farklı. Kol 1 için lezyon seviyesinde anotasyon ve patoloji
doğrulaması yeterli; Kol 2 için uzun takipli tarama kohortu zorunlu, çünkü etiket
"ilerleyen yıllarda kanser çıktı mı" sorusunun cevabı.</p>

{T['veri']}

<p class="tblnote">INbreast'te lezyonların çoğu biyopsi ile doğrulanmamış; kötü huyluluk
BI-RADS kategorisinden türetiliyor. Bu, küçük kümede bildirilen yüksek doğrulukların nasıl
okunması gerektiğini doğrudan etkiliyor. RSNA kümesinde pozitif oranı yaklaşık yüzde 2 ve bu
sınıf dengesizliği doğruluk metriğini kullanılamaz hale getiriyor.</p>

<h2>6. Ana eğilimler, 2024&ndash;2026</h2>

<ol>
<li>Transformer benimsemesi. ViT, Swin ve DeiT ile tek görünümlü sınıflandırmadan çok
görünümlü CC+MLO füzyonuna geçiş; hibrit CNN-transformer mimarileri [1, 9].</li>
<li>Temel modeller. Mammo-CLIP gibi vizyon-dil ve MammoDINO gibi self-supervised modeller
etiketli veri ihtiyacını düşürüyor [10, 12].</li>
<li>Longitudinal modelleme. Risk tahmininde tek zaman noktasından temporal dizilere geçiş;
LRP-NET, LoMaR, TRINet, VMRA-MaR ve LongiMam bu hattı oluşturuyor. Prior mamogramların
katkısı özellikle dört ve beş yıllık ufuklarda belirginleşiyor [22&ndash;26].</li>
<li>Açıklanabilirlik. AsymMirai'nin iki taraflı benzemezliği ve Mammo-FActOR'ün cümle
seviyesi öznitelik atfı, kara kutu skorunun yerine klinik olarak okunabilir bir sinyal
koymayı hedefliyor [10, 20].</li>
<li>Çok modlu adaptasyon. Görüntü, radyoloji raporu metni ve klinik risk faktörlerinin
birlikte kullanımı. Buradaki dikkat çekici bulgu, görüntü tabanlı modellere klinik veri
eklemenin ayrım performansını genelde artırmaması; mamogramın kendisi zaten zengin prognostik
sinyal taşıyor [29, 30].</li>
</ol>

<h2>7. Performans iddialarının değerlendirmesi</h2>

<p>Küçük, dengesiz ve ROI merkezli veri setlerinde bildirilen yüzde 99&ndash;100 doğruluk
değerleri klinik olarak anlamlı değil. Bu değerler tipik olarak dört tuzaktan birini
yansıtıyor. Veri sızıntısı, yani aynı hastanın görüntülerinin hem eğitim hem test kümesine
karışması. Lezyon merkezli kırpma, yani tam mamogram taramasının asıl zorluğunun atlanması.
Sınıf dengesizliğinin doğruluk metriğini şişirmesi. Ve dış doğrulama eksikliği.</p>

<p>Gerçekçi kıyaslama Kol 1 için RSNA 2023 yarışması. İki merkezden (ABD ve Avustralya)
gelen 10.830 tek meme muayenesinden oluşan gizli test kümesinde 1.537 algoritma
değerlendirildi. En iyi algoritma yüzde 48,6 duyarlılık, yüzde 99,5 özgüllük, yüzde 64,6 PPV
ve yüzde 1,5 geri çağırma oranı verdi. Medyan algoritma ise yüzde 27,6 duyarlılık ve yüzde
98,7 özgüllükte kaldı. En iyi üç ve en iyi on algoritmanın ensemble'ı duyarlılığı yüzde 60,7
ve yüzde 67,8'e çıkardı; en iyi on, Avrupa ve Avustralya'daki ortalama tarama radyoloğuna
yakın performans gösterdi [3]. Yarışmanın sıralama metriği probabilistik F1 idi; kamuya açık
lider tablosu skorları bu raporda doğrulanamadığı için burada yalnızca hakemli yayında
raporlanan değerler veriliyor.</p>

<p>Kol 2 için gerçekçi aralık iki derlemeden okunuyor. Görüntü tabanlı modellerde medyan AUC
0,71&ndash;0,72, geleneksel yoğunluk ve klinik risk araçlarında yaklaşık 0,61 [29, 30]. Yani
alanın gerçekçi tabanı 0,65, tavanı 0,85 civarında.</p>

<div class="callout">
<p>Pratik kural. ROI kırpma sınıflandırması ile tam mamogram tarama sınıflandırması asla
karıştırılmamalı. Birincisinde lezyon zaten bulunmuş durumda, ikincisinde lezyonu bulmak
görevin kendisi. Bunlar farklı zorluk seviyeleri ve doğrudan karşılaştırılamaz.</p>
</div>

<h2>8. Öneriler ve karar eşikleri</h2>

<p>Alana giriş için üç aşamalı bir sıra öneriliyor. Sıralamanın gerekçesi, her aşamanın bir
sonrakinin değerlendirme zeminini kurması.</p>

<ol>
<li>Temel oturtma. İki sistematik derleme okunarak alan haritalanır; Kol 1 için Amin ve
arkadaşları [1], Kol 2 için Lowry ve arkadaşları [30]. Mirai deposu klonlanıp referans risk
modeli bir örnek mamogramda çalıştırılır ve zihinsel taban çizgisi olarak sabitlenir [17].</li>
<li>Kıyaslama kurulumu. VinDr-Mammo ve RSNA 2023 üzerinde bir EfficientNet ya da ConvNeXt
taban çizgisi kurulur. Metrik seti yalnızca doğruluk olmamalı: AUC, probabilistik F1,
kalibrasyon ve sabit özgüllükteki duyarlılık birlikte raporlanmalı. Papers with Code
üzerinden CBIS-DDSM ve VinDr-Mammo sıralamaları izlenir.</li>
<li>İleri yöntemler. Kol 1'de çok görünümlü transformer [9] ve temel model ince ayarı [10],
Kol 2'de longitudinal modeller [23, 24] denenir. EMBED erişimi risk tahmini deneyleri için
kritik [34].</li>
</ol>

<div class="callout">
<p>Karar eşikleri. Bir tespit ya da sınıflandırma modelini umut verici saymak için tek veri
setindeki yüzde 99 doğruluk yeterli değil; en az bir harici kohortta gerçekçi metrik
aranmalı. Bir risk modelini klinik potansiyeli var saymak için en az bir harici popülasyonda
AUC 0,75 ve üzeri ile birlikte kalibrasyon değerlendirmesi olmalı. Lowry derlemesinde 41
çalışmanın yalnızca altısı kalibrasyon raporlamış [30]; bu boşluk bir farklılaşma fırsatı.
Dış doğrulama yoksa veya veri seti yaklaşık 1.000 hastadan azsa, iddia ön sonuç olarak
işaretlenmeli.</p>
</div>

<h2>9. Sınırlılıklar</h2>

<ul>
<li>Bazı performans değerleri hakemli olmayan, küçük veya ROI merkezli kaynaklardan geliyor
ve klinik geçerlilik göstermiyor. Bu rapor bu tür değerleri metne almadı.</li>
<li>Amin ve arkadaşlarının derlemesinde görev bazlı toplam sayılar ağırlıklı olarak şekiller
ve ek tablolarda sunulmuş; ana metinde konsolide bir sayısal doğruluk aralığı bulunmuyor [1].
Bu nedenle Kol 1 için tek bir alan ortalaması verilemiyor.</li>
<li>TRINet, LoMaR, VMRA-MaR ve LongiMam gibi longitudinal modeller henüz geliştirici dışı
bağımsız dış doğrulamadan geçmedi. Bildirilen AUC değerleri geliştiricinin kendi test
bölmelerinden geliyor ve iyimser olabilir [23&ndash;26].</li>
<li>Lowry derlemesindeki 41 çalışmanın tümü retrospektif [30]. Alanın hiçbir modeli için
olgun prospektif klinik fayda kanıtı yok.</li>
<li>Veri çeşitliliği kritik bir kör nokta. Risk çalışmaları ağırlıklı olarak beyaz, Hispanik
olmayan popülasyonları temsil ediyor [30]. Belirli bir popülasyona hizmet edecek bir sistem
için hedef popülasyonda yerel doğrulama vazgeçilmez.</li>
<li>Rapor 2020&ndash;2026 penceresiyle ve derin öğrenme tabanlı yaklaşımlarla sınırlı.
Klasik radyomik ve el yapımı öznitelik çalışmaları kapsam dışında.</li>
</ul>

<h2>10. Sonuç</h2>

<p>İki kol ayrı tutulduğunda tablo netleşiyor. Kol 1'de mimari ilerlemesi hızlı ama gerçek
tarama koşullarındaki performans, yayınlanan tekil doğruluk değerlerinin çok altında.
Yarışma verisi bunu doğrudan gösteriyor: 1.537 algoritmanın medyanı yüzde 27,6 duyarlılıkta,
en iyisi yüzde 48,6'da [3]. Kol 2'de referans model kurumlar arası tutarlılığını koruyor ama
farklı popülasyonlarda gerileme belirgin ve alanın gerçekçi ayrım gücü medyan AUC
0,71&ndash;0,72 civarında [17, 27, 29, 30].</p>

<p>İki kolun ortak eksiği aynı ve teknik değil metodolojik. Dış doğrulama az, kalibrasyon
neredeyse hiç yok, prospektif kanıt yok. Bu, alandaki bir sonraki değerli katkının daha
büyük bir model olmayabileceğini gösteriyor. Harici bir kohortta kalibrasyonu da raporlanan
mütevazı bir model, tek merkezde parlayan bir modelden daha fazla şey söylüyor.</p>

<h2>Kaynaklar</h2>
<ol class="refs">

<li>Amin A, Acharya U D, Koteshwara P, Siddalingaswamy PC, Mathew S (2025). A systematic
literature review on mammography: deep learning techniques for breast cancer detection with
global and Asian perspectives. BMC Cancer, 25, 1627.
<a href="https://doi.org/10.1186/s12885-025-14876-5">10.1186/s12885-025-14876-5</a></li>

<li>Fajrin HR, Min SD (2025). From machine learning to ensemble approaches: a systematic
review of mammogram classification methods. Diagnostics, 15(22), 2829.
<a href="https://doi.org/10.3390/diagnostics15222829">10.3390/diagnostics15222829</a></li>

<li>Chen Y, vd. (2025). Performance of algorithms submitted in the 2023 RSNA screening
mammography breast cancer detection AI challenge. Radiology.
<a href="https://doi.org/10.1148/radiol.241447">10.1148/radiol.241447</a> &mdash; bölüm 7'deki
duyarlılık, özgüllük ve PPV değerleri bu çalışmadan.</li>

<li>Open-source dataset for the RSNA screening mammography cancer detection challenge (2025).
Radiology: Artificial Intelligence.
<a href="https://doi.org/10.1148/ryai.250375">10.1148/ryai.250375</a> &mdash; veri seti
<a href="https://www.kaggle.com/competitions/rsna-breast-cancer-detection">Kaggle</a> ve
<a href="https://github.com/RSNA/AI-Challenge-Data/wiki/RSNA-Screening-Mammography-Breast-Cancer-Detection">GitHub
wiki</a> üzerinden erişilebilir.</li>

<li>Shia W-C, Ku T-H (2024). Enhancing microcalcification detection in mammography with
YOLO-v8: performance and clinical implications. Diagnostics, 14(24), 2875.
<a href="https://doi.org/10.3390/diagnostics14242875">10.3390/diagnostics14242875</a></li>

<li>Ho P-S, Tsai H-Y, Liu I, Lee Y-Y, Chan S-W (2025). Improving YOLO-based breast mass
detection with transfer learning pretraining on the OPTIMAM Mammography Image Database.
Computers in Biology and Medicine.
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0010482525009321">sciencedirect.com/science/article/abs/pii/S0010482525009321</a></li>

<li>Agarwal R, Diaz O, Lladó X, Yap MH, Martí R (2019). Automatic mass detection in
mammograms using deep convolutional neural networks. Journal of Medical Imaging, 6(3),
031409. <a href="https://doi.org/10.1117/1.JMI.6.3.031409">10.1117/1.JMI.6.3.031409</a>
&mdash; INbreast üzerinde 0,3 FP/görüntüde yüzde 92 duyarlılık sonucunun asıl kaynağı.</li>

<li>Famouri S, Morra L, Mangia L, Lamberti F (2021). Breast mass detection with Faster R-CNN:
on the feasibility of learning from noisy annotations. IEEE Access.
<a href="https://arxiv.org/abs/2104.12218">arXiv:2104.12218</a> &mdash; INbreast sonucunu
yalnızca alıntılıyor; kendi deneyleri CBIS-DDSM üzerinde.</li>

<li>Sarker S, Sarker P, Bebis G, Tavakkoli A (2024). MV-Swin-T: mammogram classification with
multi-view Swin transformer. IEEE International Symposium on Biomedical Imaging (ISBI).
<a href="https://arxiv.org/abs/2402.16298">arXiv:2402.16298</a></li>

<li>Ghosh S, Poynton CB, Visweswaran S, Batmanghelich K (2024). Mammo-CLIP: a vision language
foundation model to enhance data efficiency and robustness in mammography. MICCAI 2024.
<a href="https://arxiv.org/abs/2405.12255">arXiv:2405.12255</a> &mdash;
<a href="https://papers.miccai.org/miccai-2024/488-Paper0926.html">MICCAI bildiri sayfası</a></li>

<li>Yang S, Zhang C, Liang X, vd. (2025). Mammo-Clustering: context clustering based
multi-view tri-level information fusion for lesion location and classification in mammography.
<a href="https://arxiv.org/abs/2507.18642">arXiv:2507.18642</a></li>

<li>MammoDINO: anatomically aware self-supervision for mammographic images (2025).
<a href="https://arxiv.org/abs/2510.11883">arXiv:2510.11883</a> &mdash;
<a href="https://research.gehealthcare.com/patient-care-pathways/unveiling-mammodino-an-anatomically-aware-vision-foundation-model-for-mammography/">GE
HealthCare tanıtım sayfası</a></li>

<li>Khara G, vd. (2024). Generalisable deep learning method for mammographic density
prediction across imaging techniques and self-reported race. Communications Medicine, 4.
<a href="https://doi.org/10.1038/s43856-024-00446-6">10.1038/s43856-024-00446-6</a></li>

<li>Anant K, Hernández López J, Cui J, Das Gupta S, Bennett DL, Gastounioti A (2026).
Head-to-head comparisons of breast density assessment models using deep learning on digital
and synthetic mammograms. Journal of Medical Imaging, 13(2), 024503.
<a href="https://doi.org/10.1117/1.JMI.13.2.024503">10.1117/1.JMI.13.2.024503</a></li>

<li>Abrantes J, Bento e Silva MJN, Meneses JP, Oliveira C, Calisto FMGF, Filice RW (2023).
External validation of a deep learning model for breast density classification. ECR 2023
posteri, C-16014.
<a href="https://epos.myesr.org/poster/esr/ecr2023/C-16014">epos.myesr.org/poster/esr/ecr2023/C-16014</a>
&mdash; hakemli yayın değil, poster.</li>

<li>RSNA (2025). AI challenge models can independently interpret mammograms.
<a href="https://www.rsna.org/news/2025/august/ai-challenge-models-performance">rsna.org</a>
&mdash; kaynak 3'ün kurum duyurusu; sayısal değerler için 3 kullanılmalı.</li>

<li>Yala A, Mikhael PG, Strand F, vd. (2021). Toward robust mammography-based models for
breast cancer risk. Science Translational Medicine, 13(578), eaba4373.
<a href="https://doi.org/10.1126/scitranslmed.aba4373">10.1126/scitranslmed.aba4373</a></li>

<li>Yala A, Mikhael PG, Strand F, vd. (2022). Multi-institutional validation of a
mammography-based breast cancer risk model. Journal of Clinical Oncology, 40(16),
1732&ndash;1740.
<a href="https://doi.org/10.1200/JCO.21.01337">10.1200/JCO.21.01337</a></li>

<li>Mirai model deposu.
<a href="https://github.com/yala/Mirai">github.com/yala/Mirai</a></li>

<li>Donnelly J, Moffett L, Barnett AJ, Trivedi H, Schwartz F, Lo J, Rudin C (2024).
AsymMirai: interpretable mammography-based deep learning model for 1&ndash;5-year breast
cancer risk prediction. Radiology, 310(3), e232780.
<a href="https://doi.org/10.1148/radiol.232780">10.1148/radiol.232780</a></li>

<li>RSNA (2024). Deep learning for predicting breast cancer.
<a href="https://www.rsna.org/news/2024/march/deep-learning-for-predicting-breast-cancer">rsna.org</a>
&mdash; kaynak 20'nin kurum duyurusu.</li>

<li>Dadsetan S, Arefan D, Berg WA, Zuley ML, Sumkin JH, Wu S (2022). Deep learning of
longitudinal mammogram examinations for breast cancer risk prediction. Pattern Recognition,
132, 108919.
<a href="https://doi.org/10.1016/j.patcog.2022.108919">10.1016/j.patcog.2022.108919</a></li>

<li>Karaman BK, Dodelzon K, Akar GB, Sabuncu MR (2024). Longitudinal mammogram risk
prediction. MICCAI 2024.
<a href="https://arxiv.org/abs/2404.19083">arXiv:2404.19083</a> &mdash;
<a href="https://papers.miccai.org/miccai-2024/474-Paper3369.html">MICCAI bildiri sayfası</a></li>

<li>Yeoh HH, Strand F, Phan R, Rahmat K, Tan M (2025). A new time-decay radiomics integrated
network (TRINet) for breast cancer risk prediction. Medical Image Analysis.
<a href="https://www.sciencedirect.com/science/article/pii/S1361841525003755">sciencedirect.com/science/article/pii/S1361841525003755</a>
&mdash; <a href="https://arxiv.org/abs/2412.03081">arXiv:2412.03081</a></li>

<li>Sun Z, Thrun S, Kampffmeyer M (2025). VMRA-MaR: an asymmetry-aware temporal framework for
longitudinal breast cancer risk prediction.
<a href="https://arxiv.org/abs/2506.17412">arXiv:2506.17412</a></li>

<li>Rakez M, Louis T, Guillaumin J, Chamming's F, Fillard P, Amadeo B, Rondeau V (2025). The
LongiMam model for improved breast cancer risk prediction using longitudinal mammograms.
<a href="https://arxiv.org/abs/2509.21383">arXiv:2509.21383</a></li>

<li>Avendano D, Marino MA, Bosques-Palomo BA, vd. (2024). Validation of the Mirai model for
predicting breast cancer risk in Mexican women. Insights into Imaging, 15, 244.
<a href="https://doi.org/10.1186/s13244-024-01808-3">10.1186/s13244-024-01808-3</a></li>

<li>Gastounioti A, Eriksson M, Cohen EA, vd. (2022). External validation of a
mammography-derived AI-based risk model in a U.S. breast cancer screening cohort of white and
black women. Cancers, 14(19), 4803.
<a href="https://doi.org/10.3390/cancers14194803">10.3390/cancers14194803</a> &mdash; Mirai'yi
değil iCAD ProFound AI Risk 1.0 modelini doğruluyor.</li>

<li>Schopf CM, vd. (2024). Artificial intelligence-driven mammography-based future breast
cancer risk prediction: a systematic review. Journal of the American College of Radiology,
21(2), 319&ndash;328.
<a href="https://doi.org/10.1016/j.jacr.2023.10.018">10.1016/j.jacr.2023.10.018</a></li>

<li>Lowry KP, Jeong HE, Kim KH, Hughes KS, Lee CI, Yala A, Kerlikowske K, Vachon CM (2026).
Current state of mammography-based artificial intelligence for future breast cancer risk
prediction: a systematic review. JNCI: Journal of the National Cancer Institute, 118(3),
392&ndash;403. <a href="https://doi.org/10.1093/jnci/djag002">10.1093/jnci/djag002</a></li>

<li>Lee RS, Gimenez F, Hoogi A, Miyake KK, Gorovoy M, Rubin DL (2017). A curated mammography
data set for use in computer-aided detection and diagnosis research. Scientific Data, 4,
170177. <a href="https://doi.org/10.1038/sdata.2017.177">10.1038/sdata.2017.177</a> &mdash;
<a href="https://complexity.cecs.ucf.edu/cbis-ddsm/">CBIS-DDSM dağıtım sayfası</a></li>

<li>Moreira IC, Amaral I, Domingues I, Cardoso A, Cardoso MJ, Cardoso JS (2012). INbreast:
toward a full-field digital mammographic database. Academic Radiology, 19(2),
236&ndash;248.
<a href="https://doi.org/10.1016/j.acra.2011.09.014">10.1016/j.acra.2011.09.014</a></li>

<li>Nguyen HT, vd. (2023). VinDr-Mammo: a large-scale benchmark dataset for computer-aided
diagnosis in full-field digital mammography. Scientific Data, 10, 277.
<a href="https://doi.org/10.1038/s41597-023-02100-7">10.1038/s41597-023-02100-7</a> &mdash;
<a href="https://physionet.org/content/vindr-mammo/1.0.0/">PhysioNet dağıtımı</a></li>

<li>Jeong JJ, vd. (2023). The EMory BrEast imaging Dataset (EMBED): a racially diverse,
granular dataset of 3.4 million screening and diagnostic mammographic images. Radiology:
Artificial Intelligence, 5(1), e220047.
<a href="https://doi.org/10.1148/ryai.220047">10.1148/ryai.220047</a> &mdash;
<a href="https://registry.opendata.aws/emory-breast-imaging-dataset-embed/">AWS Open Data
kaydı</a></li>

<li>Halling-Brown MD, Warren LM, Ward D, vd. (2021). OPTIMAM mammography image database: a
large-scale resource of mammography images and clinical data. Radiology: Artificial
Intelligence, 3(1), e200103.
<a href="https://doi.org/10.1148/ryai.2020200103">10.1148/ryai.2020200103</a> &mdash;
<a href="https://arxiv.org/abs/2004.04742">arXiv:2004.04742</a></li>

</ol>
"""
