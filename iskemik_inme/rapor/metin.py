# -*- coding: utf-8 -*-
"""Rapor metni. Tablolar ve figürler build_report.py'den geliyor.

Metin burada ayrı tutuluyor çünkü build_report.py veri yükleme ve biçimlendirme işi
yapıyor; ikisini aynı dosyada tutmak her ikisini de okunmaz hale getiriyordu.
"""


def govde(T, F):
    """T: tablo sözlüğü, F: figür sözlüğü (hazır HTML parçaları)."""
    return f"""
<h1>İskemik İnme Lezyonlarının Gerçek Zamanlı Segmentasyonu</h1>

<h2>0. Çalışma nasıl ilerledi</h2>

<p>Bu bölüm çalışmanın kısa anlatımı. Her adımda ne yaptığımı, neden öyle yaptığımı, ne
çıktığını ve ne öğrendiğimi sırayla aktarıyor. Kısa sonuç şu. Kalite tarafında denenen hiçbir
şey en basit modeli geçemedi. Hız tarafında ise kazanç modelden değil ön işlemeden geldi ve
vaka başına 138 milisaniyelik bir hat elde edildi. Sayıların nasıl ölçüldüğü ve istatistiksel
karşılaştırmalar sonraki bölümlerde.</p>

<h3>Önce veriye baktım</h3>

<p>İlk iş 250 vakanın dağılımlarını çıkarmak oldu. Bunu eğitime başlamadan önce yaptım, çünkü
veri setinin kendisi hangi kararların alınabileceğini belirliyor. Dört şey çıktı. Voxel boyutu
vakadan vakaya değişiyordu, ADC üç farklı ölçekte geliyordu, lezyon hacmi aşırı çarpıktı ve
vakaların dörtte üçü çok odaklıydı. İlk ikisi düzeltilmeseydi model ADC kanalını kullanamazdı.
Son ikisi ise sonuçların tek bir ortalama Dice ile verilemeyeceğini baştan gösterdi. Öğrendiğim
şey basit. Yarım günlük keşif, sonradan fark edilseydi bütün koşuları çöpe atacak iki hatayı
önledi.</p>

<h3>Sonra dokuz model eğittim</h3>

<p>Beş mimari ve dört ablasyon eğittim. Hepsinde veri bölünmesi, augmentasyon, optimizer ve
epoch bütçesi sabit kaldı, çünkü değişen tek şeyin mimari olması gerekiyordu. Sonuç beklediğim
gibi çıkmadı. Dokuz koşunun tamamı 0,64 ile 0,70 arasında toplandı ve hiçbiri en basit modeli,
yani 2B U-Net taban çizgisini geçemedi. Transformer encoder, hafif encoder ve FLAIR eklemek
işleri kötüleştirdi. Buradan çıkardığım şu. Bu görevde mimari seçimi düşündüğüm kadar
belirleyici değil, ve negatif sonuç da bir sonuç.</p>

<h3>Farkların gerçek olup olmadığını sınadım</h3>

<p>Sıralamanın en üstüne bakıp 3B U-Net en iyisi demek kolaydı. Bunu yapmadım, çünkü farklar
küçüktü ve ölçüm gürültüsünün ne kadar olduğunu bilmiyordum. Aynı mimarinin üç varyantı 0,024
genişliğinde bir banda yayılıyordu, yani modeller arası farkların çoğu bu bandın içinde
kalıyordu. Eşleştirilmiş testlerden sonra yalnızca üç modelin taban çizgisinden ayrıştığı
görüldü ve üçü de kötü yöndeydi. Öğrendiğim şey, sıralama tablosunun tek başına bir şey
söylemediği. Önce gürültüyü ölçmek gerekiyor.</p>

<h3>Sonra hattı hızlandırdım</h3>

<p>Gerçek zamanlılık için önce zamanın nereye gittiğini ölçtüm. Optimizasyon planım tamamen
modeli hedefliyordu, ama ölçüm sürenin dörtte üçünün ön işlemeye gittiğini gösterdi. Bu yüzden
planı değiştirip yeniden örneklemeyi ve beyin maskesini GPU'ya taşıdım. Vaka başına süre 385
milisaniyeden 138 milisaniyeye indi. ONNX ve TensorRT ise izole ölçümde hızlı görünmelerine
rağmen gerçek hatta hiçbir kazanç vermedi ve yığından çıkarıldı. Buradaki ders en pahalı olanı.
Ölçmeden optimize etmeye başlasaydım, bütün emeği yanlış yere harcayacaktım.</p>

<h3>En sonunda servise bağladım</h3>

<p>Modeli, ham dosyaları alıp lezyon maskesi, hacim, lezyon sayısı ve hemisfer bilgisi döndüren
bir servise dönüştürdüm. Servis ayrıca vaka başına okunabilir bir rapor üretiyor. Burada
ölçtüğüm şey ilk isteğin süresiydi. Isıtma yapılmadığında ilk istek 331 milisaniye sürüyordu,
yani kararlı halin üç katı. Açılışta sentetik bir hacmi tüm hattan geçirince bu süre 85
milisaniyeye indi. Öğrendiğim şey, çalışan bir modelle çalışan bir servisin aynı şey
olmadığı.</p>

<h2>1. Amaç, veri ve protokol</h2>

<p>Akut iskemik inmede tıkalı damarın beslediği doku dakikalar içinde kayboluyor. Tedavi kararı
büyük ölçüde geri döndürülemez biçimde ölmüş çekirdek bölgenin büyüklüğüne dayanıyor. Bu hacim
büyüdükçe damar açma girişiminin beklenen faydası azalıyor ve kanama riski artıyor. Dolayısıyla
hacim ölçümünün hem hızlı hem de tekrarlanabilir olması gerekiyor. Elle çizim ikisini de
sağlamıyor, çünkü vaka başına dakikalar sürüyor ve gözlemciden gözlemciye değişiyor. Difüzyon
ağırlıklı MR bu ölçümün ana dizisi, çünkü infarkt çekirdeği ilk dakikalardan itibaren belirgin
sinyal veriyor. Gerçek zamanlılık hedefi de buradan geliyor. Ölçüm, klinik kararın önüne
geçmeyecek kadar kısa sürmek zorunda.</p>

<p>Hedef, difüzyon ağırlıklı MR üzerinde akut ve subakut infarkt lezyonunun gerçek zamanlı
segmentasyonu. Çalışma iki soru soruyor. Hangi mimari daha iyi segmentliyor, ve hangi hat
klinik akışta kabul edilebilir bir sürede çalışıyor. Veri kaynağı ISLES 2022 (Ischemic Stroke
Lesion Segmentation Challenge 2022) eğitim kümesi. Küme 250 vakadan oluşuyor ve her vakada DWI
(b=1000), ADC ve FLAIR serileri ile uzman çizimli ikili lezyon maskesi bulunuyor. Challenge'ın
150 vakalık test kümesi yayınlanmadı. Bu nedenle değerlendirme, 250 vaka içinde vaka düzeyinde
ayrılan bir test kümesiyle yapıldı.</p>

<p>Veri setinin taşıdığı bilgi çalışmanın kapsamını da belirliyor. Küme yalnızca MR
içeriyor ve NCCT serisi barındırmıyor, dolayısıyla çalışma difüzyon MR üzerinde kurgulandı.
NCCT taşıyan ikinci bir veri seti eklemek değerlendirildi ama kapsam dışında bırakıldı. Tek bir
modalite üzerinde derinlemesine karşılaştırma yapmak tercih edildi. Perfüzyon serisi de
bulunmadığından core ve penumbra ayrımı üretilemiyor ve çıktı tek bir infarkt maskesi. Bu ayrım
klinikte reperfüzyon kararının dayandığı büyüklük. Sınırlılık olarak kayda geçiyor ve raporun
sonunda tekrar ele alınıyor.</p>

<h3>1.1 Veri keşfi ve alınan kararlar</h3>

<p>250 vakanın tamamında dört serinin dördü de mevcut. Ölçümler dört karar doğurdu.</p>

{F['veri_kesfi']}

<p>Voxel boyutu tek değil. DWI ve ADC serilerinde on yedi farklı matris boyutu ve yedi farklı
voxel aralığı var. 193 vaka 2×2×2 mm izotropik, buna karşılık 52 vaka 4,8 mm dilim
kalınlığında. Ortak bir grid olmadan ne hacim karşılaştırılabilir ne de komşu dilim bağlamı
anlamlı olur. Beş dilim bir vakada 10 mm'ye, diğerinde 24 mm'ye karşılık gelir. Bütün hacimler
2×2×2 mm'ye yeniden örneklendi. Çoğunluk zaten orada olduğu için bu, veriye en az müdahale eden
seçim.</p>

<p>ADC birimi tutarsız. Beyin içi ADC üst değerlerinin dağılımı tek bir birimle
açıklanamıyordu. Vakaların bir kısmı mm²/s, bir kısmı ×10⁻³, bir kısmı ×10⁻⁶ ölçeğinde geliyor.
Düzeltme öncesi bu değerin varyasyon katsayısı 1,52 idi. Ölçek grubu vaka bazında belirlenip
ortak birime taşındıktan sonra 0,09'a düştü. Düzeltme yapılmasaydı model ikinci kanalda aynı
anda üç farklı ölçek görecekti. ADC bilgisi de pratikte kullanılamaz hale gelecekti.</p>

{F['adc']}

<p>Lezyon dağılımı aşırı çarpık. Medyan lezyon hacmi 6,66 mL, çeyrekler arası aralık
1,58–21,17 mL, en büyük vaka 482 mL. Vakaların 46'sında lezyon 1 mL'nin altında. Ayrıca
vakaların yüzde 79,6'sı çok odaklı ve medyan bağlı bileşen sayısı beş. Bu iki ölçüm, sonuçların
tek bir ortalama Dice ile verilemeyeceğini baştan belirledi. Bütün karşılaştırmalar hacim
tertiline göre kırılarak ve lezyon bazlı F1 ile birlikte raporlanıyor.</p>

<p>Merkez sayısı iki ve dağılım dengesiz (198 ve 52 vaka). Bölme merkez ve hacim tertiline göre
stratifiye edildi. Buna karşılık merkezler arası performans farkı üzerine kurulacak her cümle
küçük merkezden gelen az sayıda vakaya dayanacak. Bu nedenle söz konusu kırılım raporda öne
çıkarılmıyor.</p>

<h3>1.2 Veri hazırlığı ve bölme</h3>

<p>Ön işleme sırası şöyle. İki seri 2 mm izotropik grid'e yeniden örnekleniyor, DWI üzerinden
beyin maskesi çıkarılıyor ve hacim maskenin sınırlayıcı kutusuna kırpılıyor. Ardından DWI maske
içinde z-score ile normalize ediliyor, ADC ise ortak birime taşınıp sabit aralığa ölçekleniyor.
Kırpma sonrası düzlem boyutları en fazla 87×101 olduğu için girdi 96×128'e dolduruldu. Kare bir
boyut seçmek yüzde otuz üç fazla piksel demek olurdu.</p>

{F['onisleme']}

<p>Bölme vaka düzeyinde yapıldı. Eğitim 161, validation 39, test 50 vaka. Aynı vakanın
dilimlerinin iki kümeye dağılması, dilim tabanlı eğitimde en kolay yapılan sızıntı hatası.
Lezyon içermeyen üç vaka yalnızca eğitim kümesinde tutuldu. Bu vakalar modele yanlış pozitif
baskılamayı öğretiyor, ama validation ve test'te Dice'ı tanımsız yapıp ortalamayı
kirletirlerdi.</p>

<h2>2. Yöntem</h2>

<p>Dokuz koşu yapıldı. Beşi mimari karşılaştırması, dördü ablasyon. Karşılaştırmanın mimariyi
ölçmesi için veri bölünmesi, ön işleme, augmentasyon zinciri, optimizer, learning rate
zamanlaması ve epoch bütçesi dokuz koşuda da sabit tutuldu.</p>

<p>Mimariler. Taban çizgisi 2B U-Net ve ResNet34 encoder. Buna ek olarak aynı mimarinin 2.5D
varyantı (beş komşu dilim, on kanal), transformer tabanlı SegFormer-B0, hafif encoder'lı
MobileNetV3 ve tam hacim işleyen bir 3B U-Net denendi. 3B model başlangıçta yalnızca doğruluk
tavanı olarak, gerçek zamanlı olmayan bir referans olarak konumlandırılmıştı. Ölçümler bunun
yanlış olduğunu gösterdi ve konu dördüncü bölümde ele alınıyor.</p>

<p>Ablasyonlar. A1, FLAIR'i üçüncü modalite olarak ekliyor. A2, bölge kaybını Dice yerine
focal-Tversky ile değiştiriyor ve küçük lezyonlarda recall'u hedefliyor. A3, 2.5D bağlam
derinliğini k=1 ve k=3 ile tarayarak taban çizgisiyle 2.5D arasındaki farkın gerçek olup
olmadığını sınıyor.</p>

<p>Metrikler vaka düzeyinde hesaplanıyor. Dilim başına Dice ortalaması yanıltıcı olurdu, çünkü
lezyonu iki dilimde olan vaka ile kırk dilimde olan vaka eşit ağırlık alırdı. Her vakanın bütün
dilimleri geçirilip hacim yeniden kuruluyor. Dice, IoU, precision, recall, HD95, mutlak hacim
farkı ve lezyon bazlı F1 o hacim üzerinde ölçülüyor.</p>

<h3>2.1 Eğitim sırasında düzeltilen iki hata</h3>

<p>Focal-Tversky kaybı ilk uygulamasında modeli çökertti. Test Dice 0,044, precision 0,024
çıktı ve model beynin büyük bölümünü lezyon işaretliyordu. Kaybın dört senaryoda ölçülmesi
sebebi gösterdi. Saf bölge kaybı bir doygunluk tuzağı yaratıyor. Model her şeyi lezyon
dediğinde kayıp değeri yüksek görünüyor ama gradyan pratikte sıfır. Sigmoid doymuş ve yanlış
negatif terimi sıfırlanmış durumda. Çıkış gradyanı, yarım doğru bir tahmininkinin beş milyonda
biri. Taban çizgisindeki BCE terimi tam olarak bunu engelliyormuş. İkinci hata gamma
parametresinin literatürdeki değerin tersi alınmasıydı. Kayıp BCE ile birleştirilip gamma
düzeltildikten sonra koşu normal davrandı.</p>

<p>İkinci hata ölçüm tarafındaydı ve raporun sayılarını sessizce bozabilirdi. Yarım hassasiyet
ölçümü için model nesnesi yerinde dönüştürülüp sonra geri çevriliyordu. Bu işlem geri
dönüşümlü olmadığı için tam hassasiyet ölçümleri aslında yarım hassasiyet ağırlıklarıyla
yapılıyordu. Hata, üretilen maskelerin referans değerlerle karşılaştırıldığı kalite kilidi
tarafından yakalandı.</p>

<p>Bu iki olay raporun genel yaklaşımını da özetliyor. Her aşamada üretilen sonucun bağımsız
bir referansla karşılaştırıldığı bir kontrol var ve bu kontroller birden fazla kez gerçek hata
yakaladı.</p>

<h2>3. Model sonuçları</h2>

{T['kars']}
<p class="tblnote">Test kümesi 50 vaka. Val Dice checkpoint seçimi için kullanıldı, model
seçimi için değil. SEM, vakalar arası standart hatayı gösteriyor.</p>

{F['dice_boyut']}

<p>Dokuz koşunun tamamı 0,64 ile 0,70 arasında test Dice veriyor. Nominal sıralamada 3B U-Net
önde, taban çizgisi hemen arkasında. Ancak bu sıralamayı yorumlamadan önce farkların ölçüm
gürültüsünün üzerinde olup olmadığına bakmak gerekiyor.</p>

<h3>3.1 Farklar ayırt edilebiliyor mu</h3>

<p>Bir mimarinin kendi varyantları arasındaki fark, mimariler arasında ölçmeye çalıştığımız
farkla aynı mertebede olursa sıralama anlamını yitirir. A3 ablasyonu bunu dolaylı olarak
ölçmeyi sağladı. 2.5D bağlam derinliği k=1, k=2 ve k=3 için aynı mimarinin üç varyantı
eğitildi. Test Dice değerleri 0,6652 ile 0,6896 arasında, yani 0,024 genişliğinde bir banda
yayıldı. Bütçe nedeniyle hiçbir koşu aynı ayarla tekrarlanmadı. Dolayısıyla bu bant, elimizdeki
en iyi koşu varyansı tahmini.</p>

<p>Aynı ablasyon ikinci bir sonuç daha verdi. Bağlam derinliği arttıkça tutarlı bir eğilim yok.
k=1 taban çizgisine çok yakın, k=2 en düşük, k=3 arada. Bağlam gerçekten katkı sağlıyor ya da
zarar veriyor olsaydı monoton bir yön beklenirdi. Dolayısıyla 2.5D girdinin bu görevde
ölçülebilir bir katkısı yok, ama zarar verdiği de gösterilemedi.</p>

{T['boot']}
<p class="tblnote">Güven aralıkları, vaka düzeyinde 10.000 yeniden örnekleme ile hesaplandı.
Fark sütunları eşleştirilmiş. Her yeniden örnekleme adımında iki model aynı vaka indekslerini
kullanıyor. Holm düzeltmesi sekiz karşılaştırma üzerinden yapıldı.</p>

{F['forest']}

<p>Mutlak Dice değerlerinin güven aralıkları dokuz modelde de geniş ölçüde üst üste biniyor ve
tek başlarına hiçbir modeli diğerinden ayıramıyorlar. Eşleştirilmiş farkların aralıkları ise
belirgin şekilde dar. Sebep, varyansın büyük kısmının vaka zorluğundan gelmesi. Aynı vakalar
üzerinde eşleştirme bu ortak varyansı düşürüyor.</p>

<p>Holm düzeltmesinden sonra taban çizgisinden ayrışan üç model var. Bunlar SegFormer-B0, FLAIR
ablasyonu ve MobileNetV3 varyantı. Üçü de daha kötü yönde. Hiçbir alternatif taban çizgisini
geçmedi. 3B U-Net'in nominal üstünlüğü ise anlamlılık eşiğine ulaşmıyor ve etkisi koşu varyansı
bandının içinde kalıyor.</p>

<p>İki prosedür iki modelde birbiriyle çelişiyor. MobileNetV3 için Holm anlamlı diyor, ama
farkın güven aralığı sıfırı içeriyor. 2.5D varyantı için tersi geçerli. Sebep, Wilcoxon
testinin sıra tabanlı olması ve tutarlı işaret örüntüsüne duyarlı olması. Bootstrap aralığı ise
ortalama üzerinde çalışıyor ve birkaç uç vakadan etkileniyor. Bu iki model sınırda sayılıyor ve
tek bir prosedüre dayanarak anlamlı ilan edilmiyor.</p>

<h3>3.2 Lezyon boyutuna göre kırılım</h3>

{T['tertil']}
<p class="tblnote">Hacim tertilleri 250 vaka üzerinden tanımlandı. Test kümesindeki dağılım 16,
17 ve 17 vaka. Lezyon F1, bağlı bileşen eşleştirmesiyle hesaplandı (IoU eşiği 0,1).</p>

<p>Ortalama Dice'ın gizlediği en büyük örüntü burada. Taban çizgisinde küçük lezyonlarda Dice
0,564, büyük lezyonlarda 0,823. Aradaki 0,26'lık fark, modeller arasındaki en büyük farkın on
katından fazla. Dokuz koşunun hiçbiri küçük tertilde 0,564'ü geçemedi ve bu değeri veren model
taban çizgisi. Küçük lezyonları doğrudan hedefleyen tek ablasyon focal-Tversky, o tertilde
durumu kötüleştirdi. Kaybın ürettiği ek recall küçük lezyonlara değil, zaten bulunan lezyonların
çevresine gitti ve mutlak hacim farkı büyüdü.</p>

<p>Lezyon bazlı F1 ters yönde çalışıyor. Büyük lezyonlarda Dice en yüksek olmasına rağmen
lezyon F1 en düşük. Sebep, büyük infarktların tek kütle olmaması. Ana lezyonun yanında uydu
bileşenler bulunuyor ve bunların hepsini yakalamak bileşen düzeyinde sayımda daha zor.</p>

<h3>3.3 Eğitim davranışı</h3>

{F['curves_taban']}

{F['curves_3b']}

<p>İki koşu farklı davranıyor. Taban çizgisinde train ve validation kayıpları hiç ayrışmıyor.
Son epoch'ta validation kaybı train'in bile hafif altında, çünkü train augmente edilmiş dilimler
üzerinde ölçülüyor. Yani kapasite ya da epoch bütçesi sınırlayıcı değil. 3B modelde ise makas
açılıyor ve bu, çalışmada aşırı öğrenme gösteren tek koşu. 161 hacim, dilim tabanlı eğitimden
çok daha az çeşitlilik sunuyor.</p>

<p>Learning rate zamanlaması iki koşuda tersine çalıştı. Taban çizgisinde validation kaybı
erken platoya girdi ve learning rate sekiz kez düşürüldü. Val Dice zirvesine ulaştığında öğrenme
oranı başlangıç değerinin on altıda birine inmişti. 3B modelde ise validation kaybı plato
koşulunu hiç sağlamadı ve learning rate hiç düşmedi. Koşu yakınsadığı için değil, val Dice on
beş epoch iyileşmediği için durdu. Protokol dokuz koşuda sabit tutulduğu için değiştirilmedi,
ancak bu iki modelin epoch bütçesine sıkışmış olabileceği kayda geçiyor.</p>

{F['cases']}

<h2 class="brk">4. Gecikme</h2>

<p>Bu bölümün sorusu, hattın klinik akışta kabul edilebilir bir sürede çalışıp çalışmadığı.
Literatürdeki referans noktaları iki uç veriyor. Aynı veri setinin yayınlanmış en iyi modeli
olan DeepISLES vaka başına yaklaşık iki dakika sürüyor. Buna karşılık ONNX ve yarım hassasiyetle
optimize edilmiş bir nnU-Net dağıtım aracı olan StrokeSeg, hacim başına 0,9 saniye bildiriyor.
Hedef, vaka başına uçtan uca iki saniyenin altı, tercihen bir saniyenin altı olarak
belirlendi.</p>

<p>Ölçüm protokolü şöyle. Ölçülen şey ön işleme, ileri geçiş ve son işlemenin toplamı. Diskten
okuma bütçenin dışında tutuluyor, çünkü servis senaryosunda hacim görüntüleme sisteminden
geliyor. Yine de ayrı bir sütunda raporlanıyor. Isınma koşuları atılıyor ve her vaka tek tek
senkronlanarak ölçülüyor. Bütün ölçümler yerel bir RTX 2060 üzerinde, eğitimin yapıldığı
ortamdan bağımsız olarak alındı.</p>

<h3>4.1 Darboğaz modelde değil ön işlemede</h3>

{T['merdiven']}
<p class="tblnote">3B U-Net, 25 test vakası. Her satırda model taze yüklenip ilgili hassasiyete
çevriliyor.</p>

{F['merdiven']}

<p>İlk ölçüm, optimizasyon çabasının yanlış yere yöneldiğini gösterdi. Tam hassasiyette ve CPU
ön işlemesiyle vaka başına sürenin dörtte üçü ön işlemeye gidiyor. Ağ payı yüzde yirmi
civarında. Planlanan optimizasyon merdiveni ise tamamen ağı hedefliyordu. Ağ süresi tamamen
sıfırlansaydı bile toplam ancak dörtte bir azalırdı.</p>

<p>Ön işlemenin kırılımı üç adıma işaret ediyor. Bunlar iki yeniden örnekleme ve beyin maskesi.
Üçü de CPU'da scipy ve nibabel ile çalışıyordu. Yeniden örnekleme, nibabel'in uygulamasıyla aynı
çıktı grid'ini kullanan bir grid_sample çağrısına dönüştürüldü. Hedef grid nibabel'in kendi
fonksiyonundan alındığı için değişen tek şey örneklemenin nerede yapıldığı. On iki vakada bağıl
sapma 10⁻⁵ mertebesinde kaldı.</p>

<p>Bu dönüşüm sırasında bir tuzak çıktı. scipy, girdi sınırının dışındaki noktalara sabit değer
verirken grid_sample kenarda iç değerle sıfırı harmanlıyor. Eğik affine taşıyan vakalarda bu,
hacim sınırında birkaç yüz voxel'de büyük farklar üretiyordu. Geçerlilik maskesi eklenerek iki
davranış eşitlendi.</p>

<p>Beyin maskesinde eşikleme ve morfolojik kapama GPU'ya taşındı. Delik doldurma ve bağlı
bileşen etiketleme ise GPU'da ucuz karşılığı olmadığı için tam çözünürlükte CPU'da bırakıldı. Bu
iki adımı alt örneklenmiş bir maskede yapmak daha hızlıydı, ancak kırpma kutusunu bir voxel
kaydırıp çıktıyı değiştiriyordu. Hız uğruna sonucu değiştirmek kazanç sayılmadığı için geri
alındı.</p>

<h3>4.2 Ağ optimizasyonu ve arka uç seçimi</h3>

{T['srv']}
<p class="tblnote">Servis hattında, 25 test vakası. Kalite kilidi, üretilen maskenin vaka başına
referans Dice değerinden sapmasını kontrol ediyor. Tolerans 0,002.</p>

{F['arka_uc']}

<p>Yarım hassasiyet, ağ süresini yaklaşık yarıya indiriyor ve Dice'a etkisi ölçülemez düzeyde
kalıyor. Referanstan sapma tam hassasiyetteki sapmadan büyük değil. ONNX Runtime ise tam
hassasiyette hiçbir kazanç sağlamıyor. Bu, benzer bir çalışmada ONNX'in çerçeve yükünü
kaldırarak iki ila üç kat hızlanma vermesiyle çelişiyor gibi görünüyor. Aradaki fark, orada kare
başına Python yükünün amorti edilmemiş olması. Burada 3B model vaka başına tek ileri geçiş
yapıyor, dolayısıyla kaldırılacak bir çerçeve yükü yok.</p>

{T['izole']}
<p class="tblnote">İzole ölçüm, aynı girdi dizisi otuz kez arka arkaya beslenerek yapıldı.
Servis hattı sütunu, vaka başına taze girdiyle ölçülen süreler.</p>

<p>TensorRT sonucu özellikle dikkat çekici. İzole ölçümde 3B model için TensorRT, PyTorch'un
yarım hassasiyetli halinden yaklaşık iki kat hızlı görünüyordu. Gerçek hatta ikisi aynı süreyi
veriyor ve oturum yükü nedeniyle TensorRT toplamda birkaç milisaniye geride kalıyor. İzole
ölçümde aynı dizinin tekrar tekrar beslenmesi, çalışma zamanının giriş bağlamalarını ve cihaz
tamponlarını amorti etmesine yol açıyor. Vaka başına taze veriyle bu kazanç ortadan kalkıyor.
Sonuç olarak dağıtım için PyTorch ve yarım hassasiyet seçildi. Aynı hız, çok daha basit bir
yığın. ONNX dışa aktarımı, TensorRT sürüm eşleşmesi ve motor önbelleği gereksiz hale
geliyor.</p>

<div class="callout"><p>İzole ağ ölçümleri dağıtılabilir sayı değil. Bu çalışmada iki kez,
birbirinin tersi yönde yanıltıcı sonuç verdiler. Karar her seferinde uçtan uca ölçümle
verildi.</p></div>

<h2>5. Eşik, son işleme ve metrik doğrulaması</h2>

<p>Karar eşiği ve küçük bileşen eleme eşiği validation kümesinde seçilip test'e tek sefer
uygulandı. Test üzerinde tarama yapmak elli vakaya uydurmak olur ve raporlanan sayıları iyimser
gösterirdi.</p>

{F['esik']}

{T['esik']}
<p class="tblnote">Validation kümesinde seçilen ayarın test kümesindeki etkisi. Her model için
üst satır varsayılan ayar, alt satır seçilen ayar.</p>

<p>Eşik taraması iki modeli ters yönlere götürdü. 3B model aşırı segmentasyon eğiliminde olduğu
için daha yüksek bir eşik seçti. 2B taban çizgisi ise muhafazakâr olduğu için daha düşük bir
eşiğe indi. Ancak validation'daki kazanç iki modelde de binde üç mertebesinde kaldı. Test'e
taşındığında yalnızca 3B modelde küçük bir iyileşme üretti. Eşik ayarı bu görevde anlamlı bir
kaldıraç değil.</p>

<p>Küçük bileşen eleme her iki modelde de zarar verdi ve uygulanmadı. Bunun sebebi veri
keşfinde görülmüştü. Referans maskelerin kendisinde beş voxel'in altında 897 bağlı bileşen var,
dolayısıyla eleme yalnızca yanlış pozitifleri değil gerçek lezyonları da siliyor. Yine de bir
takas görünür oldu. Taban çizgisinde beş voxel'lik eleme Dice'ı binde beş düşürürken lezyon
F1'i altmış binde yükseltiyor. Seçim Dice üzerinden yapıldığı için eleme yok. Öncelik lezyon
tespiti olsaydı karar tersine dönerdi.</p>

<h3>5.1 Lezyon bazlı F1'in doğrulanması</h3>

{T['pan']}
<p class="tblnote">Rastgele seçilen on test vakası. panoptica, ISLES challenge'ının resmi
değerlendirme aracı. RQ metriği bileşen düzeyinde F1'e karşılık geliyor.</p>

<p>Eğitim döngüsünde hızlı bir kendi uygulamamız kullanıldı. Bu uygulamanın resmi araçla aynı
tanımı hesapladığı on vakada doğrulandı. Değerler ondalık basamağına kadar aynı ve korelasyon
1,000. Buna karşılık metriğin kendisi eşleştirme eşiğine güçlü biçimde bağlı. Aynı tahminler
üzerinde herhangi bir örtüşme yeterli sayıldığında lezyon F1 0,609 çıkıyor. IoU eşiği 0,5'e
çıkarıldığında aynı değer 0,399'a düşüyor. Bu nedenle lezyon F1 değerleri raporda daima
eşleştirme eşiğiyle birlikte veriliyor ve literatürdeki değerlerle doğrudan
karşılaştırılmıyor.</p>

<h2 class="brk">6. Servis</h2>

<p>Çıkarım hattı çalışır bir servise dönüştürüldü. Servis, ham NIfTI dosyalarını alıp infarkt
maskesi, lezyon hacmi, bağlı lezyon sayısı, en büyük lezyonun hacmi ve hemisfer bilgisini
döndürüyor. Ayrıca tek dosyalık bir vaka raporu üretiyor. Rapor dilim mozaiğini ve vakanın
hacminin veri seti dağılımındaki konumunu gösteriyor. Her yanıtta işlem süresi bileşenlere
ayrılmış olarak loglanıyor.</p>

<p>Motor uygulama açılışında bir kez yükleniyor. Bunun önemi ölçüldü. İlk gerçek istek ısıtma
olmadan 331 milisaniye sürüyordu, yani kararlı halin üç katından fazla. Yalnızca ağı ısıtmak
yeterli olmadı, çünkü yeniden örnekleme ve morfoloji çekirdekleri de ilk çağrılarında yavaş.
Açılışta sentetik bir hacim tüm hattan geçirilerek ısıtma yapıldığında ilk istek 85 milisaniyeye
indi.</p>

<p>Ayrıca ön işleme ile ağ arasındaki gereksiz bellek transferi kaldırıldı. Ön işleme GPU'da
bitip sonuç ana belleğe iniyor, ağ için tekrar GPU'ya çıkıyordu. 3B modelde girdi doğrudan
GPU'da kurulabildiği için bu gidiş dönüş ortadan kalktı.</p>

<h3>6.1 Klinik çıktı ve sınırlılıkları</h3>

<p>Hacim, 2 mm yeniden örneklenmiş grid üzerinden hesaplanıyor. Orijinal grid'e göre ölçülen
sistematik fark ortalamada ihmal edilebilir. Buna karşılık standart sapması yüzde 2,3 ve çok
küçük lezyonlarda yüzde on ikiye kadar çıkabiliyor. Klinik hacim raporlanırken bu taban
belirtiliyor.</p>

<p>Hemisfer bilgisi, yeniden örnekleme sırasında hacim kanonik yöne getirildiği için lezyon
voxel'lerinin sol-sağ ekseni üzerindeki dağılımından çıkarılıyor. Üçte ikiden fazlası bir
tarafta yoğunlaşmıyorsa bilateral olarak işaretleniyor.</p>

<h3>6.2 FLAIR ablasyonunun maliyeti</h3>

{F['flair']}

<p>FLAIR serisi hiçbir vakada DWI ile aynı grid'de değil. Yirmi iki farklı matris boyutu ve
yirmi bir farklı voxel aralığı var. Kullanabilmek için vaka başına hizalama gerekiyor ve bu
hizalama, ölçülen değerlere göre vaka başına 670 milisaniye sürüyor. Bu, tüm çıkarım hattının
beş katından fazla. Ablasyon ayrıca doğruluk tarafında da olumsuz sonuç verdi. Bu iki gerekçe
FLAIR'i gerçek zamanlı yapılandırmanın dışında bırakıyor. Yalnız bir ayrım korunmalı. Hizalama
NIfTI header'ları üzerinden yapıldı ve yoğunluk tabanlı kayıt denenmedi. Dolayısıyla sonuç,
FLAIR'in bilgi taşımadığını değil, ucuz hizalamayla zarar verdiğini gösteriyor.</p>

<h2>7. Sınırlılıklar</h2>

<ul>
<li>Test kümesi 50 vaka. Bu boyutta 0,02 mertebesindeki farklar güvenilir biçimde ayrılamıyor.
Raporun sıralama yerine ayırt edilebilirlik üzerine kurulmasının sebebi bu.</li>
<li>Hiçbir koşu tekrarlanmadı. Koşu varyansı, aynı mimarinin üç varyantından dolaylı olarak
tahmin edildi (yaklaşık 0,024). Bu tahmin üç ölçümden geliyor ve kendisi de gürültülü.</li>
<li>Tek veri kaynağı. ISLES 2022 iki merkezden geliyor ve dış doğrulama yapılmadı. Farklı cihaz
ve protokollere genelleme bu çalışmayla gösterilmiş değil.</li>
<li>Core ve penumbra ayrımı üretilmiyor. Veri setinde perfüzyon serisi bulunmadığı için
reperfüzyon kararının dayandığı mismatch oranı hesaplanamıyor.</li>
<li>Küçük lezyonlar çözülmedi. En küçük hacim tertilinde Dice 0,55 civarında kalıyor ve bunu
hedefleyen ablasyon durumu kötüleştirdi. Sebep araştırılmadı.</li>
<li>FLAIR ablasyonu yalnızca ucuz hizalamayla yapıldı. Yoğunluk tabanlı kayıt denenmedi.</li>
<li>Epoch bütçesi bazı koşularda sınırlayıcı olmuş olabilir. 3B model ve focal-Tversky
ablasyonu bütçe tavanına dayandı. Protokol dokuz koşuda sabit tutulduğu için uzatılmadı.</li>
<li>INT8 nicemleme ve pruning ölçülmedi. Yarım hassasiyetten sonra ağ, uçtan uca bütçenin üçte
birinden azını oluşturuyor, dolayısıyla beklenen kazanç birkaç milisaniye. Buna karşılık
kalibrasyon kümesinin hazırlanması ve doğrulanması ayrı bir iş kalemi. Literatürde 3B
segmentasyon modellerinde INT8'in Dice'ı büyük ölçüde koruduğu bildiriliyor. Yani atlanan adım
kalite riski değil, ölçülmemiş bir kazanç.</li>
<li>Gecikme ölçümleri, üzerinde başka uygulamalar çalışan bir masaüstü makinede alındı. GPU
tarafındaki ölçümler oturumlar arasında tekrarlanabilir çıktı. CPU'ya bağlı ön işleme süreleri
ise sistem yüküne duyarlı ve oturumlar arasında yüzde otuza varan oynama gösterdi.</li>
</ul>

<h2>8. Sonuç</h2>

<p>Kalite. Dokuz koşunun hiçbiri basit 2B U-Net taban çizgisini geçemedi. Denenen dört fikirden
üçü, yani transformer encoder, hafif encoder ve FLAIR eklenmesi, istatistiksel olarak anlamlı
biçimde daha kötü sonuç verdi. 2.5D bağlam ise ne fayda ne zarar gösterdi. 3B U-Net nominal
olarak en yüksek Dice'ı veriyor, ancak fark koşu varyansı bandının içinde kalıyor ve anlamlılık
eşiğine ulaşmıyor. Tek bir en iyi model ilan etmek eldeki ölçümlerle desteklenmiyor.</p>

<p>Seçim ölçütü. Kalite ayırt edici olmadığı için karar maliyet eksenine kayıyor. 3B U-Net 1,40
milyon parametreyle taban çizgisinin on yedide biri büyüklüğünde ve nominal Dice'ı biraz daha
yüksek. Dağıtım için bu model seçildi. Gerekçe hız değil, boyut ve kalite açısından kötü
olmaması.</p>

<p>Hız. Gerçek zamanlılık modelden değil ön işlemeden geldi. Yeniden örnekleme ve beyin
maskesinin GPU'ya taşınması vaka başına süreyi yaklaşık dörtte birine indirdi. Yarım hassasiyet
ağ süresini bir kat daha azalttı. Sonuç, vaka başına yaklaşık 138 milisaniyelik bir hat. Bu,
hedeflenen bir saniyelik üst sınırın yedide birinden az. Aynı veri setinin yayınlanmış en iyi
modelinin bildirdiği sürenin ise yaklaşık sekiz yüzde biri. ONNX ve TensorRT bu hatta kazanç
sağlamadı ve dağıtım yığınından çıkarıldı.</p>

<p>Klinik kullanım. Üretilen çıktı tek bir infarkt maskesi ve ondan türetilen hacim bilgisi.
Core ve penumbra ayrımı yapılamıyor, küçük lezyonlarda doğruluk belirgin biçimde düşük ve model
tek bir veri kaynağının dağılımına göre eğitildi. Bu haliyle sistem, uzman değerlendirmesini
destekleyen bir ön işaretleme aracı olarak değerlendirilebilir. Tanı ya da tedavi kararı için
kullanılamaz.</p>

<p>Sonraki aşama. Üç yön öne çıkıyor. Küçük lezyonlardaki düşük doğruluk, çalışmanın
çözülmemiş ana problemi ve ayrı bir çalışma konusu. Koşuların tekrarlanması, sıralama hakkında
söylenebilecekleri güçlendirir. Farklı bir merkezden gelen bağımsız bir veri kümesiyle dış
doğrulama ise klinik değerlendirmenin ön koşulu.</p>

<h2>Kaynaklar</h2>
<ol class="refs">
<li>Hernández Petzsche MR, de la Rosa E, Hanning U, vd. (2022). ISLES 2022: A multi-center
magnetic resonance imaging stroke lesion segmentation dataset. Scientific Data, 9, 762.
<a href="https://doi.org/10.1038/s41597-022-01875-5">10.1038/s41597-022-01875-5</a>
&mdash; <a href="https://zenodo.org/records/7960856">zenodo.org/records/7960856</a></li>
<li>de la Rosa E, vd. (2025). DeepISLES: a clinically validated ischemic stroke segmentation
model from the ISLES'22 challenge. Nature Communications, 16, 7357.
<a href="https://doi.org/10.1038/s41467-025-62373-x">10.1038/s41467-025-62373-x</a></li>
<li>Abraham N, Khan NM (2019). A novel focal Tversky loss function with improved attention
U-Net for lesion segmentation. IEEE International Symposium on Biomedical Imaging (ISBI).
<a href="https://doi.org/10.1109/ISBI.2019.8759329">10.1109/ISBI.2019.8759329</a></li>
<li>Kofler F, Möller H, Buchner JA, de la Rosa E, vd. (2023). Panoptica: instance-wise
evaluation of 3D semantic and instance segmentation maps.
<a href="https://arxiv.org/abs/2312.02608">arXiv:2312.02608</a></li>
<li>Kerverdo Y, Leray F, Mahé Y, Leplaideur S, Galassi F (2025). Stroke Lesion Segmentation in
Clinical Workflows: A Modular, Lightweight, and Deployment-Ready Tool.
<a href="https://arxiv.org/abs/2510.24378">arXiv:2510.24378</a> &mdash; bölüm 4'te alıntılanan
vaka başına 0,9 saniyelik süre bu çalışmadan.</li>
<li>Qu C, Zhao R, Yu Y, vd. (2025). Post-Training Quantization for 3D Medical Image
Segmentation: A Practical Study on Real Inference Engines.
<a href="https://arxiv.org/abs/2501.17343">arXiv:2501.17343</a></li>
<li>Ischemic Stroke Lesion Segmentation Challenge.
<a href="https://www.isles-challenge.org/">isles-challenge.org</a></li>
</ol>
"""
