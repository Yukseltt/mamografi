# -*- coding: utf-8 -*-
"""Mamografi literatur degerlendirmesi -> tek dosyalik HTML (Chrome ile PDF'e basilir).

Rapor figur icermiyor; tablolar burada kuruluyor, metin metin.py'de. Ayrimin sebebi
diger raporlarla ayni: veri/bicimlendirme isi ile metin ayni dosyada okunmaz hale geliyor.
"""
from pathlib import Path

BURASI = Path(__file__).resolve().parent
CIKTI = BURASI / 'literatur_raporu.html'


def tablo(basliklar, satirlar, genislik=None):
    kol = ''.join(f'<col style="width:{w}">' for w in genislik) if genislik else ''
    bas = ''.join(f'<th>{c}</th>' for c in basliklar)
    gvd = ''.join('<tr>' + ''.join(f'<td>{h}</td>' for h in s) + '</tr>' for s in satirlar)
    return (f'<table>{kol}<thead><tr>{bas}</tr></thead>'
            f'<tbody>{gvd}</tbody></table>')


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
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 8.5pt;
        page-break-inside: auto; table-layout: fixed; }
tr { page-break-inside: avoid; page-break-after: auto; }
thead { display: table-header-group; }
th { background: #eef2f8; text-align: left; font-weight: 600; }
th, td { border: 1px solid #c3ccd9; padding: 3.5pt 6pt; vertical-align: top;
         word-wrap: break-word; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.callout { border-left: 3px solid #2c5aa0; background: #f6f9fd; padding: 7pt 11pt; margin: 9pt 0;
           page-break-inside: avoid; }
.callout p:last-child { margin-bottom: 0; }
.uyari { border-left: 3px solid #c0392b; background: #fdf6f5; padding: 7pt 11pt; margin: 9pt 0;
         page-break-inside: avoid; }
.uyari p:last-child { margin-bottom: 0; }
.tblnote { font-size: 8.5pt; color: #555; margin: -5pt 0 10pt; line-height: 1.35; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f4f6f9;
       padding: 1px 4px; border-radius: 2px; }
.mono-block { font-family: Consolas, monospace; font-size: 8.8pt; background: #f4f6f9;
              border: 1px solid #dde3ec; padding: 7pt 10pt; margin: 8pt 0; white-space: pre-wrap;
              page-break-inside: avoid; line-height: 1.4; }
ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
li { margin-bottom: 3pt; }
a { color: #2c5aa0; text-decoration: none; }
ol.refs { font-size: 9.5pt; }
ol.refs li { margin-bottom: 6pt; }
.brk { page-break-before: always; }
'''

# ------------------------------------------------------------------ tablolar

# --- tablo 1: Kol 1 sentez matrisi
t_kol1 = tablo(
    ['Çalışma', 'Model / mimari', 'Veri', 'Metrik ve performans', 'Dış doğrulama'],
    [
        ['Amin ve ark. 2025 [1]', 'PRISMA sistematik derleme',
         '1.051 kayıttan 287 makale', 'Görev ve eğilim analizi', 'Yok (derleme)'],
        ['Fajrin ve Min 2025 [2]', 'PRISMA sistematik derleme',
         '2018&ndash;2025 arası 50 çalışma', 'Yöntem ve kısıt analizi', 'Yok (derleme)'],
        ['Chen ve ark. 2025 [3]', 'RSNA 2023 yarışması, 1.537 algoritma',
         '10.830 tek meme muayenesi, 2 merkez (ABD, Avustralya)',
         'En iyi algoritma: duyarlılık %48,6 &middot; özgüllük %99,5 &middot; PPV %64,6. '
         'Medyan: %27,6 &middot; %98,7 &middot; %36,9',
         'Evet (gizli test kümesi)'],
        ['Shia ve Ku 2024 [5]', 'YOLOv8-m', '11.303 görüntü, 5.712 kadın',
         'mAP50 0,921 &middot; mAP50-95 0,709 &middot; F1 0,82 &middot; recall 0,796', 'Hayır'],
        ['Ho ve ark. 2025 [6]', 'YOLOv9, OPTIMAM ön eğitimi', 'OMI-DB + özel veri',
         'mAP %73,3 (&plusmn;16,7) &middot; F1 %76,0 (&plusmn;13,4)', 'Kısmi (transfer)'],
        ['Agarwal ve ark. 2019 [7]', 'Faster R-CNN', 'INbreast (harici eğitim verisiyle)',
         'Görüntü başına 0,3 yanlış pozitifte %92 duyarlılık', 'Evet (INbreast harici)'],
        ['Sarker ve ark. 2024 [9]', 'MV-Swin-T, çok görünümlü Swin', 'CBIS-DDSM, VinDr-Mammo',
         'Karşılaştırmalı sınıflandırma', 'Kısmi (2 public küme)'],
        ['Ghosh ve ark. 2024 [10]', 'Mammo-CLIP, vizyon-dil temel modeli + Mammo-FActOR',
         'Özel mamogram-rapor çiftleri, VinDr-Mammo, RSNA',
         'Sınıflandırma ve lokalizasyon; veri verimliliği', 'Evet (2 public küme)'],
        ['Khara ve ark. 2024 [13]', 'ResNet tabanlı yoğunluk modeli',
         '69.697 çalışma, 451.642 görüntü, 23.057 kadın',
         '4 sınıflı BI-RADS doğruluk %80,5 (FFDM) &middot; %79,4 (görülmemiş 2DS)',
         'Evet (FFDM&rarr;2DS, ırk alt grupları)'],
        ['Abrantes ve ark. 2023 [15]', 'Ticari yoğunluk modeli, dış doğrulama',
         'Harici klinik kohort', '4 sınıflı doğruluk %56,7 &middot; kappa 0,325', 'Evet'],
    ],
    genislik=['15%', '20%', '18%', '32%', '15%'])

# --- tablo 2: Kol 2 sentez matrisi
t_kol2 = tablo(
    ['Çalışma', 'Model / mimari', 'Veri', 'Metrik ve performans', 'Dış doğrulama'],
    [
        ['Yala ve ark. 2021 [17]', 'Mirai: ResNet-18 + transformer toplayıcı + additif hazard',
         'MGH (eğitim); MGH, Karolinska, CGMH (test)',
         'C-indeksi 0,76 (0,74&ndash;0,80) &middot; 0,81 (0,79&ndash;0,82) &middot; '
         '0,79 (0,79&ndash;0,83)', 'Evet (2 ülke)'],
        ['Yala ve ark. 2022 [18]', 'Mirai, çok kurumlu doğrulama',
         '7 hastane, 5 ülke; 128.793 mamogram, 62.185 hasta',
         'C-indeksi 0,75&ndash;0,84; Tyrer-Cuzick üzerinde üstünlük', 'Evet (7 hastane)'],
        ['Donnelly ve ark. 2024 [20]', 'AsymMirai: yerel iki taraflı benzemezlik',
         'EMBED; 210.067 mamogram, 81.824 hasta',
         '1/3/5 yıl AUC 0,79 / 0,68 / 0,66 (Mirai: 0,84 / 0,72 / 0,71)', 'Evet (EMBED)'],
        ['Dadsetan ve ark. 2022 [22]', 'LRP-NET: CNN + dondurulmuş VGG16 gövdesi',
         '200 hastalık dengeli case-control, 4 ardışık muayene',
         'Tek muayene modellerine üstün', 'Hayır (küçük, dengeli kohort)'],
        ['Karaman ve ark. 2024 [23]', 'LoMaR: Mirai + keyfi uzunlukta longitudinal transformer',
         'Büyük ölçekli tarama kümesi',
         '5. yılda Mirai üzerinde ~%10 ROCAUC üstünlük (tarama-tespitli kanserler hariç)',
         'Hayır (geliştirici bölmesi)'],
        ['Yeoh ve ark. 2025 [24]', 'TRINet: zaman-azalımlı dikkat + radyomik + AMIL',
         'EMBED (8.528 hasta), CSAW (8.723 hasta)',
         '1&ndash;5 yıl AUC 0,851 / 0,811 / 0,796 / 0,793 / 0,789', 'Kısmi (2 küme)'],
        ['Sun ve ark. 2025 [25]', 'VMRA-MaR: Vision-Mamba RNN + asimetri',
         'CSAW-CC', '4 ve 5 yıl ROCAUC 0,84; aynı noktalarda C-indeksi 0,82',
         'Hayır (geliştirici bölmesi)'],
        ['Avendano ve ark. 2024 [27]', 'Mirai, dış doğrulama',
         'Meksika; 3.110 hasta, 76 kanser', 'C-indeksi 0,63 (0,6&ndash;0,7)', 'Evet'],
        ['Gastounioti ve ark. 2022 [28]', 'ProFound AI Risk 1.0 (iCAD)',
         'ABD; 176 kanser, 4.963 kontrol',
         'AUC 0,68 (0,64&ndash;0,72); beyaz 0,67 &middot; siyah 0,70', 'Evet'],
        ['Schopf ve ark. 2024 [29]', 'PRISMA sistematik derleme', '16 çalışma',
         'Görüntü tabanlı modellerde medyan AUC 0,72; yoğunluk ve klinik araçlarda 0,61',
         'Yok (derleme)'],
        ['Lowry ve ark. 2026 [30]', 'PRISMA sistematik derleme', '41 çalışma, tümü retrospektif',
         'Medyan AUC 0,71 (&le;2 yıl) &middot; 0,72 (3&ndash;4 yıl) &middot; 0,71 (&ge;5 yıl); '
         'kalibrasyon 41 çalışmanın 6 tanesinde', 'Yok (derleme)'],
    ],
    genislik=['15%', '21%', '17%', '32%', '15%'])

# --- tablo 3: veri setleri
t_veri = tablo(
    ['Veri seti', 'Ölçek', 'Kaynak', 'Etiket', 'Kullanım'],
    [
        ['CBIS-DDSM [31]', '3.103 taranmış film mamogram; 753 kalsifikasyon, 891 kitle vakası',
         'ABD, DDSM türevi', 'ROI segmentasyonu, patoloji onaylı iyi ve kötü huylu ayrımı',
         'Kol 1 &mdash; en sık kullanılan kıyaslama'],
        ['INbreast [32]', '410 FFDM görüntüsü, 115 vaka',
         'Portekiz, tek merkez',
         'BI-RADS ve uzman konturu; lezyonların çoğu biyopsi onaylı değil',
         'Kol 1 &mdash; küçük ölçekli kıyaslama'],
        ['VinDr-Mammo [33]', '5.000 muayene, 20.000 görüntü',
         'Vietnam, 2 hastane', 'Meme seviyesi BI-RADS, yoğunluk, lezyon bounding box; '
         'çift okuma ve üçüncü radyolog arbitrajı',
         'Kol 1 &mdash; hemen indirilebilir public küme'],
        ['RSNA 2023 [4]', '10.830 tek meme muayenesi (değerlendirme kümesi)',
         'ABD ve Avustralya', 'Patoloji ground truth; yaklaşık %2 pozitif oran',
         'Kol 1 &mdash; gerçekçi tarama kıyaslaması'],
        ['EMBED [34]', '3,4 milyon görüntü, 110.000+ hasta, 60.000 anotasyonlu lezyon',
         'ABD, Emory; siyah ve beyaz dengeli', '2D, sentetik 2D ve DBT; BI-RADS ve patoloji',
         'Her iki kol &mdash; risk modeli dış doğrulaması'],
        ['OPTIMAM (OMI-DB) [35]',
         '2,5 milyondan fazla görüntü, 173.319 kadın; 9.690 tarama-tespitli ve '
         '1.888 interval kanser', 'İngiltere NHS',
         'ROI, klinik ve patoloji verisi, longitudinal takip',
         'Her iki kol &mdash; erişim başvuru gerektirir'],
        ['CSAW-CC [24, 25]', 'Karolinska tarama kohortu',
         'İsveç', 'Uzun takipli tarama etiketleri',
         'Kol 2 &mdash; longitudinal risk modelleri'],
    ],
    genislik=['14%', '24%', '15%', '28%', '19%'])

T = {'kol1': t_kol1, 'kol2': t_kol2, 'veri': t_veri}

# ------------------------------------------------------------------ cikti
import metin  # noqa: E402

HTML = ('<!DOCTYPE html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        '<title>Mamografi ile Meme Kanseri Tahmini &mdash; Literatür Değerlendirmesi</title>\n'
        f'<style>{CSS}</style></head><body>\n'
        + metin.govde(T) + '\n</body></html>')

CIKTI.write_text(HTML, encoding='utf-8')
print('yazildi:', CIKTI, f'{CIKTI.stat().st_size/1e3:.1f} kB')
print('kaynak sayisi:', metin.KAYNAK_SAYISI)
