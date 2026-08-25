"""Vaka raporu - overlay mozaigi + klinik ozet, tek dosyalik HTML.

PLAN.md Bolum 7: lezyon hacmi (mL), lezyon sayisi, en buyuk lezyon, hemisfer, overlay.
Gorseller base64 gomulu - rapor tek dosya olarak tasinabilir (diger projelerdeki desen).
"""
import base64
import io
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

UYARI = ('Bu rapor bir tibbi tani araci degildir. Arastirma amaclidir, uzman '
         'degerlendirmesinin yerine gecmez ve klinik karar icin kullanilamaz.')


def mozaik(dwi, maske, gt=None, n=9):
    """Lezyon iceren dilimlerden esit araliklarla n tane secip overlay cizer."""
    lez = np.nonzero(maske.sum(axis=(0, 1)))[0]
    if len(lez) == 0:
        lez = np.array([maske.shape[2] // 2])
    idx = lez[np.linspace(0, len(lez) - 1, min(n, len(lez))).astype(int)]

    satir = int(np.ceil(len(idx) / 3))
    fig, ax = plt.subplots(satir, 3, figsize=(11, 3.6 * satir))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.set_axis_off()
    for i, z in enumerate(idx):
        a = ax[i]
        a.imshow(dwi[:, :, z].T, cmap='gray', origin='lower')
        if gt is not None and gt[:, :, z].any():
            a.contour(gt[:, :, z].T, levels=[0.5], colors='lime', linewidths=0.9)
        m = maske[:, :, z]
        if m.any():
            a.imshow(np.ma.masked_where(~m.T, m.T), cmap='autumn', alpha=0.45, origin='lower')
        a.set_title(f'z = {int(z)}', fontsize=9)
    fig.suptitle('DWI + tahmin (kirmizi)' + (' / referans (yesil kontur)' if gt is not None else ''),
                 fontsize=11)
    plt.tight_layout()
    tampon = io.BytesIO()
    plt.savefig(tampon, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(tampon.getvalue()).decode()


def hacim_dagilimi(hacim_ml, medyan=6.66, iqr=(1.58, 21.17)):
    """Vakanin hacmini veri setinin dagilimina yerlestirir - klinik baglam."""
    fig, a = plt.subplots(figsize=(7, 1.6))
    a.axvspan(iqr[0], iqr[1], color='tab:blue', alpha=0.15, label=f'IQR {iqr[0]}-{iqr[1]} mL')
    a.axvline(medyan, color='tab:blue', ls='--', label=f'medyan {medyan} mL')
    a.axvline(hacim_ml, color='tab:red', lw=2.5, label=f'bu vaka {hacim_ml:.1f} mL')
    a.set_xscale('symlog', linthresh=1)
    a.set_xlim(0, 500)
    a.set_yticks([])
    a.set_xlabel('lezyon hacmi (mL, log olcek) - ISLES 2022, 250 vaka')
    a.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    tampon = io.BytesIO()
    plt.savefig(tampon, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(tampon.getvalue()).decode()


STIL = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:2rem auto;
padding:0 1.2rem;color:#1a1a1a;line-height:1.55}
h1{font-size:1.5rem;margin-bottom:.2rem}
.alt{color:#666;font-size:.9rem;margin-top:0}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #ddd;padding:.5rem .7rem;text-align:left;font-size:.92rem}
th{background:#f5f5f5;width:38%}
.vurgu{font-size:1.35rem;font-weight:600;color:#b00020}
.uyari{background:#fff4e5;border-left:4px solid #ff9800;padding:.8rem 1rem;margin:1.4rem 0;
font-size:.88rem}
.kucuk{color:#666;font-size:.82rem}
img{max-width:100%;height:auto}
"""


def html_rapor(vaka, klinik, sure, dwi, maske, gt=None, model='unet3d', arka_uc='torch-fp16'):
    ov = mozaik(dwi, maske, gt)
    dag = hacim_dagilimi(klinik['hacim_ml'])
    h = klinik['hemisfer'] or '-'
    sat = [
        ('Lezyon hacmi', f"<span class='vurgu'>{klinik['hacim_ml']:.2f} mL</span>"),
        ('Lezyon sayisi', f"{klinik['lezyon_sayisi']}"),
        ('En buyuk lezyon', f"{klinik['en_buyuk_lezyon_ml']:.2f} mL"),
        ('Hemisfer', f"{h} (sag oran {klinik['sag_hemisfer_orani']})"
                     if klinik['sag_hemisfer_orani'] is not None else h),
        ('Voxel sayisi', f"{klinik['voxel']:,}".replace(',', '.')),
    ]
    sure_sat = ''.join(
        f"<tr><th>{k.replace('_sn','').replace('_',' ')}</th><td>{v*1000:.1f} ms</td></tr>"
        for k, v in sure.items())
    return f"""<!doctype html><meta charset="utf-8">
<title>{vaka} - iskemik inme segmentasyon raporu</title><style>{STIL}</style>
<h1>Iskemik inme lezyon segmentasyonu</h1>
<p class="alt">Vaka <b>{vaka}</b> &middot; {datetime.now():%Y-%m-%d %H:%M} &middot;
model <code>{model}</code> / <code>{arka_uc}</code></p>

<div class="uyari"><b>Uyari.</b> {UYARI}</div>

<h2>Klinik ozet</h2>
<table>{''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in sat)}</table>

<img src="data:image/png;base64,{dag}" alt="hacim dagilimi">
<p class="kucuk">Vakanin hacmi, ISLES 2022 egitim kumesinin (250 vaka) dagilimina gore
konumlandirildi.</p>

<h2>Gorsel</h2>
<img src="data:image/png;base64,{ov}" alt="overlay">

<h2>Islem suresi</h2>
<table>{sure_sat}</table>
<p class="kucuk">Diskten okuma gecikme butcesinin disindadir; PACS'tan gelen hacim modelin
maliyeti degildir.</p>

<h2>Yontem ve sinirliliklar</h2>
<ul>
<li>Girdi: DWI (b=1000) + ADC, 2&times;2&times;2 mm izotropik grid.</li>
<li>Cikti tek bir <b>infarkt maskesi</b>dir; <b>core/penumbra ayrimi yapilmaz</b> -
    bunun icin perfuzyon goruntulemesi gerekir.</li>
<li>Hacim, 2 mm yeniden orneklenmis grid uzerinden hesaplanir; orijinal grid'e gore
    &plusmn;%2,3 sistematik fark tasir.</li>
<li>Test kumesi 50 vaka: Dice 0,70 [0,63-0,76]. <b>Kucuk lezyonlarda dogruluk belirgin
    dusuktur</b> (en kucuk hacim tertilinde Dice ~0,55).</li>
<li>Model tek merkezli bir veri setinin dagilimina gore egitildi; farkli cihaz ve
    protokollerde gecerliligi dogrulanmamistir.</li>
</ul>
"""
