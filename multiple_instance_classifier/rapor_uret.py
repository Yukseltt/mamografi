"""Raporu bastan sona YERELDE uretir: vaka gorselleri + HTML + PDF.

  python rapor_uret.py            -> eksikleri listeler, hazirsa raporu uretir
  python rapor_uret.py --kontrol  -> yalnizca eksik dosya kontrolu

Colab'a ihtiyac yok. Vaka gorselleri CPU'da uretilir (6 meme x 2 view, tam cozunurluk;
makinaya gore 5-15 dk). Mantik kopyalanmaz: gmic_cmmd_rapor.ipynb hucreleri calistirilir.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

KOK = Path(__file__).parent
NB = KOK / 'gmic_cmmd_rapor.ipynb'

# Drive'dan gelmesi gereken dosyalar (yerelde uretilemeyenler)
GEREKLI = {
    'gmic/checkpoints/best.pt': 'vaka gorselleri icin egitilmis model',
    'karsilastirma_ozeti.xlsx': 'Parca 8 ozet tablosu (kohort sayilari)',
    'degerlendirme/karsilastirma_tablosu.xlsx': 'AUC + CI + DeLong',
    'degerlendirme/altgrup_tablosu.xlsx': 'alt grup kirilimi',
    'degerlendirme/forest_auc.png': 'Sekil 4',
    'arama/secim.json': 'secilen hiperparametreler',
    'arama/arama_ozeti.xlsx': 'arama tablosu',
    'arama/arama_dagilim.png': 'Sekil 3',
    'gmic/gmic_curves.png': 'Sekil 5',
    'resnet22_baseline/resnet22_baseline_curves.png': 'Sekil 6',
}
# yerelde zaten var olmasi gerekenler
YEREL = {
    'hazir/manifest.csv': 'Parca 2 ciktisi',
    'hazir/config.json': 'Parca 2 ciktisi',
    'hazir/eval_cases.json': 'ornek vakalar',
    'ozet/veri_kesfi.png': 'Sekil 1',
    'ozet/onisleme_dogrulama.png': 'Sekil 2',
    'gmic_cmmd_egitim.ipynb': 'model tanimlari buradan okunuyor',
}


def kontrol():
    eksik = []
    print('--- Drive\'dan gelmesi gerekenler ---')
    for yol, ne in GEREKLI.items():
        var = (KOK / yol).exists()
        print(f'  {"OK " if var else "YOK"}  {yol:52s} {ne}')
        if not var:
            eksik.append(yol)
    print('--- yerelde olmasi gerekenler ---')
    for yol, ne in YEREL.items():
        var = (KOK / yol).exists()
        print(f'  {"OK " if var else "YOK"}  {yol:52s} {ne}')
        if not var:
            eksik.append(yol)
    return eksik


def uret():
    import os
    os.chdir(KOK)
    import matplotlib
    matplotlib.use('Agg')

    nb = json.loads(NB.read_text(encoding='utf-8'))
    g = {'__name__': '__main__'}
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = '\n'.join(l for l in ''.join(c['source']).split('\n')
                        if not l.strip().startswith('!'))
        print(f'\n===== hucre {i} =====', flush=True)
        t0 = time.time()
        try:
            exec(compile(src, f'<rapor nb hucre {i}>', 'exec'), g)
        except Exception as e:
            print(f'!!! HUCRE {i} PATLADI: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            return 1
        print(f'({time.time() - t0:.0f} sn)', flush=True)

    html = KOK / 'rapor' / 'gmic_cmmd_raporu.html'
    eksik_fig = html.read_text(encoding='utf-8').count('figur bulunamadi')
    print(f'\nHTML: {html}  ({html.stat().st_size / 1e6:.1f} MB), '
          f'eksik figur: {eksik_fig}')

    print('\n--- PDF ---', flush=True)
    r = subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
                        str(KOK / 'pdf_olustur.ps1')], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return 0 if 'PDF olusturuldu' in r.stdout else 1


if __name__ == '__main__':
    eksik = kontrol()
    if '--kontrol' in sys.argv:
        sys.exit(0)
    if eksik:
        print(f'\n{len(eksik)} dosya eksik, rapor uretilemez.')
        print('Colab hucresinden paketi indirip bu klasore acin.')
        sys.exit(1)
    print('\ntum dosyalar yerinde, rapor uretiliyor...\n')
    sys.exit(uret())
