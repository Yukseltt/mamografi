"""Uretilen HTML raporu yerelde inceler.

  python rapor_incele.py [yol]

Rapor 7-8 MB ve buyuk kismi base64 gomulu figur. Bu script:
  - yapisal kontrol yapar (bolumler, gomulu figur sayisi, eksik figur uyarilari)
  - base64'u ayiklayip yalnizca metni ayri bir dosyaya yazar (okunabilir olsun)
  - gomulu figurleri PNG olarak cikarir (tek tek bakilabilsin)
"""
import base64
import re
import sys
from pathlib import Path

VARSAYILAN = Path(__file__).parent / 'rapor' / 'gmic_cmmd_raporu.html'


def main(yol):
    yol = Path(yol)
    if not yol.exists():
        print(f'bulunamadi: {yol}')
        print('Colab hucresinden indirip bu klasore koyun (rapor/ altina).')
        return 1

    ham = yol.read_text(encoding='utf-8')
    print(f'{yol}  ({yol.stat().st_size / 1e6:.1f} MB)')

    # --- gomulu figurler
    gomulu = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', ham)
    eksik = re.findall(r'\[figur bulunamadi: ([^\]]+)\]', ham)
    print(f'gomulu figur: {len(gomulu)}')
    if eksik:
        print(f'EKSIK FIGUR ({len(eksik)}):')
        for e in eksik:
            print(f'  - {e}')
    else:
        print('eksik figur yok')

    cikis = yol.parent / '_cikarilan'
    cikis.mkdir(exist_ok=True)
    for i, b in enumerate(gomulu):
        (cikis / f'figur_{i:02d}.png').write_bytes(base64.b64decode(b))
    print(f'figurler cikarildi: {cikis}')

    # --- bolumler
    print('\nbolumler:')
    for etiket, baslik in re.findall(r'<(h[12])>(.*?)</\1>', ham, re.S):
        temiz = re.sub(r'<[^>]+>', '', baslik).strip()
        print(('  ' if etiket == 'h2' else '') + temiz)

    # --- metin: base64 atilir, etiketler sadelestirilir
    metin = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', '[FIGUR]', ham)
    metin = re.sub(r'<style>.*?</style>', '', metin, flags=re.S)
    metin = re.sub(r'<(script|!--).*?(</script>|-->)', '', metin, flags=re.S)
    metin = re.sub(r'<br\s*/?>', '\n', metin)
    metin = re.sub(r'</(p|li|h1|h2|h3|tr|div|figcaption)>', '\n', metin)
    metin = re.sub(r'</t[dh]>', ' | ', metin)
    metin = re.sub(r'<[^>]+>', '', metin)
    metin = re.sub(r'&middot;', '.', metin)
    metin = re.sub(r'&mdash;', '--', metin).replace('&ndash;', '-').replace('&rarr;', '->')
    metin = re.sub(r'&[a-z]+;', ' ', metin)
    metin = re.sub(r'\n{3,}', '\n\n', metin)
    metin = '\n'.join(l.rstrip() for l in metin.split('\n'))

    metin_yolu = yol.with_name('rapor_metin.txt')
    metin_yolu.write_text(metin.strip(), encoding='utf-8')
    print(f'\nmetin: {metin_yolu}  ({len(metin.split())} kelime)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else VARSAYILAN))
