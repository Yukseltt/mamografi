"""FastAPI servisi - NIfTI al, maske + klinik cikti + rapor dondur.

Motor uygulama basladiginda BIR KEZ yukleniyor; her istekte model yuklemek gecikmeyi
on kat artirirdi. Arka uc ortam degiskeniyle secilebiliyor.

    ISLES_ARKA_UC=torch-fp16 uvicorn servis.api:app --host 0.0.0.0 --port 8000

Uc noktalar:
    GET  /saglik          motor hazir mi, hangi arka uc
    POST /segment         dwi + adc dosyasi -> JSON (klinik cikti + bilesen bazli sure)
    POST /rapor           ayni girdi -> tek dosyalik HTML rapor
    POST /maske           ayni girdi -> maske NIfTI (2 mm grid)
"""
import os
import sys
import tempfile
import time
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / 'ortak'))
sys.path.insert(0, str(KOK / 'servis'))

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from motor import Motor
import rapor as rp

ARKA_UC = os.environ.get('ISLES_ARKA_UC', 'torch-fp16')
MODEL = os.environ.get('ISLES_MODEL', 'unet3d')

app = FastAPI(title='ISLES iskemik inme segmentasyonu',
              description='DWI + ADC -> infarkt maskesi, lezyon hacmi ve rapor. '
                          'Tibbi tani araci degildir.')
MOTOR = None


@app.on_event('startup')
def yukle():
    global MOTOR
    t0 = time.perf_counter()
    MOTOR = Motor(model_key=MODEL, arka_uc=ARKA_UC).isit()
    print(f'motor hazir ve isitildi: {MODEL} / {ARKA_UC} '
          f'({time.perf_counter()-t0:.1f} sn)')


def _kaydet(dosya: UploadFile, dizin: str, ad: str) -> str:
    if not dosya.filename.endswith(('.nii', '.nii.gz')):
        raise HTTPException(400, f'{ad}: NIfTI bekleniyor (.nii / .nii.gz), '
                                 f'gelen: {dosya.filename}')
    yol = os.path.join(dizin, ad + '.nii.gz')
    with open(yol, 'wb') as f:
        f.write(dosya.file.read())
    return yol


def _calistir(dwi: UploadFile, adc: UploadFile):
    with tempfile.TemporaryDirectory() as d:
        dy, ay = _kaydet(dwi, d, 'dwi'), _kaydet(adc, d, 'adc')
        try:
            return MOTOR.calistir(dy, ay)
        except Exception as e:
            raise HTTPException(422, f'islem basarisiz: {type(e).__name__}: {e}')


@app.get('/saglik')
def saglik():
    return dict(hazir=MOTOR is not None, model=MODEL, arka_uc=ARKA_UC,
                cihaz=getattr(MOTOR, 'cihaz', None),
                esik=getattr(MOTOR, 'secim', {}).get('esik'))


@app.post('/segment')
def segment(dwi: UploadFile = File(...), adc: UploadFile = File(...)):
    r = _calistir(dwi, adc)
    return dict(klinik=r['klinik'],
                sure_ms={k: round(v * 1000, 2) for k, v in r['sure'].items()},
                model=r['model'], arka_uc=r['arka_uc'], esik=r['esik'],
                maske_sekli=list(r['maske'].shape))


@app.post('/rapor', response_class=HTMLResponse)
def rapor_uc(dwi: UploadFile = File(...), adc: UploadFile = File(...)):
    ad = Path(dwi.filename).name.replace('.nii.gz', '')
    with tempfile.TemporaryDirectory() as d:
        dy, ay = _kaydet(dwi, d, 'dwi'), _kaydet(adc, d, 'adc')
        r = MOTOR.calistir(dy, ay)
        import hat
        import hat_gpu  # noqa: F401
        ham, _ = hat.oku(dy, ay)
        x, _ = hat.on_isle(ham, MOTOR.cfg)
    return rp.html_rapor(ad, r['klinik'], r['sure'], x[0], r['maske'],
                         model=r['model'], arka_uc=r['arka_uc'])


@app.post('/maske')
def maske_uc(dwi: UploadFile = File(...), adc: UploadFile = File(...)):
    r = _calistir(dwi, adc)
    hs = MOTOR.cfg['hedef_spacing']
    affine = np.diag(list(hs) + [1.0])
    im = nib.Nifti1Image(r['maske'].astype(np.uint8), affine)
    with tempfile.TemporaryDirectory() as d:
        yol = os.path.join(d, 'maske.nii.gz')
        nib.save(im, yol)
        veri = open(yol, 'rb').read()
    return Response(veri, media_type='application/gzip',
                    headers={'Content-Disposition': 'attachment; filename=maske.nii.gz'})
