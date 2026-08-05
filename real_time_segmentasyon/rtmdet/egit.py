"""RTMDet-Ins egitimini baslatir.

Kullanim (PowerShell, proje kokunden):

    $py = "d:\\mamografi\\.venv_mmdet\\Scripts\\python.exe"
    & $py -u real_time_segmentasyon\\rtmdet\\egit.py

    # 512 girdi ile:
    & $py -u real_time_segmentasyon\\rtmdet\\egit.py --girdi 512 --batch 4

    # kesilen kosuya devam:
    & $py -u real_time_segmentasyon\\rtmdet\\egit.py --resume

Windows notlari:
  - `if __name__ == '__main__'` korumasi zorunlu (multiprocessing spawn).
  - `-u` ile calistir, yoksa yonlendirilmis ciktida loglar tamponlanip gorunmez.
  - num_workers varsayilan 0: dogrulamada olculen 0.175 sn/adim zaten veri
    yuklemeyi iceriyor, veri darbogaz degil. Windows'ta worker spawn maliyeti
    ve riski gereksiz.
"""
import argparse
import sys
from pathlib import Path

BURASI = Path(__file__).parent
sys.path.insert(0, str(BURASI))          # ozel_transformlar icin

import torch_uyum  # noqa: F401  torch 2.6 checkpoint uyumu, resume icin sart

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(BURASI / 'rtmdet_ins_busi.py'))
    ap.add_argument('--girdi', type=int, default=None, help='256 | 512')
    ap.add_argument('--aug', default=None, choices=['hafif', 'orta'])
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    # config icindeki degerleri komut satirindan ezmek: work_dir da buna gore yenilenir
    girdi = args.girdi if args.girdi else cfg.GIRDI
    aug = args.aug if args.aug else cfg.AUG_SEVIYE
    if args.girdi or args.aug:
        raise SystemExit(
            'Girdi boyutu ve augmentasyon seviyesi pipeline icinde de kullaniliyor.\n'
            'Komut satirindan ezmek yerine rtmdet_ins_busi.py icindeki GIRDI / AUG_SEVIYE\n'
            'degiskenlerini degistir -- boylece work_dir, Resize ve Albu tutarli kalir.')

    if args.batch:
        cfg.train_dataloader.batch_size = args.batch
    if args.epoch:
        cfg.train_cfg.max_epochs = args.epoch
    cfg.train_dataloader.num_workers = args.workers
    cfg.train_dataloader.persistent_workers = args.workers > 0
    cfg.val_dataloader.num_workers = args.workers
    cfg.val_dataloader.persistent_workers = args.workers > 0
    cfg.resume = args.resume

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    print('=' * 60, flush=True)
    print('girdi     :', girdi, flush=True)
    print('aug       :', aug, flush=True)
    print('batch     :', cfg.train_dataloader.batch_size, flush=True)
    print('epoch     :', cfg.train_cfg.max_epochs, flush=True)
    print('lr        :', cfg.optim_wrapper.optimizer.lr, flush=True)
    print('work_dir  :', cfg.work_dir, flush=True)
    print('resume    :', cfg.resume, flush=True)
    print('=' * 60, flush=True)

    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
