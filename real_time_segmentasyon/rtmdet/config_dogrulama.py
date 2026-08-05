"""Egitime baslamadan once config'i uctan uca dogrula.

1. Config parse olup _base_ cozuluyor mu
2. Dataset dogru sayida goruntu/anotasyon yukluyor mu (bos maskeliler dahil)
3. Pipeline'dan gecen batch'in sekilleri dogru mu
4. Gercek egitim iterasyonu calisiyor mu, VRAM ne kadar, epoch ne kadar surer

Windows notu: `if __name__ == '__main__'` korumasi zorunlu. num_workers>0 oldugunda
multiprocessing spawn kullaniyor, cocuk surecler ana modulu yeniden import ediyor;
koruma olmadan surec uretimi kendini tekrarlar. Ayrica -u ile calistirilmali,
yoksa yonlendirilmis ciktida printler tamponlanip gorunmez.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))   # ozel_transformlar icin
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=r'd:\mamografi\real_time_segmentasyon\rtmdet\rtmdet_ins_busi.py')
    ap.add_argument('--iter', type=int, default=20)
    ap.add_argument('--workers', type=int, default=0, help='olcum icin 0, gercekci epoch suresi icin 2')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope('mmdet')
    cfg.train_dataloader.num_workers = args.workers
    cfg.train_dataloader.persistent_workers = args.workers > 0

    print('--- 1) config ---', flush=True)
    print('girdi boyutu :', cfg.GIRDI, flush=True)
    print('batch        :', cfg.train_dataloader.batch_size, flush=True)
    print('aug seviye   :', cfg.AUG_SEVIYE, flush=True)
    print('base lr      :', cfg.optim_wrapper.optimizer.lr, flush=True)
    print('max epoch    :', cfg.train_cfg.max_epochs, flush=True)

    print('\n--- 2) dataset ---', flush=True)
    dl = Runner.build_dataloader(cfg.train_dataloader)
    ds = dl.dataset
    bos = sum(1 for i in range(len(ds)) if len(ds.get_data_info(i)['instances']) == 0)
    inst = sum(len(ds.get_data_info(i)['instances']) for i in range(len(ds)))
    print(f'train goruntu: {len(ds)}', flush=True)
    print(f'bos maskeli  : {bos}', flush=True)
    print(f'instance     : {inst}', flush=True)
    assert len(ds) == 538, f'beklenen 538, gelen {len(ds)}'
    assert bos == 92, f'beklenen 92 bos, gelen {bos}'
    assert inst == 455, f'beklenen 455 instance, gelen {inst}'

    print('\n--- 3) pipeline ---', flush=True)
    batch = next(iter(dl))
    inputs = batch['inputs']
    ornek = batch['data_samples'][0]
    print('tek goruntu  :', tuple(inputs[0].shape), inputs[0].dtype, flush=True)
    print('deger araligi: [%d, %d]' % (int(inputs[0].min()), int(inputs[0].max())), flush=True)
    print('gt bbox      :', tuple(ornek.gt_instances.bboxes.shape), flush=True)
    if len(ornek.gt_instances):
        print('gt mask      :', ornek.gt_instances.masks.masks.shape, flush=True)
    assert inputs[0].shape[-1] == cfg.GIRDI and inputs[0].shape[-2] == cfg.GIRDI, 'girdi boyutu yanlis'

    # 538 goruntunun 92'sinde hic instance yok. Albu bos anotasyonda cokebiliyor ve
    # bu ancak o ornek batch'e dustugunde ortaya cikiyor -- egitimin ortasinda
    # saatler sonra vurmasin diye tum veri seti tek tek pipeline'dan geciriliyor.
    print('\n--- 3b) tum veri seti pipeline testi ---', flush=True)
    hatalar, bos_gecen, dolu_gecen = [], 0, 0
    for i in range(len(ds)):
        try:
            r = ds[i]
            n = len(r['data_samples'].gt_instances)
            if n == 0:
                bos_gecen += 1
            else:
                dolu_gecen += 1
        except Exception as e:
            hatalar.append((i, ds.get_data_info(i)['img_path'].split('\\')[-1],
                            type(e).__name__, str(e)[:70]))
    print(f'gecen: dolu {dolu_gecen}, bos {bos_gecen} | hata {len(hatalar)}', flush=True)
    for h in hatalar[:8]:
        print('   ', h, flush=True)
    assert not hatalar, f'{len(hatalar)} ornek pipeline\'dan gecemedi'

    print('\n--- 4) egitim iterasyonu ---', flush=True)
    cfg.train_cfg.max_epochs = 1
    cfg.load_from = None                 # agirlik indirmeden yapi testi
    cfg.work_dir = cfg.work_dir + '_smoke'
    cfg.default_hooks.checkpoint = None
    cfg.custom_hooks = []
    runner = Runner.from_cfg(cfg)
    runner.model.cuda().train()

    # mmengine optim_wrapper'i tembel kuruyor: runner.train() cagrilmadan hala
    # ConfigDict olarak duruyor, elle kurulmasi gerekiyor
    opt = runner.build_optim_wrapper(cfg.optim_wrapper)

    torch.cuda.reset_peak_memory_stats()
    sureler = []
    it = iter(dl)
    for k in range(args.iter):
        data = next(it)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with opt.optim_context(runner.model):
            veri = runner.model.data_preprocessor(data, True)
            kayip = runner.model(**veri, mode='loss')
            toplam, _ = runner.model.parse_losses(kayip)
        opt.update_params(toplam)
        torch.cuda.synchronize()
        sureler.append(time.perf_counter() - t0)
        if k == 0:
            ozet = {}
            for a, b in kayip.items():
                v = sum(x.mean() for x in b) if isinstance(b, (list, tuple)) else b.mean()
                ozet[a] = round(float(v), 4)
            print('kayip bilesenleri:', ozet, flush=True)

    sureler = np.array(sureler[3:])       # ilk 3 isinma
    adim = sureler.mean()
    n_adim = int(np.ceil(len(ds) / cfg.train_dataloader.batch_size))
    print('adim suresi  : %.3f sn (n=%d)' % (adim, len(sureler)), flush=True)
    print('epoch tahmini: %.0f sn = %.1f dk  (%d adim, val haric)' % (
        adim * n_adim, adim * n_adim / 60, n_adim), flush=True)
    print('%d epoch     : %.1f saat' % (cfg.MAX_EPOCH, adim * n_adim * cfg.MAX_EPOCH / 3600), flush=True)
    print('tepe VRAM    : %.0f MB / 6144 MB' % (torch.cuda.max_memory_allocated() / 1e6), flush=True)

    print('\nConfig dogrulandi.', flush=True)


if __name__ == '__main__':
    main()
