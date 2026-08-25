"""Model tanimlari - yerel olcum scriptleri icin.

isles_egitim.ipynb'deki tanimlarin AYNISI. Notebook uzak Colab kernel'inde kosuyor,
yerel scriptler onun hucrelerini import edemiyor; bu yuzden mimariler burada tekrar
tanimli. Parametre sayilari `dagitim/*_cikarim.pt` yuklenirken dogrulaniyor - iki
tanim ayrisirsa state_dict yuklemesi hata verir, sessizce farkli bir model olculmez.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """Kompakt 3B U-Net, tam hacim (96x128x96)."""

    def __init__(self, kanal=2, taban=16):
        super().__init__()
        blok = lambda gi, go: nn.Sequential(
            nn.Conv3d(gi, go, 3, padding=1), nn.InstanceNorm3d(go), nn.LeakyReLU(inplace=True),
            nn.Conv3d(go, go, 3, padding=1), nn.InstanceNorm3d(go), nn.LeakyReLU(inplace=True))
        f = [taban, taban * 2, taban * 4, taban * 8]
        self.e1, self.e2, self.e3 = blok(kanal, f[0]), blok(f[0], f[1]), blok(f[1], f[2])
        self.alt = blok(f[2], f[3])
        self.u3 = nn.ConvTranspose3d(f[3], f[2], 2, 2); self.d3 = blok(f[3], f[2])
        self.u2 = nn.ConvTranspose3d(f[2], f[1], 2, 2); self.d2 = blok(f[2], f[1])
        self.u1 = nn.ConvTranspose3d(f[1], f[0], 2, 2); self.d1 = blok(f[1], f[0])
        self.cikis = nn.Conv3d(f[0], 1, 1)
        self.havuz = nn.MaxPool3d(2)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.havuz(e1)); e3 = self.e3(self.havuz(e2))
        a = self.alt(self.havuz(e3))
        d = self.d3(torch.cat([self.u3(a), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.cikis(d)


def model_yap(konfig):
    if konfig['mimari'] == 'unet3d':
        return UNet3D(kanal=2)
    if konfig['mimari'] == 'unet':
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=konfig['encoder'], encoder_weights=None,
                        in_channels=konfig['kanal'], classes=1)
    raise ValueError(konfig['mimari'])


def model_yukle(pt_yolu, cihaz='cpu'):
    """dagitim/*_cikarim.pt yukler. state_dict uyusmazsa hata verir (sessiz gecmez)."""
    d = torch.load(pt_yolu, map_location='cpu', weights_only=False)
    c = d['konfig']
    m = model_yap(c)
    m.load_state_dict(d['model'])          # strict=True: mimari ayrisirsa burada patlar
    m.eval().to(cihaz)
    return m, c, d['secim']


def parametre_sayisi(m):
    return sum(p.numel() for p in m.parameters())
