# Decoder + autoencoder
# The decoder generates images from latent vectors.
# The autoencoder compresses and reconstructs the generated images, minimizing the reconstruction error.
# The decoder maximizes the autoencoder's error.

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import utils
import IPython.display


def get_args_parser():
    parser = argparse.ArgumentParser("ICM", add_help=False)

    parser.add_argument("--id", type=int)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument("--dim_x", default=10, type=int)
    parser.add_argument("--n_clusters", default=10, type=int)
    parser.add_argument("--sigma_x", default=0.1, type=float)
    parser.add_argument("--n_ch", default=1, type=int)
    parser.add_argument("--n_maps_dec", default=16, type=int)
    parser.add_argument("--n_maps_enc", default=16, type=int)

    parser.add_argument("--lr_start", default=1e-2, type=float)
    parser.add_argument("--lr_end", default=1e-3, type=float)
    
    parser.add_argument("--lossR_factor", default=1.0, type=float)
    parser.add_argument("--lossE_factor", default=0.0, type=float)

    return parser


class Decoder(nn.Module):

    def __init__(self, dim_z, n_ch, n_maps):
        super().__init__()
        self.dim_z = dim_z
        self.n_ch = n_ch
        self.n_maps = n_maps

        self.deconv = nn.Sequential(
            # shape: (dim_z) x 1 x 1

            nn.ConvTranspose2d(dim_z, n_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_maps * 16),
            nn.ReLU(True),
            # shape: (n_maps x 16) x 4 x 4

            nn.ConvTranspose2d(n_maps * 16, n_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_maps * 8),
            nn.ReLU(True),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_maps x 8) x 8 x 8

            nn.ConvTranspose2d(n_maps * 8, n_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_maps * 4),
            nn.ReLU(True),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_maps x 4) x 16 x 16

            nn.ConvTranspose2d(n_maps * 4, n_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_maps * 2),
            nn.ReLU(True),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_maps x 2) x 32 x 32

            nn.ConvTranspose2d(n_maps * 2, n_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_maps),
            nn.ReLU(True),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_maps) x 64 x 64

            nn.Conv2d(n_maps, n_maps, 3, 1, 1),
            nn.BatchNorm2d(n_maps),
            nn.ReLU(True),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_maps) x 64 x 64

            nn.Conv2d(n_maps, n_maps, 3, 1, 1),
            nn.BatchNorm2d(n_maps),
            nn.ReLU(True),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_maps) x 64 x 64

            nn.Conv2d(n_maps, n_ch, 3, 1, 1),
            nn.Tanh(),
            nn.AvgPool2d(3, 1, 1),
            # shape: (n_ch) x 64 x 64
        )

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.deconv(x)
        return x


class Encoder(nn.Module):

    def __init__(self, dim_z, n_ch, n_maps):
        super().__init__()
        self.dim_z = dim_z
        self.n_ch = n_ch
        self.n_maps = n_maps

        self.conv = nn.Sequential(
            # shape: (n_ch) x 64 x 64

            nn.Conv2d(n_ch, n_maps, 11, 4, 4, bias=False),
            nn.BatchNorm2d(n_maps),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (n_maps) x 16 x 16

            nn.Conv2d(n_maps, n_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (n_maps x 2) x 8 x 8

            nn.Conv2d(n_maps * 2, n_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # shape: (n_maps x 4) x 4 x 4

            nn.Conv2d(n_maps * 4, dim_z, 4, 1, 0, bias=False),
            # shape: (dim_z) x 1 x 1
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return x


class InputSampler:

    def __init__(self, dim_x, n_clusters, sigma, device):
        self.dim_x = dim_x
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.device = device

        if n_clusters != 0:
            self.mu = torch.randn(n_clusters, dim_x, device=device)
            self.mu = self.mu / torch.norm(self.mu, dim=1, keepdim=True)
    
    def sample(self, n):
        x = torch.randn(n, self.dim_x, device=self.device)
        if self.n_clusters != 0:
            rand_ids = torch.randint(0, self.n_clusters, (n,))
            x = x * self.sigma + self.mu[rand_ids]
        return x


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0")
    sampler = InputSampler(args.dim_x, args.n_clusters, args.sigma_x, device)
    gen = Decoder(args.dim_x, args.n_ch, args.n_maps_dec)
    enc = Encoder(args.dim_x, args.n_ch, args.n_maps_enc)
    dec = Decoder(args.dim_x, args.n_ch, args.n_maps_dec)
    mse = nn.MSELoss()
    gen.to(device)
    enc.to(device)
    dec.to(device)
    optimG = torch.optim.SGD(
        gen.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimE = torch.optim.SGD(
        enc.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimD = torch.optim.SGD(
        dec.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    lr_schedulerG = CosineAnnealingLR(optimG, args.n_epochs, args.lr_end)
    lr_schedulerE = CosineAnnealingLR(optimE, args.n_epochs, args.lr_end)
    lr_schedulerD = CosineAnnealingLR(optimD, args.n_epochs, args.lr_end)
    res = Results(args, sampler, gen, enc)
    for epoch in tqdm(range(args.n_epochs)):

        x = sampler.sample(args.batch_size)
        g = gen(x)
        y = enc(g.detach())
        r = dec(y)

        lossR = mse(r, g)

        optimE.zero_grad()
        optimD.zero_grad()

        lossR.backward(retain_graph=True)
        if args.lossR_factor != 1.0:
            for p in enc.parameters():
                p.grad *= args.lossR_factor
        if args.lossE_factor != 0.0:
            lossE = args.lossE_factor * mse(y, x)
            lossE.backward()
        else:
            lossE = torch.tensor(0)

        optimE.step()
        optimD.step()

        y = enc(g)
        r = dec(y)

        lossG = 1 / (1 + mse(r, g))

        optimG.zero_grad()
        lossG.backward()
        optimG.step()

        res.add_loss(lossG.item(), lossR.item(), lossE.item())
        res.update_imgs(epoch)

        lr_schedulerG.step()
        lr_schedulerE.step()
        lr_schedulerD.step()
    return res


class Results:

    def __init__(self, args, sampler, gen, enc):
        self.args = args
        self.sampler = sampler
        self.gen = gen
        self.enc = enc
        self.device = next(self.gen.parameters()).device
        self.hlossG = []
        self.hlossR = []
        self.hlossE = []

        self.n_imgs = 2
        self.img_freq = 5
        self.xs = self.sampler.sample(self.n_imgs)
        self.imgs = []

    def add_loss(self, lossG, lossR, lossE):
        self.hlossG.append(lossG)
        self.hlossR.append(lossR)
        self.hlossE.append(lossE)
    
    def update_imgs(self, epoch):
        if not (epoch % self.img_freq == 0 or epoch == self.args.n_epochs - 1): return
        gs = self._forward(self.xs)
        self.imgs.append(self._to_img(gs, 1))

    def plot_loss(self):
        fig, axs = plt.subplots(1, 2, figsize=(8, 2))
        fig.tight_layout()
        epochs = np.arange(1, len(self.hlossG) + 1)

        axs[0].plot(epochs, self.hlossG, label="G")
        axs[0].plot(epochs, self.hlossR, label="R")
        axs[0].legend()
        axs[0].set_title("Loss G & R")

        axs[1].plot(epochs, self.hlossE, label="E")
        axs[1].legend()
        axs[1].set_title("Loss E")

        plt.show()

    def _to_numpy(self, t):
        return t.cpu().detach().numpy()

    @torch.no_grad()
    def _forward(self, x):
        g = self.gen(x)
        return self._to_numpy(g)

    def _to_img(self, g, ch_dim=0):
        im = np.moveaxis(g, ch_dim, -1)
        im = (im + 1) / 2
        return im

    def plot_imgs(self):
        n_im = 30
        x = self.sampler.sample(n_im)
        g = self._forward(x)
        ncols = 5
        nrows = int(np.ceil(n_im / ncols))
        _, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        for i in range(n_im):
            im = self._to_img(g[i])
            ax = axs[i // ncols, i % ncols]
            ax.imshow(im)
            ax.axis("off")
        utils.save_plot_args(args, "images/exp1")
        plt.show()
    
    def plot_anim(self):
        anim = utils.animate(self.imgs)
        path = utils.get_path(args.id, "images/exp1")
        anim.save(f'{path}.gif', writer='imagemagick')
        display = IPython.display.HTML(anim.to_jshtml())
        IPython.display.display(display)


if __name__ == "__main__":

    args = [
        "--id=21",

        "--seed=12",
        "--n_epochs=1000",
        "--batch_size=32",

        "--dim_x=100",
        "--n_clusters=0",
        "--sigma_x=0.1",
        "--n_ch=3",
        "--n_maps_enc=32",
        "--n_maps_dec=32",

        "--lr_start=1e-2",
        "--lr_end=1e-2",

        "--lossR_factor=1.0",
        "--lossE_factor=0.0",
    ]

    parser = argparse.ArgumentParser("ICM", parents=[get_args_parser()])
    args = parser.parse_args(args)
    res = train(args)

    res.plot_loss()
    res.plot_imgs()
    res.plot_anim()