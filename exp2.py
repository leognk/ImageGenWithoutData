# 2 x (decoder + autoencoder)
# The 2 decoders generate images from latent vectors.
# Each autoencoder compresses and reconstructs the images generated from both decoders, minimizing the reconstruction error.
# The 2 decoders maximize the autoencoders' errors.

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

    sampler1 = InputSampler(args.dim_x, args.n_clusters, args.sigma_x, device)
    sampler2 = InputSampler(args.dim_x, args.n_clusters, args.sigma_x, device)
    gen1 = Decoder(args.dim_x, args.n_ch, args.n_maps_dec)
    gen2 = Decoder(args.dim_x, args.n_ch, args.n_maps_dec)
    enc1 = Encoder(args.dim_x, args.n_ch, args.n_maps_enc)
    enc2 = Encoder(args.dim_x, args.n_ch, args.n_maps_enc)
    dec1 = Decoder(args.dim_x, args.n_ch, args.n_maps_dec)
    dec2 = Decoder(args.dim_x, args.n_ch, args.n_maps_dec)
    mse = nn.MSELoss()
    gen1.to(device)
    gen2.to(device)
    enc1.to(device)
    enc2.to(device)
    dec1.to(device)
    dec2.to(device)

    optimG1 = torch.optim.SGD(
        gen1.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimG2 = torch.optim.SGD(
        gen2.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimE1 = torch.optim.SGD(
        enc1.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimE2 = torch.optim.SGD(
        enc2.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimD1 = torch.optim.SGD(
        dec1.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    optimD2 = torch.optim.SGD(
        dec2.parameters(),
        lr=args.lr_start, momentum=0.9
    )
    lr_schedulerG1 = CosineAnnealingLR(optimG1, args.n_epochs, args.lr_end)
    lr_schedulerG2 = CosineAnnealingLR(optimG2, args.n_epochs, args.lr_end)
    lr_schedulerE1 = CosineAnnealingLR(optimE1, args.n_epochs, args.lr_end)
    lr_schedulerE2 = CosineAnnealingLR(optimE2, args.n_epochs, args.lr_end)
    lr_schedulerD1 = CosineAnnealingLR(optimD1, args.n_epochs, args.lr_end)
    lr_schedulerD2 = CosineAnnealingLR(optimD2, args.n_epochs, args.lr_end)

    res = Results(args, sampler1, sampler2, gen1, gen2)
    for epoch in tqdm(range(args.n_epochs)):

        x1 = sampler1.sample(args.batch_size)
        x2 = sampler2.sample(args.batch_size)
        g1 = gen1(x1)
        g2 = gen2(x2)

        yE1_g1 = enc1(g1.detach())
        yE1_g2 = enc1(g2.detach())
        yE2_g1 = enc2(g1.detach())
        yE2_g2 = enc2(g2.detach())
        rD1_g1 = dec1(yE1_g1)
        rD1_g2 = dec1(yE1_g2)
        rD2_g1 = dec1(yE2_g1)
        rD2_g2 = dec1(yE2_g2)

        lossR1 = (mse(rD1_g1, g1) + mse(rD1_g2, g2)) / 2
        lossR2 = (mse(rD2_g1, g1) + mse(rD2_g2, g2)) / 2

        optimE1.zero_grad()
        optimE2.zero_grad()
        optimD1.zero_grad()
        optimD2.zero_grad()

        lossR1.backward(retain_graph=True)
        lossR2.backward(retain_graph=True)
        for p in enc1.parameters():
            p.grad *= args.lossR_factor
        for p in enc2.parameters():
            p.grad *= args.lossR_factor
        lossE1 = (mse(yE1_g1, x1) + mse(yE1_g2, x2)) / 2
        lossE2 = (mse(yE2_g1, x1) + mse(yE2_g2, x2)) / 2
        lossE1 = args.lossE_factor * lossE1
        lossE2 = args.lossE_factor * lossE2
        lossE1.backward()
        lossE2.backward()

        optimE1.step()
        optimE2.step()
        optimD1.step()
        optimD2.step()

        yE1_g1 = enc1(g1)
        yE1_g2 = enc1(g2)
        yE2_g1 = enc2(g1)
        yE2_g2 = enc2(g2)
        rD1_g1 = dec1(yE1_g1)
        rD1_g2 = dec1(yE1_g2)
        rD2_g1 = dec1(yE2_g1)
        rD2_g2 = dec1(yE2_g2)

        lossG1 = (mse(rD1_g1, g1) + mse(rD2_g1, g1)) / 2
        lossG2 = (mse(rD1_g2, g2) + mse(rD2_g2, g2)) / 2
        lossG1 = 1 / (1 + lossG1)
        lossG2 = 1 / (1 + lossG2)

        optimG1.zero_grad()
        optimG2.zero_grad()
        lossG1.backward()
        lossG2.backward()
        optimG1.step()
        optimG2.step()

        res.add_loss(
            lossG1.item(), lossG2.item(),
            lossR1.item(), lossR2.item(),
            lossE1.item(), lossE2.item()
        )
        res.update_imgs(epoch)

        lr_schedulerG1.step()
        lr_schedulerG2.step()
        lr_schedulerE1.step()
        lr_schedulerE2.step()
        lr_schedulerD1.step()
        lr_schedulerD2.step()

    return res


class Results:

    def __init__(self, args, sampler1, sampler2, gen1, gen2):
        self.args = args
        self.samplers = [sampler1, sampler2]
        self.gen1 = gen1
        self.gen2 = gen2
        self.device = next(self.gen1.parameters()).device
        self.hlossG = ([], [])
        self.hlossR = ([], [])
        self.hlossE = ([], [])

        self.n_imgs = 2
        self.img_freq = 10
        self.xs = [self.samplers[i].sample(self.n_imgs) for i in range(2)]
        self.imgs = [[], []]

    def add_loss(self, lossG1, lossG2, lossR1, lossR2, lossE1, lossE2):
        self.hlossG[0].append(lossG1)
        self.hlossG[1].append(lossG2)
        self.hlossR[0].append(lossR1)
        self.hlossR[1].append(lossR2)
        self.hlossE[0].append(lossE1)
        self.hlossE[1].append(lossE2)
    
    def _update_imgs_i(self, i, epoch):
        if not (epoch % self.img_freq == 0 or epoch == self.args.n_epochs - 1): return
        gs = self._forward(i, self.xs[i - 1])
        self.imgs[i - 1].append(self._to_img(gs, 1))
    
    def update_imgs(self, epoch):
        self._update_imgs_i(1, epoch)
        self._update_imgs_i(2, epoch)

    def plot_loss(self):
        fig, axs = plt.subplots(1, 3, figsize=(12, 2))
        fig.tight_layout()
        epochs = np.arange(1, len(self.hlossG[0]) + 1)

        axs[0].plot(epochs, self.hlossG[0], label="G1")
        axs[0].plot(epochs, self.hlossG[1], label="G2")
        axs[0].legend()
        axs[0].set_title("Loss G")

        axs[1].plot(epochs, self.hlossR[0], label="R1")
        axs[1].plot(epochs, self.hlossR[1], label="R2")
        axs[1].legend()
        axs[1].set_title("Loss R")

        axs[2].plot(epochs, self.hlossE[0], label="E1")
        axs[2].plot(epochs, self.hlossE[1], label="E2")
        axs[2].legend()
        axs[2].set_title("Loss E")

        plt.show()

    def _to_numpy(self, t):
        return t.cpu().detach().numpy()

    @torch.no_grad()
    def _forward(self, i, x):
        if i == 1:
            g = self.gen1(x)
        elif i == 2:
            g = self.gen2(x)
        return self._to_numpy(g)

    def _to_img(self, g, ch_dim=0):
        im = np.moveaxis(g, ch_dim, -1)
        im = (im + 1) / 2
        return im

    def _plot_imgs_i(self, i):
        n_im = 30
        x = self.samplers[i - 1].sample(n_im)
        g = self._forward(i, x)
        ncols = 5
        nrows = int(np.ceil(n_im / ncols))
        _, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        for k in range(n_im):
            im = self._to_img(g[k])
            ax = axs[k // ncols, k % ncols]
            ax.imshow(im)
            ax.axis("off")
        utils.save_plot_args(args, "images/exp2", i)
        plt.show()
    
    def plot_imgs(self):
        self._plot_imgs_i(1)
        self._plot_imgs_i(2)
    
    def _plot_anim_i(self, i):
        anim = utils.animate(self.imgs[i - 1])
        path = utils.get_path(args.id, "images/exp2", i)
        anim.save(f'{path}.gif', writer='imagemagick')
        display = IPython.display.HTML(anim.to_jshtml())
        IPython.display.display(display)
    
    def plot_anim(self):
        self._plot_anim_i(1)
        self._plot_anim_i(2)


if __name__ == "__main__":

    args = [
        "--id=10",

        "--seed=14",
        "--n_epochs=2000",
        "--batch_size=64",

        "--dim_x=1000",
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