"""Small anisotropic 3D U-Net used as a drop-in replacement for the ZephOD CNN.

Output is a logit heatmap with the same (Z, Y, X) shape as the input.
Z is never pooled because typical worm volumes have only ~20 slices.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, (1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(inplace=True),
        nn.Conv3d(c_out, c_out, (3, 3, 3), padding=(1, 1, 1), bias=False),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, n_channels_in=1, n_channels_out=1, init_nodes=16, depth=3):
        super().__init__()
        ch = [init_nodes * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        c = n_channels_in
        for c_out in ch:
            self.down.append(block(c, c_out))
            c = c_out
        self.up = nn.ModuleList()
        for c_skip in reversed(ch[:-1]):
            self.up.append(block(c + c_skip, c_skip))
            c = c_skip
        self.head = nn.Conv3d(c, n_channels_out, 1)

    def preprocess(self, vol):
        """(C, Z, Y, X) numpy -> (1, C, Z, Y, X) tensor, each channel scaled to max 1."""
        vol = np.asarray(vol, dtype=np.float32)
        vol = vol / (vol.max(axis=(1, 2, 3), keepdims=True) + 1E-8)
        return torch.from_numpy(vol)[None].to(self.head.weight.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = self.preprocess(x)
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool3d(x, (1, 2, 2), ceil_mode=True)
        for u, s in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=s.shape[2:], mode='nearest')
            x = u(torch.cat([x, s], 1))
        return self.head(x)
