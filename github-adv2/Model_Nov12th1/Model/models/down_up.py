# physics_factory.py (revised: roll-based Neumann BC everywhere)
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1) MDE block (small kernels, large effective RF)
# ---------------------------
class MDEBlock(nn.Module):
    def __init__(self, c, se_ratio=0.125):
        super().__init__()
        self.dw1 = nn.Conv2d(c, c, 3, padding=1, groups=c, dilation=1)
        self.dw2 = nn.Conv2d(c, c, 3, padding=2, groups=c, dilation=2)
        self.dw3 = nn.Conv2d(c, c, 3, padding=3, groups=c, dilation=3)
        self.pw  = nn.Conv2d(3*c, c, 1)
        # squeeze-excite (very light)
        r = max(8, int(c*se_ratio))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(c, r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(r, c, 1), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.Conv2d(c, 2*c, 1), nn.GELU(),
                                 nn.Conv2d(2*c, c, 1))
        self.n1 = nn.GroupNorm(1, c)
        self.n2 = nn.GroupNorm(1, c)

    def forward(self, x):
        s = x
        x = self.n1(x)
        x = torch.cat([self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.pw(x)
        x = x * self.se(x) + s
        s = x
        x = self.ffn(self.n2(x))
        return s + x

def mde_stage(in_ch, out_ch, depth=2):
    layers = [nn.Conv2d(in_ch, out_ch, 1)]
    for _ in range(depth):
        layers.append(MDEBlock(out_ch))
    return nn.Sequential(*layers)

class MDEDown(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.body = mde_stage(in_ch, out_ch, depth)
    def forward(self, x): return self.body(self.pool(x))

class MDEUp(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.fuse = nn.Conv2d(in_ch, out_ch, 1)
        self.body = mde_stage(out_ch, out_ch, depth)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.body(self.fuse(torch.cat([x2, x1], 1)))




# ----------------------------- factory -----------------------------
DOWN_REGISTRY = {
    "mde":MDEDown, # kwargs: depth
}
UP_REGISTRY = {
    "mde": MDEUp,
}
BLOCK_REGISTRY = {
    "mde": mde_stage,
}


def make_down(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in DOWN_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(DOWN_REGISTRY.keys())}")
    return DOWN_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_up(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in UP_REGISTRY:
        raise ValueError(f"Unknown up type '{name}'. Available: {list(UP_REGISTRY.keys())}")
    return UP_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_block(name: str, in_ch: int, out_ch: int, **kwargs):
    name = name.lower()
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[name](in_ch, out_ch, **kwargs)
