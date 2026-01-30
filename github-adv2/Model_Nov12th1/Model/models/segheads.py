import math, torch
import torch.nn as nn
import torch.nn.functional as F

from .down_up import make_down, make_up, make_block


MODEL_REGISTRY = {}
def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator



# ---------------------------
# ADV 2: MDE-UNet + TRADES-Gate + AT-FiLM
# ---------------------------
def tv(x):
    return (x[...,1:,:]-x[...,:-1,:]).abs().mean() + (x[...,:,1:]-x[...,:, :-1]).abs().mean()

class TRADESGate(nn.Module):
    def __init__(self, ch, reduction=4, eps=0.05, steps=1):
        super().__init__()
        mid = max(ch//reduction, 8)
        self.eps, self.steps = eps, steps
        self.phi = nn.Sequential(nn.Conv2d(2*ch, mid, 1), nn.GELU(), nn.Conv2d(mid, ch, 1))
        # tiny surrogate to get gradients fast
        self.surr = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 1, 1))

    def forward(self, f_m, f_a, y=None):
        z = torch.cat([f_m, f_a], 1)                       # (N,2C,H,W)
        g = torch.sigmoid(self.phi(z))
        fused = g * f_m + (1-g) * f_a

        loss_tv = tv(g)
        loss_trades = fused.new_zeros(())

        if y is not None and self.steps > 0 and self.training:
            z_adv = z.clone().detach().requires_grad_(True)
            for _ in range(self.steps):
                g_adv = torch.sigmoid(self.phi(z_adv))
                f_adv = g_adv * f_m + (1-g_adv) * f_a
                logit_adv = self.surr(f_adv)
                # maximize seg surrogate => ascend on z
                loss_adv = dice_loss(logit_adv, y) + bce_loss(logit_adv, y)
                grad = torch.autograd.grad(loss_adv, z_adv, retain_graph=False, create_graph=False)[0]
                z_adv = (z_adv + self.eps * grad.sign()).detach().requires_grad_(True)
            with torch.no_grad():
                p_clean = self.surr(fused)
                p_adv   = self.surr(g_adv * f_m + (1-g_adv) * f_a)
            loss_trades = kl_div_with_logits(p_clean, p_adv)         # TRADES-style KL

        return fused, g, {"gate_tv": loss_tv, "gate_trades": loss_trades}

class ATFiLM(nn.Module):
    def __init__(self, ch, reduction=4, eps=0.05):
        super().__init__()
        mid = max(ch//reduction, 8)
        self.eps = eps
        self.head = nn.Sequential(nn.Conv2d(ch, mid, 1), nn.GELU(), nn.Conv2d(mid, 2*ch, 1))
        self.surr = nn.Conv2d(ch, 1, 1)  # tiny head for KL consistency

    def forward(self, s_main, a_aux):
        if a_aux.shape[-2:] != s_main.shape[-2:]:
            a_aux = F.interpolate(a_aux, size=s_main.shape[-2:], mode='bilinear', align_corners=False)
        gb = self.head(a_aux)
        C = s_main.size(1)
        gamma, beta = gb[:, :C], gb[:, C:]
        s_clean = (1 + gamma) * s_main + beta

        # adversarial perturb on (gamma,beta)
        cons = s_clean.new_zeros(())
        if self.training:  # 仅训练态做对抗
            with torch.enable_grad():
                gb_adv = gb.clone().detach().requires_grad_(True)
                gamma_a, beta_a = gb_adv[:, :C], gb_adv[:, C:]
                s_adv = (1 + gamma_a) * s_main + beta_a

                # 内环攻击：干净 logits 做 stop-grad
                loss_attack = kl_div_with_logits(self.surr(s_clean).detach(), self.surr(s_adv))
                grad = torch.autograd.grad(loss_attack, gb_adv, retain_graph=False, create_graph=False)[0]

                gb_adv = gb_adv + self.eps * grad.sign()
                # 若上面做了 tanh*0.5，建议 clamp 回 [-0.5, 0.5]
                gb_adv = gb_adv.clamp_(-0.5, 0.5).detach()

                gamma_b, beta_b = gb_adv[:, :C], gb_adv[:, C:]
                s_adv = (1 + gamma_b) * s_main + beta_b

                # 最终一致性：仍然对干净路 stop-grad，更稳
                cons = kl_div_with_logits(self.surr(s_clean).detach(), self.surr(s_adv))

        smooth = tv(gamma) + tv(beta)
        return s_clean, {"film_cons": cons, "film_tv": smooth}

@register_model('unet_adv2')
class UNetSegHead_ATSmall(nn.Module):
    """
    x: (N,15,H,W) -> 0:12 aux, 12:15 main
    y: (N,1,H,W) optional; if provided with compute_loss=True, model returns logits and loss dict.
    """
    def __init__(self, base=32, depth=(1,1,2,1),
                 gate_eps=0.03, gate_steps=1, film_eps=0.03,
                 w_dice=2.0, w_bce=0.5, w_gate_tv=5e-4, w_gate_tr=8e-4,
                 w_film_cons=8e-4, w_film_tv=5e-4):
        super().__init__()
        d1,d2,d3,d4 = depth
        # stems
        self.inc_aux = make_block('mde', 12, base, depth=1)
        self.inc_main = make_block('mde', 3, base, depth=1)

        # pre-encoder adversarial gate
        self.gate = TRADESGate(base, eps=gate_eps, steps=gate_steps)

        # encoders
        self.down1 = make_down('mde', base, base*2, depth=d1)
        self.down2 = make_down('mde', base*2, base*4, depth=d2)
        self.down3 = make_down('mde', base*4, base*8, depth=d3)
        self.down4 = make_down('mde', base*8, base*16, depth=d4)

        self.aux1 = make_down('mde', base, base*2, depth=1)
        self.aux2 = make_down('mde', base*2, base*4, depth=1)
        self.aux3 = make_down('mde', base*4, base*8, depth=1)
        self.aux4 = make_down('mde', base*8, base*16, depth=1)

        self.up1 = make_up('mde', base*16, base*8, depth=2)
        self.up2 = make_up('mde', base*8, base*4, depth=2)
        self.up3 = make_up('mde', base*4, base*2, depth=2)
        self.up4 = make_up('mde', base*2, base, depth=2)


        # adversarial FiLM on each skip
        self.film4 = ATFiLM(base*8,  eps=film_eps)
        self.film3 = ATFiLM(base*4,  eps=film_eps)
        self.film2 = ATFiLM(base*2,  eps=film_eps)
        self.film1 = ATFiLM(base,    eps=film_eps)

        self.outc = nn.Conv2d(base, 1, 1)

        # loss weights
        self.w_dice, self.w_bce = w_dice, w_bce
        self.w_gate_tv, self.w_gate_tr = w_gate_tv, w_gate_tr
        self.w_film_cons, self.w_film_tv = w_film_cons, w_film_tv

    def forward(self, x, y=None, compute_loss=True):
        assert x.ndim==4 and x.size(1)>=15
        xa, xm = x[:, :12], x[:, 12:15]

        f_aux0  = self.inc_aux(xa)
        f_main0 = self.inc_main(xm)

        fused, g, gate_losses = self.gate(f_main0, f_aux0, y if compute_loss else None)

        # main encoder
        e2 = self.down1(fused)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # aux pyramid
        a2 = self.aux1(f_aux0)
        a3 = self.aux2(a2)
        a4 = self.aux3(a3)
        a5 = self.aux4(a4)

        # decoder with AT-FiLM skips
        s4, L4 = self.film4(e4, a4); y_ = self.up1(e5, s4)
        s3, L3 = self.film3(e3, a3); y_ = self.up2(y_, s3)
        s2, L2 = self.film2(e2, a2); y_ = self.up3(y_, s2)
        s1, L1 = self.film1(fused, f_aux0); y_ = self.up4(y_, s1)

        logits = self.outc(y_)

        if not compute_loss or y is None:
            return logits, g, {"gate_name":'adv'}

        # gate regularizers
        L_gate = self.w_gate_tv * gate_losses["gate_tv"] + self.w_gate_tr * gate_losses["gate_trades"]
        # skip regularizers
        L_film = self.w_film_cons * (L1["film_cons"] + L2["film_cons"] + L3["film_cons"] + L4["film_cons"]) \
               + self.w_film_tv   * (L1["film_tv"]   + L2["film_tv"]   + L3["film_tv"]   + L4["film_tv"])

        L_total = L_gate + L_film
        return logits, g, {"total": L_total, "gate_name":'adv'}



