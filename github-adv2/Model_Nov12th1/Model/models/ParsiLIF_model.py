import torch
from .base_model import BaseModel
from . import networks
from .networks import get_optimizer
from .segheads import MODEL_REGISTRY
import itertools
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch
from typing import Dict, Tuple, Any, Optional
import torch


def masks_from_overlay_torch(
    overlay: torch.Tensor,              # [B,3,H,W], float or uint8, 0–1 or 0–255
    red_thresh=(150, 40, 40),
    blue_thresh=(40, 40, 150)
):
    """
    Returns:
      pos      : [B,1,H,W]  (red-only mask)
      semantic : [B,1,H,W]  (red OR blue)
    """
    assert overlay.dim() == 4 and overlay.size(1) == 3, "overlay must be [B,3,H,W]"

    # Normalize to 0–255 if needed
    if overlay.dtype.is_floating_point:
        # if already in 0–255 keep; else scale 0–1 → 0–255
        needs_scale = overlay.max() <= 1.5
        img255 = overlay * (255.0 if needs_scale else 1.0)
    else:
        img255 = overlay.float()

    R, G, B = img255[:, 0], img255[:, 1], img255[:, 2]

    # Thresholds (broadcast scalars)
    rt0, rt1, rt2 = red_thresh
    bt0, bt1, bt2 = blue_thresh

    red_mask  = (R >= rt0) & (G <= rt1) & (B <= rt2)
    blue_mask = (B >= bt2) & (R <= bt0) & (G <= bt1)

    pos = red_mask.unsqueeze(1).float()                 # [B,1,H,W]
    semantic = (red_mask | blue_mask).unsqueeze(1).float()

    return pos, semantic


class ParsiLIFModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # ------- defaults / flags -------
        if not hasattr(opt, 'seg_joint_backprop'):
            opt.seg_joint_backprop = False   # True to let seg loss flow into Gs
        opt.lr_s = 6e-4
        opt.lambda_seg_bce = 0.5
        opt.lambda_seg_dice = 1-opt.lambda_seg_bce

        self.seg_weights      = opt.seg_weights
        self.loss_G_weights   = opt.loss_G_weights
        self.loss_D_weights   = opt.loss_D_weights

        # gpu_ids handling unchanged ...
        if not opt.is_train:
            self.gpu_ids = []
        else:
            self.gpu_ids = opt.gpu_ids

        # ------- losses & visuals to log -------
        self.loss_names = []
        self.visual_names = ['real_A']

        for i in range(1, self.opt.modalities_no + 1 + 1):
            self.loss_names.extend(['G_GAN_' + str(i), 'G_L1_' + str(i), 'D_real_' + str(i), 'D_fake_' + str(i)])
            # keep per-modality visuals
            self.visual_names.extend(['fake_B_' + str(i), 'real_B_' + str(i)])

        # add a single segmentation visual/output pair
        self.visual_names.extend(['overlay'])

        if self.is_train:
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend(['G' + str(i), 'D' + str(i)])
        else:
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend(['G' + str(i)])

        # include segmentation head in save/load lists
        self.model_names.extend(['S'])

        # ------- define nets -------
        if isinstance(opt.netG, str):
            opt.netG = [opt.netG] * 4

        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[0], opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[1], opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[2], opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)
        self.netG4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG[3], opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding)

        # segmentation head (produces logits) print(MODEL_REGISTRY)
        self.seghead_name = opt.seghead
        self.netS  = MODEL_REGISTRY[self.seghead_name]().to(self.device)

        if hasattr(self, 'gpu_ids') and isinstance(self.gpu_ids, (list, tuple)) and len(self.gpu_ids) > 0:
            self.netS = torch.nn.DataParallel(self.netS, device_ids=self.gpu_ids)

        if self.is_train:
            # ------- Discriminators -------
            self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD4 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # ------- criteria -------
            self.criterionGAN_BCE   = networks.GANLoss('vanilla').to(self.device)
            #self.criterionGAN_lsgan = networks.GANLoss('lsgan').to(self.device)
            self.criterionSmoothL1  = torch.nn.SmoothL1Loss()
            self.criterionVGG       = networks.VGGLoss().to(self.device)
            self.criterionSegBCE    = torch.nn.BCEWithLogitsLoss()

            # ------- optimizers -------
            params_G = list(self.netG1.parameters()) + list(self.netG2.parameters()) + \
                       list(self.netG3.parameters()) + list(self.netG4.parameters())
            try:
                self.optimizer_G = get_optimizer(opt.optimizer)(params_G, lr=opt.lr_g, betas=(opt.beta1, 0.999))
            except:
                self.optimizer_G = get_optimizer(opt.optimizer)(params_G, lr=opt.lr_g)

            params_D = list(self.netD1.parameters()) + list(self.netD2.parameters()) + \
                       list(self.netD3.parameters()) + list(self.netD4.parameters())
            try:
                self.optimizer_D = get_optimizer(opt.optimizer)(params_D, lr=opt.lr_d, betas=(opt.beta1, 0.999))
            except:
                self.optimizer_D = get_optimizer(opt.optimizer)(params_D, lr=opt.lr_d)

            try:
                self.optimizer_S = get_optimizer(opt.optimizer)(self.netS.parameters(), lr=opt.lr_s, betas=(opt.beta1, 0.999))
            except:
                self.optimizer_S = get_optimizer(opt.optimizer)(self.netS.parameters(), lr=opt.lr_s)

            self.optimizers += [self.optimizer_G, self.optimizer_D, self.optimizer_S]


    @staticmethod
    def dice_loss_with_logits(logits, targets, eps: float = 1e-6):
        # targets: {0,1}, logits: raw outputs
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))  # reduce over spatial & channel dims
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()


    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_A_seg = input['A_seg'].to(self.device)
        self.real_B_array = input['B']
        self.real_B_1 = self.real_B_array[0].to(self.device)
        self.real_B_2 = self.real_B_array[1].to(self.device)
        self.real_B_3 = self.real_B_array[2].to(self.device)
        self.real_B_4 = self.real_B_array[3].to(self.device)

        # Overlay → masks (all on GPU, no CPU roundtrip)
        self.overlay = self.real_B_array[4].to(self.device)  # [B,3,H,W]
        pos, semantic = masks_from_overlay_torch(self.overlay)  # [B,1,H,W], [B,1,H,W]

        self.real_B_5 = semantic
        self.real_B_5_vis = semantic.float()
        self.real_B_5_pos = pos

        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B_1 = self.netG1(self.real_A)
        self.fake_B_2 = self.netG2(self.real_A)
        self.fake_B_3 = self.netG3(self.real_A)
        self.fake_B_4 = self.netG4(self.real_A)

        self.fake_B_5_input = torch.cat([
            self.fake_B_1,
            self.fake_B_2,
            self.fake_B_3,
            self.fake_B_4,
            self.real_A_seg
        ], dim=1)

        # logits for loss; sigmoid version for visualization
        self.fake_B_5_logit, self.g, self.info = self.unpack_outputs(self.netS(self.fake_B_5_input))
        self.fake_B_5       = torch.sigmoid(self.fake_B_5_logit)
        self.fake_B_5_vis = (self.fake_B_5 > 0.5).float()

    def backward_S(self):
        with torch.enable_grad():
            if 'adv' in self.seghead_name:
                logits, g, info = self.unpack_outputs(self.netS(self.fake_B_5_input.detach(), self.real_B_5.detach()))

            else:
                logits, g, info = self.unpack_outputs(self.netS(self.fake_B_5_input.detach()))
        self.loss_gate, _ = self.gate_constraints_router((logits, g, info),)

        loss_bce  = self.criterionSegBCE(logits, self.real_B_5)
        loss_dice = self.dice_loss_with_logits(logits, self.real_B_5)

        self.loss_S_BCE  = loss_bce * self.opt.lambda_seg_bce
        self.loss_S_DICE = loss_dice * self.opt.lambda_seg_dice
        self.loss_S      = self.loss_S_BCE + self.loss_S_DICE + self.loss_gate

        if 'S_BCE' not in self.loss_names:
            self.loss_names.extend(['S_BCE', 'S_DICE', 'S'])
        self.loss_S.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1),
                              1)  # Conditional GANs; feed IHC input and Hematoxtlin output to the discriminator
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2),
                              1)  # Conditional GANs; feed IHC input and mpIF DAPI output to the discriminator
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3),
                              1)  # Conditional GANs; feed IHC input and mpIF Lap2 output to the discriminator
        fake_AB_4 = torch.cat((self.real_A, self.fake_B_4),
                              1)  # Conditional GANs; feed IHC input and mpIF Ki67 output to the discriminator

        pred_fake_1 = self.netD1(fake_AB_1.detach())
        pred_fake_2 = self.netD2(fake_AB_2.detach())
        pred_fake_3 = self.netD3(fake_AB_3.detach())
        pred_fake_4 = self.netD4(fake_AB_4.detach())

        self.loss_D_fake_1 = self.criterionGAN_BCE(pred_fake_1, False)
        self.loss_D_fake_2 = self.criterionGAN_BCE(pred_fake_2, False)
        self.loss_D_fake_3 = self.criterionGAN_BCE(pred_fake_3, False)
        self.loss_D_fake_4 = self.criterionGAN_BCE(pred_fake_4, False)

        real_AB_1 = torch.cat((self.real_A, self.real_B_1), 1)
        real_AB_2 = torch.cat((self.real_A, self.real_B_2), 1)
        real_AB_3 = torch.cat((self.real_A, self.real_B_3), 1)
        real_AB_4 = torch.cat((self.real_A, self.real_B_4), 1)

        pred_real_1 = self.netD1(real_AB_1)
        pred_real_2 = self.netD2(real_AB_2)
        pred_real_3 = self.netD3(real_AB_3)
        pred_real_4 = self.netD4(real_AB_4)

        self.loss_D_real_1 = self.criterionGAN_BCE(pred_real_1, True)
        self.loss_D_real_2 = self.criterionGAN_BCE(pred_real_2, True)
        self.loss_D_real_3 = self.criterionGAN_BCE(pred_real_3, True)
        self.loss_D_real_4 = self.criterionGAN_BCE(pred_real_4, True)

        # combine losses and calculate gradients
        self.loss_D = (self.loss_D_fake_1 + self.loss_D_real_1) * 0.5 * self.loss_D_weights[0] + \
                      (self.loss_D_fake_2 + self.loss_D_real_2) * 0.5 * self.loss_D_weights[1] + \
                      (self.loss_D_fake_3 + self.loss_D_real_3) * 0.5 * self.loss_D_weights[2] + \
                      (self.loss_D_fake_4 + self.loss_D_real_4) * 0.5 * self.loss_D_weights[3]

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)
        fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)

        pred_fake_1 = self.netD1(fake_AB_1)
        pred_fake_2 = self.netD2(fake_AB_2)
        pred_fake_3 = self.netD3(fake_AB_3)
        pred_fake_4 = self.netD4(fake_AB_4)

        self.loss_G_GAN_1 = self.criterionGAN_BCE(pred_fake_1, True)
        self.loss_G_GAN_2 = self.criterionGAN_BCE(pred_fake_2, True)
        self.loss_G_GAN_3 = self.criterionGAN_BCE(pred_fake_3, True)
        self.loss_G_GAN_4 = self.criterionGAN_BCE(pred_fake_4, True)

        # Second, G(A) = B
        self.loss_G_L1_1 = self.criterionSmoothL1(self.fake_B_1, self.real_B_1) * self.opt.lambda_L1
        self.loss_G_L1_2 = self.criterionSmoothL1(self.fake_B_2, self.real_B_2) * self.opt.lambda_L1
        self.loss_G_L1_3 = self.criterionSmoothL1(self.fake_B_3, self.real_B_3) * self.opt.lambda_L1
        self.loss_G_L1_4 = self.criterionSmoothL1(self.fake_B_4, self.real_B_4) * self.opt.lambda_L1

        self.loss_G_VGG_1 = self.criterionVGG(self.fake_B_1, self.real_B_1) * self.opt.lambda_feat
        self.loss_G_VGG_2 = self.criterionVGG(self.fake_B_2, self.real_B_2) * self.opt.lambda_feat
        self.loss_G_VGG_3 = self.criterionVGG(self.fake_B_3, self.real_B_3) * self.opt.lambda_feat
        self.loss_G_VGG_4 = self.criterionVGG(self.fake_B_4, self.real_B_4) * self.opt.lambda_feat

        self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1 + self.loss_G_VGG_1) * self.loss_G_weights[0] + \
                      (self.loss_G_GAN_2 + self.loss_G_L1_2 + self.loss_G_VGG_2) * self.loss_G_weights[1] + \
                      (self.loss_G_GAN_3 + self.loss_G_L1_3 + self.loss_G_VGG_3) * self.loss_G_weights[2] + \
                      (self.loss_G_GAN_4 + self.loss_G_L1_4 + self.loss_G_VGG_4) * self.loss_G_weights[3]

        self.loss_G.backward()

    def update(self):
        self.forward()

        # --- D ---
        self.set_requires_grad(self.netD1, True)
        self.set_requires_grad(self.netD2, True)
        self.set_requires_grad(self.netD3, True)
        self.set_requires_grad(self.netD4, True)

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # --- G ---
        self.set_requires_grad(self.netD1, False)
        self.set_requires_grad(self.netD2, False)
        self.set_requires_grad(self.netD3, False)
        self.set_requires_grad(self.netD4, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # --- S (segmentation) ---
        self.optimizer_S.zero_grad()
        self.backward_S()
        self.optimizer_S.step()

    def calculate_losses(self):
        self.forward()

        # D (no step)
        self.set_requires_grad(self.netD1, True)
        self.set_requires_grad(self.netD2, True)
        self.set_requires_grad(self.netD3, True)
        self.set_requires_grad(self.netD4, True)

        self.optimizer_D.zero_grad()
        self.backward_D()  # accumulates self.loss_D_*

        # G (no step)
        self.set_requires_grad(self.netD1, False)
        self.set_requires_grad(self.netD2, False)
        self.set_requires_grad(self.netD3, False)
        self.set_requires_grad(self.netD4, False)

        self.optimizer_G.zero_grad()
        self.backward_G()  # accumulates self.loss_G_*

        # S (no step) — compute but don't step
        self.optimizer_S.zero_grad()
        # Run the same computation but stop before stepping
        if getattr(self.opt, 'seg_joint_backprop', False):
            logits = self.fake_B_5_logit
        else:
            logits, _, _ = self.unpack_outputs(self.netS(self.fake_B_5_input.detach()))

            #logits = self.netS(self.fake_B_5_input.detach())

        self.loss_S_BCE  = self.criterionSegBCE(logits, self.real_B_5) * self.opt.lambda_seg_bce
        self.loss_S_DICE = self.dice_loss_with_logits(logits, self.real_B_5) * self.opt.lambda_seg_dice
        self.loss_S      = self.loss_S_BCE + self.loss_S_DICE
        # no backward here in validation

    def unpack_outputs(self, out):
        if isinstance(out, (tuple, list)):
            logit = out[0]
            g = out[1] if len(out) > 1 else None
            info = out[2] if len(out) > 2 else None
        else:
            logit, g, info = out, None, None
        return logit, g, info

    def gate_constraints_router(self, outputs):
        # ---- 1 ----
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            g = outputs[1] if len(outputs) > 1 else None
            info = outputs[2] if len(outputs) > 2 else None
        else:
            logits, g, info = outputs, None, None

        # ---- 2) gate_type ----
        def get_gate_type(info):
            if isinstance(info, dict):
                for k in ('gate_name', 'name', 'gate', 'type', 'kind'):
                    if k in info:
                        return str(info[k]).lower()
                if len(info):
                    return str(list(info.values())[-1]).lower()
                raise ValueError("info is an empty dict; cannot infer gate name.")
            elif isinstance(info, (list, tuple)):
                if len(info) == 0:
                    raise ValueError("info is an empty sequence; cannot infer gate name.")
                return str(info[-1]).lower()
            else:
                if hasattr(info, 'gate_name'):
                    return str(info.gate_name).lower()
                return str(info).lower()

        gate_type = get_gate_type(info)

        # ---- 3) ----
        def ig(key, default=None):
            if isinstance(info, dict):
                return info.get(key, default)
            return getattr(info, key, default)
        eps = 1e-6
        terms = {}
        total = 0.0
        # ---- 4) three gate ----
        if gate_type in ('beta', 'betagate'):
            assert g is not None, "BetaGate 需要传入 g"
            p = torch.clamp(g, eps, 1 - eps)
            # 熵正则（越大越好，这里以负号方式加入损失）
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
            # Beta 先验（对全局均值轻回归到 α/(α+β)）
            beta_alpha = float(ig('beta_alpha', 1.2))
            beta_beta = float(ig('beta_beta', 1.2))
            target = p.new_tensor(beta_alpha / (beta_alpha + beta_beta))
            beta_prior = F.mse_loss(p.mean(), target)

            lambda_entropy = float(ig('lambda_entropy', 1e-3))
            lambda_beta = float(ig('lambda_beta', 1e-3))

            total = total + lambda_entropy * entropy + lambda_beta * beta_prior
            terms.update({'entropy': entropy, 'beta_prior': beta_prior, 'gate_type': torch.tensor(0)})

        elif gate_type in ('dirichlet', 'evidence', 'ed-gate', 'edgate'):
            e_main = ig('e_main', None)
            e_aux = ig('e_aux', None)
            if e_main is None or e_aux is None:
                raise ValueError("EvidenceGate 需要在 info 中提供 e_main 与 e_aux。")

            em = torch.clamp(e_main, min=eps)
            ea = torch.clamp(e_aux, min=eps)

            g_bar = em.mean() / (em.mean() + ea.mean())
            dir_alpha = float(ig('dir_alpha', 1.1))
            dir_beta = float(ig('dir_beta', 1.1))
            dirichlet_cal = F.mse_loss(g_bar, g_bar.new_tensor(dir_alpha / (dir_alpha + dir_beta)))
            evidence_l1 = (em.mean() + ea.mean())

            lambda_dirichlet_cal = float(ig('lambda_dirichlet_cal', 1e-3))
            lambda_evidence_l1 = float(ig('lambda_evidence_l1', 1e-6))

            total = total + lambda_dirichlet_cal * dirichlet_cal + lambda_evidence_l1 * evidence_l1
            terms.update({'dirichlet_cal': dirichlet_cal, 'evidence_l1': evidence_l1, 'gate_type': torch.tensor(1)})

        elif gate_type in ('precision', 'pwg', 'precisiongate'):
            logv_m = ig('logv_m', None)
            logv_a = ig('logv_a', None)
            resid_m = ig('resid_m', None)
            resid_a = ig('resid_a', None)
            if logv_m is None or logv_a is None:
                raise ValueError("PrecisionGate 需要在 info 中提供 logv_m 与 logv_a。")

            # 高斯 NLL：0.5*(exp(-logv)*res^2 + logv)
            nll_m = 0.5 * (torch.exp(-logv_m) * (resid_m ** 2) + logv_m).mean()
            nll_a = 0.5 * (torch.exp(-logv_a) * (resid_a ** 2) + logv_a).mean()

            lambda_nll_m = float(ig('lambda_nll_m', 1e-3))
            lambda_nll_a = float(ig('lambda_nll_a', 1e-3))

            total = total + lambda_nll_m * nll_m + lambda_nll_a * nll_a
            terms.update({'nll_main': nll_m, 'nll_aux': nll_a, 'gate_type': torch.tensor(2)})

        elif gate_type in ('adv', 'al-gate', 'algate', 'adversarial', 'robust'):
            total = ig('total', None)
            terms.update({
                'total': total,
                'gate_type': torch.tensor(3)
            })

        elif gate_type in ('ssi', 'subband skip injection'):
            total = ig('ot_cost', None) * 1e-3
            terms.update({
                'total': total,
                'gate_type': torch.tensor(3)
            })

        else:
            return 0, None

        return total, terms

