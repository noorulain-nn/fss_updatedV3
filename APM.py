import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import ImprovedFPNDecoder


class MemoryModuleFSS(nn.Module):
    """
    APM memory module for binary FSS.
    Slot 0 = background, Slot 1 = foreground.
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.num_classes = 2
        self.feature_dim = feature_dim
        self.register_buffer('memory', torch.zeros(2, feature_dim))
        self.initialized = [False, False]

    def extract_prototype(self, features: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Masked average pooling over foreground pixels.
        features : (1, C, Hf, Wf)
        mask     : (H, W)  binary  — 1 = foreground pixels to pool
        returns  : (C,) L2-normalised prototype
        """
        mask_ds = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=features.shape[-2:],
            mode='nearest'
        )                                               # (1,1,Hf,Wf)
        masked   = features * mask_ds
        n_pixels = mask_ds.sum() + 1e-6
        proto    = masked.sum(dim=[2, 3]) / n_pixels   # (1,C)
        return F.normalize(proto.squeeze(0), p=2, dim=0)  # (C,)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Pixel-wise cosine similarity to both prototypes.
        query_features : (B, C, Hf, Wf)
        returns        : (B, 2, Hf, Wf)
        """
        B, C, H, W = query_features.shape
        q_norm  = F.normalize(query_features, p=2, dim=1)   # (B,C,H,W)
        m_norm  = F.normalize(self.memory,    p=2, dim=1)   # (2,C)
        q_flat  = q_norm.permute(0, 2, 3, 1).reshape(B, H * W, C)
        sim     = torch.matmul(q_flat, m_norm.t())           # (B,HW,2)
        return sim.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # (B,2,H,W)

    def update_memory(self, features: torch.Tensor,
                      mask: torch.Tensor,
                      class_label: int) -> None:
        """
        Adaptive EMA update with re-normalisation.
        Same order as APM paper: called after optimizer.step().
        """
        with torch.no_grad():
            new_proto = self.extract_prototype(
                features, mask.to(features.device))

            if not self.initialized[class_label]:
                self.memory[class_label] = new_proto
                self.initialized[class_label] = True
                return

            sim   = F.cosine_similarity(
                new_proto.unsqueeze(0),
                self.memory[class_label].unsqueeze(0),
                dim=1
            ).item()
            alpha   = 1.0 - sim
            blended = ((1.0 - alpha) * self.memory[class_label]
                       + alpha * new_proto)
            # Re-normalise: convex combination drifts off unit sphere
            self.memory[class_label] = F.normalize(blended, p=2, dim=0)


class SegAPM(nn.Module):
    """
    APM-FSS with multiplicative memory gating.

    WHY MULTIPLICATIVE INSTEAD OF ADDITIVE
    ---------------------------------------
    Additive prior (V2):
        seg_logits = decoder_out + sigmoid(sim_map)
    The decoder can produce high foreground scores from learned
    base-class appearance patterns alone. The prior is just a small
    correction. At test time, the prior is weak (K shots) but the
    decoder's base-class patterns are also wrong (novel class) — both
    fail together.

    Multiplicative gate (this version):
        fg_logit = decoder_fg_logit * sigmoid(fg_sim_map)
    The decoder can ONLY produce a high foreground score where the
    memory prototype also says "this looks like foreground."
    Decoder output is suppressed wherever prototype similarity is low.
    This mirrors APM FSIC exactly: memory decides WHAT, decoder
    decides WHERE precisely within the agreed region.

    At training time: memory is rich → gates are confident → decoder
    learns to spatially refine confident gates.
    At test time: memory comes from K support shots → gates reflect
    the K-shot prototype → decoder applies the same learned refinement.
    The mechanism is the same in both regimes.

    PRIOR DROPOUT
    -------------
    With probability prior_dropout_p (default 0.3), during training
    the gate is replaced with all-ones (no gating). This prevents the
    decoder from becoming dependent on training-time gate quality and
    forces partial robustness to sparse priors.
    """

    def __init__(self, backbone, num_classes, feature_dim,
                 output_size=(473, 473),
                 prior_dropout_p: float = 0.3):
        super().__init__()
        self.backbone        = backbone
        self.memory_module   = MemoryModuleFSS(
            num_classes=2, feature_dim=feature_dim)
        self.output_size     = output_size
        self.prior_dropout_p = prior_dropout_p
        self.decoder         = ImprovedFPNDecoder()

        # Projection layers: reduce 2048→256 and 1024→256
        self.proj4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def encode(self, imgs: torch.Tensor):
        """
        Returns (feats4_reduced, feats3_reduced, feats4_raw).
        feats4_raw is used by the memory module (full 2048 channels).
        feats4_reduced / feats3_reduced feed the FPN decoder.
        """
        x = self.backbone.conv1(imgs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feats3 = self.backbone.layer3(x)
        feats4 = self.backbone.layer4(feats3)
        return self.proj4(feats4), self.proj3(feats3), feats4

    def forward(self, imgs: torch.Tensor):
        feats4_reduced, feats3_reduced, feats4_raw = self.encode(imgs)

        # ── Memory similarity map ────────────────────────────────
        # (B, 2, Hf, Wf) — channel 0 = bg sim, channel 1 = fg sim
        similarity_map = self.memory_module(feats4_raw)

        # ── Upsample foreground gate to output resolution ────────
        fg_gate = torch.sigmoid(
            similarity_map[:, 1:2])              # (B,1,Hf,Wf) ∈ [0,1]
        fg_gate = F.interpolate(
            fg_gate,
            size=self.output_size,
            mode='bilinear',
            align_corners=True
        )                                         # (B,1,473,473)

        # ── Prior dropout — training only ────────────────────────
        # With probability prior_dropout_p, replace gate with all-ones.
        # Forces decoder to be partially robust to weak priors.
        if self.training and torch.rand(1).item() < self.prior_dropout_p:
            fg_gate = torch.ones_like(fg_gate)

        # ── Decoder prediction ───────────────────────────────────
        seg_logits = self.decoder(
            feats4_reduced, feats3_reduced)       # (B,2,473,473)

        # ── Multiplicative gate on foreground channel ────────────
        # MECHANISM (mirrors APM FSIC):
        #   bg channel  : unchanged — background is predicted freely
        #   fg channel  : multiplied by fg_gate ∈ [0,1]
        #                 high score only where prototype agrees
        # The decoder cannot fire "foreground" where the K-shot
        # prototype says "this is not the target class."
        seg_logits_bg = seg_logits[:, 0:1]                 # (B,1,H,W)
        seg_logits_fg = seg_logits[:, 1:2] * fg_gate       # (B,1,H,W)
        seg_logits    = torch.cat(
            [seg_logits_bg, seg_logits_fg], dim=1)         # (B,2,H,W)

        return seg_logits, feats4_raw, similarity_map