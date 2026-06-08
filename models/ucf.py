import torch
import torch.nn as nn

from .backbones import Xception, load_xception_pretrained


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim), nn.LeakyReLU(inplace=True), nn.Linear(hidden_dim, out_f))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        feat = self.pool(x).view(x.size(0), -1)
        return self.dropout(self.mlp(feat)), feat


class UCFDetector(nn.Module):
    """UCF dual-encoder architecture with binary common-feature head.

    DeepfakeBench's full UCF loss expects paired real/fake batches and manipulation
    labels. This project's flat binary dataset trains the shared/common head so it
    remains compatible with image, label, image_path samples.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        backbone_cfg = model_cfg.get("backbone_config", {"mode": "adjust_channel", "num_classes": 2})
        self.encoder_feat_dim = int(model_cfg.get("encoder_feat_dim", 512))
        half_dim = self.encoder_feat_dim // 2

        self.encoder_f = Xception(backbone_cfg)
        self.encoder_c = Xception(backbone_cfg)
        pretrained = model_cfg.get("pretrained", "")
        load_xception_pretrained(self.encoder_f, pretrained)
        load_xception_pretrained(self.encoder_c, pretrained)

        self.block_spe = Conv2d1x1(self.encoder_feat_dim, half_dim, half_dim)
        self.block_sha = Conv2d1x1(self.encoder_feat_dim, half_dim, half_dim)
        self.head_spe = Head(half_dim, self.encoder_feat_dim, int(model_cfg.get("specific_classes", 5)))
        self.head_sha = Head(half_dim, self.encoder_feat_dim, 2)
        self.shuffle_common_features = bool(model_cfg.get("shuffle_common_features", False))

    def forward(self, images: torch.Tensor):
        forgery_features = self.encoder_f.features(images)
        f_spe = self.block_spe(forgery_features)
        f_share = self.block_sha(forgery_features)

        if self.training and self.shuffle_common_features and f_share.size(0) > 1:
            index = torch.randperm(f_share.size(0), device=f_share.device)
            f_share = f_share[index]

        logits, sha_feat = self.head_sha(f_share)
        logits_spe, spe_feat = self.head_spe(f_spe)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "features": sha_feat,
            "logits_spe": logits_spe,
            "features_spe": spe_feat,
        }
