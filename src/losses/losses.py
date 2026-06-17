import torch
import torch.nn as nn
from typing import Any

class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets=targets.float().view_as(logits)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        sum_of_two_set_norm = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.epsilon)/ (sum_of_two_set_norm + self.epsilon)
        dice_loss = 1.0 - dice_score.mean()
        return dice_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float =2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma

        focal_loss = alpha_factor * modulating_factor * bce_loss
        return focal_loss.mean()

class HybridSegmentationLoss(nn.Module):
    def __init__(self, base_loss: str = "focal", base_weight: float = 0.5, epsilon: float = 1e-6, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.base_weight = base_weight
        self.dice = DiceLoss(epsilon=epsilon)
        
        match base_loss.lower():
            case "bce":
                self.base = nn.BCEWithLogitsLoss()
            case "focal":
                self.base = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            case _:
                raise ValueError(f"지원하지 않는 Base Loss 입니다: {base_loss}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_l = self.base(logits, targets)
        dice_l = self.dice(logits, targets)
        return (self.base_weight * base_l) + ((1.0 - self.base_weight) * dice_l)


def get_loss_fn(cfg_dict: dict[str, Any]) -> nn.Module:
    loss_name = cfg_dict.get("name", "hybrid")
    params = cfg_dict.get("params", {})

    match loss_name.lower():
        case "dice":
            return DiceLoss(**params)
        case "focal":
            return FocalLoss(**params)
        case "bce":
            return nn.BCEWithLogitsLoss(**params)
        case "hybrid":
            return HybridSegmentationLoss(**params)
        case unhandled:
            raise NotImplementedError(f"정의되지 않은 Loss입니다: {unhandled}")



