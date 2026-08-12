import torch
import torch.nn.functional as F

def nnpu_loss(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    train_prevalence: float,
    prior_multiplier: float = 1.0
) -> torch.Tensor:
    """
    Non-Negative Positive-Unlabeled (nnPU) Loss for a single label.
    Args:
        positive_logits: Logits for observed_positive pairs.
        unlabeled_logits: Logits for sampled_unlabeled controls.
        train_prevalence: Observed train prevalence for this label.
        prior_multiplier: Tunable multiplier [1.0, 2.0, 4.0].
    """
    if len(positive_logits) == 0:
        # If no positives in batch for this label, return 0 risk
        return torch.tensor(0.0, device=unlabeled_logits.device, requires_grad=True)

    positive_loss = F.softplus(-positive_logits)
    negative_loss = F.softplus(unlabeled_logits)
    positive_as_negative_loss = F.softplus(positive_logits)
    
    pi_l = torch.clamp(torch.tensor(train_prevalence * prior_multiplier), 1e-6, 0.5).to(positive_logits.device)
    
    positive_risk = pi_l * torch.mean(positive_loss)
    
    if len(unlabeled_logits) > 0:
        negative_risk = torch.mean(negative_loss) - pi_l * torch.mean(positive_as_negative_loss)
    else:
        negative_risk = torch.tensor(0.0, device=positive_logits.device)
        
    # Non-negative correction
    nnpu_risk = positive_risk + torch.clamp(negative_risk, min=0.0)
    
    return nnpu_risk
