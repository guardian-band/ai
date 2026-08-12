import torch
from src.training.losses import nnpu_loss

def test_nnpu_loss_empty_positive():
    pos = torch.tensor([], requires_grad=True)
    neg = torch.tensor([1.0, -1.0])
    loss = nnpu_loss(pos, neg, 0.1)
    
    assert loss.item() == 0.0

def test_nnpu_loss_computation():
    pos = torch.tensor([2.0, 1.0])
    neg = torch.tensor([-2.0, -1.0, 0.0])
    
    # Train prevalence = 0.5, multiplier = 1.0
    loss = nnpu_loss(pos, neg, 0.5, 1.0)
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0 # non-negative

def test_nnpu_loss_empty_negative():
    pos = torch.tensor([2.0, 1.0])
    neg = torch.tensor([])
    
    loss = nnpu_loss(pos, neg, 0.5, 1.0)
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
