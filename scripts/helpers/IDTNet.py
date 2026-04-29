import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
import cantera as ct


class IDTNet(nn.Module):
    """
    Predict IDT from a simulation from IC's
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, T0, P, phi):
        """
        T0: (N,) in K
        P:  (N,) in atm (or consistent units with training)
        phi: (N,) equivalence ratio
        Returns: IDT in seconds (or whatever units you trained on)
        """
        x = torch.stack([torch.log(P), T0, phi], dim=-1)  # (N, 3)
        ln_idt = self.mlp(x).squeeze(-1)                   # (N,)
        return torch.exp(ln_idt)                           # (N,) IDT

def idt_loss(idt_pred, idt_true):
    log_term = (torch.log(idt_pred) - torch.log(idt_true)) ** 2
    rel_term = ((idt_pred - idt_true) / idt_true) ** 2
    return (log_term + rel_term).mean()