#sage_attn.py
import os
from typing import Optional
import torch
import torch.nn.functional as F
from sageattention import sageattn
import math

__all__ = ['SageAttention', 'sage_attention']

_ORIGINAL_SDPA = F.scaled_dot_product_attention


def enable_sage_attention():
    """
    Keep SageAttention opt-in for local Stable3DGen call sites without replacing
    PyTorch's process-global scaled_dot_product_attention.
    """
    return True

def disable_sage_attention():
    """
    Restore PyTorch's original scaled_dot_product_attention function.
    """
    F.scaled_dot_product_attention = _ORIGINAL_SDPA
    return True
