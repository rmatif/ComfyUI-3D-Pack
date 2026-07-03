from typing import *
import importlib
import logging
from trellis.backend_config import get_sparse_backend, get_attention_backend, get_debug_mode, get_available_backends

logger = logging.getLogger(__name__)

BACKEND = get_sparse_backend()
DEBUG = get_debug_mode()
ATTN = get_attention_backend()

if ATTN not in ['xformers', 'flash_attn']:
    available_backends = get_available_backends()
    fallback = next((backend for backend in ['flash_attn', 'xformers'] if available_backends[backend]), None)
    if fallback is not None:
        logger.warning(f"Attention backend {ATTN} not supported for sparse attention. Defaulting to {fallback}")
        ATTN = fallback
    else:
        logger.warning(f"Attention backend {ATTN} not supported for sparse attention, and neither xformers nor flash-attn is installed")

__attributes = {
    'SparseTensor': 'basic',
    'sparse_batch_broadcast': 'basic',
    'sparse_batch_op': 'basic',
    'sparse_cat': 'basic',
    'sparse_unbind': 'basic',
    'SparseGroupNorm': 'norm',
    'SparseLayerNorm': 'norm',
    'SparseGroupNorm32': 'norm',
    'SparseLayerNorm32': 'norm',
    'SparseReLU': 'nonlinearity',
    'SparseSiLU': 'nonlinearity',
    'SparseGELU': 'nonlinearity',
    'SparseActivation': 'nonlinearity',
    'SparseLinear': 'linear',
    'sparse_scaled_dot_product_attention': 'attention',
    'SerializeMode': 'attention',
    'sparse_serialized_scaled_dot_product_self_attention': 'attention',
    'sparse_windowed_scaled_dot_product_self_attention': 'attention',
    'SparseMultiHeadAttention': 'attention',
    'SparseConv3d': 'conv',
    'SparseInverseConv3d': 'conv',
    'SparseDownsample': 'spatial',
    'SparseUpsample': 'spatial',
    'SparseSubdivide' : 'spatial'
}

__submodules = ['transformer']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import *
    from .norm import *
    from .nonlinearity import *
    from .linear import *
    from .attention import *
    from .conv import *
    from .spatial import *
    import transformer
