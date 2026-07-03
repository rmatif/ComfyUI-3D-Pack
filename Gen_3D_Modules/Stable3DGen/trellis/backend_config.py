# trellis/backend_config.py
from typing import *
import os
import logging
import importlib

# Global variables
BACKEND = 'spconv'  # Default sparse backend
DEBUG = False       # Debug mode flag
ATTN = None # Selected during module init from env or available backends
SPCONV_ALGO = 'implicit_gemm'  # Default algorithm
VALID_ATTN_BACKENDS = ['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']
DEFAULT_ATTN_BACKENDS = ['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']

def get_spconv_algo() -> str:
    """Get current spconv algorithm."""
    global SPCONV_ALGO
    return SPCONV_ALGO

def set_spconv_algo(algo: Literal['implicit_gemm', 'native', 'auto']) -> bool:
    """Set spconv algorithm with validation."""
    global SPCONV_ALGO
    
    if algo not in ['implicit_gemm', 'native', 'auto']:
        logger.warning(f"Invalid spconv algorithm: {algo}. Must be 'implicit_gemm', 'native', or 'auto'")
        return False
        
    SPCONV_ALGO = algo
    os.environ['SPCONV_ALGO'] = algo
    logger.info(f"Set spconv algorithm to: {algo}")
    return True

logger = logging.getLogger(__name__)

def _try_import_xformers() -> bool:
    try:
        import xformers.ops
        return True
    except ImportError:
        return False

def _try_import_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def _try_import_sageattention() -> bool:
    try:
        import sageattention
        return True
    except ImportError:
        return False

def _try_import_spconv() -> bool:
    try:
        import spconv
        return True
    except ImportError:
        return False

def _try_import_torchsparse() -> bool:
    try:
        import torchsparse
        return True
    except ImportError:
        return False

def get_available_backends() -> Dict[str, bool]:
    """Return dict of available attention backends and their status"""
    return {
        'xformers': _try_import_xformers(),
        'flash_attn': _try_import_flash_attn(),
        'sage': _try_import_sageattention(),
        'naive': True,
        'sdpa': True  # Always available with PyTorch >= 2.0
    }

def _is_attention_backend_available(attn: str, available_backends: Dict[str, bool] = None) -> bool:
    if attn not in VALID_ATTN_BACKENDS:
        return False

    if available_backends is None:
        available_backends = get_available_backends()

    return available_backends.get(attn, False)

def _select_attention_backend(candidates: List[str] = None) -> str:
    if candidates is None:
        candidates = DEFAULT_ATTN_BACKENDS

    available_backends = get_available_backends()
    for candidate in candidates:
        if _is_attention_backend_available(candidate, available_backends):
            return candidate

    return 'naive'

def _sync_attention_env(attn: str):
    os.environ['SPARSE_ATTN_BACKEND'] = attn
    os.environ['ATTN_BACKEND'] = attn

def get_available_sparse_backends() -> Dict[str, bool]:
    """Return dict of available sparse backends and their status"""
    return {
        'spconv': _try_import_spconv(),
        'torchsparse': _try_import_torchsparse()
    }

def get_attention_backend() -> str:
    """Get current attention backend"""
    global ATTN
    return ATTN

def get_sparse_backend() -> str:
    """Get current sparse backend"""
    global BACKEND
    return BACKEND

def get_debug_mode() -> bool:
    """Get current debug mode status"""
    global DEBUG
    return DEBUG

def __from_env():
    """Initialize settings from environment variables"""
    global BACKEND
    global DEBUG
    global ATTN
    
    env_sparse_backend = os.environ.get('SPARSE_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn = os.environ.get('SPARSE_ATTN_BACKEND') or os.environ.get('ATTN_BACKEND')
    
    if env_sparse_backend is not None and env_sparse_backend in ['spconv', 'torchsparse']:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn is not None:
        env_sparse_attn = env_sparse_attn.lower().strip()
        if _is_attention_backend_available(env_sparse_attn):
            ATTN = env_sparse_attn
        else:
            logger.warning(f"Attention backend {env_sparse_attn} not available, selecting fallback")

    if ATTN is None:
        ATTN = _select_attention_backend()

    _sync_attention_env(ATTN)
        
    logger.info(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")

def set_backend(backend: Literal['spconv', 'torchsparse']) -> bool:
    """Set sparse backend with validation"""
    global BACKEND
    
    backend = backend.lower().strip()
    logger.info(f"Setting sparse backend to: {backend}")

    if backend == 'spconv':
        try:
            import spconv
            BACKEND = 'spconv'
            os.environ['SPARSE_BACKEND'] = 'spconv'
            return True
        except ImportError:
            logger.warning("spconv not available")
            return False
            
    elif backend == 'torchsparse':
        try:
            import torchsparse
            BACKEND = 'torchsparse'
            os.environ['SPARSE_BACKEND'] = 'torchsparse'
            return True
        except ImportError:
            logger.warning("torchsparse not available")
            return False
    
    return False

def set_sparse_backend(backend: Literal['spconv', 'torchsparse'], algo: str = None) -> bool:
    """Alias for set_backend for backwards compatibility
    
    Parameters:
        backend: The sparse backend to use
        algo: The algorithm to use (only relevant for spconv backend)
    """
    # Call set_backend first
    result = set_backend(backend)
    
    # If algorithm is provided and backend was set successfully
    if algo is not None and result:
        set_spconv_algo(algo)
        
    return result

def set_debug(debug: bool):
    """Set debug mode"""
    global DEBUG
    DEBUG = debug
    if debug:
        os.environ['SPARSE_DEBUG'] = '1'
    else:
        os.environ['SPARSE_DEBUG'] = '0'

def set_attn(attn: Literal['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']) -> bool:
    """Set attention backend with validation"""
    global ATTN
    
    attn = attn.lower().strip()
    logger.info(f"Setting attention backend to: {attn}")

    if _is_attention_backend_available(attn):
        ATTN = attn
        _sync_attention_env(attn)
        return True
        
    logger.warning(f"Attention backend {attn} not available")
    return False

# Add alias for backwards compatibility 
def set_attention_backend(backend: Literal['xformers', 'flash_attn', 'sage', 'sdpa']) -> bool:
    """Alias for set_attn for backwards compatibility"""
    return set_attn(backend)

# Initialize from environment variables on module import
__from_env()
