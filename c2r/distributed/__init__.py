from .wan_usp_forward import usp_attn_forward as usp_gather_attn_forward
from .wan_usp_forward import usp_dit_forward
from .wan_usp_ring_forward import usp_attn_forward as usp_ring_attn_forward


def get_usp_backend(name: str):
    backend = name.lower().strip()
    if backend == "gather":
        return usp_gather_attn_forward, usp_dit_forward
    if backend == "ring":
        return usp_ring_attn_forward, usp_dit_forward
    raise ValueError(
        f"Unsupported usp_attention_backend='{name}'. Use 'gather' or 'ring'."
    )


usp_attn_forward = usp_gather_attn_forward

__all__ = [
    "get_usp_backend",
    "usp_attn_forward",
    "usp_dit_forward",
    "usp_gather_attn_forward",
    "usp_ring_attn_forward",
]
