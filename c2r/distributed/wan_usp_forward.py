import torch

from ..models.wan_video_dit import flash_attention, rope_apply
from .wan_usp_common import (
    _all_gather_along_sequence,
    _compile_disable,
    usp_dit_forward,
)


def _usp_attn_forward_impl(self, x, freqs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    k = _all_gather_along_sequence(k)
    v = _all_gather_along_sequence(v)
    x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
    return self.o(x)


if _compile_disable is None:
    try:
        import torch._dynamo as _torch_dynamo

        usp_attn_forward = _torch_dynamo.disable(_usp_attn_forward_impl)
    except Exception:
        usp_attn_forward = _usp_attn_forward_impl
else:
    usp_attn_forward = _compile_disable(_usp_attn_forward_impl)
