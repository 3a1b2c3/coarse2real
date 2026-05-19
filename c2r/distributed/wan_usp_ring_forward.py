import math

import torch
import torch.distributed as dist
from einops import rearrange

from ..models.wan_video_dit import rope_apply
from .wan_usp_common import (
    _compile_disable,
    _get_sequence_parallel_rank,
    _get_sequence_parallel_world_size,
    usp_dit_forward,
)


def _ring_exchange_sequence_chunk(x: torch.Tensor) -> torch.Tensor:
    world_size = _get_sequence_parallel_world_size()
    if world_size == 1:
        return x

    rank = _get_sequence_parallel_rank()
    prev_rank = (rank - 1 + world_size) % world_size
    next_rank = (rank + 1) % world_size
    received = torch.empty_like(x)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, x.contiguous(), next_rank),
            dist.P2POp(dist.irecv, received, prev_rank),
        ]
    )
    for request in requests:
        request.wait()
    return received


def _ring_attention_exact(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    world_size = _get_sequence_parallel_world_size()
    if world_size == 1:
        from ..models.wan_video_dit import flash_attention

        return flash_attention(q=q, k=k, v=v, num_heads=num_heads)

    q_heads = rearrange(q, "b s (n d) -> b n s d", n=num_heads).float()
    k_chunk = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v_chunk = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    scale = 1.0 / math.sqrt(q_heads.shape[-1])

    max_score = torch.full(
        (*q_heads.shape[:-1], 1),
        -torch.inf,
        device=q.device,
        dtype=torch.float32,
    )
    normalizer = torch.zeros_like(max_score)
    output = torch.zeros_like(q_heads, dtype=torch.float32)

    for step in range(world_size):
        scores = torch.matmul(q_heads, k_chunk.transpose(-1, -2).float()) * scale
        step_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(max_score, step_max)
        exp_scale = torch.exp(max_score - new_max)
        exp_scores = torch.exp(scores - new_max)
        normalizer = exp_scale * normalizer + exp_scores.sum(dim=-1, keepdim=True)
        output = exp_scale * output + torch.matmul(exp_scores, v_chunk.float())
        max_score = new_max

        if step + 1 < world_size:
            k_chunk = _ring_exchange_sequence_chunk(k_chunk)
            v_chunk = _ring_exchange_sequence_chunk(v_chunk)

    output = output / normalizer.clamp_min(torch.finfo(output.dtype).tiny)
    return rearrange(output.to(dtype=q.dtype), "b n s d -> b s (n d)", n=num_heads)


def _usp_ring_attn_forward_impl(self, x, freqs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    x = _ring_attention_exact(q=q, k=k, v=v, num_heads=self.num_heads)
    return self.o(x)


if _compile_disable is None:
    try:
        import torch._dynamo as _torch_dynamo

        usp_attn_forward = _torch_dynamo.disable(_usp_ring_attn_forward_impl)
    except Exception:
        usp_attn_forward = _usp_ring_attn_forward_impl
else:
    usp_attn_forward = _compile_disable(_usp_ring_attn_forward_impl)
