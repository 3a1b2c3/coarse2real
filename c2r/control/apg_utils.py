import torch


class APGMomentum:
    """
    Momentum buffer. Using a negative momentum implements reverse momentum.
    """

    def __init__(self, momentum: float = -0.5):
        self.momentum = float(momentum)
        self.buf = None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.buf is None:
            self.buf = torch.zeros_like(x)
        self.buf = x + self.momentum * self.buf
        return self.buf


def _reduce_dims(x: torch.Tensor):
    # Works for [B, C, H, W] or [B, C, F, H, W], etc.
    return tuple(range(1, x.ndim))


def _project(v: torch.Tensor, basis: torch.Tensor, eps: float = 1e-12):
    """
    Decompose v into (parallel, orthogonal) with respect to basis.
    Uses float64 internally for numerical stability.
    """

    dims = _reduce_dims(v)
    v64 = v.double()
    b64 = basis.double()

    b_norm = b64.norm(p=2, dim=dims, keepdim=True).clamp_min(eps)
    b_unit = b64 / b_norm

    v_par = (v64 * b_unit).sum(dim=dims, keepdim=True) * b_unit
    v_orth = v64 - v_par
    return v_par.to(v.dtype), v_orth.to(v.dtype)


def apg_delta_x0(
    x0_uncond: torch.Tensor,
    x0_cond: torch.Tensor,
    *,
    scale: float,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
    momentum: APGMomentum | None = None,
    eps: float = 1e-12,
):
    """
    APG in x0-space, returning the delta to add to x0_uncond.

    delta = (x0_cond - x0_uncond) + (scale - 1) * (diff_orth + eta * diff_par)
    """

    diff = x0_cond - x0_uncond

    # Optional reverse momentum.
    if momentum is not None:
        diff = momentum.apply(diff)

    # Optional norm clipping/rescaling.
    if norm_threshold and norm_threshold > 0:
        dims = _reduce_dims(diff)
        n = diff.norm(p=2, dim=dims, keepdim=True).clamp_min(eps)
        diff = diff * torch.minimum(torch.ones_like(n), norm_threshold / n)

    diff_par, diff_orth = _project(diff, x0_cond, eps=eps)
    update = diff_orth + eta * diff_par

    return (x0_cond - x0_uncond) + (scale - 1.0) * update


def flow_pred_to_x0(v_pred: torch.Tensor, x_sigma: torch.Tensor, sigma: torch.Tensor):
    # FlowMatch: x_sigma = x0 + sigma * v  =>  x0 = x_sigma - sigma * v
    return x_sigma - sigma * v_pred


def x0_to_flow_pred(x0: torch.Tensor, x_sigma: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6):
    # v = (x_sigma - x0) / sigma
    sigma_safe = sigma.clamp_min(eps)
    return (x_sigma - x0) / sigma_safe
