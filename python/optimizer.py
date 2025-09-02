import os
import sys
import torch
from typing import Iterable

# Try prebuilt module; otherwise JIT build
try:
    import flashalopex_cuda as ext
except Exception:
    _here = os.path.dirname(__file__)
    if _here not in sys.path:
        sys.path.append(_here)
    from extension_build import build_jit
    ext = build_jit()

class FlashAlopexCUDA:
    """
    CUDA-only FlashAlopex wrapper using fused kernel; semantics match AlopexOptimizer.
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], delta: float = 1e-3, seed: int = 1234):
        if ext is None:
            raise RuntimeError("flashalopex_cuda extension not available; use AlopexOptimizer instead.")
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("No parameters provided.")
        for p in self.params:
            if not p.is_cuda:
                raise RuntimeError("All parameters must be on CUDA for FlashAlopexCUDA.")
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        self.delta = float(delta)
        self.seed = int(seed)
        self.offset = 0  # advanced by numel per launch

        dev = self.params[0].device
        self.old_loss = torch.tensor(0.0, device=dev)
        self.delta_E = torch.tensor(0.0, device=dev)
        self.xstates = [ext.init_xstate_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, loss: torch.Tensor, denominator: float | torch.Tensor):
        # Same scalar p computation as the original
        self.old_loss = loss.detach().to(self.old_loss.device)
        denom_t = denominator if isinstance(denominator, torch.Tensor) else torch.tensor(float(denominator), device=self.delta_E.device)
        denom_t = denom_t.to(self.delta_E.device).clamp_min(1e-12)
        p = torch.sigmoid(-self.delta_E / denom_t).clamp(1e-6, 1 - 1e-6)
        p_flip = float(p.item())

        for i, p_ in enumerate(self.params):
            x = self.xstates[i]
            if x.device != p_.device or x.shape != p_.shape:
                x = ext.init_xstate_like(p_)
                self.xstates[i] = x
            # Call with seed+offset if available, else fallback
            try:
                ext.alopex_step(p_.data, x, self.delta, p_flip, self.seed, int(self.offset))
            except TypeError:
                ext.alopex_step(p_.data, x, self.delta, p_flip, int(self.offset))
            try:
                self.offset += int(ext.philox_offset_increment(p_.numel()))
            except AttributeError:
                self.offset += p_.numel()
        return p_flip  # convenient for logging

    def update_delta_E(self, new_loss: torch.Tensor):
        new_loss = new_loss.detach().to(self.old_loss.device)
        self.delta_E = new_loss - self.old_loss
        return self.delta_E