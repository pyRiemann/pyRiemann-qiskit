"""Canonical IGL (Inverse Green's Learning) Reference Implementation.


Architecture:
    x [N, D] → Encoder → z_raw [N, d]
             → HardConcreteGates → gates [d]
             → z = z_raw * gates [N, d]
             → GreenKernel(z, gate_mask) → Phi [N, R]
             → Phi @ W + bias → output [N, C] or [N]

Dependencies: torch, numpy only.
"""

# ============================================================================
# Section 1: Imports
# ============================================================================

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ============================================================================
# Section 2: Log-Space Kernel Functions
# ============================================================================

SUPPORTED_OPERATORS = [
    "gaussian",
    "helmholtz",
    "cauchy",
    "gabor",
    "laplacian",
    "yukawa",
    "mexican_hat",
    "multiquadric",
]


def _log_kernel_fn(
    d: torch.Tensor, sigma: torch.Tensor, operator: str
) -> torch.Tensor:
    """Evaluate log(kernel) for a given operator directly in log-space.

    All math stays in log-space to avoid numerical underflow when the product
    over many dimensions would otherwise produce values below float64 range.

    Args:
        d: distances [..., d_dim]
        sigma: scales [..., d_dim], strictly positive
        operator: kernel name from SUPPORTED_OPERATORS

    Returns:
        log kernel values, same shape as d
    """
    eps = 1e-8
    if operator == "gaussian":
        # log exp(-d^2 / (2 sigma^2)) = -d^2 / (2 sigma^2)
        return -d**2 / (2 * sigma**2 + eps)
    elif operator == "helmholtz":
        # exp(-|d|/sigma) * cos(pi*d/sigma)
        cos_val = torch.cos(math.pi * d / (sigma + eps))
        return -torch.abs(d) / (sigma + eps) + torch.log(
            cos_val.abs().clamp(min=eps)
        )
    elif operator == "cauchy":
        # log(1 / (1 + d^2/sigma^2)) = -log1p(d^2/sigma^2)
        return -torch.log1p(d**2 / (sigma**2 + eps))
    elif operator == "gabor":
        # exp(-d^2/(2*sigma^2)) * cos(pi*d/sigma)
        envelope_log = -d**2 / (2 * sigma**2 + eps)
        cos_val = torch.cos(math.pi * d / (sigma + eps))
        return envelope_log + torch.log(cos_val.abs().clamp(min=eps))
    elif operator == "laplacian":
        # log exp(-|d|/sigma) = -|d|/sigma
        return -torch.abs(d) / (sigma + eps)
    elif operator == "yukawa":
        # exp(-kappa*|d|) / (2*kappa) where kappa = 1/sigma
        # log = -|d|/sigma - log(2/sigma) = -|d|/sigma - log(2) + log(sigma)
        return -torch.abs(d) / (sigma + eps) - math.log(2) + torch.log(sigma + eps)
    elif operator == "mexican_hat":
        # (1 - d^2/sigma^2) * exp(-d^2/(2*sigma^2))
        # log-space: handle sign via abs + sign tracking
        ratio = d**2 / (sigma**2 + eps)
        envelope_log = -ratio / 2
        factor = (1 - ratio).abs().clamp(min=eps)
        return torch.log(factor) + envelope_log
    elif operator == "multiquadric":
        # sqrt(1 + d^2/sigma^2) -> log = 0.5 * log(1 + d^2/sigma^2)
        return 0.5 * torch.log1p(d**2 / (sigma**2 + eps))
    else:
        raise ValueError(
            f"Unknown operator: {operator}. Supported: {SUPPORTED_OPERATORS}"
        )


def _log_signed_kernel_fn(
    d: torch.Tensor, sigma: torch.Tensor, operator: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate kernel in log-space, preserving sign for oscillatory operators.

    For non-oscillatory kernels (gaussian, laplacian, cauchy, yukawa, multiquadric),
    the kernel is always positive, so sign is +1 everywhere.

    For oscillatory kernels (helmholtz, gabor, mexican_hat), the cosine or
    (1 - d²/σ²) factor can be negative. We return log(|factor|) and sign(factor)
    separately so the caller can track sign parity through the dimension product.

    Args:
        d: distances [..., d_dim]
        sigma: scales [..., d_dim], strictly positive
        operator: kernel name from SUPPORTED_OPERATORS

    Returns:
        (log_abs, sign): both same shape as d.
            log_abs = log of absolute kernel value per dimension
            sign = +1 or -1 per dimension
    """
    eps = 1e-8
    ones = torch.ones_like(d)

    if operator == "gaussian":
        return -d**2 / (2 * sigma**2 + eps), ones
    elif operator == "helmholtz":
        cos_val = torch.cos(math.pi * d / (sigma + eps))
        log_abs = -torch.abs(d) / (sigma + eps) + torch.log(
            cos_val.abs().clamp(min=eps)
        )
        sign = torch.where(cos_val >= 0, ones, -ones)
        return log_abs, sign
    elif operator == "cauchy":
        return -torch.log1p(d**2 / (sigma**2 + eps)), ones
    elif operator == "gabor":
        cos_val = torch.cos(math.pi * d / (sigma + eps))
        log_abs = -d**2 / (2 * sigma**2 + eps) + torch.log(
            cos_val.abs().clamp(min=eps)
        )
        sign = torch.where(cos_val >= 0, ones, -ones)
        return log_abs, sign
    elif operator == "laplacian":
        return -torch.abs(d) / (sigma + eps), ones
    elif operator == "yukawa":
        # Yukawa: exp(-|d|/σ) / (2/σ). The 1/(2κ) normalization factor is
        # absorbed by learnable γ (scale mixing) and rank_importance, so the
        # physics-correct form is preserved without affecting expressiveness.
        return -torch.abs(d) / (sigma + eps) - math.log(2) + torch.log(sigma + eps), ones
    elif operator == "mexican_hat":
        ratio = d**2 / (sigma**2 + eps)
        factor = 1 - ratio
        log_abs = torch.log(factor.abs().clamp(min=eps)) + (-ratio / 2)
        sign = torch.where(factor >= 0, ones, -ones)
        return log_abs, sign
    elif operator == "multiquadric":
        return 0.5 * torch.log1p(d**2 / (sigma**2 + eps)), ones
    else:
        raise ValueError(
            f"Unknown operator: {operator}. Supported: {SUPPORTED_OPERATORS}"
        )


def list_operators() -> list[str]:
    """Return list of all supported kernel operator names."""
    return list(SUPPORTED_OPERATORS)


# ============================================================================
# Section 3: Hard Concrete Gating
# ============================================================================


class HardConcreteGates(nn.Module):
    """Hard Concrete gates for auto dimension discovery (Louizos et al., 2018).

    Produces exact zeros during training via stretched concrete distribution
    with hard-sigmoid clamping. Provides a smooth, differentiable L0 penalty.

    During training: stochastic binary gates via reparameterized binary concrete.
    During eval: deterministic gates from distribution mean.

    Args:
        max_dim: Maximum number of latent dimensions to gate.
        init_bias: Initial bias for log_alpha. Positive = biased ON.
            First dims get +init_bias, last dims get -init_bias.
    """

    # Fixed hyperparameters (not learned)
    _beta: float = 2 / 3  # temperature
    _gamma: float = -0.1  # stretch lower bound
    _zeta: float = 1.1  # stretch upper bound

    def __init__(self, max_dim: int, init_bias: float = 2.0):
        super().__init__()
        self.max_dim = max_dim
        # Linearly interpolate: first dims biased ON, last dims biased OFF
        self.log_alpha = nn.Parameter(
            torch.linspace(init_bias, -init_bias, max_dim)
        )

    def sample(self) -> torch.Tensor:
        """Sample gate values.

        Returns:
            gates: [max_dim], values in [0, 1] with exact zeros/ones possible.
        """
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self._beta
            )
            s_bar = s * (self._zeta - self._gamma) + self._gamma
            return s_bar.clamp(0.0, 1.0)
        else:
            # Deterministic: E[s_bar] clamped
            mean = (
                torch.sigmoid(self.log_alpha) * (self._zeta - self._gamma)
                + self._gamma
            )
            return mean.clamp(0.0, 1.0)

    def get_p_active(self) -> torch.Tensor:
        """P(gate_j > 0) per dimension, differentiable.

        Returns:
            p_active: [max_dim] tensor of probabilities
        """
        return torch.sigmoid(
            self.log_alpha - self._beta * math.log(-self._gamma / self._zeta)
        )

    def get_active_dims(self, threshold: float = 0.5) -> int:
        """Count dimensions where P(gate > 0) > threshold."""
        with torch.no_grad():
            return int((self.get_p_active() > threshold).sum().item())

    def get_sparsity_loss(self) -> torch.Tensor:
        """Normalized L0 regularization: sum P(gate_j > 0) / max_dim.

        Returns:
            scalar in [0, 1]: 0 = all gates off, 1 = all gates on.
        """
        return self.get_p_active().sum() / self.max_dim


# ============================================================================
# Section 4: Encoders
# ============================================================================


class MLPEncoder(nn.Module):
    """MLP encoder: LayerNorm + SiLU activations, no final nonlinearity.

    Architecture: Linear → LayerNorm → SiLU → Linear → LayerNorm → SiLU → Linear

    Output is unbounded (no sigmoid) — works with all kernel types and lets
    the optimizer find the natural scale of the latent space.

    Args:
        input_dim: Ambient dimension D
        output_dim: Latent dimension d (max_dim)
        hidden: Hidden layer width
    """

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearEncoder(nn.Module):
    """Single linear layer encoder (for ablations / low-complexity baselines).

    Args:
        input_dim: Ambient dimension D
        output_dim: Latent dimension d
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================================
# Section 5: Green Kernel
# ============================================================================


class GreenKernel(nn.Module):
    """Multi-scale Green's kernel producing design matrix Phi [N, R].

    Computes:
        Phi[n, r] = sum_k gamma_k * prod_j G_kj(z_nj - anchor_rj; sigma_kj)
                    * importance[r]

    Uses log-space computation with optional gate masking for numerical
    stability at high dimensions.

    Multi-operator mode: pass a list of operator names to split K evenly
    across operators (no separate MultiScaleGreenCP needed).

    Args:
        latent_dim: Dimension of latent space d
        n_anchors: Number of source anchors R
        n_scales: Number of kernel scales K
        operator: Kernel type string or list of strings for multi-operator
    """

    def __init__(
        self,
        latent_dim: int,
        n_anchors: int = 128,
        n_scales: int = 6,
        operator: str | list[str] = "gaussian",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_anchors = n_anchors
        self.n_scales = n_scales

        # Resolve multi-operator configuration
        if isinstance(operator, list):
            self._ops: list[str] = operator
            # Split scales evenly across operators
            base = n_scales // len(operator)
            remainder = n_scales % len(operator)
            self._op_k: list[int] = [
                base + (1 if i < remainder else 0) for i in range(len(operator))
            ]
        else:
            self._ops = [operator]
            self._op_k = [n_scales]

        total_k = sum(self._op_k)

        # Anchor positions [R, d] — learnable
        self.anchor_positions = nn.Parameter(
            torch.randn(n_anchors, latent_dim) * 0.5
        )

        # Per-anchor importance [R]
        self.rank_importance = nn.Parameter(torch.ones(n_anchors))

        # Log-sigma [K, d] — learnable, initialized log-spaced
        self.log_sigma = nn.Parameter(
            torch.linspace(-1.5, 1.5, total_k)
            .unsqueeze(1)
            .expand(total_k, latent_dim)
            .clone()
        )

        # Log-gamma [K] — scale mixing weights (softmax applied at forward)
        self.log_gamma = nn.Parameter(torch.zeros(total_k))

    def compute_design_matrix(
        self,
        z: torch.Tensor,
        gate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build design matrix from latent coordinates.

        Log-space computation with sign tracking for oscillatory kernels.
        Vectorized across all K scales within each operator type.

        Gate masking in log-space: multiplying log(k) by gate g ∈ [0,1] gives
        exp(g · log(k)) = k^g — a power transform, not a hard mask. At Hard
        Concrete convergence (g ∈ {0,1}), this is exact selection (k^0=1, k^1=k).
        During training, intermediate g values smoothly interpolate influence.
        Gradient: ∂(k^g)/∂g = k^g · log(k). When k ≈ 0 (large distances),
        log(k) is large negative, creating sharp gradients that push gates
        decisively toward 0 or 1 — a desirable property for dimension selection.

        Args:
            z: [N, d] latent coordinates
            gate_mask: Optional [d] tensor from Hard Concrete gates.
                Values in [0, 1]; used to mask log-kernel contributions.

        Returns:
            Phi: [N, R] design matrix
        """
        N = z.shape[0]

        # Distances: [N, R, d]
        dist = z.unsqueeze(1) - self.anchor_positions.unsqueeze(0)

        # Sigma: [K, d] -> positive
        sigma = torch.exp(self.log_sigma)  # [K, d]
        gamma = F.softmax(self.log_gamma, dim=0)  # [K]

        # Accumulate weighted kernel products across scales.
        # Vectorized per-operator: all K_op scales computed in one kernel call.
        # The outer loop over operators remains (can't vectorize different fns).
        Phi = torch.zeros(N, self.n_anchors, device=z.device, dtype=z.dtype)
        k_offset = 0
        for op_name, k_count in zip(self._ops, self._op_k):
            # Slice sigma and gamma for this operator's scales
            sigma_op = sigma[k_offset : k_offset + k_count]  # [K_op, d]
            gamma_op = gamma[k_offset : k_offset + k_count]  # [K_op]

            # Broadcast: dist [N, R, 1, d] x sigma_op [1, 1, K_op, d]
            log_kvals, signs = _log_signed_kernel_fn(
                dist.unsqueeze(2),
                sigma_op.unsqueeze(0).unsqueeze(0),
                op_name,
            )  # both [N, R, K_op, d]

            # Apply gate mask in log-space (see docstring for k^g interpretation)
            if gate_mask is not None:
                log_kvals = log_kvals * gate_mask[None, None, None, :]

            # Sign parity: count negative signs per (n, r, k) across dimensions
            neg_count = (signs < 0).sum(dim=-1)  # [N, R, K_op]
            total_sign = torch.where(neg_count % 2 == 1, -1.0, 1.0)

            # Product over dimensions in log-space, then exp
            prod_k = total_sign * torch.exp(log_kvals.sum(dim=-1))  # [N, R, K_op]

            # Weighted sum over scales for this operator
            Phi = Phi + (prod_k * gamma_op[None, None, :]).sum(dim=-1)  # [N, R]

            k_offset += k_count

        # Apply rank importance
        importance = torch.sigmoid(self.rank_importance)  # [R]
        Phi = Phi * importance.unsqueeze(0)

        return Phi

    def forward(
        self,
        z: torch.Tensor,
        gate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for compute_design_matrix."""
        return self.compute_design_matrix(z, gate_mask=gate_mask)


# ============================================================================
# Section 6: Models
# ============================================================================


class _IGLBase(nn.Module):
    """Shared base for IGL models (private, not part of public API).

    Holds the encoder → gates → GreenKernel pipeline and common methods.
    Subclasses define source_weights, bias, forward(), and task-specific logic.

    Args:
        input_dim: Ambient dimension D
        max_dim: Maximum latent dimension d (gates will prune unused)
        n_anchors: R, number of source anchors
        n_scales: K, number of kernel scales
        operator: Kernel operator name or list for multi-operator
        hidden: Encoder hidden width
        use_gates: Enable Hard Concrete dimension gates
        encoder: 'mlp', 'linear', or nn.Module instance
    """

    def __init__(
        self,
        input_dim: int,
        max_dim: int,
        n_anchors: int,
        n_scales: int = 6,
        operator: str | list[str] = "gaussian",
        hidden: int = 256,
        use_gates: bool = True,
        encoder: str | nn.Module = "mlp",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_dim = max_dim
        self.n_anchors = n_anchors
        self.use_gates = use_gates

        # Encoder
        if isinstance(encoder, str):
            if encoder == "linear":
                self.encoder = LinearEncoder(input_dim, max_dim)
            else:
                self.encoder = MLPEncoder(input_dim, max_dim, hidden=hidden)
        else:
            self.encoder = encoder

        # Hard Concrete gates
        self.gates = HardConcreteGates(max_dim) if use_gates else None

        # Green kernel
        self.green = GreenKernel(
            latent_dim=max_dim,
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,
        )

        # Store last gate sample for external access
        self._last_gates: torch.Tensor | None = None

    def _encode_and_gate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode input and apply gates. Returns (z, gate_mask)."""
        z = self.encoder(x)
        gate_mask = None
        if self.gates is not None:
            gate_mask = self.gates.sample()
            self._last_gates = gate_mask
            z = z * gate_mask.unsqueeze(0)
        return z, gate_mask

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x → z (gated latent coordinates)."""
        z, _ = self._encode_and_gate(x)
        return z

    def get_design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute design matrix Phi from raw input."""
        z, gate_mask = self._encode_and_gate(x)
        return self.green.compute_design_matrix(z, gate_mask=gate_mask)

    def get_active_dims(self, threshold: float = 0.5) -> int:
        """Number of active latent dimensions."""
        if self.gates is not None:
            return self.gates.get_active_dims(threshold)
        return self.max_dim

    def get_gate_values(self) -> np.ndarray:
        """P(gate_j > 0) per dimension as numpy array."""
        if self.gates is not None:
            with torch.no_grad():
                return self.gates.get_p_active().cpu().numpy()
        return np.ones(self.max_dim)

    def get_reg_loss(self) -> torch.Tensor:
        """Combined regularization: gate L0 + rank importance sparsity."""
        if self.gates is not None:
            gate_loss = self.gates.get_sparsity_loss()
        else:
            gate_loss = torch.tensor(0.0, device=self.source_weights.device)
        rank_loss = (
            torch.sigmoid(self.green.rank_importance).sum() / self.n_anchors
        )
        return gate_loss + 0.1 * rank_loss

    @torch.no_grad()
    def reinitialize_anchors(self, x_batch: torch.Tensor) -> None:
        """Reinitialize anchor positions from encoded data samples.

        Samples R positions from the latent encoding of x_batch (without
        gradient). Useful when random initialization places anchors far
        from the data manifold.

        Args:
            x_batch: [M, D] input batch (M >= n_anchors recommended)
        """
        self.eval()
        z = self.encoder(x_batch)
        R = self.green.n_anchors
        if z.shape[0] >= R:
            idx = torch.randperm(z.shape[0], device=z.device)[:R]
            self.green.anchor_positions.data.copy_(z[idx])
        else:
            # Fewer samples than anchors: tile and add noise
            repeats = (R // z.shape[0]) + 1
            z_tiled = z.repeat(repeats, 1)[:R]
            z_tiled = z_tiled + torch.randn_like(z_tiled) * 0.01
            self.green.anchor_positions.data.copy_(z_tiled)
        self.train()


class IGLModel(_IGLBase):
    """IGL model for regression: x → y (scalar or multi-output).

    Architecture:
        x → Encoder → z_raw [N, d]
        → HardConcreteGates → gates [d]
        → z = z_raw * gates [N, d]
        → GreenKernel(z, gates) → Phi [N, R]
        → Phi @ source_weights + bias → output [N] or [N, n_outputs]

    Args:
        input_dim: Ambient dimension D
        max_dim: Maximum latent dimension d (gates will prune unused)
        n_outputs: Number of output dimensions (1 for scalar regression)
        n_anchors: R, number of source anchors
        n_scales: K, number of kernel scales
        operator: Kernel operator name or list for multi-operator
        hidden: Encoder hidden width
        use_gates: Enable Hard Concrete dimension gates
        encoder: 'mlp', 'linear', or nn.Module instance
    """

    def __init__(
        self,
        input_dim: int,
        max_dim: int = 64,
        n_outputs: int = 1,
        n_anchors: int = 64,
        n_scales: int = 6,
        operator: str | list[str] = "gaussian",
        hidden: int = 256,
        use_gates: bool = True,
        encoder: str | nn.Module = "mlp",
    ):
        super().__init__(
            input_dim, max_dim, n_anchors,
            n_scales=n_scales, operator=operator,
            hidden=hidden, use_gates=use_gates, encoder=encoder,
        )
        self.n_outputs = n_outputs

        # Linear readout: Phi @ W + b
        self.source_weights = nn.Parameter(
            torch.randn(n_anchors, n_outputs) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x → predictions.

        Returns:
            [N] if n_outputs == 1, else [N, n_outputs]
        """
        z, gate_mask = self._encode_and_gate(x)
        Phi = self.green.compute_design_matrix(z, gate_mask=gate_mask)
        out = Phi @ self.source_weights + self.bias
        if self.n_outputs == 1:
            return out.squeeze(-1)
        return out

    # Factory methods

    @classmethod
    def additive_green(cls, input_dim: int, max_dim: int, **kwargs) -> "IGLModel":
        """Additive Green model (K=1, single scale)."""
        return cls(input_dim, max_dim, n_scales=1, **kwargs)

    @classmethod
    def multi_scale(cls, input_dim: int, max_dim: int, **kwargs) -> "IGLModel":
        """Multi-scale model: Gaussian + Helmholtz."""
        return cls(
            input_dim, max_dim, operator=["gaussian", "helmholtz"], **kwargs
        )


class IGLClassifier(_IGLBase):
    """IGL classifier for binary/multi-class classification.

    Architecture:
        x → Encoder → z_raw [N, d]
        → HardConcreteGates → gates [d]
        → z = z_raw * gates [N, d]
        → GreenKernel(z, gates) → Phi [N, R]
        → Phi @ source_weights + bias → logits [N, C]

    In VP training, source_weights are solved via least-squares; encoder is
    learned by gradient descent.

    Args:
        input_dim: Ambient dimension D
        max_dim: Maximum latent dimension d
        n_classes: Number of output classes C
        n_anchors: R, number of source anchors
        n_scales: K, number of kernel scales
        operator: Kernel operator name or list for multi-operator
        hidden: Encoder hidden width
        use_gates: Enable Hard Concrete dimension gates
        encoder: 'mlp', 'linear', or nn.Module instance
    """

    def __init__(
        self,
        input_dim: int,
        max_dim: int = 128,
        n_classes: int = 2,
        n_anchors: int = 128,
        n_scales: int = 4,
        operator: str | list[str] = "gaussian",
        hidden: int = 256,
        use_gates: bool = True,
        encoder: str | nn.Module = "mlp",
    ):
        super().__init__(
            input_dim, max_dim, n_anchors,
            n_scales=n_scales, operator=operator,
            hidden=hidden, use_gates=use_gates, encoder=encoder,
        )
        self.n_classes = n_classes

        # Per-class source weights [R, C] and bias [C]
        self.source_weights = nn.Parameter(
            torch.randn(n_anchors, n_classes) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x → logits [N, C]."""
        z, gate_mask = self._encode_and_gate(x)
        Phi = self.green.compute_design_matrix(z, gate_mask=gate_mask)
        return Phi @ self.source_weights + self.bias

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted class labels."""
        return self.forward(x).argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities (softmax)."""
        return F.softmax(self.forward(x), dim=1)


class TimeIGLClassifier(nn.Module):
    """IGL with time as the PDE domain variable, K-component source, endpoint query.

    Solves u_k(T) = Σ_s G(T,s) · f_k(x(s)) where:
      - T: final (query) time, normalized to 1.0
      - G(T,s) = Φ(T) · Φ(s): Green's kernel in factorized form
      - f_k(x(s)): k-th component of the learned source (spatial filter k)

    Architecture:
      x[n,t,:] ∈ R^C  →  source_enc (Linear(C,K))  →  f[n,t,:] ∈ R^K
      t_grid ∈ [0,1]^T  →  GreenKernel  →  Φ ∈ [T, R]
      G_row[s] = Φ[-1] · Φ[s]           (G(T,s) for all source times s)
      emb[n,k] = Σ_s G_row[s] · f[n,s,k]   →  [N, K]
      logits   = emb @ W_out + bias          →  [N, n_classes]

    No attention mechanism needed: G(T,s) naturally weights each source time by
    its temporal relationship to the endpoint (e.g. Gaussian kernel → recent
    events dominate; Helmholtz → oscillatory weighting).

    In VP training W_out is solved via Tikhonov-regularized lstsq; source_enc
    and GreenKernel parameters are optimized by gradient descent.

    Args:
        n_channels: Number of EEG channels C
        T: Number of time samples
        n_components: K, number of source components (spatial filter bank width)
        n_anchors: R, number of temporal anchor points
        n_scales: number of kernel scales
        operator: Kernel operator name
        n_classes: Number of output classes
    """

    def __init__(
        self,
        n_channels: int,
        T: int,
        n_components: int = 16,
        n_anchors: int = 64,
        n_scales: int = 3,
        operator: str = "gaussian",
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.T = T
        self.n_components = n_components
        self.n_classes = n_classes

        # Spatial filter bank: x(t) ∈ R^C → f(t) ∈ R^K
        self.source_enc = nn.Linear(n_channels, n_components, bias=False)

        # Temporal Green's kernel (1D domain: time)
        self.green = GreenKernel(
            latent_dim=1,
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,
        )
        # Initialize anchors uniformly in [0, 1] (the time range)
        nn.init.uniform_(self.green.anchor_positions, 0.0, 1.0)

        # Shared time grid t ∈ [0,1]^T  [T, 1]
        t_grid = torch.linspace(0, 1, T).unsqueeze(1)
        self.register_buffer("t_grid", t_grid)

        # Classification head: [K, n_classes]  (replaced by lstsq in VP training)
        self.W_out = nn.Parameter(torch.randn(n_components, n_classes) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def _compute_phi(self) -> torch.Tensor:
        """Temporal basis Φ ∈ [T, R]. Shared across trials."""
        return self.green.compute_design_matrix(self.t_grid)

    def _compute_embedding(self, X: torch.Tensor) -> torch.Tensor:
        """Green's solution evaluated at final time T → [N, K].

        emb[n,k] = Σ_s G(T,s) · f_k[n,s]
        G(T,s)   = Φ[-1] · Φ[s]   (inner product between query and source bases)

        Args:
            X: [N, C, T]  (MOABB/pyriemann convention: channels before time)
        """
        X_tc = X.transpose(1, 2)          # [N, C, T] → [N, T, C]
        N, T, C = X_tc.shape

        # K-dimensional source at each time: [N, T, K]
        f = self.source_enc(X_tc.reshape(N * T, C)).reshape(N, T, self.n_components)

        # Temporal basis: [T, R]
        Phi = self._compute_phi()

        # G(T, s) = Φ[-1] · Φ[s]  for all s → [T]
        G_row = Phi @ Phi[-1]

        # emb[n,k] = Σ_s G_row[s] · f[n,s,k] → [N, K]
        return torch.einsum("t,ntk->nk", G_row, f)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: [N, C, T] → logits [N, n_classes]."""
        return self._compute_embedding(X) @ self.W_out + self.bias

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(X), dim=1)

    def get_active_dims(self) -> int:
        """Compatibility with train_joint: time is always 1D."""
        return 1

    def get_reg_loss(self) -> torch.Tensor:
        """Compatibility with train_joint: no L0 penalty for 1D time."""
        return torch.tensor(0.0, device=self.attn_v.device)


# ============================================================================
# Section 7: Training Configs
# ============================================================================


@dataclass
class VPConfig:
    """Configuration for VP (Variational Projection) training."""

    encoder_lr: float = 1e-3
    epochs: int = 1500
    warmup_epochs: int = 300
    batch_size: int = 256
    gate_weight: float = 0.015
    source_l2: float = 1e-3
    inner_batch_size: int = 4096
    grad_clip: float = 1.0
    scheduler: str = "cosine_warm_restarts"
    log_every: int = 100
    verbose: bool = True


@dataclass
class JointConfig:
    """Configuration for joint (end-to-end) training."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1500
    batch_size: int = 256
    gate_weight: float = 0.015
    grad_clip: float = 1.0
    scheduler: str = "cosine_warm_restarts"
    early_stopping_patience: int = 0  # 0 = disabled
    log_every: int = 100
    verbose: bool = True


# ============================================================================
# Section 8: Training Functions
# ============================================================================


def _get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def direct_solve_weights(
    Phi: torch.Tensor,
    y: torch.Tensor,
    l2: float = 1e-3,
) -> torch.Tensor:
    """Solve w* = argmin ||Phi @ w - y||^2 + l2 * ||w||^2 via lstsq on CPU.

    CPU is used deliberately: torch.linalg.lstsq has stability issues on MPS
    (incorrect results for some matrix sizes). For R≤256, N≤10K, the CPU
    transfer + lstsq is <10ms — negligible vs the gradient step.
    TODO: For large-scale use, a GPU Cholesky path (Phi.T @ Phi + λI)⁻¹ Phi.T y
    would avoid the transfer, but needs careful conditioning checks.

    Works for both classification (y = one-hot [N, C]) and regression (y = [N, 1]).

    Args:
        Phi: Design matrix [N, R]
        y: Targets [N, C] or [N, 1]
        l2: Tikhonov regularization strength

    Returns:
        w: Optimal weights [R, C] or [R, 1]
    """
    Phi_cpu = Phi.detach().cpu().float()
    y_cpu = y.detach().cpu().float()

    R = Phi_cpu.shape[1]
    # Tikhonov: augment Phi with sqrt(l2)*I, y with zeros
    Phi_aug = torch.cat([Phi_cpu, (l2**0.5) * torch.eye(R)], dim=0)
    y_aug = torch.cat([y_cpu, torch.zeros(R, y_cpu.shape[1])], dim=0)

    result = torch.linalg.lstsq(Phi_aug, y_aug)
    return result.solution  # [R, C]


def train_vp(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    config: VPConfig | None = None,
) -> dict:
    """Two-stage VP (Variational Projection) training.

    Warmup phase: joint training to initialize encoder.
    VP phase:
        Stage 1: gates.eval() → deterministic gates → Phi → lstsq → copy weights
        Stage 2: gates.train() → gradient step on encoder+Green (source_weights detached)

    Auto-detects task type from model: IGLClassifier → classification, IGLModel → regression.

    Args:
        model: IGLModel or IGLClassifier
        X_train: Training inputs [N, D]
        y_train: Training targets [N] (class labels or regression values)
        X_val: Optional validation inputs
        y_val: Optional validation targets
        config: VPConfig (defaults provided)

    Returns:
        history dict with train_loss, val_loss, val_acc (classification) or
        val_mse (regression), active_dims
    """
    config = config or VPConfig()
    device = _get_device()

    model = model.to(device)
    X_train = X_train.to(device)

    # Detect task type
    is_classification = isinstance(model, IGLClassifier)

    if is_classification:
        y_train = y_train.to(device).long()
        n_classes = model.n_classes
    else:
        y_train = y_train.to(device).float()

    if X_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device).long() if is_classification else y_val.to(device).float()

    N = X_train.shape[0]
    warmup_epochs = min(config.warmup_epochs, config.epochs // 5)

    # Optimizer for encoder + green kernel + bias (NOT source_weights)
    encoder_params = list(model.encoder.parameters()) + list(model.green.parameters())
    if model.gates is not None:
        encoder_params += list(model.gates.parameters())
    encoder_params.append(model.bias)
    optimizer = AdamW(encoder_params, lr=config.encoder_lr)

    scheduler = None
    if config.scheduler == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1)

    metric_key = "val_acc" if is_classification else "val_mse"
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        metric_key: [],
        "active_dims": [],
    }

    for epoch in range(config.epochs):
        model.train()
        is_warmup = epoch < warmup_epochs

        # Clear accumulated source_weights gradients at warmup→VP transition
        if epoch == warmup_epochs:
            model.source_weights.grad = None

        # --- Stage 1: Direct solve for source weights ---
        if not is_warmup:
            # Random subsample for lstsq — resampled each epoch. This is a
            # design choice: per-epoch resampling works in practice, though
            # a fixed subset would reduce epoch-to-epoch noise in w*.
            inner_n = min(config.inner_batch_size, N)
            inner_idx = torch.randperm(N, device=device)[:inner_n]
            # Deterministic gates for stable lstsq
            model.eval()
            with torch.no_grad():
                z = model.get_latent(X_train[inner_idx])
                gate_mask = model._last_gates if hasattr(model, "_last_gates") else None
                Phi_all = model.green.compute_design_matrix(z, gate_mask=gate_mask)
                if is_classification:
                    y_target = F.one_hot(y_train[inner_idx], n_classes).float()
                else:
                    y_target = y_train[inner_idx]
                    if y_target.dim() == 1:
                        y_target = y_target.unsqueeze(-1)
                w_star = direct_solve_weights(Phi_all, y_target, l2=config.source_l2)
                model.source_weights.data.copy_(w_star.to(device))
            model.train()

        # --- Stage 2: Gradient step on encoder ---
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for i in range(0, N, config.batch_size):
            idx = perm[i : i + config.batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()

            if is_warmup:
                # Warmup: encoder + gates + green + bias only (source_weights
                # excluded from optimizer — they'll be set by lstsq after warmup).
                # source_weights act as a random projection during warmup, giving
                # the encoder noisy but nonzero gradients to initialize geometry.
                output = model(X_batch)
            else:
                # VP: encoder gradient only, source_weights frozen from Stage 1
                z, gate_mask = model._encode_and_gate(X_batch)
                Phi = model.green.compute_design_matrix(z, gate_mask=gate_mask)
                output = Phi @ model.source_weights.detach() + model.bias

            if is_classification:
                loss = F.cross_entropy(output, y_batch)
            else:
                target = y_batch if y_batch.dim() == output.dim() else y_batch.unsqueeze(-1)
                loss = F.mse_loss(output, target)

            if config.gate_weight > 0:
                loss = loss + config.gate_weight * model.get_reg_loss()

            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= N
        history["train_loss"].append(epoch_loss)

        # --- Validation ---
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                if is_classification:
                    val_loss = F.cross_entropy(val_output, y_val).item()
                    val_metric = (val_output.argmax(1) == y_val).float().mean().item()
                else:
                    target = y_val if y_val.dim() == val_output.dim() else y_val.unsqueeze(-1)
                    val_loss = F.mse_loss(val_output, target).item()
                    val_metric = val_loss  # MSE as metric for regression
            history["val_loss"].append(val_loss)
            history[metric_key].append(val_metric)
        else:
            val_loss = epoch_loss
            val_metric = 0.0

        active = model.get_active_dims()
        history["active_dims"].append(active)

        if config.verbose and (epoch + 1) % config.log_every == 0:
            phase = "WARM" if is_warmup else "VP"
            msg = (
                f"[{phase}] Epoch {epoch+1}/{config.epochs} "
                f"train_loss={epoch_loss:.4f}"
            )
            if X_val is not None:
                msg += f" val_loss={val_loss:.4f} {metric_key}={val_metric:.4f}"
            msg += f" d_eff={active}"
            print(msg)

    return history


def train_joint(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    config: JointConfig | None = None,
    task: str = "auto",
) -> dict:
    """Standard joint training (all params via AdamW).

    Unified for both regression (MSE) and classification (cross-entropy).
    Supports early stopping, gradient clipping, and cosine/warm-restart scheduling.

    Args:
        model: IGLModel or IGLClassifier
        X_train: Training inputs [N, D]
        y_train: Training targets [N]
        X_val: Optional validation inputs
        y_val: Optional validation targets
        config: JointConfig (defaults provided)
        task: "classification", "regression", or "auto" (detect from model type)

    Returns:
        history dict with train_loss, val_loss, val_acc/val_mse, active_dims
    """
    config = config or JointConfig()
    device = _get_device()

    model = model.to(device)
    X_train = X_train.to(device)

    # Detect task type
    if task == "auto":
        is_classification = isinstance(model, IGLClassifier)
    else:
        is_classification = task == "classification"

    if is_classification:
        y_train = y_train.to(device).long()
    else:
        y_train = y_train.to(device).float()

    if X_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device).long() if is_classification else y_val.to(device).float()

    N = X_train.shape[0]
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = None
    if config.scheduler == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1)

    metric_key = "val_acc" if is_classification else "val_mse"
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        metric_key: [],
        "active_dims": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for i in range(0, N, config.batch_size):
            idx = perm[i : i + config.batch_size]
            optimizer.zero_grad()
            output = model(X_train[idx])
            y_batch = y_train[idx]

            if is_classification:
                loss = F.cross_entropy(output, y_batch)
            else:
                target = y_batch if y_batch.dim() == output.dim() else y_batch.unsqueeze(-1)
                loss = F.mse_loss(output, target)

            if config.gate_weight > 0:
                loss = loss + config.gate_weight * model.get_reg_loss()

            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= N
        history["train_loss"].append(epoch_loss)

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                if is_classification:
                    val_loss = F.cross_entropy(val_output, y_val).item()
                    val_metric = (val_output.argmax(1) == y_val).float().mean().item()
                else:
                    target = y_val if y_val.dim() == val_output.dim() else y_val.unsqueeze(-1)
                    val_loss = F.mse_loss(val_output, target).item()
                    val_metric = val_loss
            history["val_loss"].append(val_loss)
            history[metric_key].append(val_metric)

            # Early stopping
            if config.early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        if config.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        else:
            val_loss = epoch_loss
            val_metric = 0.0

        history["active_dims"].append(model.get_active_dims())

        if config.verbose and (epoch + 1) % config.log_every == 0:
            msg = (
                f"[JOINT] Epoch {epoch+1}/{config.epochs} "
                f"train_loss={epoch_loss:.4f}"
            )
            if X_val is not None:
                msg += f" val_loss={val_loss:.4f} {metric_key}={val_metric:.4f}"
            msg += f" d_eff={model.get_active_dims()}"
            print(msg)

    return history


def train_vp_timeseries(
    model: "TimeIGLClassifier",
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: "VPConfig | None" = None,
) -> dict:
    """VP training for TimeIGLClassifier.

    Warmup phase: joint gradient descent on all parameters.
    VP phase: alternating
        Stage 1 — gradient on encoder params (source_enc, GreenKernel, attn_v)
        Stage 2 — Tikhonov lstsq for W_out given the current embeddings

    Args:
        model: TimeIGLClassifier instance
        X_train: [N, C, T] EEG epochs (MOABB convention)
        y_train: [N] integer class labels
        config: VPConfig (defaults used if None)

    Returns:
        history dict with train_loss and active_dims keys
    """
    config = config or VPConfig()
    device = _get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device).long()

    N = X_train.shape[0]
    n_classes = model.n_classes
    warmup_epochs = min(config.warmup_epochs, config.epochs // 5)
    vp_epochs = config.epochs - warmup_epochs

    # Warmup optimizer: all parameters
    optimizer_all = AdamW(model.parameters(), lr=config.encoder_lr,
                          weight_decay=1e-5)

    # VP-phase optimizer: encoder only (W_out updated via lstsq)
    encoder_params = (
        list(model.source_enc.parameters())
        + list(model.green.parameters())
    )
    optimizer_enc = AdamW(encoder_params, lr=config.encoder_lr,
                          weight_decay=1e-5)

    history: dict = {"train_loss": [], "active_dims": []}

    # ── Warmup: joint training ──────────────────────────────────────────────
    for epoch in range(warmup_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, config.batch_size):
            idx = perm[i : i + config.batch_size]
            optimizer_all.zero_grad()
            loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.grad_clip)
            optimizer_all.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= N
        history["train_loss"].append(epoch_loss)
        history["active_dims"].append(1)
        if config.verbose and (epoch + 1) % config.log_every == 0:
            print(f"[TS-IGL WARMUP] Epoch {epoch+1}/{warmup_epochs} "
                  f"loss={epoch_loss:.4f}")

    # ── VP phase ────────────────────────────────────────────────────────────
    y_onehot = F.one_hot(y_train, n_classes).float()  # [N, n_classes]

    for epoch in range(vp_epochs):
        # Stage 2: lstsq for W_out (run in eval/deterministic mode)
        model.eval()
        with torch.no_grad():
            emb_all = model._compute_embedding(X_train)  # [N, R]

        emb_cpu = emb_all.cpu()
        y_cpu = y_onehot.cpu()
        lam = config.source_l2
        A = emb_cpu.T @ emb_cpu + lam * torch.eye(emb_cpu.shape[1])
        B = emb_cpu.T @ y_cpu
        W_solved = torch.linalg.solve(A, B)  # [R, n_classes]
        model.W_out.data.copy_(W_solved.to(device))
        model.bias.data.zero_()

        # Stage 1: gradient on encoder params
        model.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, config.batch_size):
            idx = perm[i : i + config.batch_size]
            optimizer_enc.zero_grad()
            loss = F.cross_entropy(model(X_train[idx]), y_train[idx])
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder_params, config.grad_clip)
            optimizer_enc.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= N
        history["train_loss"].append(epoch_loss)
        history["active_dims"].append(1)
        if config.verbose and (epoch + 1) % config.log_every == 0:
            print(f"[TS-IGL VP] Epoch {epoch+1}/{vp_epochs} "
                  f"loss={epoch_loss:.4f}")

    return history


# ============================================================================
# Section 9: Sklearn Wrapper
# ============================================================================


class IGLSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around IGLClassifier + train_vp/train_joint.

    input_dim and n_classes are inferred from data at fit() time, so this
    estimator drops in directly into any sklearn Pipeline.

    Args:
        max_dim: Maximum latent dimension d
        n_anchors: R, number of source anchors
        n_scales: K, number of kernel scales
        operator: Kernel operator name or list of names
        hidden: Encoder hidden width (MLP encoder only)
        use_gates: Enable Hard Concrete dimension gates
        encoder: 'mlp' or 'linear'
        training: 'vp' (recommended) or 'joint'
        vp_config: VPConfig instance (None = use defaults)
        joint_config: JointConfig instance (None = use defaults)
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        max_dim: int = 32,
        n_anchors: int = 128,
        n_scales: int = 4,
        operator: str | list[str] = "gaussian",
        hidden: int = 256,
        use_gates: bool = True,
        encoder: str = "mlp",
        training: str = "vp",
        vp_config: VPConfig | None = None,
        joint_config: JointConfig | None = None,
        random_state: int | None = None,
    ):
        self.max_dim = max_dim
        self.n_anchors = n_anchors
        self.n_scales = n_scales
        self.operator = operator
        self.hidden = hidden
        self.use_gates = use_gates
        self.encoder = encoder
        self.training = training
        self.vp_config = vp_config
        self.joint_config = joint_config
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_map[c] for c in y])

        self.model_ = IGLClassifier(
            input_dim=X.shape[1],
            max_dim=self.max_dim,
            n_classes=n_classes,
            n_anchors=self.n_anchors,
            n_scales=self.n_scales,
            operator=self.operator,
            hidden=self.hidden,
            use_gates=self.use_gates,
            encoder=self.encoder,
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        if self.training == "vp":
            train_vp(self.model_, X_t, y_t, config=self.vp_config)
        else:
            train_joint(self.model_, X_t, y_t, config=self.joint_config)

        return self

    def predict(self, X):
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            idx = self.model_.predict(
                torch.tensor(X, dtype=torch.float32).to(device)
            ).cpu().numpy()
        return self.classes_[idx]

    def predict_proba(self, X):
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            proba = self.model_.predict_proba(
                torch.tensor(X, dtype=torch.float32).to(device)
            ).cpu().numpy()
        return proba

    def effective_dimension(self) -> int:
        """Return d_eff discovered by the gates after fitting."""
        return self.model_.get_active_dims()


class IGLTimeSeriesSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for TimeIGLClassifier.

    Accepts raw EEG epochs in MOABB/pyriemann format [N, C, T].
    n_channels, T, and n_classes are inferred from data at fit() time.

    Args:
        n_components: K, spatial filter bank width (source components)
        n_anchors: R, number of temporal anchor points
        n_scales: number of kernel scales
        operator: Kernel operator name
        training: 'vp' (recommended) or 'joint'
        vp_config: VPConfig (None = defaults)
        joint_config: JointConfig (None = defaults)
        random_state: Seed for reproducibility
    """

    def __init__(
        self,
        n_components: int = 16,
        n_anchors: int = 64,
        n_scales: int = 3,
        operator: str = "gaussian",
        training: str = "vp",
        vp_config: VPConfig | None = None,
        joint_config: JointConfig | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.n_anchors = n_anchors
        self.n_scales = n_scales
        self.operator = operator
        self.training = training
        self.vp_config = vp_config
        self.joint_config = joint_config
        self.random_state = random_state

    def fit(self, X, y):
        """Fit on raw EEG epochs.

        Args:
            X: [N, C, T]  (channels × time — MOABB convention)
            y: [N] class labels
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        N, C, T = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_map[c] for c in y])

        self.model_ = TimeIGLClassifier(
            n_channels=C,
            T=T,
            n_components=self.n_components,
            n_anchors=self.n_anchors,
            n_scales=self.n_scales,
            operator=self.operator,
            n_classes=n_classes,
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        if self.training == "vp":
            train_vp_timeseries(self.model_, X_t, y_t, config=self.vp_config)
        else:
            train_joint(self.model_, X_t, y_t,
                        config=self.joint_config, task="classification")

        return self

    def predict(self, X):
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            idx = self.model_.predict(
                torch.tensor(X, dtype=torch.float32).to(device)
            ).cpu().numpy()
        return self.classes_[idx]

    def predict_proba(self, X):
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            proba = self.model_.predict_proba(
                torch.tensor(X, dtype=torch.float32).to(device)
            ).cpu().numpy()
        return proba

    def effective_dimension(self, threshold: float = 0.1) -> int:
        """Return the number of spatially active filters after fitting.

        Since TimeIGLClassifier has no HardConcreteGates, spatial filter
        activity is proxied by the combined contribution of each filter:

            contribution_k = ||source_enc.weight[:, k]||_2
                             * ||W_out[k, :]||_2

        A filter is considered active if its contribution exceeds
        ``threshold`` * max(contribution).

        Args:
            threshold: Fraction of the maximum contribution below which a
                filter is considered inactive (default: 0.1).

        Returns:
            Number of active spatial filters.
        """
        filter_norms = self.model_.source_enc.weight.detach().norm(dim=1)
        wout_norms = self.model_.W_out.detach().norm(dim=1)
        contrib = filter_norms * wout_norms
        return int((contrib > threshold * contrib.max()).sum().item())


# ============================================================================
# Section 10: Exports
# ============================================================================

__all__ = [
    # Kernel functions
    "SUPPORTED_OPERATORS",
    "list_operators",
    "_log_kernel_fn",
    "_log_signed_kernel_fn",
    # Gates
    "HardConcreteGates",
    # Encoders
    "MLPEncoder",
    "LinearEncoder",
    # Green kernel
    "GreenKernel",
    # Models
    "IGLModel",
    "IGLClassifier",
    "TimeIGLClassifier",
    "IGLSklearnClassifier",
    "IGLTimeSeriesSklearnClassifier",
    # Training configs
    "VPConfig",
    "JointConfig",
    # Training functions
    "direct_solve_weights",
    "train_vp",
    "train_joint",
    "train_vp_timeseries",
    # Utilities
    "_get_device",
]
