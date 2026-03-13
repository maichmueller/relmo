from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.resolver import activation_resolver

from .mlp import ArityMLPFactory


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: h' = gamma(c) * h + beta(c)
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.to_gamma(c) * h + self.to_beta(c)


class FiLMBlock(nn.Module):
    """
    Linear -> FiLM -> Activation
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        activation: str | None = "mish",
        norm_class: type[nn.Module] | None = None,
        norm_kwargs: dict | None = None,
        dropout: float | None = None,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
    ):
        super().__init__()
        layer_kwargs = dict(layer_kwargs or {})
        self.lin = layer_class(in_dim, out_dim, **layer_kwargs)
        self.film = FiLM(cond_dim, out_dim)
        if norm_class is not None:
            resolved_norm_kwargs = dict(norm_kwargs or {})
            if (
                norm_class is nn.LayerNorm
                and "normalized_shape" not in resolved_norm_kwargs
            ):
                resolved_norm_kwargs["normalized_shape"] = out_dim
            self.norm: nn.Module | None = norm_class(**resolved_norm_kwargs)
        else:
            self.norm = None
        self.act = activation_resolver(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        h = self.film(h, cond)
        if self.norm is not None:
            h = self.norm(h)
        h = self.act(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class FiLMMLP(nn.Module):
    """
    FiLM-conditioned MLP: each hidden layer is modulated by `cond`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        cond_dim: int,
        activation: str | None = "mish",
        norm_class: type[nn.Module] | None = None,
        norm_kwargs: dict | list[dict] | None = None,
        dropout: float | None = None,
        bias_last_layer: bool = True,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = in_dim
        for idx, h in enumerate(hidden_dims):
            if isinstance(norm_kwargs, list):
                block_norm_kwargs = (
                    norm_kwargs[idx]
                    if idx < len(norm_kwargs)
                    else (norm_kwargs[-1] if norm_kwargs else {})
                )
            else:
                block_norm_kwargs = norm_kwargs
            self.blocks.append(
                FiLMBlock(
                    prev,
                    h,
                    cond_dim,
                    activation,
                    norm_class=norm_class,
                    norm_kwargs=block_norm_kwargs,
                    dropout=dropout,
                    layer_class=layer_class,
                    layer_kwargs=layer_kwargs,
                )
            )
            prev = h
        out_layer_kwargs = dict(layer_kwargs or {})
        if "bias" not in out_layer_kwargs:
            out_layer_kwargs["bias"] = bias_last_layer
        self.out = layer_class(prev, out_dim, **out_layer_kwargs)
        self.out_activation = (
            activation_resolver(activation) if activate_last_layer else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_activation(self.out(h))


class FiLMConcatMLP(nn.Module):
    """
    FiLM-conditioned MLP that accepts concatenated input [cond, x] or [x, cond].

    This is compatible with existing call sites that concatenate condition features
    and expects a single input tensor. Internally it splits out the condition and
    applies FiLM at each hidden layer.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        cond_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        condition_position: str = "pre",
        activation: str | None = "mish",
        norm_class: type[nn.Module] | None = None,
        norm_kwargs: dict | list[dict] | None = None,
        dropout: float | None = None,
        bias_last_layer: bool = True,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
    ):
        super().__init__()
        if condition_position not in ("pre", "post"):
            raise ValueError(
                "condition_position must be 'pre' or 'post', "
                f"got {condition_position!r}."
            )
        if cond_dim < 1:
            raise ValueError(f"cond_dim must be >= 1, got {cond_dim}")
        self.in_dim = int(in_dim)
        self.cond_dim = int(cond_dim)
        self.out_dim = int(out_dim)
        self.condition_position = condition_position
        self.film_mlp = FiLMMLP(
            in_dim=self.in_dim,
            hidden_dims=hidden_dims,
            out_dim=self.out_dim,
            cond_dim=self.cond_dim,
            activation=activation,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            bias_last_layer=bias_last_layer,
            layer_class=layer_class,
            layer_kwargs=layer_kwargs,
            activate_last_layer=activate_last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.in_dim + self.cond_dim
        if x.size(-1) != expected:
            raise ValueError(
                f"Expected input feature size {expected}, got {x.size(-1)}."
            )
        if self.condition_position == "pre":
            cond = x[..., : self.cond_dim]
            feats = x[..., self.cond_dim :]
        else:
            feats = x[..., : self.in_dim]
            cond = x[..., self.in_dim :]
        return self.film_mlp(feats, cond)


class FiLMResFFNBlock(nn.Module):
    """
    Pre-norm residual FiLM feed-forward block (Transformer-style FFN):

      x = x + gate * FFN(FiLM(Norm(x), cond))

    where FFN is Linear(dim->hidden)->act->dropout->Linear(hidden->dim)->dropout.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        cond_dim: int,
        activation: str | None = "mish",
        norm_class: type[nn.Module] | None = nn.LayerNorm,
        norm_kwargs: dict | None = None,
        dropout: float | None = None,
        gate: str = "layerscale",
        gate_init: float = 1e-4,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.cond_dim < 1:
            raise ValueError(f"cond_dim must be >= 1, got {self.cond_dim}")
        if gate not in ("layerscale", "rezero", "none"):
            raise ValueError(
                f"gate must be one of {{'layerscale','rezero','none'}}, got {gate!r}"
            )
        self.gate = gate

        resolved_norm_kwargs = dict(norm_kwargs or {})
        if (
            norm_class is nn.LayerNorm
            and "normalized_shape" not in resolved_norm_kwargs
        ):
            resolved_norm_kwargs["normalized_shape"] = self.dim
        self.norm = None if norm_class is None else norm_class(**resolved_norm_kwargs)
        self.film = FiLM(self.cond_dim, self.dim)

        layer_kwargs = dict(layer_kwargs or {})
        self.fc1 = layer_class(self.dim, self.hidden_dim, **layer_kwargs)
        self.act = activation_resolver(activation)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.fc2 = layer_class(self.hidden_dim, self.dim, **layer_kwargs)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0 else None

        if self.gate == "layerscale":
            # Per-channel residual scaling (small init helps stability at depth).
            self.alpha = nn.Parameter(torch.full((self.dim,), float(gate_init)))
        elif self.gate == "rezero":
            # Identity at init; network learns to "turn on" residual branch.
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("alpha", None)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        if self.norm is not None:
            h = self.norm(h)
        h = self.film(h, cond)
        h = self.fc1(h)
        h = self.act(h)
        if self.dropout1 is not None:
            h = self.dropout1(h)
        h = self.fc2(h)
        if self.dropout2 is not None:
            h = self.dropout2(h)

        if self.gate == "none":
            return x + h
        return x + h * self.alpha


class FiLMResMLP(nn.Module):
    """
    Residual FiLM MLP that conditions every block on `cond`.

    Runs a stack of FiLMResFFNBlocks in a constant-width space (model_dim), with
    optional input/output projections when dimensions differ.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        cond_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        model_dim: int | None = None,
        activation: str | None = "mish",
        norm_class: type[nn.Module] | None = nn.LayerNorm,
        norm_kwargs: dict | list[dict] | None = None,
        dropout: float | None = None,
        gate: str = "layerscale",
        gate_init: float = 1e-4,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
        bias_last_layer: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.cond_dim = int(cond_dim)
        self.out_dim = int(out_dim)
        self.model_dim = int(model_dim) if model_dim is not None else self.in_dim
        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be > 0, got {self.model_dim}")

        layer_kwargs = dict(layer_kwargs or {})
        self.in_proj = (
            nn.Identity()
            if self.in_dim == self.model_dim
            else layer_class(self.in_dim, self.model_dim, **layer_kwargs)
        )

        blocks: list[nn.Module] = []
        for idx, h in enumerate(hidden_dims):
            if isinstance(norm_kwargs, list):
                block_norm_kwargs = (
                    norm_kwargs[idx]
                    if idx < len(norm_kwargs)
                    else (norm_kwargs[-1] if norm_kwargs else {})
                )
            else:
                block_norm_kwargs = norm_kwargs
            blocks.append(
                FiLMResFFNBlock(
                    dim=self.model_dim,
                    hidden_dim=int(h),
                    cond_dim=self.cond_dim,
                    activation=activation,
                    norm_class=norm_class,
                    norm_kwargs=block_norm_kwargs,
                    dropout=dropout,
                    gate=gate,
                    gate_init=gate_init,
                    layer_class=layer_class,
                    layer_kwargs=layer_kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        out_layer_kwargs = dict(layer_kwargs)
        if "bias" not in out_layer_kwargs:
            out_layer_kwargs["bias"] = bias_last_layer
        self.out = (
            nn.Identity()
            if self.model_dim == self.out_dim
            else layer_class(self.model_dim, self.out_dim, **out_layer_kwargs)
        )
        self.out_activation = (
            activation_resolver(activation) if activate_last_layer else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_activation(self.out(h))


class FiLMConcatResMLP(nn.Module):
    """
    Residual FiLM MLP that accepts concatenated input [cond, x] or [x, cond].
    Mirrors FiLMConcatMLP's public interface.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        cond_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        condition_position: str = "pre",
        model_dim: int | None = None,
        activation: str | None = "mish",
        norm_class: type[nn.Module] | None = nn.LayerNorm,
        norm_kwargs: dict | list[dict] | None = None,
        dropout: float | None = None,
        gate: str = "layerscale",
        gate_init: float = 1e-4,
        bias_last_layer: bool = True,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
    ) -> None:
        super().__init__()
        if condition_position not in ("pre", "post"):
            raise ValueError(
                "condition_position must be 'pre' or 'post', "
                f"got {condition_position!r}."
            )
        if cond_dim < 1:
            raise ValueError(f"cond_dim must be >= 1, got {cond_dim}")
        self.in_dim = int(in_dim)
        self.cond_dim = int(cond_dim)
        self.condition_position = condition_position
        self.res_mlp = FiLMResMLP(
            in_dim=self.in_dim,
            cond_dim=self.cond_dim,
            hidden_dims=hidden_dims,
            out_dim=int(out_dim),
            model_dim=model_dim,
            activation=activation,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            gate=gate,
            gate_init=gate_init,
            bias_last_layer=bias_last_layer,
            layer_class=layer_class,
            layer_kwargs=layer_kwargs,
            activate_last_layer=activate_last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.in_dim + self.cond_dim
        if x.size(-1) != expected:
            raise ValueError(
                f"Expected input feature size {expected}, got {x.size(-1)}."
            )
        if self.condition_position == "pre":
            cond = x[..., : self.cond_dim]
            feats = x[..., self.cond_dim :]
        else:
            feats = x[..., : self.in_dim]
            cond = x[..., self.in_dim :]
        return self.res_mlp(feats, cond)


class CentralFiLMFactory:
    """
    Build a centralized FiLM module with the same layer DSL used by ArityMLPFactory.

    The produced module expects concatenated inputs where condition is either
    prepended or appended, controlled by `condition_position`.
    """

    def __init__(
        self,
        layers: int | list[int | str] | tuple[int | str, ...] = 3,
        architecture: str = "residual",
        activation: str | None = "mish",
        out_dim: int | None = None,
        norm_class: type[nn.Module] | None = None,
        norm_kwargs: dict | list[dict] | None = None,
        dropout: float | None = None,
        residual_gate: str = "layerscale",
        residual_gate_init: float = 1e-4,
        residual_model_dim: int | None = None,
        bias_last_layer: bool = True,
        layer_class: type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
    ) -> None:
        self.layers = [-1] * layers if isinstance(layers, int) else list(layers)
        if architecture not in ("plain", "residual"):
            raise ValueError(
                f"architecture must be 'plain' or 'residual', got {architecture!r}."
            )
        self.architecture = architecture
        self.activation = activation
        self.out_dim = out_dim
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs
        self.dropout = dropout
        self.residual_gate = residual_gate
        self.residual_gate_init = float(residual_gate_init)
        self.residual_model_dim = residual_model_dim
        self.bias_last_layer = bias_last_layer
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs
        self.activate_last_layer = activate_last_layer

    def _hidden_dims(self, feature_dim: int) -> list[int]:
        hidden_dims: list[int] = []
        for layer in self.layers:
            hidden_dims.append(
                ArityMLPFactory.make_layer_size(
                    layer_mode=layer,
                    arity_feature_size=feature_dim,
                    prev_size=feature_dim if not hidden_dims else hidden_dims[-1],
                )
            )
        return hidden_dims

    def __call__(
        self,
        embedding_size: int,
        max_arity: int,
        cond_dim: int,
        mask_dim: int = 0,
        condition_position: str = "pre",
    ) -> nn.Module:
        feature_dim = int(embedding_size) * int(max_arity) + int(mask_dim)
        if feature_dim < 0:
            raise ValueError(f"feature_dim must be >= 0, got {feature_dim}.")
        out_dim = (
            self.out_dim if self.out_dim is not None else embedding_size * max_arity
        )
        hidden_dims = self._hidden_dims(feature_dim)
        if self.architecture == "plain":
            return FiLMConcatMLP(
                in_dim=feature_dim,
                cond_dim=cond_dim,
                hidden_dims=hidden_dims,
                out_dim=out_dim,
                condition_position=condition_position,
                activation=self.activation,
                norm_class=self.norm_class,
                norm_kwargs=self.norm_kwargs,
                dropout=self.dropout,
                bias_last_layer=self.bias_last_layer,
                layer_class=self.layer_class,
                layer_kwargs=self.layer_kwargs,
                activate_last_layer=self.activate_last_layer,
            )
        # Residual architecture: default to LayerNorm if the caller didn't specify a norm.
        norm_class = self.norm_class if self.norm_class is not None else nn.LayerNorm
        return FiLMConcatResMLP(
            in_dim=feature_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            condition_position=condition_position,
            model_dim=self.residual_model_dim,
            activation=self.activation,
            norm_class=norm_class,
            norm_kwargs=self.norm_kwargs,
            dropout=self.dropout,
            gate=self.residual_gate,
            gate_init=self.residual_gate_init,
            bias_last_layer=self.bias_last_layer,
            layer_class=self.layer_class,
            layer_kwargs=self.layer_kwargs,
            activate_last_layer=self.activate_last_layer,
        )

# --- Tuple-aware, conditional-aware modules and helper ---
class CondLinear(nn.Module):
    """Tuple-aware FiLM Linear.

    Forward signature: (x, cond) -> (y, cond)
    - x: [..., in_features]
    - cond: [..., cond_dim]

    Applies nn.Linear to x, then FiLM-modulates with cond, and returns the pair.
    """

    def __init__(
        self, in_features: int, out_features: int, cond_dim: int, bias: bool = True
    ):
        super().__init__()
        if cond_dim < 0:
            raise ValueError(f"cond_dim must be >= 0, got {cond_dim}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cond_dim = int(cond_dim)
        self.lin = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.film = FiLM(max(1, self.cond_dim), self.out_features)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cond_dim == 0:
            # pass a neutral zero-vector of size 1 to FiLM
            c = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        else:
            c = cond
        h = self.lin(x)
        h = self.film(h, c)
        return h, cond


class CondActivation(nn.Module):
    """Tuple-aware activation: applies act(x) and returns (y, cond)."""

    def __init__(self, activation: str | None = "mish"):
        super().__init__()
        self.act = activation_resolver(activation)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.act(x), cond


class CondNorm(nn.Module):
    """Tuple-aware norm wrapper: applies norm(x) and returns (y, cond)."""

    def __init__(self, norm_class: type[nn.Module] | None = None, **norm_kwargs):
        super().__init__()
        self.norm = None if norm_class is None else norm_class(**norm_kwargs)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm is None:
            return x, cond
        return self.norm(x), cond


class CondDropout(nn.Module):
    """Tuple-aware dropout wrapper: applies dropout(x) and returns (y, cond)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p) if p and p > 0 else None

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dropout is None:
            return x, cond
        return self.dropout(x), cond


class ConditionalSequential(nn.Sequential):
    """
    Tuple-aware sequential that propagates (x, cond).
    Each submodule can either accept (x, cond) or just x.
    If it accepts just x, cond is carried through unchanged.
    """

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self:
            try:
                out = layer(x, cond)
                if isinstance(out, tuple) and len(out) == 2:
                    x, cond = out
                else:
                    x = out
            except TypeError:
                x = layer(x)
        return x, cond
