"""
irAEGIS model architectures.

- PathwayAE             : Pathway-masked denoising autoencoder
- LinearIraeClassifier  : Single linear layer used as the per-CT irAE classifier

Activation: GELU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCondBatchNorm1d(nn.Module):
    # BatchNorm with separate running stats per cell type.

    def __init__(self, num_features: int, n_ct: int, **bn_kwargs):
        super().__init__()
        self.num_features = num_features
        self.n_ct = n_ct
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features, **bn_kwargs) for _ in range(n_ct)]
        )
        self._ct_ctx: int | None = None

    def set_ct_context(self, ct_idx: int | None):
        self._ct_ctx = ct_idx

    def forward(self, x: torch.Tensor,
                ct_ids: torch.Tensor | None = None) -> torch.Tensor:
        if ct_ids is not None:
            out = torch.empty_like(x)
            for k in range(self.n_ct):
                mask = ct_ids == k
                if mask.any():
                    if mask.sum() == 1 and self.training:
                        self.bns[k].eval()
                        out[mask] = self.bns[k](x[mask])
                        self.bns[k].train()
                    else:
                        out[mask] = self.bns[k](x[mask])
            return out

        if self._ct_ctx is not None:
            return self.bns[self._ct_ctx](x)

        return self.bns[0](x)


class PathwayAE(nn.Module):
    """
    Pathway-masked autoencoder.

    Encoder
    -------
    x (G,) -> [W . mask] (G x P linear, only in-pathway connections) -> Norm -> GELU
            -> h (P,)  <- pathway activation vector
            -> BN + GELU + Dropout
            -> z (L,)  <- denoised latent

    Decoder
    -------
    z (L,) -> Linear(L, 4P) -> GELU -> Linear(4P, G) -> x_recon

    mask shape: (G, P) -- gene x pathway, 1 where gene g belongs to pathway p.
    """

    def __init__(self, n_genes: int, n_pathways: int,
                 mask: torch.Tensor,       # (G, P)
                 latent_dim: int = 32,
                 dropout: float = 0.1,
                 norm: str = "bn",         # "bn", "ln", or "ctbn"
                 act: str = "gelu",        # "gelu" or "relu"
                 n_ct: int = 0,            # required when norm="ctbn"
                 ):
        super().__init__()
        self.n_genes    = n_genes
        self.n_pathways = n_pathways
        self.latent_dim = latent_dim
        self._act_name  = act
        self._norm_name = norm

        # Encoder: pathway-masked linear  (G -> P)
        self.pw_weight = nn.Parameter(torch.empty(n_genes, n_pathways))
        nn.init.xavier_uniform_(self.pw_weight)
        self.register_buffer("mask", mask.float())   # (G, P)

        if norm == "ctbn":
            assert n_ct > 0, "n_ct required for CT-conditional BatchNorm"
            self.pw_norm = CTCondBatchNorm1d(n_pathways, n_ct)
        elif norm == "ln":
            self.pw_norm = nn.LayerNorm(n_pathways)
        else:
            self.pw_norm = nn.BatchNorm1d(n_pathways)

        self._act_fn = F.gelu if act == "gelu" else F.relu

        # Encoder: latent projection  (P -> L)
        act_module = nn.GELU() if act == "gelu" else nn.ReLU()
        self.pw_to_z = nn.Sequential(
            nn.Linear(n_pathways, latent_dim),
            nn.BatchNorm1d(latent_dim),
            act_module,
            nn.Dropout(dropout),
        )

        # Decoder:  (L -> G)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_pathways * 4),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Linear(n_pathways * 4, n_genes),
        )

        # Auxiliary cell-type head on h
        self.ct_head = None

    def set_ct_context(self, ct_idx: int | None):
        # Set CT context for inference with CT-conditional BN.
        if isinstance(self.pw_norm, CTCondBatchNorm1d):
            self.pw_norm.set_ct_context(ct_idx)

    def masked_weight(self) -> torch.Tensor:
        """Effective encoder weight (G, P) -- zeros outside pathway mask.
        Uses softplus to constrain weights non-negative.
        """
        return F.softplus(self.pw_weight) * self.mask

    def encode_to_h(self, x: torch.Tensor,
                    ct_ids: torch.Tensor | None = None) -> torch.Tensor:
        """x (cells, G) -> h (cells, P) -- pathway activations."""
        h_pre = x @ self.masked_weight()     # (cells, P)
        if isinstance(self.pw_norm, CTCondBatchNorm1d):
            return self._act_fn(self.pw_norm(h_pre, ct_ids=ct_ids))
        return self._act_fn(self.pw_norm(h_pre))

    def encode(self, x: torch.Tensor,
               ct_ids: torch.Tensor | None = None):
        """x -> (h, z)."""
        h = self.encode_to_h(x, ct_ids=ct_ids)
        z = self.pw_to_z(h)
        return h, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)               # (cells, G)

    def attach_ct_head(self, n_ct: int):
        """Create a linear cell-type classifier on h (training only)."""
        self.ct_head = nn.Linear(self.n_pathways, n_ct)

    def forward(self, x: torch.Tensor,
                ct_ids: torch.Tensor | None = None):
        h, z = self.encode(x, ct_ids=ct_ids)
        return h, z, self.decode(z)


class LinearIraeClassifier(nn.Module):
    """Single linear layer for per-CT irAE classification (logistic regression).
    Equivalent to sklearn LogisticRegression but torch-compatible for
    gradient-based attribution in the explainability pipeline.
    """

    def __init__(self, n_pathways: int):
        super().__init__()
        self.linear = nn.Linear(n_pathways, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
