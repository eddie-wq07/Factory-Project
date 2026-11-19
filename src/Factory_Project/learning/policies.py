"""Behaviour-cloning policies for the docked manipulation skills.

Two models share one interface (:meth:`ChunkPolicy.forward` returns a chunk of future actions):

* :class:`MLPPolicy`, the small baseline: a plain network predicting the next action (a chunk of 1).
* :class:`ChunkTransformer`, in the spirit of ACT: a transformer that predicts the next ``horizon``
  actions at once from the current observation. Predicting chunks shortens the effective horizon and
  smooths over pauses in the demonstrations. It is trained with an L1 loss and has no latent
  variable, since the scripted teacher is close to deterministic.

:class:`PolicyRunner` wraps either for control: it normalises inputs, and blends overlapping chunk
predictions (temporal ensembling) into one action per step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ChunkPolicy(nn.Module):
    horizon: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) -> (B, horizon, A)
        raise NotImplementedError


class MLPPolicy(ChunkPolicy):
    def __init__(self, input_dim: int, action_dim: int, hidden: int = 512, layers: int = 3, horizon: int = 1):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        blocks, width = [], input_dim
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ELU()]
            width = hidden
        self.net = nn.Sequential(*blocks, nn.Linear(width, action_dim * horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.horizon, self.action_dim)


class ChunkTransformer(ChunkPolicy):
    def __init__(
        self, input_dim: int, action_dim: int, horizon: int = 20, width: int = 256, layers: int = 4, heads: int = 4
    ):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.embed = nn.Sequential(nn.Linear(input_dim, width), nn.ELU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(horizon, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.embed(x).unsqueeze(1)
        tokens = torch.cat([state, self.queries.unsqueeze(0).expand(len(x), -1, -1)], dim=1)
        return self.head(self.encoder(tokens)[:, 1:])


def build_policy(kind: str, input_dim: int, action_dim: int, horizon: int) -> ChunkPolicy:
    if kind == "mlp":
        return MLPPolicy(input_dim, action_dim, horizon=1)
    if kind == "act":
        return ChunkTransformer(input_dim, action_dim, horizon=horizon)
    raise ValueError(f"Unknown policy kind {kind!r}; choose 'mlp' or 'act'")


def save_checkpoint(path: Path | str, policy: ChunkPolicy, kind: str, stats: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=kind, horizon=policy.horizon, state=policy.state_dict(), stats=stats, meta=meta), path)


class PolicyRunner:
    """Runs a trained policy one control step at a time."""

    def __init__(self, checkpoint: Path | str, device: str = "cpu", ensemble_decay: float = 0.1):
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        stats = data["stats"]
        self.in_mean, self.in_std = stats["in_mean"], stats["in_std"]
        self.out_mean, self.out_std = stats["out_mean"], stats["out_std"]
        self.kind, self.meta = data["kind"], data["meta"]
        self.policy = build_policy(self.kind, len(self.in_mean), len(self.out_mean), data["horizon"]).to(device)
        self.policy.load_state_dict(data["state"])
        self.policy.eval()
        self.device, self.decay = device, ensemble_decay
        self.reset()

    def reset(self) -> None:
        """Forget earlier chunk predictions; call when the policy takes over control."""
        self._chunks: list[np.ndarray] = []

    @torch.no_grad()
    def act(self, model_input: np.ndarray) -> np.ndarray:
        """Action for this step from the conditioned input (see :func:`dataset.condition`)."""
        x = (np.asarray(model_input, dtype=np.float32) - self.in_mean) / self.in_std
        chunk = self.policy(torch.as_tensor(x, device=self.device).unsqueeze(0))[0].cpu().numpy()
        chunk = chunk * self.out_std + self.out_mean
        # Temporal ensembling: every earlier chunk also predicted this step; average them with the
        # oldest prediction weighted most, as in ACT.
        self._chunks = [c[1:] for c in self._chunks if len(c) > 1] + [chunk]
        votes = np.stack([c[0] for c in self._chunks])
        weights = np.exp(-self.decay * np.arange(len(votes)))
        return (votes * weights[:, None]).sum(axis=0) / weights.sum()
