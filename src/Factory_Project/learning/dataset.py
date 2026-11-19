"""Demonstration files: one HDF5 file per recording run, one group per episode.

Each episode holds, per 20 Hz control step, the environment observation, the action the teacher
commanded, and the teacher's skill label. Everything a policy is conditioned on is in the file, so
training needs no simulator.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

SKILLS = ("idle", "pick", "place", "retreat", "done")
"""Skill labels the teacher reports. Driving between docks happens inside ``pick`` and ``place``."""
MANIPULATION_SKILLS = ("pick", "place")


def skill_index(name: str) -> int:
    return SKILLS.index(name) if name in SKILLS else 0


class DemoWriter:
    """Appends episodes to an HDF5 demonstration file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "a")
        self._obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._skills: list[int] = []

    def add(self, observation: np.ndarray, action: np.ndarray, skill: str) -> None:
        self._obs.append(np.asarray(observation, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._skills.append(skill_index(skill))

    def end_episode(self, **attributes) -> str:
        """Store the buffered steps as one episode with the given attributes (seed, success, ...)."""
        name = f"episode_{len(self.file):04d}"
        group = self.file.create_group(name)
        group.create_dataset("obs", data=np.stack(self._obs), compression="gzip")
        group.create_dataset("actions", data=np.stack(self._actions), compression="gzip")
        group.create_dataset("skill", data=np.asarray(self._skills, dtype=np.int8), compression="gzip")
        for key, value in attributes.items():
            group.attrs[key] = value
        self.file.flush()
        self._obs, self._actions, self._skills = [], [], []
        return name

    def close(self) -> None:
        self.file.close()


def load_episodes(paths, successful_only: bool = False) -> list[dict]:
    """Read episodes from the given files as ``dict(obs, actions, skill, source)``.

    A mission that failed part-way still holds clean demonstrations of the items sorted before the
    failure. Unless ``successful_only`` is set, such an episode is kept up to the step at which its
    last item was verified in place (its ``good_steps``), and dropped if no item got that far.
    """
    episodes = []
    for path in sorted(Path(p) for p in paths):
        with h5py.File(path, "r") as file:
            for name in sorted(file):
                group = file[name]
                success = bool(group.attrs.get("success", False))
                keep = len(group["obs"]) if success else int(group.attrs.get("good_steps", 0))
                if keep == 0 or (successful_only and not success):
                    continue
                episodes.append(
                    dict(
                        obs=group["obs"][:keep],
                        actions=group["actions"][:keep],
                        skill=group["skill"][:keep].astype(np.int64),
                        source=f"{path.name}/{name}",
                    )
                )
    return episodes


def condition(obs: np.ndarray, skill: np.ndarray) -> np.ndarray:
    """Policy input: the observation followed by a one-hot of the skill the executive has selected."""
    one_hot = np.zeros((len(obs), len(SKILLS)), dtype=np.float32)
    one_hot[np.arange(len(obs)), skill] = 1.0
    return np.concatenate([obs.astype(np.float32), one_hot], axis=1)


def chunk_windows(episodes: list[dict], horizon: int, skills=MANIPULATION_SKILLS, stride: int = 1):
    """Training arrays: inputs (N, D), action chunks (N, horizon, A) and a mask of real (unpadded) steps.

    Only steps whose skill is in ``skills`` start a window. A window that runs past the end of its
    episode repeats the last action and masks the repeats out of the loss.
    """
    wanted = {skill_index(name) for name in skills}
    inputs, chunks, masks = [], [], []
    for episode in episodes:
        x = condition(episode["obs"], episode["skill"])
        actions = episode["actions"]
        length = len(actions)
        starts = [t for t in range(0, length, stride) if int(episode["skill"][t]) in wanted]
        for t in starts:
            end = min(t + horizon, length)
            chunk = np.repeat(actions[end - 1 : end], horizon, axis=0)
            chunk[: end - t] = actions[t:end]
            mask = np.zeros(horizon, dtype=np.float32)
            mask[: end - t] = 1.0
            inputs.append(x[t])
            chunks.append(chunk)
            masks.append(mask)
    return np.stack(inputs), np.stack(chunks), np.stack(masks)
