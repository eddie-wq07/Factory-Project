"""Chase-camera recording of a mission, for ``run_mission.py --record``."""

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

CAMERA = "chase"
RESOLUTION = (1280, 720)
RECORDINGS = Path(__file__).resolve().parents[1] / "recordings" / "sorting-mission"


class ChaseRecorder:
    """Films the robot from behind its right shoulder and writes an MP4.

    The environment must have been created with ``cfg.viewer_cameras = {CAMERA: RESOLUTION}``.
    The camera eases toward its place behind the robot, so a chassis that turns on the spot does not
    whip the picture around.
    """

    def __init__(
        self, env, label: str, every: int = 2, offset=(-2.4, -1.5, 1.9), look=(0.7, 0.0, 0.85), ease: float = 0.08
    ):
        self.env = env
        self.every = max(1, every)
        self.offset = np.asarray(offset, dtype=float)
        self.look = np.asarray(look, dtype=float)
        self.ease = ease
        self.count = 0
        self.eye = None
        self.target = None
        RECORDINGS.mkdir(parents=True, exist_ok=True)
        self.path = RECORDINGS / f"{label}.mp4"
        fps = round(1.0 / (env.cfg.sim.dt * env.cfg.decimation * self.every))
        self.writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=7, macro_block_size=None)

    def _wanted(self):
        position, yaw = self.env.base_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        base = np.array([position[0], position[1], 0.0])
        return base + turn @ self.offset, base + turn @ self.look

    def capture(self) -> None:
        """Call once per control step; every ``every``-th call adds a frame."""
        eye, target = self._wanted()
        if self.eye is None:
            self.eye, self.target = eye, target
        else:
            self.eye += self.ease * (eye - self.eye)
            self.target += 0.3 * (target - self.target)
        self.count += 1
        if self.count % self.every:
            return
        self.writer.append_data(self.env.view(CAMERA, self.eye, self.target))

    def close(self) -> Path:
        self.writer.close()
        return self.path
