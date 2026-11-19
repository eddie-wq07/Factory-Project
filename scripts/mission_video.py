"""Mission recording for ``run_mission.py --record``: a chase camera, optionally with the robot's own cameras."""

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


TILE = (960, 540)
"""Width and height at which each of the four views is rendered [px]."""
SIDE = (480, 270)
"""Size of each onboard view in the finished frame [px]; the chase view takes the remaining 1440 x 810."""
ONBOARD = (("head", "Head camera"), ("left_wrist", "Left wrist camera"), ("right_wrist", "Right wrist camera"))


class QuadRecorder(ChaseRecorder):
    """Chase camera on the left three quarters of the frame, head and wrist cameras stacked on the right.

    The environment must have been created with ``cfg.viewer_cameras = {CAMERA: TILE}``,
    ``cfg.enable_cameras = True`` and ``cfg.camera_resolution = TILE``. Frames are 1920 x 810.
    """

    @staticmethod
    def _tile(image: np.ndarray, size: tuple[int, int], text: str) -> np.ndarray:
        import cv2

        image = cv2.resize(np.ascontiguousarray(image), size, interpolation=cv2.INTER_AREA)
        scale = size[0] / 960
        cv2.rectangle(image, (0, 0), (int((20 + 15 * len(text)) * scale), int(40 * scale)), (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (int(10 * scale), int(28 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            (255, 255, 255),
            max(1, round(2 * scale)),
            cv2.LINE_AA,
        )
        return image

    def capture(self) -> None:
        eye, target = self._wanted()
        if self.eye is None:
            self.eye, self.target = eye, target
        else:
            self.eye += self.ease * (eye - self.eye)
            self.target += 0.3 * (target - self.target)
        self.count += 1
        if self.count % self.every:
            return
        chase = self.env.view(CAMERA, self.eye, self.target)  # renders, so the onboard images are fresh too
        onboard = self.env.camera_images(refresh=True)
        main = self._tile(chase, (3 * SIDE[0], 3 * SIDE[1]), "Third person")
        side = np.vstack([self._tile(onboard[name], SIDE, text) for name, text in ONBOARD])
        self.writer.append_data(np.hstack([main, side]))
