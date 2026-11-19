"""Small pose helpers shared by the simulator-free sorting logic. Quaternions are (x, y, z, w)."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi] [rad]."""
    return math.remainder(angle, math.tau)


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def yaw_from_quat(quat: np.ndarray) -> float:
    x, y, z, w = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotation_from_quat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tilt_from_quat(quat: np.ndarray) -> float:
    """Angle between the body's Z axis and world up [rad]."""
    return float(math.acos(max(-1.0, min(1.0, rotation_from_quat(quat)[2, 2]))))


class Pose:
    """Rigid transform with a rotation matrix and a translation."""

    __slots__ = ("position", "rotation")

    def __init__(self, position, rotation=None):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)

    @classmethod
    def from_quat(cls, position, quat) -> Pose:
        return cls(position, rotation_from_quat(np.asarray(quat, dtype=float)))

    @classmethod
    def from_yaw(cls, position, yaw: float) -> Pose:
        return cls(position, rotation_z(yaw))

    @property
    def yaw(self) -> float:
        return math.atan2(self.rotation[1, 0], self.rotation[0, 0])

    @property
    def tilt(self) -> float:
        return float(math.acos(max(-1.0, min(1.0, self.rotation[2, 2]))))

    def __mul__(self, other: Pose) -> Pose:
        return Pose(self.position + self.rotation @ other.position, self.rotation @ other.rotation)

    def inverse(self) -> Pose:
        return Pose(-self.rotation.T @ self.position, self.rotation.T)

    def apply(self, point) -> np.ndarray:
        return self.position + self.rotation @ np.asarray(point, dtype=float)

    def __repr__(self) -> str:
        return f"Pose(position={np.round(self.position, 3)}, yaw={self.yaw:.3f})"
