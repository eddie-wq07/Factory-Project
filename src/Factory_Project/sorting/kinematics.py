"""Whole-upper-body inverse kinematics for the Astribot S1 (torso + both arms).

Simulator-free: built on Pinocchio and Pink from the bundled URDF, so the same solver serves the
scripted teacher inside Isaac Lab and the unit tests. All poses are expressed in the chassis frame
(``astribot_torso_base``: X forward, Y left, Z up, origin 95 mm above the floor).

Each hand has a *palm* frame at the centre of the gripper housing. Its axes follow the gripper base:
Z points along the fingers and Y is the housing's thin direction, i.e. the normal of the two broad
side faces used to clamp a carton. A clamp target fixes the palm position and that Y axis and leaves
the rotation about Y (the forearm pitch) to the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pink
import pinocchio as pin
from pink.limits import ConfigurationLimit
from pink.tasks import ComTask, FrameTask, PostureTask

URDF_PATH = Path(__file__).resolve().parents[1] / "assets/data/astribot_s1/urdf/astribot_s1.urdf"

TORSO_JOINTS = tuple(f"astribot_torso_joint_{i}" for i in range(1, 5))
LEFT_ARM_JOINTS = tuple(f"astribot_arm_left_joint_{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"astribot_arm_right_joint_{i}" for i in range(1, 8))
HEAD_JOINTS = ("astribot_head_joint_1", "astribot_head_joint_2")
GRIPPER_JOINTS = ("astribot_gripper_left_joint_L1", "astribot_gripper_right_joint_L1")
WHEEL_JOINTS = ("wheel_RF_Joint", "wheel_LF_Joint", "wheel_RR_Joint", "wheel_LR_Joint")
IK_JOINTS = TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
"""Joint order of every configuration vector handled by :class:`UpperBodyIK` (18 values)."""

PALM_OFFSET = (0.0, 0.0, 0.045)
"""Palm frame origin in the gripper-base frame [m]: the centre of the housing's side faces."""
BASE_HEIGHT = 0.0952
"""Height of the chassis frame above the floor with the wheels on the ground [m]."""
SHIN, THIGH, TRUNK = 0.38, 0.39, 0.399
"""Leg and trunk segment lengths in the sagittal plane, from the URDF joint offsets [m]."""
ANKLE_HEIGHT = 0.12
"""Height of the first torso joint in the chassis frame [m]."""
HEAD_RADIUS, HEAD_OFFSET, HEAD_TILT_LIMIT = 0.11, 0.15, 1.0
"""Radius of the head's collision sphere [m], its distance from the head tilt joint [m], that joint's range [rad]."""
ARM_STIFFNESS = (800.0, 800.0, 800.0, 800.0, 400.0, 200.0, 200.0)
"""Position gains of the seven arm joints, shoulder to wrist [N·m/rad]. Simulation gains, used by the
asset configuration too. They are deliberately moderate: a two-handed clamp is held by commanding the
palms a few centimetres inside the carton, and softer arms turn the unavoidable millimetres of path
error into a small change of clamping force instead of a momentary release."""
ARM_DAMPING = (80.0, 80.0, 80.0, 80.0, 40.0, 20.0, 20.0)
"""Velocity gains of the seven arm joints [N·m·s/rad]."""
ELBOW_RADIUS = 0.06
"""Radius of the elbow's collision sphere [m]."""
STANDING_SHOULDER = 1.289
"""Shoulder height in the chassis frame with the torso upright [m]."""


def torso_for_shoulder(height: float, knee: float, forward: float = 0.03) -> np.ndarray:
    """Torso joint angles that put the shoulders at a chassis-frame height with the hips set back.

    The shin leans forward by ``knee`` radians and the thigh folds back from it, so the knee stays over
    the chassis and nothing below the shoulders projects toward a shelf or a stocked pallet.

    Args:
        height: Shoulder height in the chassis frame [m].
        knee: Forward lean of the shin, i.e. the first torso joint [rad].
        forward: Forward position of the shoulders in the chassis frame [m].

    Returns:
        The four torso joint angles [rad] (the waist yaw is zero).
    """
    import math

    knee_x, knee_z = SHIN * math.sin(knee), ANKLE_HEIGHT + SHIN * math.cos(knee)
    dx, dz = forward - knee_x, min(height, STANDING_SHOULDER - 1e-3) - knee_z
    reach = min(math.hypot(dx, dz), THIGH + TRUNK - 1e-6)
    bearing = math.atan2(dx, dz)
    fold = math.acos(max(-1.0, min(1.0, (THIGH**2 + reach**2 - TRUNK**2) / (2 * THIGH * reach))))
    thigh = bearing - fold
    hip_x, hip_z = knee_x + THIGH * math.sin(thigh), knee_z + THIGH * math.cos(thigh)
    trunk = math.atan2(knee_x + reach * math.sin(bearing) - hip_x, knee_z + reach * math.cos(bearing) - hip_z)
    return np.array([knee, min(0.0, max(-2.35, thigh - knee)), min(2.4, max(0.0, trunk - thigh)), 0.0])


@dataclass
class HandTarget:
    """Desired palm placement in the chassis frame."""

    position: np.ndarray
    """Palm origin [m]."""
    normal: np.ndarray | None = None
    """Direction the palm's +Y axis must take (unit vector). ``None`` leaves the orientation free."""
    fingers: np.ndarray | None = None
    """Direction the fingers (+Z) should take. With :attr:`normal` this fixes the full orientation."""
    free_pitch: bool = True
    """Leave the rotation about the palm normal to the solver, using :attr:`fingers` only as a seed."""


class UpperBodyIK:
    """Differential IK over the torso and both arms with joint limits and a balance task."""

    def __init__(self, urdf: Path | str = URDF_PATH):
        full = pin.buildModelFromUrdf(str(urdf))
        locked = [i for i in range(1, full.njoints) if full.names[i] not in IK_JOINTS]
        self.model = pin.buildReducedModel(full, locked, pin.neutral(full))
        for side in ("left", "right"):
            parent = self.model.getFrameId(f"astribot_gripper_{side}_base")
            frame = self.model.frames[parent]
            placement = frame.placement * pin.SE3(np.eye(3), np.array(PALM_OFFSET))
            self.model.addFrame(pin.Frame(f"{side}_palm", frame.parentJoint, parent, placement, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        names = [self.model.names[i] for i in range(1, self.model.njoints)]
        self._to_model = np.array([IK_JOINTS.index(n) for n in names])  # model order -> IK_JOINTS order
        self._from_model = np.argsort(self._to_model)
        self.lower = self.model.lowerPositionLimit[self._from_model].copy()
        self.upper = self.model.upperPositionLimit[self._from_model].copy()
        self._limit = ConfigurationLimit(self.model, config_limit_gain=0.5)

    # -- conversions -------------------------------------------------------------------------------------

    def _q_model(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float)[self._to_model]

    def _q_ik(self, q_model: np.ndarray) -> np.ndarray:
        return q_model[self._from_model]

    def clip(self, q: np.ndarray, margin: float = 1e-3) -> np.ndarray:
        return np.clip(q, self.lower + margin, self.upper - margin)

    # -- forward kinematics ------------------------------------------------------------------------------

    def forward(self, q: np.ndarray) -> dict[str, pin.SE3]:
        """Return the chassis-frame placements of both palms and the torso end frame."""
        pin.forwardKinematics(self.model, self.data, self._q_model(q))
        pin.updateFramePlacements(self.model, self.data)
        frames = ("left_palm", "right_palm", "astribot_torso_end_effector", "astribot_head_base_link")
        return {name: self.data.oMf[self.model.getFrameId(name)].copy() for name in frames}

    @staticmethod
    def torso_lean(q: np.ndarray) -> float:
        """Forward pitch of the trunk from vertical [rad]: the three sagittal torso joints add up."""
        return float(q[0] + q[1] + q[2])

    @classmethod
    def level_head_tilt(cls, q: np.ndarray) -> float:
        """Head tilt joint angle that keeps the head upright while the trunk leans [rad]."""
        return float(np.clip(-cls.torso_lean(q), -HEAD_TILT_LIMIT, HEAD_TILT_LIMIT))

    def upper_body_front(self, q: np.ndarray) -> float:
        """Forward extent of the head and chest in the chassis frame [m], from their collision shapes.

        Assumes the head is held level with :meth:`level_head_tilt`. Left at its zero angle, the head
        sphere would swing forward with the trunk and reach up to 0.1 m farther.
        """
        frames = self.forward(q)
        lean = self.torso_lean(q)
        residual = lean + self.level_head_tilt(q)
        head = frames["astribot_head_base_link"].translation[0] + HEAD_OFFSET * np.sin(residual) + HEAD_RADIUS
        chest = (
            frames["astribot_torso_end_effector"].translation[0] + 0.093 * np.cos(lean) + 0.113 * np.sin(max(lean, 0.0))
        )
        return float(max(head, chest))

    def lateral_stiffness(self, q: np.ndarray, lead: float = 0.0) -> float:
        """Stiffness of a position-controlled arm against a sideways push on its palm [N/m].

        The joint gains act as torsional springs, so the palm yields by ``J K^-1 J^T`` per unit force.
        The torso is treated as rigid; its gains are far higher. Returns the softer arm's value.

        Args:
            q: Posture in :data:`IK_JOINTS` order.
            lead: Distance of the pushed point ahead of the palm origin, along the fingers [m].
        """
        model, data = self.model, self.data
        pin.computeJointJacobians(model, data, self._q_model(q))
        pin.updateFramePlacements(model, data)
        softness = np.diag(1.0 / np.array(ARM_STIFFNESS))
        values = []
        for side, joints in (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)):
            frame = model.getFrameId(f"{side}_palm")
            jacobian = pin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)
            arm = data.oMf[frame].rotation @ np.array([0.0, 0.0, lead])
            columns = [model.joints[model.getJointId(name)].idx_v for name in joints]
            linear = (jacobian[:3] + np.cross(jacobian[3:].T, arm).T)[:, columns]
            values.append(1.0 / float((linear @ softness @ linear.T)[1, 1]))
        return min(values)

    def elbow_height(self, q: np.ndarray) -> float:
        """Height above the floor of the top of the higher elbow's collision sphere [m]."""
        pin.forwardKinematics(self.model, self.data, self._q_model(q))
        pin.updateFramePlacements(self.model, self.data)
        heights = [
            self.data.oMf[self.model.getFrameId(f"astribot_arm_{side}_link_4")].translation[2]
            for side in ("left", "right")
        ]
        return float(max(heights)) + BASE_HEIGHT + ELBOW_RADIUS

    def com(self, q: np.ndarray) -> np.ndarray:
        """Centre of mass of the upper body and chassis links in the chassis frame [m]."""
        return pin.centerOfMass(self.model, self.data, self._q_model(q)).copy()

    # -- inverse kinematics ------------------------------------------------------------------------------

    @staticmethod
    def _rotation(normal: np.ndarray, fingers: np.ndarray) -> np.ndarray:
        y = normal / np.linalg.norm(normal)
        z = fingers - y * float(fingers @ y)
        z /= np.linalg.norm(z)
        return np.column_stack([np.cross(y, z), y, z])

    def solve(
        self,
        seed: np.ndarray,
        left: HandTarget | None = None,
        right: HandTarget | None = None,
        *,
        posture: np.ndarray | None = None,
        posture_cost: float = 0.02,
        torso: np.ndarray | None = None,
        free_trunk: bool = False,
        knee: float | None = None,
        com_x: float | None = 0.0,
        iterations: int = 200,
        dt: float = 0.1,
        position_tolerance: float = 4e-3,
        angle_tolerance: float = 0.03,
    ) -> tuple[np.ndarray, bool]:
        """Find joint angles placing the palms on their targets.

        Args:
            seed: Start configuration in :data:`IK_JOINTS` order. Solutions stay near it.
            left: Left palm target, or ``None`` to leave that arm to the posture task.
            right: Right palm target.
            posture: Regularisation posture; defaults to the seed.
            posture_cost: Weight of the posture task relative to unit palm costs.
            torso: If given, the four torso joints are pinned to these angles [rad].
            free_trunk: With ``torso``, leave the trunk pitch (third torso joint) to the solver and
                pin only the legs and the waist. The trunk then leans as the reach requires.
            knee: If given, only the first torso joint is pinned to this angle [rad]. It alone sets
                how far the knee projects in front of the chassis.
            com_x: Desired forward centre-of-mass offset [m]; ``None`` disables the balance task.
            iterations: Maximum differential-IK steps.
            dt: Integration step of each differential-IK iteration [s].
            position_tolerance: Accepted palm position error [m].
            angle_tolerance: Accepted palm orientation error [rad].

        Returns:
            The configuration in :data:`IK_JOINTS` order and whether every target was met.
        """
        q = self.clip(np.array(seed, dtype=float))
        if torso is not None:
            trunk = q[2]
            q[:4] = torso
            if free_trunk:
                q[2] = trunk
        if knee is not None:
            q[0] = knee
        reference = self.clip(np.array(seed if posture is None else posture, dtype=float))
        configuration = pink.Configuration(self.model, self.data, self._q_model(q))
        tasks = []
        checks = []
        for side, target in (("left", left), ("right", right)):
            if target is None:
                continue
            name = f"{side}_palm"
            current = configuration.get_transform_frame_to_world(name)
            if target.normal is None:
                task = FrameTask(name, position_cost=1.0, orientation_cost=0.0)
                rotation = current.rotation
            else:
                fingers = target.fingers if target.fingers is not None else current.rotation[:, 2]
                rotation = self._rotation(np.asarray(target.normal, float), np.asarray(fingers, float))
                free = target.free_pitch
                # Orientation costs act on the palm's own axes: zero about Y frees the forearm pitch.
                task = FrameTask(name, position_cost=1.0, orientation_cost=[0.3, 0.004 if free else 0.3, 0.3])
            task.set_target(pin.SE3(rotation, np.asarray(target.position, float)))
            tasks.append(task)
            checks.append((name, target, rotation))
        posture_task = PostureTask(cost=posture_cost)
        posture_task.set_target(self._q_model(reference))
        tasks.append(posture_task)
        if com_x is not None:
            balance = ComTask(cost=[0.2, 0.0, 0.0])
            balance.set_target(np.array([com_x, 0.0, 0.0]))
            tasks.append(balance)
        # Pin joints by collapsing their position limits, so the QP itself keeps them still and the
        # remaining joints absorb the whole task.
        if torso is not None:
            pinned = [0, 1, 3] if free_trunk else [0, 1, 2, 3]
        else:
            pinned = [0] if knee is not None else []
        model_ids = [self.model.joints[self.model.getJointId(TORSO_JOINTS[i])].idx_q for i in pinned]
        lower, upper = self.model.lowerPositionLimit.copy(), self.model.upperPositionLimit.copy()
        for index, model_id in zip(pinned, model_ids, strict=True):
            self.model.lowerPositionLimit[model_id] = q[index] - 5e-4
            self.model.upperPositionLimit[model_id] = q[index] + 5e-4
        try:
            for _ in range(iterations):
                velocity = pink.solve_ik(
                    configuration, tasks, dt, solver="daqp", damping=1e-4, limits=[self._limit], safety_break=False
                )
                configuration.integrate_inplace(velocity, dt)
                if np.linalg.norm(velocity) * dt < 1e-5:
                    break
        finally:
            self.model.lowerPositionLimit[:] = lower
            self.model.upperPositionLimit[:] = upper
        q = self.clip(self._q_ik(configuration.q.copy()))
        return q, self._converged(q, checks, position_tolerance, angle_tolerance)

    def _converged(self, q, checks, position_tolerance, angle_tolerance) -> bool:
        frames = self.forward(q)
        for name, target, rotation in checks:
            pose = frames[name]
            if np.linalg.norm(pose.translation - target.position) > position_tolerance:
                return False
            if target.normal is not None:
                if target.free_pitch:
                    normal = np.asarray(target.normal, float) / np.linalg.norm(target.normal)
                    angle = float(np.arccos(np.clip(pose.rotation[:, 1] @ normal, -1.0, 1.0)))
                else:
                    angle = float(np.linalg.norm(pin.log3(rotation.T @ pose.rotation)))
                if angle > angle_tolerance:
                    return False
        return True
