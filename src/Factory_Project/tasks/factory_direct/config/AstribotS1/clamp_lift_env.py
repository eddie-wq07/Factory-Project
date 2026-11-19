"""Clamp-and-lift skill environment: many docked S1 robots learn the two-handed carton grasp."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from Factory_Project.sorting import kinematics
from Factory_Project.sorting.geometry import quat_from_yaw
from Factory_Project.sorting.teacher import CONTACT_LEAD, HOUSING_HALF_THICKNESS, clamp_pregrasp

from .clamp_lift_env_cfg import CARTON_SIZE, DOCK, SHELF_TOP

if TYPE_CHECKING:
    from .clamp_lift_env_cfg import ClampLiftEnvCfg

ARM_JOINTS = kinematics.LEFT_ARM_JOINTS + kinematics.RIGHT_ARM_JOINTS
GRIPPER_CLOSED = 0.93


class ClampLiftEnv(DirectRLEnv):
    """Each robot is docked at a shelf with its hands open either side of a carton.

    Actions (14): changes of the left and right arm joint targets, scaled by ``cfg.action_scale``.
    The torso holds the teacher's shelf posture and the grippers stay closed into fists.

    Observations (57): arm joint angles relative to the start posture (14) and speeds (14), each
    palm's position relative to the carton (6), the carton's displacement from its start (3), its up
    axis (3) and forward axis (3), and the previous action (14).

    Reward: palms near the carton's sides, then closer together than the carton is wide, then the
    carton rising level off the deck. An episode ends early if the carton tips or is pushed away.
    """

    cfg: ClampLiftEnvCfg

    def __init__(self, cfg: ClampLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        robot = self._robot
        self._arm_ids = torch.as_tensor(robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0], device=self.device)
        self._palm_bodies = robot.find_bodies(
            ["astribot_gripper_left_base", "astribot_gripper_right_base"], preserve_order=True
        )[0]
        centre_height = SHELF_TOP + CARTON_SIZE[2] / 2
        opened, _ = clamp_pregrasp(CARTON_SIZE, centre_height, DOCK)
        names = robot.joint_names
        start = torch.zeros(len(names), device=self.device)
        for name, value in zip(kinematics.IK_JOINTS, opened, strict=True):
            start[names.index(name)] = float(value)
        for name in kinematics.GRIPPER_JOINTS:
            start[names.index(name)] = GRIPPER_CLOSED
        self._start_joints = start
        self._arm_start = start[self._arm_ids]
        limits = robot.data.joint_pos_limits.torch[0, self._arm_ids]
        self._arm_low, self._arm_high = limits[:, 0], limits[:, 1]
        self._targets = self._arm_start.repeat(self.num_envs, 1)
        self._previous = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._palm_offset = torch.tensor(kinematics.PALM_OFFSET, device=self.device).repeat(self.num_envs, 1)
        self._carton_start = torch.zeros(self.num_envs, 3, device=self.device)
        half = CARTON_SIZE[1] / 2 + HOUSING_HALF_THICKNESS
        # Where each palm bears in the carton's frame: its front edge level with the centre of mass.
        self._grip_points = torch.tensor(
            [[-CONTACT_LEAD, half, 0.06], [-CONTACT_LEAD, -half, 0.06]], device=self.device
        )

    def _setup_scene(self) -> None:
        self._robot = self.scene["robot"]
        self._carton = self.scene["carton"]

    # -- step ------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        self._targets = torch.clamp(self._targets + self.cfg.action_scale * self.actions, self._arm_low, self._arm_high)

    def _apply_action(self) -> None:
        self._robot.actuators.target_command.set_position_index(value=self._targets, joint_ids=self._arm_ids)

    def _palms(self) -> torch.Tensor:
        """Palm positions relative to the carton, in the world frame, shape (N, 2, 3)."""
        data = self._robot.data
        positions = data.body_pos_w.torch[:, self._palm_bodies]
        quats = data.body_quat_w.torch[:, self._palm_bodies]
        offsets = quat_apply(quats.reshape(-1, 4), self._palm_offset.repeat_interleave(2, dim=0)).view(
            self.num_envs, 2, 3
        )
        return positions + offsets - self._carton.data.root_pos_w.torch[:, None, :]

    def _carton_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self._carton.data.root_quat_w.torch
        unit_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unit_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        return quat_apply(quat, unit_x), quat_apply(quat, unit_z)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        data = self._robot.data
        forward, up = self._carton_axes()
        observation = torch.cat(
            [
                data.joint_pos.torch[:, self._arm_ids] - self._arm_start,
                data.joint_vel.torch[:, self._arm_ids] * 0.2,
                self._palms().reshape(self.num_envs, 6),
                self._carton.data.root_pos_w.torch - self._carton_start,
                up,
                forward,
                self._previous,
            ],
            dim=-1,
        )
        return {"policy": observation}

    def _task_terms(self) -> dict[str, torch.Tensor]:
        palms = self._palms()
        distance = (palms - self._grip_points).norm(dim=-1)
        reach = (1.0 - torch.tanh(6.0 * distance)).mean(dim=-1)
        # Squeeze: the palms closer together than the carton with a hand on each side, but only once
        # both are level with it, so shoving the carton sideways with one hand earns nothing.
        beside = ((palms - self._grip_points)[..., [0, 2]].norm(dim=-1) < 0.06).all(dim=-1).float()
        separation = palms[:, 0, 1] - palms[:, 1, 1]
        squeeze = beside * torch.clamp((self._grip_points[0, 1] * 2 - separation) / 0.06, 0.0, 1.0)
        displacement = self._carton.data.root_pos_w.torch - self._carton_start
        lift = torch.clamp(displacement[:, 2] / self.cfg.lift_goal, 0.0, 1.0)
        _, up = self._carton_axes()
        tilt = torch.acos(up[:, 2].clamp(-1.0, 1.0))
        push = displacement[:, :2].norm(dim=-1)
        return dict(reach=reach, squeeze=squeeze, lift=lift, tilt=tilt, push=push, height=displacement[:, 2])

    def _get_rewards(self) -> torch.Tensor:
        cfg, terms = self.cfg, self._task_terms()
        lifted = ((terms["lift"] >= 1.0) & (terms["tilt"] < 0.1)).float()
        rate = (self.actions - self._previous).square().sum(dim=-1)
        self._previous = self.actions.clone()
        reward = (
            cfg.reach_weight * terms["reach"]
            + cfg.squeeze_weight * terms["squeeze"]
            + cfg.lift_weight * terms["lift"] * (terms["tilt"] < 0.25).float()
            + cfg.success_bonus * lifted
            - cfg.tilt_weight * terms["tilt"]
            - cfg.push_weight * terms["push"]
            - cfg.action_rate_weight * rate
        )
        self.extras["log"] = {
            "reach": terms["reach"].mean(),
            "squeeze": terms["squeeze"].mean(),
            "lift_m": terms["height"].mean(),
            "lifted_fraction": lifted.mean(),
        }
        return reward * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terms = self._task_terms()
        failed = (terms["tilt"] > 0.7) | (terms["push"] > 0.25) | (terms["height"] < -0.08)
        return failed, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        robot, cfg = self._robot, self.cfg
        joints = self._start_joints.repeat(count, 1)
        noise = (torch.rand(count, len(self._arm_ids), device=self.device) * 2 - 1) * cfg.joint_noise
        joints[:, self._arm_ids] += noise
        robot.write_joint_position_to_sim_index(position=joints, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), env_ids=env_ids)
        robot.actuators.target_command.set_position_index(value=joints, env_ids=env_ids, full_data=False)
        self._targets[env_ids] = joints[:, self._arm_ids]
        self._previous[env_ids] = 0.0

        pose = self._carton.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, :2] += (torch.rand(count, 2, device=self.device) * 2 - 1) * cfg.carton_xy_noise
        yaws = (np.random.rand(count) * 2 - 1) * cfg.carton_yaw_noise
        pose[:, 3:] = torch.as_tensor(
            np.stack([quat_from_yaw(float(y)) for y in yaws]), dtype=pose.dtype, device=self.device
        )
        self._carton.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._carton.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )
        self._carton_start[env_ids] = pose[:, :3]
