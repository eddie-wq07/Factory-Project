# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The six-item sorting mission: one S1 in the full warehouse."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv

from Factory_Project.assets import SORTING_TASK_USD
from Factory_Project.sorting import kinematics
from Factory_Project.sorting.geometry import Pose, quat_from_yaw, wrap_angle, yaw_from_quat
from Factory_Project.sorting.manifest import load_manifest
from Factory_Project.sorting.stacking import (
    NoPlacementError,
    Placement,
    check_placement,
    plan_placement,
    stack_heights,
)

from .env_cfg import ROBOT_PRIM_PATH, SITE_PRIM_PATH

if TYPE_CHECKING:
    from .env_cfg import FactoryEnvCfg

UPPER_JOINTS = (
    kinematics.TORSO_JOINTS
    + kinematics.HEAD_JOINTS
    + kinematics.LEFT_ARM_JOINTS
    + kinematics.RIGHT_ARM_JOINTS
    + kinematics.GRIPPER_JOINTS
)
"""Order of the 22 joint-position targets in an action, after the three base-twist values."""
FORWARD_SIGNS = (-1.0, 1.0, -1.0, 1.0)
LEFT_SIGNS = (-1.0, -1.0, 1.0, 1.0)
TURN_SIGNS = (-1.0, -1.0, -1.0, -1.0)
"""Hub directions for chassis motion, in :data:`kinematics.WHEEL_JOINTS` order (RF, LF, RR, LR)."""

ON_RACK, IN_TRANSIT, PLACED, DROPPED = "rack", "transit", "placed", "dropped"


class FactoryEnv(DirectRLEnv):
    """Sort six rack items, alphabetically by address, onto their stocked drop pallets.

    Actions (25): chassis twist in the chassis frame (forward m/s, left m/s, yaw rad/s) followed by
    joint-position targets [rad] for :data:`UPPER_JOINTS`. Each gripper takes one driver target:
    0 opens it and 0.93 closes it.

    The environment owns the task state. It tracks which item is next in alphabetical order, plans
    that item's resting pose from the live stacks with the putaway rules, and marks it placed once it
    is released, at rest and inside tolerance. A scripted or learned controller only supplies actions.
    """

    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        robot = self._robot
        names = robot.joint_names
        self._upper_ids = torch.tensor([names.index(n) for n in UPPER_JOINTS], device=self.device)
        self._wheel_ids = torch.tensor([names.index(n) for n in kinematics.WHEEL_JOINTS], device=self.device)
        self._palm_bodies = [
            robot.body_names.index("astribot_gripper_left_base"),
            robot.body_names.index("astribot_gripper_right_base"),
        ]
        limits = robot.data.joint_pos_limits.torch[0, self._upper_ids]
        self._upper_low, self._upper_high = limits[:, 0].clone(), limits[:, 1].clone()
        # Wheel torque per unit chassis force and yaw moment. The four omni wheels sit on the chassis
        # diagonals, so each contributes cos(45 deg) of its traction to forward or lateral motion and
        # all of it to turning.
        force_to_torque = self.cfg.wheel_radius / (4.0 * math.cos(math.pi / 4))
        moment_to_torque = self.cfg.wheel_radius / (4.0 * self.cfg.wheel_offset)
        self._wheel_matrix = torch.tensor(
            [
                [force_to_torque * f, force_to_torque * lateral, moment_to_torque * t]
                for f, lateral, t in zip(FORWARD_SIGNS, LEFT_SIGNS, TURN_SIGNS, strict=True)
            ],
            device=self.device,
        )
        self._velocity_gain = torch.tensor(self.cfg.base_velocity_gain, device=self.device)
        self._force_limit = torch.tensor(self.cfg.base_force_limit, device=self.device)
        self._velocity_integral = torch.zeros(self.num_envs, 3, device=self.device)
        self._speed_limit = torch.tensor(self.cfg.base_speed_limit, device=self.device)
        self._twist = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_targets = torch.zeros(self.num_envs, len(UPPER_JOINTS), device=self.device)

        # The saved scene, not a config, defines where every object starts.
        self._initial_object_poses = {
            name: asset.data.root_pose_w.torch.clone() for name, asset in self._objects.items()
        }
        self._rng = np.random.default_rng(self.cfg.seed)
        self._status: dict[str, str] = {}
        self._placements: dict[str, Placement] = {}
        self._placed_time: dict[str, float] = {}
        self._rest_steps: dict[str, int] = {}
        self._metrics: dict[str, float] = {}
        self._clear_task_state()

    # -- scene -------------------------------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.manifest = load_manifest()
        self._robot = self.scene["robot"]
        site = self.sim.stage.DefinePrim(SITE_PRIM_PATH, "Xform")
        if not site.GetReferences().AddReference(str(SORTING_TASK_USD)):
            raise RuntimeError(f"Could not reference the sorting scene {SORTING_TASK_USD}")
        self.scene.filter_collisions(global_prim_paths=[SITE_PRIM_PATH])
        self._objects: dict[str, RigidObject] = {}
        for name, spec in self.manifest.objects.items():
            path = SITE_PRIM_PATH + spec.prim.removeprefix("/World")
            asset = RigidObject(RigidObjectCfg(prim_path=path, spawn=None))
            self.scene.rigid_objects[name] = asset
            self._objects[name] = asset
        self._cameras = {}
        if self.cfg.enable_cameras:
            from isaaclab.sensors.camera import Camera, CameraCfg

            width, height = self.cfg.camera_resolution
            for name, link in (
                ("head", "astribot_head_link_2/head_rgbd"),
                ("left_wrist", "astribot_arm_left_link_7/left_wrist_rgbd"),
                ("right_wrist", "astribot_arm_right_link_7/right_wrist_rgbd"),
            ):
                camera = Camera(
                    CameraCfg(
                        prim_path=ROBOT_PRIM_PATH + "/Geometry/" + link,
                        spawn=None,
                        data_types=["rgb"],
                        width=width,
                        height=height,
                        update_period=self.cfg.sim.dt * self.cfg.decimation,
                    )
                )
                self.scene.sensors[name + "_camera"] = camera
                self._cameras[name] = camera
        self._viewers = {}
        if self.cfg.viewer_cameras:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors.camera import Camera, CameraCfg

            for name, (width, height) in self.cfg.viewer_cameras.items():
                viewer = Camera(
                    CameraCfg(
                        prim_path=f"/World/Viewers/{name}",
                        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 200.0)),
                        data_types=["rgb"],
                        width=width,
                        height=height,
                    )
                )
                self.scene.sensors[f"viewer_{name}"] = viewer
                self._viewers[name] = viewer

    # -- privileged state accessors (simulator truth, for the scripted teacher and for checks) ---------

    @property
    def robot(self):
        return self._robot

    def base_pose(self) -> tuple[np.ndarray, float]:
        """Chassis position [m] and heading [rad] in the world."""
        position = self._robot.data.root_pos_w.torch[0].cpu().numpy()
        return position, yaw_from_quat(self._robot.data.root_quat_w.torch[0].cpu().numpy())

    def base_frame(self) -> Pose:
        position, yaw = self.base_pose()
        return Pose.from_yaw(position, yaw)

    def base_twist(self) -> np.ndarray:
        """Measured forward, lateral and yaw speed in the chassis frame."""
        linear = self._robot.data.root_lin_vel_b.torch[0].cpu().numpy()
        angular = self._robot.data.root_ang_vel_b.torch[0].cpu().numpy()
        return np.array([linear[0], linear[1], angular[2]])

    def upper_joint_positions(self) -> np.ndarray:
        return self._robot.data.joint_pos.torch[0, self._upper_ids].cpu().numpy()

    def ik_joint_positions(self) -> np.ndarray:
        """Torso and arm angles in :data:`kinematics.IK_JOINTS` order."""
        q = self.upper_joint_positions()
        return np.concatenate([q[:4], q[6:20]])

    def palm_poses(self) -> tuple[Pose, Pose]:
        """World poses of the left and right palm frames."""
        positions = self._robot.data.body_pos_w.torch[0, self._palm_bodies].cpu().numpy()
        quats = self._robot.data.body_quat_w.torch[0, self._palm_bodies].cpu().numpy()
        offset = Pose(kinematics.PALM_OFFSET)
        return tuple(Pose.from_quat(positions[i], quats[i]) * offset for i in range(2))

    def object_pose(self, name: str) -> Pose:
        """Live world pose of an object's box frame."""
        data = self._objects[name].data
        body = Pose.from_quat(data.root_pos_w.torch[0].cpu().numpy(), data.root_quat_w.torch[0].cpu().numpy())
        return self.manifest.objects[name].box_pose(body)

    def object_poses(self) -> dict[str, Pose]:
        return {name: self.object_pose(name) for name in self._objects}

    def object_speed(self, name: str) -> tuple[float, float]:
        data = self._objects[name].data
        return (
            float(torch.linalg.norm(data.root_lin_vel_w.torch[0])),
            float(torch.linalg.norm(data.root_ang_vel_w.torch[0])),
        )

    def camera_images(self) -> dict[str, np.ndarray]:
        """Latest onboard RGB images as ``uint8`` arrays of shape (height, width, 3)."""
        return {name: camera.data.output["rgb"][0, ..., :3].cpu().numpy() for name, camera in self._cameras.items()}

    def view(self, name: str, eye, target) -> np.ndarray:
        """Aim a viewer camera and return its freshly rendered RGB image, shape (height, width, 3)."""
        viewer = self._viewers[name]
        eye = torch.as_tensor([eye], dtype=torch.float32, device=self.device)
        target = torch.as_tensor([target], dtype=torch.float32, device=self.device)
        viewer.set_world_poses_from_view(eye, target)
        self.sim.render()
        viewer.update(dt=0.0, force_recompute=True)
        return viewer.data.output["rgb"][0, ..., :3].cpu().numpy()

    # -- task state --------------------------------------------------------------------------------------

    @property
    def status(self) -> dict[str, str]:
        """Status of every item: ``rack``, ``transit``, ``placed`` or ``dropped``."""
        return dict(self._status)

    @property
    def active_address(self) -> str | None:
        """The next item in alphabetical order that is not yet placed."""
        for address in self.manifest.order:
            if self._status[address] != PLACED:
                return address
        return None

    def placement(self, address: str) -> Placement:
        """The planned resting pose of an item, chosen once from the live stacks."""
        if address not in self._placements:
            self._placements[address] = plan_placement(self.manifest, address, self.object_poses())
        return self._placements[address]

    def is_released(self, address: str) -> bool:
        """True when neither palm is close enough to the item to be holding it."""
        item = self.manifest.item(address)
        inverse = self.object_pose(address).inverse()
        half = np.array(item.size) / 2
        for palm in self.palm_poses():
            outside = np.abs(inverse.apply(palm.position)) - half
            if float(np.linalg.norm(np.clip(outside, 0.0, None))) < self.cfg.release_clearance:
                return False
        return True

    def is_settled(self, address: str) -> bool:
        linear, angular = self.object_speed(address)
        return linear < self.cfg.settle_linear_speed and angular < self.cfg.settle_angular_speed

    def task_metrics(self) -> dict[str, float]:
        placed = sum(status == PLACED for status in self._status.values())
        heights = stack_heights(self.manifest, self.object_poses())
        tallest = max((max(counts) for counts in heights.values() if counts), default=0)
        return {
            **self._metrics,
            "items_placed": float(placed),
            "success": float(placed == len(self._status)),
            "tallest_stack": float(tallest),
            "stack_limit_respected": float(tallest <= self.manifest.max_stack),
            "elapsed_s": float(self.episode_length_buf[0]) * self.step_dt,
        }

    def _clear_task_state(self) -> None:
        self._status = {address: ON_RACK for address in self.manifest.order}
        self._placements = {}
        self._placed_time = {}
        self._rest_steps = {address: 0 for address in self.manifest.order}
        self._metrics = {"drops": 0.0, "order_violations": 0.0, "wrong_address": 0.0, "distance_m": 0.0}
        self._newly_placed = 0
        self._newly_dropped = 0

    def _update_task_state(self) -> None:
        self._newly_placed = self._newly_dropped = 0
        speed = self.base_twist()
        self._metrics["distance_m"] += float(math.hypot(speed[0], speed[1])) * self.step_dt
        for address in self.manifest.order:
            status = self._status[address]
            if status in (PLACED, DROPPED):
                continue
            item = self.manifest.item(address)
            pose = self.object_pose(address)
            start = np.array(item.center)
            if status == ON_RACK and np.linalg.norm(pose.position - start) > 0.05:
                status = IN_TRANSIT
            underside = pose.position[2] - item.size[2] / 2
            if pose.tilt > self.cfg.drop_tilt or (status == IN_TRANSIT and underside < self.cfg.floor_height):
                self._status[address] = DROPPED
                self._metrics["drops"] += 1
                self._newly_dropped += 1
                continue
            if status == IN_TRANSIT and self.is_released(address) and self.is_settled(address):
                self._rest_steps[address] += 1
            else:
                self._rest_steps[address] = 0
            # Half a second of continuous rest, hands off, before the placement is judged.
            if self._rest_steps[address] * self.step_dt >= 0.5:
                try:
                    check = check_placement(self.placement(address), pose)
                except NoPlacementError:
                    check = None
                if check is not None and check.placed:
                    status = PLACED
                    self._newly_placed += 1
                    self._placed_time[address] = float(self.episode_length_buf[0]) * self.step_dt
                    if address != self.active_address:
                        self._metrics["order_violations"] += 1
                elif self._on_wrong_pallet(address, pose):
                    self._metrics["wrong_address"] += 1
                    self._rest_steps[address] = 0
            self._status[address] = status

    def _on_wrong_pallet(self, address: str, pose: Pose) -> bool:
        for zone in self.manifest.destinations:
            if zone.address == address:
                continue
            for pallet in zone.pallets:
                if np.all(np.abs(pose.position[:2] - np.array(pallet["center"])) < np.array(pallet["half_extents"])):
                    return True
        return False

    # -- MDP ---------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.to(self.device, dtype=torch.float32).reshape(self.num_envs, -1).clone()
        self._twist = torch.clamp(self.actions[:, :3], -self._speed_limit, self._speed_limit)
        self._joint_targets = torch.clamp(self.actions[:, 3:], self._upper_low, self._upper_high)

    def _apply_action(self) -> None:
        """Runs every physics step: a PI loop turns the commanded chassis twist into wheel torques."""
        data = self._robot.data
        measured = torch.cat([data.root_lin_vel_b.torch[:, :2], data.root_ang_vel_b.torch[:, 2:3]], dim=-1)
        error = self._twist - measured
        self._velocity_integral += error * self.physics_dt / self.cfg.base_velocity_integral_time
        # Bleed the integral when asked to stand still, so the chassis parks instead of creeping.
        stopped = (self._twist.abs().sum(dim=-1, keepdim=True) < 1e-6).float()
        self._velocity_integral *= 1.0 - 0.1 * stopped
        wrench = self._velocity_gain * (error + self._velocity_integral)
        clipped = torch.clamp(wrench, -self._force_limit, self._force_limit)
        # Anti-windup: stop integrating in the direction that is already saturated.
        self._velocity_integral -= (wrench - clipped) / self._velocity_gain
        torque = torch.clamp(clipped @ self._wheel_matrix.T, -self.cfg.wheel_torque_limit, self.cfg.wheel_torque_limit)
        command = self._robot.actuators.target_command
        command.set_effort_index(value=torque, joint_ids=self._wheel_ids)
        command.set_position_index(value=self._joint_targets, joint_ids=self._upper_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        base = self.base_frame()
        inverse = base.inverse()
        twist = self.base_twist()
        joint_pos = self._robot.data.joint_pos.torch[0, self._upper_ids].cpu().numpy()
        joint_vel = self._robot.data.joint_vel.torch[0, self._upper_ids].cpu().numpy()
        left, right = self.palm_poses()
        address = self.active_address
        item_block = np.zeros(8)
        goal_block = np.zeros(5)
        one_hot = np.zeros(len(self.manifest.order))
        held = 0.0
        if address is not None:
            item = self.manifest.item(address)
            pose = inverse * self.object_pose(address)
            item_block = np.concatenate([pose.position, [math.cos(pose.yaw), math.sin(pose.yaw)], item.size])
            try:
                goal = inverse * self.placement(address).pose
                goal_block = np.concatenate([goal.position, [math.cos(goal.yaw), math.sin(goal.yaw)]])
            except NoPlacementError:
                pass
            one_hot[self.manifest.order.index(address)] = 1.0
            held = float(self._status[address] == IN_TRANSIT and not self.is_released(address))
        placed = sum(status == PLACED for status in self._status.values()) / len(self._status)
        observation = np.concatenate(
            [
                base.position[:2],
                [math.cos(base.yaw), math.sin(base.yaw)],
                twist,
                joint_pos,
                joint_vel,
                item_block,
                goal_block,
                inverse.apply(left.position),
                inverse.apply(right.position),
                [held, placed],
                one_hot,
            ]
        )
        return {"policy": torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)}

    def _get_rewards(self) -> torch.Tensor:
        self._update_task_state()
        reward = self.cfg.delivery_reward * self._newly_placed + self.cfg.drop_penalty * self._newly_dropped
        return torch.full((self.num_envs,), float(reward), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = all(status == PLACED for status in self._status.values())
        dropped = self.cfg.terminate_on_drop and any(status == DROPPED for status in self._status.values())
        terminated = torch.full((self.num_envs,), bool(success or dropped), dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if bool(terminated[0]) or bool(time_out[0]):
            self.extras["task"] = {**self.task_metrics(), "status": self.status, "placed_time": dict(self._placed_time)}
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        robot = self._robot
        root_pose = robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )
        joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        self._twist[env_ids] = 0.0
        self._velocity_integral[env_ids] = 0.0
        self._joint_targets[env_ids] = 0.0
        command = robot.actuators.target_command
        command.set_position_index(value=joint_pos, env_ids=env_ids, full_data=False)
        command.set_effort_index(value=torch.zeros_like(joint_pos), env_ids=env_ids, full_data=False)

        zero = torch.zeros(len(env_ids), 6, device=self.device)
        for name, asset in self._objects.items():
            pose = self._initial_object_poses[name][env_ids].clone()
            if name in self._status and (self.cfg.reset_item_xy_noise > 0 or self.cfg.reset_item_yaw_noise > 0):
                # Shift along the shelf and twist about the vertical; the depth stays flush with the beam.
                pose[:, 0] += float(self._rng.uniform(-1, 1)) * self.cfg.reset_item_xy_noise
                yaw = yaw_from_quat(pose[0, 3:].cpu().numpy())
                yaw = wrap_angle(yaw + float(self._rng.uniform(-1, 1)) * self.cfg.reset_item_yaw_noise)
                pose[:, 3:] = torch.as_tensor(quat_from_yaw(yaw), dtype=pose.dtype, device=self.device)
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=zero, env_ids=env_ids)
        self._clear_task_state()
