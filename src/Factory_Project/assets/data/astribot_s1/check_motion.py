"""Exercise the shipped S1 drives in an isolated Isaac Sim PhysX scene."""

import argparse
import math
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WHEEL_ORDER = ["wheel_RF_Joint", "wheel_LF_Joint", "wheel_RR_Joint", "wheel_LR_Joint"]


def yaw(pose):
    x, y, z, w = pose[3:]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def check_motion(asset, hz, device):
    import carb.settings
    import omni.physics.tensors as tensors
    import omni.usd
    import torch
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    gpu = device == "cuda"
    physx_scene.CreateEnableGPUDynamicsAttr(gpu)
    physx_scene.CreateBroadphaseTypeAttr("GPU" if gpu else "MBP")
    carb.settings.get_settings().set_bool("/physics/suppressReadback", gpu)

    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set((0, 0, -0.05))
    floor.AddScaleOp().Set((20, 20, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    robot = UsdGeom.Xform.Define(stage, "/World/S1")
    if not robot.GetPrim().GetReferences().AddReference(str(asset.resolve())):
        raise RuntimeError(f"Failed to reference {asset}")
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bottom = cache.ComputeWorldBound(robot.GetPrim()).ComputeAlignedRange().GetMin()[2]
    robot.AddTranslateOp(opSuffix="testSpawn").Set((0, 0, -bottom + 0.003))
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation, found {roots}")
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation = get_physx_simulation_interface()
    simulation.attach_stage(stage_id)
    try:
        simulation.simulate(1 / hz, 0.0)
        simulation.fetch_results()
        views = tensors.create_simulation_view("torch", stage_id)
        if torch.device(views.device).type != device:
            raise RuntimeError(f"Expected {device} physics tensors, got {views.device}")
        art = views.create_articulation_view(str(roots[0].GetPath()))
        if art.count != 1 or art.shared_metatype.fixed_base:
            raise RuntimeError("Expected one floating-base robot")
        meta = art.shared_metatype
        urdf = ET.parse(HERE / "urdf" / "astribot_s1.urdf")
        joints = {j.attrib["name"]: j for j in urdf.findall("joint") if j.get("type") != "fixed"}
        indices = meta.dof_indices
        if set(indices) != set(joints):
            raise RuntimeError(f"PhysX/URDF DOF mismatch: {sorted(set(joints) ^ set(indices))}")
        frame = 1

        def step(seconds):
            nonlocal frame
            for _ in range(round(seconds * hz)):
                simulation.simulate(1 / hz, frame / hz)
                simulation.fetch_results()
                frame += 1
            if (
                not torch.isfinite(art.get_link_transforms()).all()
                or not torch.isfinite(art.get_dof_velocities()).all()
            ):
                raise RuntimeError(f"Nonfinite physics state after step {frame}")
            pose = art.get_root_transforms()[0].cpu().numpy()
            if abs(pose[3]) + abs(pose[4]) > 0.15:
                raise RuntimeError(f"Unstable base after step {frame}: {pose}")

        step(2)
        settled = art.get_root_transforms()[0].cpu().numpy().copy()
        step(2)
        drift = np.linalg.norm(art.get_root_transforms()[0, :2].cpu().numpy() - settled[:2])
        if drift > 0.03:
            raise RuntimeError(f"Chassis drifted {drift:.3f} m at rest")
        print(f"Idle drift over 2 s: {drift:.4f} m")
        home = art.get_dof_positions().cpu().numpy().copy()
        target = art.get_dof_position_targets().clone()
        instances = torch.tensor([0], dtype=torch.int32, device=views.device)
        slaves = {}
        driven = {}
        passive = {name for name in joints if "_roller_joint_" in name}
        for name, joint in joints.items():
            follower = joint.find("mimic")
            if follower is not None:
                slaves[name] = (follower.attrib["joint"], float(follower.get("multiplier", "1")))
            elif name not in WHEEL_ORDER and name not in passive:
                limit = joint.find("limit")
                direction = -1 if float(limit.attrib["upper"]) <= 0 else 1
                delta = 0.35 if "gripper" in name else 0.15
                driven[name] = direction * delta
                target[0, indices[name]] += driven[name]
        art.set_dof_position_targets(target, instances)
        step(2)
        q = art.get_dof_positions()[0].cpu().numpy()
        weak = {}
        for name, delta in driven.items():
            motion = q[indices[name]] - home[0, indices[name]]
            if motion / delta < 0.35 or abs(q[indices[name]] - target[0, indices[name]].item()) > 0.11:
                weak[name] = round(float(motion), 4)
        if weak:
            raise RuntimeError(f"Joint tracking failed. Measured travel (radians): {weak}")
        for slave, (master, ratio) in slaves.items():
            if abs(q[indices[slave]] - ratio * q[indices[master]]) > 0.05:
                raise RuntimeError(f"{slave} failed to follow {master}")
        if len(driven) != 22 or len(slaves) != 10 or len(passive) != 64:
            raise RuntimeError("Incomplete upper body, grippers, or wheel roller assemblies")
        print(f"PASS {len(driven)} position drives, {len(slaves)} gripper followers, {len(passive)} passive rollers.")
        wheels = [indices[n] for n in WHEEL_ORDER]
        for label, signs, axis in (
            ("forward", [-1, 1, -1, 1], 0),
            ("sideways", [-1, -1, 1, 1], 1),
            ("turn", [-1, -1, -1, -1], 2),
        ):
            before = art.get_root_transforms()[0].cpu().numpy().copy()
            speeds = torch.zeros_like(target)
            speeds[0, wheels] = torch.tensor(signs, dtype=target.dtype, device=views.device) * 3.0
            art.set_dof_velocity_targets(speeds, instances)
            angles = [art.get_dof_positions()[0, wheels].cpu().numpy().copy()]
            for _ in range(6):
                step(0.25)
                angles.append(art.get_dof_positions()[0, wheels].cpu().numpy().copy())
            after = art.get_root_transforms()[0].cpu().numpy().copy()
            # Estimate rate from net wheel travel. Instantaneous velocity oscillates at roller handoff.
            unwrapped = np.unwrap(angles, axis=0)
            actual = (unwrapped[-1] - unwrapped[0]) / 1.5
            if min(actual * signs) < 0.8:
                raise RuntimeError(f"{label} wheel speeds (rad/s) {actual}")
            yaw0, yaw1 = yaw(before), yaw(after)
            c, s = math.cos(yaw0), math.sin(yaw0)
            dx, dy = after[:2] - before[:2]
            displacement = (c * dx + s * dy, -s * dx + c * dy, math.remainder(yaw1 - yaw0, math.tau))
            print(f"{label}: body-frame dx, dy, yaw = {tuple(round(float(v), 3) for v in displacement)}", flush=True)
            if displacement[axis] < (0.5 if axis == 2 else 0.20):
                raise RuntimeError(f"Insufficient chassis motion during {label}")
            if axis < 2 and (abs(displacement[1 - axis]) > 0.09 or abs(displacement[2]) > 0.25):
                raise RuntimeError(f"Excessive sideways drift or turning during {label}")
            if axis == 2 and math.hypot(dx, dy) > 0.08:
                raise RuntimeError("Chassis translated excessively during in-place turn")
            art.set_dof_velocity_targets(torch.zeros_like(target), instances)
            step(1)
        print(f"PASS floating-base locomotion over {frame / hz:.1f} simulated seconds.")
    finally:
        simulation.detach_stage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=HERE / "astribot_s1.usd")
    parser.add_argument("--hz", type=int, choices=(60, 120, 240), default=120, help="Physics steps per second")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PhysX simulation device")
    args = parser.parse_args()
    asset = args.asset
    if not asset.is_file():
        parser.error(f"Missing USD {asset}")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    try:
        check_motion(asset, args.hz, args.device)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()
