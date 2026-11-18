"""Check both S1 grippers can retain and release a free block in Isaac Sim."""

import traceback
from pathlib import Path

import numpy as np

ASSET = Path(__file__).with_name("astribot_s1.usd")
BLOCK_WIDTH = 0.035
TIMESTEP = 1 / 240


def check_grasp():
    import omni.physics.tensors as tensors
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdPhysics.Scene.Define(stage, "/Physics")
    scene.CreateGravityDirectionAttr((0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    api.CreateEnableGPUDynamicsAttr(False)
    api.CreateBroadphaseTypeAttr("MBP")

    robot = UsdGeom.Xform.Define(stage, "/S1")
    robot.GetPrim().GetReferences().AddReference(str(ASSET.resolve()))
    root = next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    # Clamp the chassis to a test bench to isolate the grippers under gravity.
    bench = UsdPhysics.FixedJoint.Define(stage, "/BenchMount")
    bench.CreateBody1Rel().SetTargets([root.GetPath()])
    sides = ("left", "right")
    for index, side in enumerate(sides):
        cube = UsdGeom.Cube.Define(stage, f"/Block_{side}")
        cube.CreateSizeAttr(BLOCK_WIDTH)
        cube.AddTranslateOp().Set((2 + index, 0, 1))
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.020)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation = get_physx_simulation_interface()
    simulation.attach_stage(stage_id)
    frame = 0

    def step(count):
        nonlocal frame
        for _ in range(count):
            simulation.simulate(TIMESTEP, frame * TIMESTEP)
            simulation.fetch_results()
            frame += 1

    try:
        step(1)
        views = tensors.create_simulation_view("numpy", stage_id)
        articulation = views.create_articulation_view(str(root.GetPath()))
        blocks = {side: views.create_rigid_body_view(f"/Block_{side}") for side in sides}
        meta = articulation.shared_metatype
        indices = np.array([0], dtype=np.int32)
        targets = articulation.get_dof_position_targets().copy()

        def pad_frame(side):
            poses = articulation.get_link_transforms()[0]
            centers = []
            for jaw, x in (("L", -0.003), ("R", 0.003)):
                pose = poses[meta.link_indices[f"astribot_gripper_{side}_Link_{jaw}11"]].astype(float)
                orientation = Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6]))
                centers.append(pose[:3] + orientation.Transform((x, 0, 0.036)))
            midpoint = np.mean(centers, axis=0)
            gap = np.linalg.norm(centers[0] - centers[1]) - 0.010
            return midpoint, pose[3:], gap

        step(200)
        placements = {}
        for side in sides:
            center, orientation, gap = pad_frame(side)
            if gap < 0.085:
                raise ValueError(f"{side} gripper did not start open ({gap:.4f} m gap)")
            placements[side] = np.array([[*center, *orientation]], dtype=np.float32)
            targets[0, meta.dof_indices[f"astribot_gripper_{side}_joint_L1"]] = 0.93
        articulation.set_dof_position_targets(targets, indices)

        # A positioning fixture supports each block during approach. Once closed,
        # the blocks are fully dynamic and only fingertip contacts can hold them.
        stationary = np.zeros((1, 6), dtype=np.float32)
        for _ in range(180):
            for side, block in blocks.items():
                block.set_transforms(placements[side], indices)
                block.set_velocities(stationary, indices)
            step(1)
        step(480)
        heights = {}
        for side, block in blocks.items():
            position = block.get_transforms()[0, :3]
            center, _, gap = pad_frame(side)
            drift = np.linalg.norm(position - placements[side][0, :3])
            if not (drift < 0.010 and np.linalg.norm(position - center) < 0.030 and abs(gap - BLOCK_WIDTH) < 0.003):
                raise ValueError(f"{side} did not retain its block for 2 s (gap {gap:.4f} m, drift {drift:.4f} m)")
            heights[side] = float(position[2])
            targets[0, meta.dof_indices[f"astribot_gripper_{side}_joint_L1"]] = 0
            print(f"{side}: retained 20 g block for 2 s, pad gap {gap * 1000:.2f} mm, drift {drift * 1000:.2f} mm")
        articulation.set_dof_position_targets(targets, indices)
        step(120)
        for side, block in blocks.items():
            drop = heights[side] - block.get_transforms()[0, 2]
            if not drop > 0.50:
                raise ValueError(f"{side} failed to release its block ({drop:.3f} m drop)")
            print(f"{side}: released block fell {drop:.3f} m in 0.5 s", flush=True)
    finally:
        simulation.detach_stage()


if __name__ == "__main__":
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    status = 0
    try:
        check_grasp()
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
