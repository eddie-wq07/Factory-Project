"""Record the S1 tracing a cross on the Full Warehouse floor.

Run `python scripts/drive_demo.py` with Isaac Sim Python. Each spoke has a
two-second forward leg, an in-place half-turn, and a two-second forward return.
The fixed camera records the complete route under recordings/MM-DD-YYYY/.
"""

import math

import numpy as np

PHYSICS_HZ = 120
WHEEL_NAMES = ("wheel_RF_Joint", "wheel_LF_Joint", "wheel_RR_Joint", "wheel_LR_Joint")
FORWARD_SIGNS = np.array([-1, 1, -1, 1])
SPOKE_LENGTH = 0.8  # meters from the initial chassis center


def heading(pose):
    x, y, z, w = pose[3:]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def angle_error(target, current):
    return math.remainder(target - current, math.tau)


def run_drive(stage, robot_prim_path, capture_frame, *, fps=30):
    """Step the wheel drives on a fixed clock. Capture callbacks may render with the timeline stopped."""
    import omni.physics.tensors as tensors
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from pxr import Usd, UsdPhysics, UsdUtils

    if fps <= 0 or PHYSICS_HZ % fps:
        raise ValueError(f"Video FPS must divide the physics rate of {PHYSICS_HZ} Hz")
    robot = stage.GetPrimAtPath(robot_prim_path)
    roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation under {robot_prim_path}, found {len(roots)}")

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    sim = get_physx_simulation_interface()
    sim.attach_stage(stage_id)
    ticks = 0

    def step(count):
        nonlocal ticks
        for _ in range(count):
            sim.simulate(1 / PHYSICS_HZ, ticks / PHYSICS_HZ)
            sim.fetch_results()
            ticks += 1

    try:
        step(1)
        view = tensors.create_simulation_view("numpy", stage_id)
        articulation = view.create_articulation_view(str(roots[0].GetPath()))
        if articulation.count != 1 or articulation.shared_metatype.fixed_base:
            raise RuntimeError("Expected one floating-base S1")
        indices = [articulation.shared_metatype.dof_indices[name] for name in WHEEL_NAMES]
        target = articulation.get_dof_velocity_targets().copy()
        instance = np.array([0], dtype=np.int32)
        physx = get_physx_interface()

        def pose():
            return articulation.get_root_transforms()[0].copy()

        def command(forward=0.0, turn=0.0):
            # Positive turn drives all four diagonal wheels in the negative hub direction.
            target[0, indices] = forward * FORWARD_SIGNS - turn
            articulation.set_dof_velocity_targets(target, instance)

        def frame():
            step(PHYSICS_HZ // fps)
            physx.update_transformations(False, True)
            capture_frame()

        def brake():
            command()
            for _ in range(max(1, round(0.4 * fps))):
                frame()

        def travel(goal, target_yaw, label):
            start = pose()
            forward = np.array([math.cos(target_yaw), math.sin(target_yaw)])
            left = np.array([-forward[1], forward[0]])
            distance = np.dot(goal - start[:2], forward)
            # Track a two-second straight reference using wheel speed and measured chassis yaw.
            for i in range(2 * fps):
                current = pose()
                to_goal = goal - current[:2]
                remaining = np.dot(to_goal, forward)
                # A smoothstep distance profile comes to rest at the target before the turn.
                t = (i + 0.5) / (2 * fps)
                fraction = t * t * (3 - 2 * t)
                ref_speed = 3 * distance * t * (1 - t)
                schedule_error = remaining - distance * (1 - fraction)
                velocity = articulation.get_root_velocities()[0, :2]
                along_speed = np.dot(velocity, forward)
                forward_rate = np.clip(8.5 * ref_speed + 15 * schedule_error + 10 * (ref_speed - along_speed), 0, 7)
                course = target_yaw + np.clip(3 * np.dot(to_goal, left), -0.25, 0.25)
                turn_rate = np.clip(8 * angle_error(course, heading(current)), -2, 2)
                command(forward_rate, turn_rate)
                frame()
            brake()
            end = pose()
            travel_m = np.dot(end[:2] - start[:2], forward)
            miss = np.linalg.norm(goal - end[:2])
            print(f"{label}, 2.00 s forward: {travel_m:.3f} m; endpoint error {miss:.3f} m", flush=True)
            if not (travel_m > 0.55 and miss < 0.14 and abs(end[2] - start[2]) < 0.05):
                raise RuntimeError(f"Unexpected displacement during {label}")

        def turn_to(target_yaw):
            start = pose()
            previous_yaw = heading(start)
            requested = angle_error(target_yaw, previous_yaw)
            if abs(requested) < math.radians(1):
                return
            # Use the same turning direction at each tip and unwrap the measured yaw across +/-pi.
            if requested < -math.pi + math.radians(10):
                requested += math.tau
            turned = 0.0
            settled = 0
            for _ in range(8 * fps):
                error = requested - turned
                if abs(error) < math.radians(0.5):
                    command()
                    settled += 1
                else:
                    command(turn=math.copysign(np.clip(8 * abs(error), 1.5, 6), error))
                    settled = 0
                frame()
                current_yaw = heading(pose())
                turned += angle_error(current_yaw, previous_yaw)
                previous_yaw = current_yaw
                if settled >= max(1, round(0.25 * fps)):
                    break
            command()
            drift = np.linalg.norm(pose()[:2] - start[:2])
            print(f"In-place turn: {math.degrees(turned):+.2f} degrees; drift {drift:.3f} m", flush=True)
            if abs(requested - turned) > math.radians(1) or drift > 0.15:
                raise RuntimeError("Chassis missed its turn setpoint")

        command()
        step(PHYSICS_HZ)
        origin = pose()
        center = origin[:2].copy()
        initial_yaw = heading(origin)
        for name, offset in (("+X", 0), ("-X", math.pi), ("+Y", math.pi / 2), ("-Y", -math.pi / 2)):
            outward = initial_yaw + offset
            tip = center + SPOKE_LENGTH * np.array([math.cos(outward), math.sin(outward)])
            turn_to(outward)
            travel(tip, outward, f"{name} outbound")
            # The return leg also drives forward, following a physical half-turn at the tip.
            turn_to(heading(pose()) + math.pi)
            travel(center, outward + math.pi, f"{name} return to center")
    finally:
        sim.detach_stage()


WAREHOUSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.2/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
ROBOT_PATH = "/World/Astribot"
VIDEO_SIZE = (1280, 720)
VIDEO_FPS = 24


def setup_warehouse(stage, robot_asset):
    import carb.settings
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    factory = UsdGeom.Xform.Define(stage, "/World/Factory").GetPrim()
    factory.GetReferences().AddReference(WAREHOUSE_USD)
    if not factory.GetChild("SM_floor63"):
        raise RuntimeError("The Full Warehouse floor at the starting location did not load")

    # The warehouse supplies the floor colliders. All distances here are in meters.
    center = Gf.Vec3d(-12.0, 6.0, 0.0)
    robot = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    robot.GetPrim().GetReferences().AddReference(str(robot_asset))
    robot.AddTranslateOp(opSuffix="start").Set(center + Gf.Vec3d(0, 0, 0.09523))
    robot.AddRotateZOp(opSuffix="heading").Set(-90)
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateEnableGPUDynamicsAttr(False)
    physx.CreateBroadphaseTypeAttr("MBP")
    physx.CreateEnableEnhancedDeterminismAttr(True)
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/suppressReadback", False)
    settings.set_bool("/app/viewport/grid/enabled", False)

    camera = UsdGeom.Camera.Define(stage, "/World/CrossCamera")
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateFocalLengthAttr(32)
    camera.CreateClippingRangeAttr((0.05, 150.0))
    eye = center + Gf.Vec3d(0, -4.2, 4.4)
    aim = center + Gf.Vec3d(0, 0, 0.7)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, aim, Gf.Vec3d(0, 0, 1)).GetInverse())
    viewport = get_active_viewport()
    viewport.resolution = VIDEO_SIZE
    viewport.camera_path = camera.GetPath()
    return viewport


def main():
    import sys
    import traceback
    from pathlib import Path

    from isaacsim import SimulationApp

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from Factory_Project.assets.data.screenshot import ROBOT_USD, record_viewport_video

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": True,
            "width": VIDEO_SIZE[0],
            "height": VIDEO_SIZE[1],
            "multi_gpu": False,
            "sync_loads": True,
        }
    )
    status = 0
    try:
        import omni.timeline
        import omni.usd

        omni.timeline.get_timeline_interface().stop()
        stage = omni.usd.get_context().get_stage()
        viewport = setup_warehouse(stage, ROBOT_USD)
        for _ in range(60):
            app.update()
        with record_viewport_video(app.update, viewport, "s1-wheel-cross", VIDEO_FPS) as video:
            run_drive(stage, ROBOT_PATH, video.capture_frame, fps=VIDEO_FPS)
        print(f"Saved {video.frame_count} frames at {VIDEO_FPS} fps to {video.path}", flush=True)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
