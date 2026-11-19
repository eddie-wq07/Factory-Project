"""Run the scripted teacher on the six-item sorting mission and report the outcome.

From the repository root, with the Isaac Sim Python environment::

    ~/env_isaaclab/bin/python scripts/run_mission.py                  # full mission, headless
    ~/env_isaaclab/bin/python scripts/run_mission.py --items A2 F8    # only these items, in this order
    ~/env_isaaclab/bin/python scripts/run_mission.py --record mission # also record a chase-camera MP4
    ~/env_isaaclab/bin/python scripts/run_mission.py --viz kit        # watch it in the Isaac Sim window

The teacher reads simulator poses, so this is the privileged-state scripted baseline.
"""

import argparse
import json
import time
from pathlib import Path

import sim_bootstrap  # noqa: F401  (must come before NumPy and Isaac Sim)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--items", nargs="*", default=None, help="Addresses to sort, in order. Default: all six.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--item-noise", type=float, default=0.0, help="Rack item start offset along the shelf [m].")
parser.add_argument("--yaw-noise", type=float, default=0.0, help="Rack item start twist [rad].")
parser.add_argument("--dock-noise", type=float, default=0.0, help="Chassis docking offset [m].")
parser.add_argument(
    "--record", default=None, help="Record a chase-camera MP4 under recordings/sorting-mission/ with this label."
)
parser.add_argument("--record-every", type=int, default=2, help="Record one frame every N control steps.")
parser.add_argument("--report", type=Path, default=None, help="Write the mission report to this JSON file.")
parser.add_argument("--quiet", action="store_true")
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.record:
    args.enable_cameras = True

import gymnasium as gym
import torch

import Factory_Project.tasks  # noqa: F401
from Factory_Project.sorting.teacher import Teacher, TeacherConfig
from Factory_Project.tasks.factory_direct.config.AstribotS1.env_cfg import FactoryEnvCfg


def main() -> int:
    cfg = FactoryEnvCfg()
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.reset_item_xy_noise = args.item_noise
    cfg.reset_item_yaw_noise = args.yaw_noise
    if args.record:
        from mission_video import CAMERA, RESOLUTION

        cfg.viewer_cameras = {CAMERA: RESOLUTION}
    with launch_simulation(cfg, args):
        env = gym.make("FactoryProject-Factory-ASTRIBOTS1-Direct", cfg=cfg).unwrapped
        env.reset(seed=args.seed)
        teacher = Teacher(env, TeacherConfig(dock_noise=args.dock_noise), seed=args.seed, verbose=not args.quiet)
        recorder = None
        if args.record:
            from mission_video import ChaseRecorder

            recorder = ChaseRecorder(env, args.record, every=args.record_every)

        def mission():
            if args.items is None:
                return (yield from teacher.run())
            yield from teacher.move_joints(teacher.travel_posture(), 2.0)
            for address in args.items:
                if not (yield from teacher.sort_item(address)):
                    return False
            return True

        generator = mission()
        started = time.time()
        completed = False
        terminated = truncated = False
        try:
            action = next(generator)
            while True:
                _, _, done, time_out, _ = env.step(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0))
                if recorder is not None:
                    recorder.capture()
                terminated, truncated = bool(done[0]), bool(time_out[0])
                if terminated or truncated:
                    break
                action = generator.send(None)
        except StopIteration as stop:
            completed = bool(stop.value)
        metrics = env.extras.get("task") or {**env.task_metrics(), "status": env.status}
        wall = time.time() - started
        report = {
            "completed": completed or metrics.get("success", 0.0) == 1.0,
            "terminated": terminated,
            "timed_out": truncated,
            "wall_time_s": round(wall, 1),
            "metrics": metrics,
            "events": teacher.log.events,
        }
        print("\nMISSION REPORT", flush=True)
        for key, value in metrics.items():
            print(f"  {key}: {value}", flush=True)
        print(f"  teacher_completed: {completed}   wall time: {wall:.0f} s", flush=True)
        if recorder is not None:
            print(f"  video: {recorder.close()}", flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, default=float) + "\n")
        env.close()
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
