"""Record scripted-teacher demonstrations of the sorting mission to an HDF5 file.

No teleoperation is involved: the teacher drives the simulated robot and every 20 Hz control step is
stored as (observation, action, skill). Start positions and docking are varied per episode so the
data covers more than one trajectory::

    ~/env_isaaclab/bin/python scripts/record_demos.py --episodes 4 --out outputs/demos/run0.h5
    ~/env_isaaclab/bin/python scripts/record_demos.py --episodes 4 --seed 100 --out outputs/demos/run1.h5

Runs are independent, so several can record in parallel into different files.
"""

import argparse
import time
from pathlib import Path

import sim_bootstrap  # noqa: F401  (must come before NumPy and Isaac Sim)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode; later ones count up.")
parser.add_argument("--out", type=Path, default=Path("outputs/demos/demos.h5"))
parser.add_argument("--item-noise", type=float, default=0.01, help="Rack item start offset along the shelf [m].")
parser.add_argument("--yaw-noise", type=float, default=0.02, help="Rack item start twist [rad].")
parser.add_argument("--dock-noise", type=float, default=0.005, help="Chassis docking offset [m].")
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()

import gymnasium as gym
import torch

import Factory_Project.tasks  # noqa: F401
from Factory_Project.learning.dataset import DemoWriter
from Factory_Project.sorting.teacher import Teacher, TeacherConfig
from Factory_Project.tasks.factory_direct.config.AstribotS1.env_cfg import FactoryEnvCfg


def main() -> int:
    cfg = FactoryEnvCfg()
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.reset_item_xy_noise = args.item_noise
    cfg.reset_item_yaw_noise = args.yaw_noise
    writer = DemoWriter(args.out)
    successes = 0
    with launch_simulation(cfg, args):
        env = gym.make("FactoryProject-Factory-ASTRIBOTS1-Direct", cfg=cfg).unwrapped
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, _ = env.reset(seed=seed)
            teacher = Teacher(env, TeacherConfig(dock_noise=args.dock_noise), seed=seed, verbose=False)
            generator = teacher.run()
            started = time.time()
            steps = good_steps = placed = 0
            try:
                action = next(generator)
                while True:
                    writer.add(observation["policy"][0].cpu().numpy(), action, teacher.skill)
                    observation, _, done, time_out, _ = env.step(
                        torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
                    )
                    steps += 1
                    now_placed = sum(status == "placed" for status in env.status.values())
                    if now_placed > placed:
                        placed, good_steps = now_placed, steps
                    if bool(done[0]) or bool(time_out[0]):
                        break
                    action = generator.send(None)
            except StopIteration:
                pass
            metrics = env.extras.get("task") or env.task_metrics()
            success = float(metrics.get("success", 0.0)) == 1.0
            successes += success
            name = writer.end_episode(
                seed=seed,
                success=success,
                good_steps=steps if success else good_steps,
                items_placed=float(metrics.get("items_placed", 0.0)),
                drops=float(metrics.get("drops", 0.0)),
                item_noise=args.item_noise,
                yaw_noise=args.yaw_noise,
                dock_noise=args.dock_noise,
            )
            print(
                f"{name}: seed {seed}, {steps} steps, placed {metrics.get('items_placed', 0):.0f}/6, "
                f"success {success}, {time.time() - started:.0f} s wall",
                flush=True,
            )
        env.close()
    writer.close()
    print(f"DEMOS {successes}/{args.episodes} successful episodes in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
