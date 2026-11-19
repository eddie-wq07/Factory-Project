"""Evaluate the sorting mission with a learned policy doing the manipulation.

The split follows docs/policy-arch.md: the deterministic executive (the teacher's sequencing, route
planning, docking and verification) stays in charge, and during the ``pick`` and ``place`` skills
the learned policy commands the torso, head, arms and grippers from observations. With no
``--policy`` this is the scripted baseline under the same conditions::

    ~/env_isaaclab/bin/python scripts/evaluate_policy.py --episodes 3
    ~/env_isaaclab/bin/python scripts/evaluate_policy.py --policy outputs/policies/act.pt --episodes 3

``--blend`` below 1 mixes the policy's command with the teacher's, which shows how far the policy
can be trusted before it loses a carton. With ``--dagger-out`` the states the blended controller visits are
saved with the teacher's own command as the label; adding that file to the training set teaches the policy
to recover from its own drift (DAgger).
"""

import argparse
import json
import time
from pathlib import Path

import sim_bootstrap  # noqa: F401  (must come before NumPy and Isaac Sim)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--policy", type=Path, default=None, help="Checkpoint from train_bc.py. Omit for the scripted baseline."
)
parser.add_argument("--blend", type=float, default=1.0, help="Share of the policy in the upper-body command, 0 to 1.")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument(
    "--seed", type=int, default=1000, help="Seed of the first episode; keep it apart from the demo seeds."
)
parser.add_argument("--item-noise", type=float, default=0.01)
parser.add_argument("--yaw-noise", type=float, default=0.02)
parser.add_argument("--dock-noise", type=float, default=0.005)
parser.add_argument("--report", type=Path, default=None)
parser.add_argument(
    "--dagger-out",
    type=Path,
    default=None,
    help="Also save every visited state with the teacher's command for it (DAgger), as a demonstration file.",
)
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()

import gymnasium as gym
import numpy as np
import torch

import Factory_Project.tasks  # noqa: F401
from Factory_Project.learning.dataset import MANIPULATION_SKILLS, DemoWriter, condition, skill_index
from Factory_Project.learning.policies import PolicyRunner
from Factory_Project.sorting.teacher import Teacher, TeacherConfig
from Factory_Project.tasks.factory_direct.config.AstribotS1.env_cfg import FactoryEnvCfg


def main() -> int:
    cfg = FactoryEnvCfg()
    cfg.sim.device = args.device
    cfg.reset_item_xy_noise = args.item_noise
    cfg.reset_item_yaw_noise = args.yaw_noise
    runner = PolicyRunner(args.policy) if args.policy else None
    results = []
    writer = DemoWriter(args.dagger_out) if args.dagger_out else None
    with launch_simulation(cfg, args):
        env = gym.make("FactoryProject-Factory-ASTRIBOTS1-Direct", cfg=cfg).unwrapped
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, _ = env.reset(seed=seed)
            teacher = Teacher(env, TeacherConfig(dock_noise=args.dock_noise), seed=seed, verbose=False)
            generator = teacher.run()
            started = time.time()
            steps = policy_steps = 0
            deviation = 0.0
            was_active = False
            try:
                action = np.asarray(next(generator), dtype=np.float32)
                while True:
                    if writer is not None:
                        writer.add(observation["policy"][0].cpu().numpy(), action, teacher.skill)
                    active = runner is not None and teacher.skill in MANIPULATION_SKILLS
                    if active:
                        if not was_active:
                            runner.reset()
                        model_input = condition(
                            observation["policy"].cpu().numpy(), np.array([skill_index(teacher.skill)])
                        )[0]
                        predicted = runner.act(model_input)
                        deviation += float(np.abs(predicted[:20] - action[3:23]).mean())
                        action = action.copy()
                        action[3:] = (1.0 - args.blend) * action[3:] + args.blend * predicted
                        policy_steps += 1
                    was_active = active
                    observation, _, done, time_out, _ = env.step(
                        torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
                    )
                    steps += 1
                    if bool(done[0]) or bool(time_out[0]):
                        break
                    action = np.asarray(generator.send(None), dtype=np.float32)
            except StopIteration:
                pass
            metrics = env.extras.get("task") or env.task_metrics()
            if writer is not None:
                # Every label is the teacher's own command, so the whole episode is usable.
                writer.end_episode(seed=seed, success=True, good_steps=steps, dagger=True, blend=args.blend)
            result = dict(
                seed=seed,
                steps=steps,
                policy_steps=policy_steps,
                mean_joint_deviation_rad=deviation / max(policy_steps, 1),
                wall_s=round(time.time() - started, 1),
                status=dict(env.status),
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            )
            results.append(result)
            print(
                f"episode {episode}: seed {seed}, placed {result.get('items_placed', 0):.0f}/6, "
                f"drops {result.get('drops', 0):.0f}, "
                f"success {result.get('success', 0):.0f}, policy on {policy_steps}/{steps} steps, "
                f"mean deviation from teacher {result['mean_joint_deviation_rad'] * 1000:.1f} mrad",
                flush=True,
            )
        env.close()
    if writer is not None:
        writer.close()
    placed = sum(r.get("items_placed", 0.0) for r in results)
    summary = dict(
        policy=str(args.policy) if args.policy else "scripted teacher",
        blend=args.blend if args.policy else 0.0,
        episodes=len(results),
        mission_success_rate=float(np.mean([r.get("success", 0.0) for r in results])),
        items_placed=placed,
        items_total=6 * len(results),
        drops=sum(r.get("drops", 0.0) for r in results),
        results=results,
    )
    print(
        f"EVALUATION {summary['policy']}: missions {summary['mission_success_rate'] * 100:.0f}% of {len(results)}, "
        f"items {placed:.0f}/{summary['items_total']}, drops {summary['drops']:.0f}",
        flush=True,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
