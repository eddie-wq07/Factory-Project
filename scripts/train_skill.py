"""Train the clamp-and-lift skill with PPO (rsl_rl): reinforcement learning, no demonstrations.

    ~/env_isaaclab/bin/python scripts/train_skill.py --num-envs 32 --iterations 400
    ~/env_isaaclab/bin/python scripts/train_skill.py --play outputs/skills/clamp_lift/model_399.pt --num-envs 8

Logs and checkpoints go to outputs/skills/clamp_lift/. ``--play`` runs a trained policy without
exploration noise and reports how many episodes end with the carton lifted and level.
"""

import argparse
import importlib.metadata
from pathlib import Path

import sim_bootstrap  # noqa: F401  (must come before NumPy and Isaac Sim)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--iterations", type=int, default=None, help="PPO iterations. Default: the agent configuration's.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log-dir", type=Path, default=Path("outputs/skills/clamp_lift"))
parser.add_argument("--play", type=Path, default=None, help="Evaluate this checkpoint instead of training.")
parser.add_argument("--play-episodes", type=int, default=32)
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

import Factory_Project.tasks  # noqa: F401
from Factory_Project.tasks.factory_direct.config.AstribotS1.agents.rsl_rl_clamp_lift_ppo_cfg import (
    ClampLiftPPORunnerCfg,
)
from Factory_Project.tasks.factory_direct.config.AstribotS1.clamp_lift_env_cfg import ClampLiftEnvCfg

TASK = "FactoryProject-ClampLift-ASTRIBOTS1-Direct"


def main() -> int:
    cfg = ClampLiftEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    agent = ClampLiftPPORunnerCfg()
    agent.seed = args.seed
    agent.device = args.device
    if args.iterations is not None:
        agent.max_iterations = args.iterations
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        if args.play is None:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            runner = create_rsl_rl_runner(wrapped, agent, log_dir=str(args.log_dir))
            runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
            print(f"TRAINED {agent.max_iterations} iterations, checkpoints in {args.log_dir}", flush=True)
        else:
            runner = create_rsl_rl_runner(wrapped, agent, log_dir=None)
            runner.load(str(args.play))
            policy = runner.get_inference_policy(device=args.device)
            observations = wrapped.get_observations()
            base = env.unwrapped
            finished = lifted = 0
            best = torch.zeros(base.num_envs, device=base.device)
            while finished < args.play_episodes:
                with torch.inference_mode():
                    observations, _, dones, _ = wrapped.step(policy(observations))
                terms = base._task_terms()
                success = ((terms["lift"] >= 1.0) & (terms["tilt"] < 0.1)).float()
                best = torch.maximum(best, success)
                for index in torch.nonzero(dones).flatten().tolist():
                    finished += 1
                    lifted += int(best[index] > 0)
                    best[index] = 0.0
            print(f"PLAY {args.play}: carton lifted level in {lifted}/{finished} episodes", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
