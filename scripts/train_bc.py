"""Train a behaviour-cloning policy for the docked manipulation skills from recorded demonstrations.

    ~/env_isaaclab/bin/python scripts/train_bc.py --demos outputs/demos/*.h5 --kind mlp
    ~/env_isaaclab/bin/python scripts/train_bc.py --demos outputs/demos/*.h5 --kind act --horizon 20

The policy sees the environment observation plus the skill the executive selected, and predicts the
teacher's upper-body command (torso, head, both arms, grippers). The chassis stays with the
navigation stack, as proposed in docs/policy-arch.md. Whole episodes are held out for validation.
No simulator is needed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from Factory_Project.learning.dataset import SKILLS, chunk_windows, load_episodes
from Factory_Project.learning.policies import build_policy, save_checkpoint

UPPER = slice(3, 25)
"""Action entries the policy controls: everything but the chassis twist."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, nargs="+", required=True)
    parser.add_argument("--kind", choices=("mlp", "act"), default="mlp")
    parser.add_argument("--horizon", type=int, default=20, help="Action chunk length for --kind act [steps].")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-episodes", type=int, default=1, help="Episodes held out for validation.")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th step as a training sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None, help="Checkpoint path. Default: outputs/policies/<kind>.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    episodes = load_episodes(args.demos)
    if len(episodes) <= args.val_episodes:
        raise SystemExit(f"Need more than {args.val_episodes} usable episodes, found {len(episodes)}")
    order = np.random.permutation(len(episodes))
    val = [episodes[i] for i in order[: args.val_episodes]]
    train = [episodes[i] for i in order[args.val_episodes :]]
    horizon = args.horizon if args.kind == "act" else 1
    x_train, y_train, m_train = chunk_windows(train, horizon, stride=args.stride)
    x_val, y_val, m_val = chunk_windows(val, horizon)
    y_train, y_val = y_train[..., UPPER], y_val[..., UPPER]

    stats = dict(
        in_mean=x_train.mean(0),
        in_std=x_train.std(0) + 1e-3,
        out_mean=y_train.reshape(-1, y_train.shape[-1]).mean(0),
        out_std=y_train.reshape(-1, y_train.shape[-1]).std(0) + 1e-3,
    )
    # One-hot entries keep their scale: a rarely seen bit must not be blown up by a tiny deviation.
    stats["in_mean"][-len(SKILLS) :] = 0.0
    stats["in_std"][-len(SKILLS) :] = 1.0

    def tensors(x, y, m):
        x = torch.as_tensor((x - stats["in_mean"]) / stats["in_std"], dtype=torch.float32)
        y = torch.as_tensor((y - stats["out_mean"]) / stats["out_std"], dtype=torch.float32)
        return x, y, torch.as_tensor(m)

    x_train_t, y_train_t, m_train_t = tensors(x_train, y_train, m_train)
    x_val_t, y_val_t, m_val_t = (t.to(args.device) for t in tensors(x_val, y_val, m_val))
    policy = build_policy(args.kind, x_train.shape[1], y_train.shape[-1], horizon).to(args.device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(x_train_t) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr, total_steps=args.epochs * steps_per_epoch)
    out_std = torch.as_tensor(stats["out_std"], device=args.device)

    def loss_of(x, y, m):
        error = (policy(x) - y).abs() * m.unsqueeze(-1)
        return error.sum() / (m.sum() * y.shape[-1])

    @torch.no_grad()
    def validate():
        policy.eval()
        losses, first = [], []
        for i in range(0, len(x_val_t), 4096):
            x, y, m = x_val_t[i : i + 4096], y_val_t[i : i + 4096], m_val_t[i : i + 4096]
            prediction = policy(x)
            losses.append(float(((prediction - y).abs() * m.unsqueeze(-1)).sum() / (m.sum() * y.shape[-1])) * len(x))
            first.append(((prediction[:, 0] - y[:, 0]).abs() * out_std).mean(0) * len(x))
        policy.train()
        return sum(losses) / len(x_val_t), (torch.stack(first).sum(0) / len(x_val_t)).cpu().numpy()

    print(
        f"{args.kind}: {len(train)} training episodes ({len(x_train)} samples), {len(val)} validation "
        f"({len(x_val)} samples), horizon {horizon}, device {args.device}",
        flush=True,
    )
    history, best, started = [], float("inf"), time.time()
    out = args.out or Path("outputs/policies") / f"{args.kind}.pt"
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(x_train_t))
        running = 0.0
        for step in range(steps_per_epoch):
            index = permutation[step * args.batch : (step + 1) * args.batch]
            loss = loss_of(
                x_train_t[index].to(args.device), y_train_t[index].to(args.device), m_train_t[index].to(args.device)
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            running += float(loss)
        val_loss, joint_error = validate()
        history.append(dict(epoch=epoch + 1, train=running / steps_per_epoch, val=val_loss))
        print(
            f"epoch {epoch + 1:3d}  train L1 {running / steps_per_epoch:.4f}  val L1 {val_loss:.4f}  "
            f"val next-step joint error {joint_error[:20].mean() * 1000:.1f} mrad",
            flush=True,
        )
        if val_loss < best:
            best = val_loss
            meta = dict(
                demos=[str(p) for p in args.demos],
                train_episodes=[e["source"] for e in train],
                val_episodes=[e["source"] for e in val],
                val_loss=val_loss,
                val_joint_error_rad=float(joint_error[:20].mean()),
                epoch=epoch + 1,
                controls="actions[3:25]",
            )
            save_checkpoint(out, policy, args.kind, stats, meta)
    out.with_suffix(".json").write_text(json.dumps(dict(history=history, best_val=best), indent=2) + "\n")
    print(
        f"TRAINED {args.kind}: best val L1 {best:.4f} (normalised), {time.time() - started:.0f} s, saved {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
