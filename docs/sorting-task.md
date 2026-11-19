# The sorting task, as built

One Astribot S1 takes six items off the “To be sorted.” rack, in alphabetical order of their address
(A2 → F8 → H4 → K3 → O8 → Q4), and puts each away on the stocked pallet at that address. Every item,
every unit of pallet stock and the robot are dynamic rigid bodies; nothing is teleported or attached.

No teleoperation data is used anywhere. A scripted teacher that reads simulator state does the task,
its runs are the demonstrations for imitation learning, and a separate skill environment trains the
two-handed grasp with PPO from reward alone.

## What has been measured

All figures are from this machine, Isaac Sim 6.2 / Isaac Lab 3.0, run on 18 November 2026.

| What | Result |
| --- | --- |
| Scripted teacher, nominal scene (seed 0, no variation) | **6 of 6 items placed**, in order, no drops, stack limits kept. 860 s simulated, 272 m driven. Same result on a second run. Videos: `recordings/sorting-mission/mission_4x.mp4` and, with head and wrist cameras, `mission_pov_4x.mp4`. |
| Scripted teacher with start and docking variation (±1 cm, ±0.02 rad, ±5 mm dock) | 5 of 15 missions complete; 55 of 90 items placed. |
| Scripted teacher on three unseen seeds at that variation | 0 of 3 missions complete; 9 of 18 items placed. |
| Scripted teacher with twice that variation | Roughly 2 of 14 missions complete. |
| Behaviour cloning, offline (held-out episodes) | MLP 3 mrad and ACT-style transformer 6 mrad mean next-step joint error. |
| Behaviour cloning, closed loop (policy commands the upper body during pick and place) | **0 of 18 items** for either model. The policies drift 0.3–0.6 rad from the teacher once in control. |
| After one DAgger round (5 relabelled episodes) | MLP drift falls to 40–65 mrad, still 0 of 18 items. The transformer does not improve. |
| PPO clamp-and-lift skill, 200 iterations, 256 environments on GPU | Carton lifted level at some point in 128 of 128 playback episodes; 92% of training episodes end lifted. Not yet tried inside the full mission. |
| Unit tests | 9 pass (`tests/`). |

So the end-to-end mission works with the scripted teacher in the nominal scene, the reinforcement-learned
grasp works in its own environment, and imitation learning does not yet work in closed loop.

## Run it

All commands run from the repository root with the Isaac Sim interpreter. The mission uses CPU PhysX.

```bash
PY=~/env_isaaclab/bin/python

# Rebuild the scene (only after changing items, drop zones or the rack)
$PY scripts/build_sorting_rack.py && $PY scripts/build_drop_zones.py && $PY scripts/build_nav_map.py

# Scripted teacher: full mission, a subset, with a chase-camera video, or in the Isaac Sim window
$PY scripts/run_mission.py
$PY scripts/run_mission.py --items A2 F8
$PY scripts/run_mission.py --record mission --record-every 4
$PY scripts/run_mission.py --record mission_pov --pov --record-every 4   # plus head and wrist cameras
$PY scripts/run_mission.py --viz kit

# Imitation learning: record teacher demonstrations, train, evaluate in closed loop
$PY scripts/record_demos.py --episodes 3 --seed 0 --out outputs/demos/run0.h5
$PY scripts/train_bc.py --demos outputs/demos/*.h5 --kind mlp
$PY scripts/train_bc.py --demos outputs/demos/*.h5 --kind act --horizon 20
$PY scripts/evaluate_policy.py --policy outputs/policies/act.pt --episodes 3
$PY scripts/evaluate_policy.py --policy outputs/policies/mlp.pt --blend 0.3 --dagger-out outputs/demos/dagger0.h5  # DAgger data

# Reinforcement learning: the clamp-and-lift skill with PPO, then play the result
$PY scripts/train_skill.py --device cuda --num-envs 256 --iterations 200
$PY scripts/train_skill.py --device cuda --num-envs 32 --play outputs/skills/clamp_lift/model_199.pt
```

Tests that need no simulator: `PYTHONPATH=src pytest tests`.

## What is in the scene

| Address | Item | Size D × W × H [m] | Mass [kg] | Drop pallet stock before the mission | Where the item goes |
| --- | --- | --- | --- | --- | --- |
| A2 | Cube carton | 0.50 × 0.50 × 0.50 | 3.0 | Columns of 1 and 2 | On the one-high column (second tier) |
| F8 | Blue tote | 0.50 × 0.62 × 0.25 | 1.8 | Two pallets, columns of 3 and 2 | Third tier of the two-high column |
| H4 | Shallow tote | 0.44 × 0.53 × 0.14 | 1.0 | Double pallet, columns of 2 and 0 | Third tier of the started column |
| K3 | Carton | 0.36 × 0.50 × 0.36 | 2.5 | Columns of 2 and 3 | Third tier of the two-high column |
| O8 | Bottle | 0.13 × 0.06 × 0.28 | 0.4 | Two nested totes at the aisle-5 end cap | Inside the top tote |
| Q4 | Flat carton | 0.50 × 0.50 × 0.25 | 2.0 | Columns of 3 and 2 | Third tier of the two-high column |

`scripts/build_sorting_rack.py` and `scripts/build_drop_zones.py` hold these definitions;
`assets/data/environments/sorting_task.json` is the manifest every other part reads.

## Putaway rules

These are the day-to-day rules the planner (`sorting/stacking.py`) and the success check enforce:

- Strict alphabetical order by address; an item only counts at its own address.
- Columns stand flush with the pallet's aisle edge. A new unit sits square on the one below, front faces flush.
- At most three units high, and no stack taller than 1.5 m.
- Like on like: cartons on cartons, a tote only on its own family. No overhang. Never heavier on lighter.
- Top up a started column before opening an empty one.
- Loose small items (the bottle) go inside a tote, never on a stack.
- An item is placed when it is released, at rest, upright, and within 6 cm, 3 cm in height and 0.21 rad of its planned pose. A drop ends the episode.

## How the teacher does it

`sorting/teacher.py` sequences short skills; `sorting/navigation.py` plans routes with A* on an
occupancy grid and follows them; `sorting/kinematics.py` solves differential IK for torso and arms.

- **Drive and dock.** Route to a standoff pose, shape the arms in the open, then let the chassis carry them in.
- **Clamp pick.** The S1's grippers open about 9 cm, so cartons and totes are clamped between the two closed hands. The hands close in small steps until the arms' position error shows the required force, lift, and are re-pitched until the carton hangs level; a carton that comes up crooked is set down and gripped again.
- **Carry.** The carton is brought to a carry pose in front of the chest and re-levelled.
- **Stack.** Held out over the pallet edge, the chassis servos until the carton itself is over its planned pose; then lower, relax the arms, open, back out.
- **Bottle.** One gripper pinches it a quarter of its height below centre and lowers it into the tote.

## Environments

| Gym id | What it is |
| --- | --- |
| `FactoryProject-Factory-ASTRIBOTS1-Direct` | The full mission. 20 Hz control; 25 actions (chassis twist, then 22 joint targets); 78 observations (base, joints, active item and goal in the chassis frame, palms, progress). Owns order, placement planning, success and drop checks. Optional head and wrist RGB cameras. |
| `FactoryProject-ClampLift-ASTRIBOTS1-Direct` | The clamp-and-lift skill for PPO. Many docked robots in parallel, each starting at the teacher's pre-grasp; 14 arm actions; shaped reward for reaching the sides, squeezing and lifting level. |

The manager-based and multi-agent task stubs from the project template are unchanged.

## Assumptions

- **Masses** are assumed (1–3.5 kg, within a two-arm payload); the source assets carry none.
- **Friction** (static / dynamic): rubber hand housings 1.1 / 1.0, cardboard 0.70 / 0.60, plastic 0.50 / 0.45, painted steel 0.5 / 0.4.
- **Arm gains** are simulation gains (800 N·m/rad at the shoulder and elbow, 200 at the wrist), chosen so a few centimetres of commanded squeeze give 60–90 N of clamping force. The real robot's controller is not modelled.
- **Hand colliders**: each gripper housing collides as a box and the wrist as a small sphere, so a flat palm can bear on a carton.
- **Chassis**: a PI velocity loop commands wheel torques; the 64 omni rollers are simulated.
- **Rack**: collides through four deck boxes and four posts instead of its 850,000-triangle mesh. Its two upper shelves are out of the S1's reach and stay empty.
- **Item limits**: rack cartons are at most 0.5 m wide and 0.5 m deep. The K3 carton is its stock line's 0.7 m carton scaled to 0.5 m.
- **Perception is privileged**: the teacher and the state-based policies read object poses from the simulator. Reading addresses from the cameras is not implemented.

## Known limits

- **The teacher is not robust to variation.** A centimetre of carton offset or docking error often loses a carton. The two-palm friction clamp behaves like a hinge: the carton can turn about the line between the palms, and a jolt can swing it out of the grip. Levelling and re-gripping help; they do not make it reliable.
- **Imitation learning fails in closed loop.** The teacher mostly replays planned joint paths, so a policy that sees joint angles learns to copy its current posture and drifts. More DAgger rounds, or action targets relative to the carton rather than absolute joint angles, are the next things to try. With a teacher this fragile, the demonstrations are also few (11 usable episodes).
- **PPO is the more promising route for the grasp**, matching the zero-teleoperation goal: it learned a 100% lift in about half an hour. It covers one carton size on one shelf, starts from the teacher's pre-grasp, and has not been connected to the mission executive.
- **A2's stack is two high, not three.** Lifting a 0.5 m cube to a third tier puts its centre at 1.4 m, beyond what this clamp holds; the 1.5 m stack-height rule keeps the planner away from it.
- **Arm gains and masses are simulation assumptions**, so none of this is evidence about the real S1.
- **No camera perception.** Cameras can be attached to the mission environment, but nothing reads them.
- **Mission environment is single-robot on CPU PhysX**; GPU PhysX was about 40 times slower on the full warehouse.
