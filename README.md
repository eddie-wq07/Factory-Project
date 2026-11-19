# Factory_Project

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `FactoryProject-Factory-ASTRIBOTS1`
- `FactoryProject-Factory-ASTRIBOTS1-Direct`
- `FactoryProject-Factory-Marl-ASTRIBOTS1-Direct`

## Load the Astribot S1

Open [astribot_s1.usd](src/Factory_Project/assets/data/astribot_s1/astribot_s1.usd) in Isaac Sim, or reference its default prim into a Z-up, meter-scale stage. Keep the contents of the `astribot_s1/` folder together when moving the asset. For a ground plane at Z = 0, place the robot at Z = 0.09523 m in its neutral pose.

From the repository root, run these with the Python interpreter from an Isaac Sim 6.2 environment:

```bash
python scripts/view_scene.py
python scripts/view_scene.py --screenshot s1-front --view front
python scripts/view_scene.py --screenshot wheel-inspection --view underside
python src/Factory_Project/assets/data/astribot_s1/check_motion.py
python src/Factory_Project/assets/data/astribot_s1/check_motion.py --device cuda
python src/Factory_Project/assets/data/astribot_s1/check_grasp.py
```

The first command opens the robot in a lit studio. Screenshot commands render a PNG headlessly and exit. Files go under `recordings/MM-DD-YYYY/` in Vancouver time, for example `recordings/11-18-2026/12:16am_s1-front.png`. Repeat captures in the same minute receive a numeric suffix. Use `--view head`, `--view left-wrist` or `--view right-wrist` to capture the onboard cameras with colored inspection targets.

The [editable USDA](src/Factory_Project/assets/data/astribot_s1/astribot_s1.usda) composes the geometry and materials from `model.usdc`. Rebuild with `python src/Factory_Project/assets/data/astribot_s1/build_usd.py` using the bundled URDF and meshes. The [source notes](src/Factory_Project/assets/data/astribot_s1/SOURCE.md) detail wheel contact, gripper coupling and camera placement.

### Driving the robot

Command the four wheel hubs with angular velocity targets. For example, at a magnitude of 3 rad/s, use these signs in the order shown. The base frame is X forward, Y left and Z up.

| Motion | `wheel_RF_Joint` | `wheel_LF_Joint` | `wheel_RR_Joint` | `wheel_LR_Joint` |
| --- | ---: | ---: | ---: | ---: |
| Forward | −3 | +3 | −3 | +3 |
| Left | −3 | −3 | +3 | +3 |
| Turn left | −3 | −3 | −3 | −3 |

The PhysX tensor interface accepts radians and radians/s. When editing a USD angular drive directly, enter degrees and degrees/s. Each gripper has one position target, `astribot_gripper_left_joint_L1` or `astribot_gripper_right_joint_L1`. Set it to 0 rad to open or 0.93 rad to close. Five joints per hand follow that target. Each wheel has 16 unpowered roller joints to allow sideways travel.

### Validation in Isaac Sim 6.2

The standalone motion check exercised the floating base at 60 and 120 Hz on a flat floor. On CPU at 120 Hz, separate 1.5-second commands moved it 0.429 m forward, 0.414 m left and 1.111 rad around its vertical axis. The bilateral grasp check used a fixed chassis to isolate the fingers. Each hand held a 35 mm, 20 g block against gravity for two seconds, then released it.

[Simulated reach](recordings/misc/11-18-2026/12:46am_s1-forward-reach.png) · [Four wheel assemblies](recordings/misc/11-18-2026/12:24am_s1-four-wheels.png) · [Upright head camera](recordings/misc/11-18-2026/12:27am_s1-head-upright.png)

## Preview environment assets

Open an empty Isaac Sim editor using the existing Isaac Sim Python environment:

```bash
cd ~/Desktop/Simulation/Projects/S1-Project
~/env_isaaclab/bin/python scripts/preview.py
```

The scene starts with only `/World` and a neutral preview light. Physics starts stopped, so you can browse and
arrange assets before running a task. This uses Isaac Sim directly and does not require the unfinished task configs.

1. In **Window > Browsers > Content**, expand **Isaac Sim > Environments**. The script opens this folder for you.
2. Start with **Simple_Warehouse** (`warehouse.usd` or `warehouse_multiple_shelves.usd`) for the factory aisle,
   or **Simple_Room** (`simple_room.usd`) for a smaller scene. Browse **Props** for individual objects.
3. Drag a USD onto **World** in the Stage panel to place it at the origin. Select it, then press **F** with the
   viewport focused to frame it. Use **File > New** to clear the scene between candidates.
4. Record each chosen asset's full USD URL in `vision.md`, along with its role (background, pickup table,
   drop-off station, obstacle). If you arrange several assets, use **File > Save As** to save a project scene
   under `src/Factory_Project/assets/data/environments/`; that scene can later be loaded by Isaac Lab.

Assets stream from NVIDIA when first opened, so larger scenes can take time to download. The editor must be
launched from a graphical desktop terminal. For a bounded check without a window:

```bash
~/env_isaaclab/bin/python scripts/preview.py --headless --max-frames 10
```

See NVIDIA's [Content Browser guide](https://docs.isaacsim.omniverse.nvidia.com/latest/utilities/content_browser.html)
and [environment catalog](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_environments.html).

## The sorting task

One S1 takes six dynamic items off the “To be sorted.” rack in alphabetical address order and stacks each on
its stocked pallet. [docs/sorting-task.md](docs/sorting-task.md) describes the scene, the putaway rules, the
scripted teacher, the imitation-learning and reinforcement-learning pipelines, measured results and known limits.

```bash
~/env_isaaclab/bin/python scripts/run_mission.py                    # scripted teacher, full mission
~/env_isaaclab/bin/python scripts/run_mission.py --record mission   # with a chase-camera video
~/env_isaaclab/bin/python scripts/record_demos.py --episodes 3      # demonstrations for imitation learning
~/env_isaaclab/bin/python scripts/train_bc.py --demos outputs/demos/*.h5 --kind act
~/env_isaaclab/bin/python scripts/evaluate_policy.py --policy outputs/policies/act.pt
~/env_isaaclab/bin/python scripts/train_skill.py --device cuda --num-envs 256   # PPO clamp-and-lift skill
```

The manager-based (`FactoryProject-Factory-ASTRIBOTS1`) and multi-agent task families are still the template's
scaffolding: their `MISSING` values and `NotImplementedError` functions remain to be filled in.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Task families live in `src/Factory_Project/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/AstribotS1`. Add another robot by
creating a sibling of `config/AstribotS1`; add another task by creating a sibling task-family directory.

## Project assets

Keep project-owned USD files and related data under `src/Factory_Project/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from Factory_Project.assets import ASTRIBOT_S1_USD

robot_usd_path = str(ASTRIBOT_S1_USD)
```

Keep asset configuration in the `Factory_Project.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.

### Persistent factory starting point

```bash
~/env_isaaclab/bin/python scripts/view_factory.py
~/env_isaaclab/bin/python scripts/view_factory.py --screenshot test-boundary
```

The saved scene is `src/Factory_Project/assets/data/environments/factory.usda`. It places S1 at `(4.45, 7.30, 0.09523)` metres, heading 180°, beside the east wall and shelf end. Reopening this USD restores the same pose and the robot asset's neutral joint configuration. The RL task templates do not yet implement episode resets.

Four references to the supplied NVIDIA SimReady tape form a square with a 0.596852 m inner side and 0.15 m tape width. Its inner edges match the full chassis-and-wheel bounding square; the narrower axis has about 4.6 mm clearance per side. Tape is decorative, with physics disabled, and raised 1 mm above the floor. `scripts/build_factory.py` regenerates the scene from measured robot bounds. Robot and tape dependencies are local; the warehouse still uses NVIDIA's hosted USD and requires network access on an uncached machine.

Requested captures go in `recordings/MM-DD-YYYY/`. Add `--diagnostic` to either screenshot viewer for unprompted inspection captures under `recordings/misc/MM-DD-YYYY/`. Earlier diagnostic screenshots were moved there; the requested render test and wheel video remain in the day folder. Capture filenames use Vancouver local time.

[Boundary screenshot](recordings/11-18-2026/12:25pm_test-boundary.png). A two-second idle physics check measured approximately 15 mm of base settling/drift; this is a fixed initial pose, not an active parking controller.
