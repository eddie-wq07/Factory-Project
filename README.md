# Factory_Project

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `FactoryProject-Factory-ASTRIBOTS1`
- `FactoryProject-Factory-ASTRIBOTS1-Direct`
- `FactoryProject-Factory-Marl-ASTRIBOTS1-Direct`

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

## Implement the Task

This project contains task scaffolding. Before running or training, replace `MISSING` configuration values and
implement functions that raise `NotImplementedError`. Define the scene and robot assets, physics and timing,
action and observation spaces, rewards, termination conditions, and reset behavior. Manager-based task terms live
in the family's `mdp` package. Define assets in the scene configuration; direct environments inherit scene creation
from the framework. If AMP is selected, implement reference-motion collection as well. Direct AMP tasks must also
update observation history and supply `self.extras["amp_obs"]` on every step.

Registrations and agent configurations are provided so you can build on the same structure as the runnable example.

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
from Factory_Project.assets import FACTORY_PROJECT_ASSETS_DIR

robot_usd_path = FACTORY_PROJECT_ASSETS_DIR / "robots" / "my_robot.usd"
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
