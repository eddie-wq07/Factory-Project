# Vision for Project

Astribot S1 doing humanoid factory task. Multi-step long-horizon task.

Visual navigation + pick and place manipulation.

## Environment

A small factory scene in Isaac Lab. Two workstations connected by an aisle.

Include a pickup table, a drop-off area, and obstacles. Vary object positions and obstacle placement as the task improves.

Source: build the floor with Isaac Lab's [GroundPlaneCfg](/home/eddie/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py) and the tables, obstacles, and box with [CuboidCfg](/home/eddie/IsaacLab/source/isaaclab/isaaclab/sim/spawners/shapes/shapes_cfg.py).

## Robot

Astribot S1

Use the mobile base to move between workstations. Start with one arm for manipulation.

Validate the robot model, gripper, and camera setup before training.

Source: combine the local [wheel URDF](/home/eddie/Desktop/Simulation/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_with_wheel.urdf) and [gripper definitions](/home/eddie/Desktop/Simulation/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf) from [Astribot-Dev](https://github.com/Astribot-Dev/astribot_descriptions), then validate mesh paths and collisions.

## Policy

Separate navigation and manipulation policies. A task controller coordinates the full sequence.

Start with scripted baselines. Train each skill before combining them into a longer task.

Measure full-task success, completion time, collisions, and dropped objects.

Source: adapt Isaac Lab's [lift_cube_sm.py](/home/eddie/IsaacLab/scripts/environments/state_machine/lift_cube_sm.py) for scripted manipulation, then train using this project's [RSL-RL PPO scaffold](/home/eddie/Desktop/Simulation/S1-Project/src/Factory_Project/tasks/factory/config/AstribotS1/agents/rsl_rl_ppo_cfg.py).

## Visual Navigation Policy

Use simulated RGB-D images to navigate toward the active workstation. Avoid obstacles and stop within manipulation reach.

Use simulator poses as an initial baseline. Evaluate visual navigation separately before adding manipulation.

Source: train a custom RGB-D navigation policy using Isaac Lab's [CameraCfg](/home/eddie/IsaacLab/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py) and [RslRlCNNModelCfg](/home/eddie/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py) with RSL-RL PPO.

## Exact task

Proposed task: move a small box between two factory workstations.

Navigate to the pickup table. Locate the box, grasp it, and lift it.

Carry it to the drop-off station. Place it inside the marked area and release it.

Return to the pickup station and repeat. Gradually vary the box position and route obstacles.

For the November 19 demo, prioritize one complete sequence. Treat robust learned policies as a later milestone.

Source: author the full sequence in this project's [factory task](/home/eddie/Desktop/Simulation/S1-Project/src/Factory_Project/tasks/factory), using Isaac Lab's [locomanipulation example](/home/eddie/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_pick_place/locomanipulation_g1_env_cfg.py) as a structural reference.
