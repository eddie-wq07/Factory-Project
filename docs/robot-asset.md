# The Astribot S1 asset

Open [astribot_s1.usd](../src/Factory_Project/assets/data/astribot_s1/astribot_s1.usd) in Isaac Sim, or reference its default prim into a Z-up, meter-scale stage. Keep the contents of the `astribot_s1/` folder together when moving the asset. For a ground plane at Z = 0, place the robot at Z = 0.09523 m in its neutral pose.

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

The [editable USDA](../src/Factory_Project/assets/data/astribot_s1/astribot_s1.usda) composes the geometry and materials from `model.usdc`. Rebuild with `python src/Factory_Project/assets/data/astribot_s1/build_usd.py` using the bundled URDF and meshes. The [source notes](../src/Factory_Project/assets/data/astribot_s1/SOURCE.md) detail wheel contact, gripper coupling and camera placement.

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

[Simulated reach](../recordings/misc/11-18-2026/12:46am_s1-forward-reach.png) · [Four wheel assemblies](../recordings/misc/11-18-2026/12:24am_s1-four-wheels.png) · [Upright head camera](../recordings/misc/11-18-2026/12:27am_s1-head-upright.png)
