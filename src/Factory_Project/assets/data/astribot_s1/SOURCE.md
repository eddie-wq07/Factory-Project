# Astribot S1 source model

The bundled URDF, CAD meshes and camera parameters derive from [Astribot-Dev/astribot_descriptions, revision 8c651a1](https://github.com/Astribot-Dev/astribot_descriptions/tree/8c651a1abbe2821824a692aa3a3aa4cbf88e4122). The upstream Apache 2.0 license is included in [LICENSE](LICENSE).

## Composition

| Local component | Upstream source and project edits |
| --- | --- |
| Body and grippers | `astribot_whole_body_maniskill.urdf` supplies the body, inertias, head, 7-axis arms and two grippers. Original DAE materials supply the panel colors and chest logo. |
| Wheels | `astribot_whole_body_with_wheel.urdf` supplies four continuous joints, their hub placements and inertias. All four original tire STLs have identical bytes, so every wheel uses one shared local STL. The rubber color comes from the gripper MJCF. |
| Wheel contact | Each wheel has 16 passive rollers to permit lateral rolling. The spherical contact proxies have a 12 mm radius. Their centers lie on a 68.228 mm radius ring at an axial offset of 2.65 mm, giving the wheel a CAD-matched outer radius of 80.228 mm. Each unpowered roller spins about the tangent to the ring and weighs 8 g. Hub mass, center of mass and inertia were adjusted using the parallel-axis theorem to preserve the upstream CAD values for the assembled wheel. The neutral wheel contact plane is 0.095228 m below the root. The hub uses simulation limits of 30 N·m and 20 rad/s, with a velocity-drive damping gain of 45 N·m·s/rad. Roller geometry and drive limits are project approximations. |
| Body contact | The chassis collision cylinder uses the vertical CAD bounds for the base. This gives the body about 21 mm of ground clearance. A bounding box represents the previously missing chest collider. |
| Gripper motion | Each gripper's MJCF model supplies joint axes, travel, palm and fingertip contact pads. Its actuator equality constraints give five mimic relationships to `astribot_gripper_{left,right}_joint_L1`. A driver angle of zero opens the fingers and a positive angle closes them. |
| Cameras | [cameras.json](urdf/cameras.json) contains the head and wrist camera poses and vertical fields of view from `model_with_effector/astribot_s1_model_with_gripper.xml`. Positions are in the named link's local frame in meters. Quaternions use scalar-first order. Each USD camera looks along local −Z with +Y up. The USD applies a 180° image roll to the head camera after the original mount looked upside down in a test render. Its position, view direction and field of view are preserved. |

For stable CPU and GPU dynamics, the USD requests 32 position and eight velocity solver iterations. The simulated reflected motor inertia is 0.06 kg·m² per upper-body joint and 0.005 kg·m² per gripper joint. These are simulation parameters. The source CAD mass and inertia tensors are preserved.

The [source URDF](urdf/astribot_s1.urdf) uses relative paths for every mesh. The chest DAE resolves its logo texture from the same directory. The source tree has 107 links and 106 joints. These include six fixed mounts, four driven wheel hubs, 64 passive roller joints and 32 limited revolute joints. Ten revolute joints follow gripper drivers, leaving 26 independently commanded joints.

## Upstream asset audit

At the revision above, the tracked binary USD references `astribot_whole_body_maniskill.usd` through an absolute path on the exporter’s workstation. Opening it here yielded zero composed physics joints. The tracked USDA is present and opens independently with 32 revolute joints. It has no wheel joints and its `OmniPBR.mdl` dependency is unresolved in standalone OpenUSD. The plain whole-body URDF also lacks wheels. The wheel-specific URDF includes four hubs, but its wheel mesh paths omit the `meshes/` prefix and the tires have no collision shape.

This project builds from the URDF, MJCF parameters and CAD meshes. The upstream generated USD directories were excluded. The shipped USD references the editable USDA, the binary model layer and a local chest texture. It uses USD Preview Surface materials.
