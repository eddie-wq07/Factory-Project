"""Scripted teacher: a deterministic mission executive plus short manipulation and driving skills.

The executive owns the long horizon. It walks the alphabetical queue, asks the environment where each
item must rest, and calls short skills (dock, clamp-pick, carry, stack, pinch-pick, bin-place, retreat),
checking the outcome of each one and retrying within a fixed budget. It reads simulator truth through
the environment's accessors, so every result it produces is a *privileged-state* baseline.

Skills are generators. Each ``yield`` hands one 25-value action to the caller, which steps the
environment and resumes the generator; a skill returns its outcome with ``return``. A learned policy
can replace any skill by producing the same actions.

How the S1 handles a carton, as found in simulation:

* Cartons and crates are far wider than the gripper opens, so both hands clamp one between their
  rubber housings. Each hand toes in and bears along the housing's front edge, a little above the
  carton's centre of mass, so the carton hangs level between two lines like a pendulum.
* The hands keep one pitch from grasp to release, because the carton turns with them.
* The torso posture is always chosen explicitly and changed gradually. At a shelf it is the posture
  that keeps the head and chest in front of the beam above; in the open it follows the carton's height.
* The arms take their extended shape in the open and the chassis carries them in and out. Folding
  them near a fixture swings the elbows into neighbouring stock.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np

from . import kinematics
from .geometry import Pose, wrap_angle
from .kinematics import BASE_HEIGHT, HandTarget, UpperBodyIK
from .manifest import Item, obstacle_rectangles
from .navigation import NoRouteError, OccupancyMap, PathFollower, PoseController, RoutePlanner
from .stacking import NoPlacementError, Placement

Action = np.ndarray
Skill = Generator[Action, None, bool]

GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.93
HOUSING_HALF_THICKNESS = 0.031
"""Half the gripper housing's thickness: the palm origin sits this far outside a clamped face [m]."""
CONTACT_LEAD = 0.042
"""Distance from the palm origin to the housing's front edge, the line that bears on a clamped carton [m]."""
PINCH_REACH = 0.085
"""Distance from the palm origin to the centre of the fingertip pads, along the fingers [m]."""
CHASSIS_RADIUS = 0.30
MAX_KNEE = 0.2
"""Largest forward lean of the shin [rad]. It puts the front of the knee 0.18 m ahead of the chassis
centre, short of a shelf beam or a stocked pallet at any dock distance the skills use."""
NATURAL_LEFT_ARM = (-0.4, -0.1, -1.4, 1.4, 0.0, 0.0, 0.0)
NATURAL_RIGHT_ARM = (0.4, -0.1, 1.4, 1.4, 0.0, 0.0, 0.0)
"""Arm seed with the upper arms hanging and the forearms pointing forward, elbows by the sides."""


@dataclass
class TeacherConfig:
    """Tunable numbers of the scripted skills. Distances in metres, times in seconds, angles in radians."""

    control_dt: float = 0.05
    # -- driving
    travel_speed: float = 0.9
    carry_speed: float = 0.55
    travel_radius: float = 0.45
    carry_radius: float = 0.62
    standoff: float = 0.65
    """Extra distance behind a dock pose where the robot shapes or folds its arms in the open."""
    # -- posture
    shoulder_above: float = 0.14
    """Height of the shoulders above a carried carton's centre when working in the open."""
    torso_step: float = 0.12
    """Largest change of any torso joint between consecutive arm waypoints. A bigger change is split
    into intermediate waypoints, because the arms cannot hold a carton steady through a torso lurch."""
    rack_shoulder_heights: tuple[float, ...] = (0.25, 0.14)
    """Candidate heights of the shoulders above the palms while working at a shelf. Higher shoulders
    reach farther; lower ones keep the elbows of a wide clamp under the beam above."""
    beam_clearance: float = 0.04
    """Gap kept between the head or chest and the front of the rack beams."""
    # -- clamp grasp
    clamp_open: float = 0.07
    """Clearance between each hand's bearing edge and the carton side while reaching in."""
    clamp_above_fraction: float = 0.3
    """Least height of the grip above a carton's centre of mass, as a fraction of the carton's height."""
    clamp_toe: float = 0.14
    """Toe-in of each hand as commanded: how far the fingertip end turns toward the carton
    [small-angle]. Under the clamping force the wrists give by about this much, so the hands end up
    flat on the carton with the forearms bearing on its near corners. That long two-point patch on
    each side, not a single line, is what keeps a carton level."""
    clamp_above: float = 0.06
    """Height of the bearing line above the carton's centre of mass."""
    clamp_pitches: tuple[float, ...] = (0.0, -0.2, -0.4, 0.2)
    """Candidate pitches of the fingers above horizontal while clamping, in order of preference."""
    squeeze_steps: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    """Depths, inside the carton's nominal face, to which the palms are commanded while closing."""
    grip_friction: float = 0.7
    """Assumed friction between the rubber housings and a carton."""
    grip_safety: float = 3.5
    """Clamping force as a multiple of the least that holds the item's weight."""
    min_grip_force: float = 60.0
    """Least clamping force per hand, however light the item [N]."""
    grip_moment: float = 1.2
    """Pitching moment the two bearing edges hold by friction [N·m]. It bounds how far short of an
    item's centre of mass the palms may grip when the centre itself is out of reach."""
    max_box_reach: float = 0.66
    """Farthest the centre of a clamped carton is held from the chassis centre."""
    reach_margin: float = 0.03
    """A planned rack dock must still work with the carton this much farther away."""
    rack_min_dock: float = 0.25
    """Closest the chassis centre may come to a rack item's front face; the chassis rolls under the shelf."""
    joint_speed: float = 0.25
    """Fastest any torso or arm joint moves along a planned path [rad/s]."""
    pinch_dock: float = 0.42
    """Distance from the chassis centre to the front of a small item picked with one gripper [m]."""
    regrips: int = 3
    """Times a carton that lifts crooked is set down and gripped again at one dock."""
    regrip_step: float = 0.025
    """Distance the grip moves along the carton on each new grip [m]."""
    level_tolerance: float = 0.06
    """Largest end-to-end tilt of a lifted carton accepted before it leaves the shelf [rad]."""
    lift: float = 0.05
    """Height a carton is raised off its shelf, where the shelf above leaves room for it."""
    deck_thickness: float = 0.035
    """Thickness of a shelf deck with its front beam."""
    elbow_headroom: float = 0.12
    """Gap kept between the elbows and the underside of the shelf above while reaching into a shelf.
    The forearms pass under that shelf's beam and must still have room to lift."""
    carry_height: float = 1.10
    """Height of a carried carton's centre above the floor: high enough that a 0.5 m carton's
    underside stays well above the knee."""
    carry_gap: float = 0.28
    """Distance from the chassis centre to the near face of a carried carton, clear of the knee and chest."""
    place_clearance: float = 0.035
    """Height above the resting pose at which a carton is brought over its column."""
    release_gaps: tuple[float, ...] = (0.05, 0.03, 0.015)
    """Hand openings tried, widest first, after the squeeze has been relaxed."""
    pallet_clearance: float = 0.33
    """Closest the chassis centre may come to a pallet's aisle edge."""
    # -- timing
    reach_time: float = 3.0
    squeeze_time: float = 1.0
    lift_time: float = 1.2
    settle_time: float = 0.4
    # -- recovery
    pick_attempts: int = 2
    place_attempts: int = 2
    # -- demonstration variation
    dock_noise: float = 0.0
    """Uniform chassis offset added to every dock pose, to diversify demonstrations."""
    dock_yaw_noise: float = 0.0


@dataclass
class Step:
    """One waypoint of an arm move: where the palms go and how the torso is held meanwhile."""

    targets: tuple
    """Left and right :class:`HandTarget` (either may be ``None``)."""
    torso: np.ndarray | None = None
    """The four torso joint angles to pin, for work at a shelf where the body must stay clear of the
    beams. ``None`` leaves the torso to the solver, biased toward the stance for :attr:`height`."""
    height: float = 1.0
    """Working height above the floor, which selects the stance a free torso is drawn toward."""


@dataclass
class MissionLog:
    events: list[tuple[float, str]] = field(default_factory=list)
    item_results: dict[str, str] = field(default_factory=dict)

    def add(self, time: float, message: str, echo: bool = True) -> None:
        self.events.append((time, message))
        if echo:
            print(f"[teacher {time:7.1f}s] {message}", flush=True)


class Teacher:
    """Scripted controller for the six-item sorting mission."""

    def __init__(self, env, config: TeacherConfig | None = None, seed: int = 0, verbose: bool = True):
        self.env = env
        self.cfg = config or TeacherConfig()
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.ik = UpperBodyIK()
        self.log = MissionLog()
        occupancy = OccupancyMap.load(rectangles=obstacle_rectangles(env.manifest))
        self.planners = {
            False: RoutePlanner(occupancy, self.cfg.travel_radius),
            True: RoutePlanner(occupancy, self.cfg.carry_radius),
        }
        self.q = env.ik_joint_positions().copy()
        self.head = np.zeros(2)
        self.grip = np.array([GRIPPER_CLOSED, GRIPPER_CLOSED])
        self.skill = "idle"
        """Name of the running skill, recorded with every demonstration step."""
        self.target = ""
        # -- the grip planned for the item in hand; see :meth:`plan_grasp`
        self.forward = 0.0
        """Grip offset ahead (+) of the carton's centre."""
        self.pitch = 0.0
        """Pitch of the fingers above horizontal, fixed from grasp to release."""
        self.dock_distance = 0.4
        """Distance from the chassis centre to the item's front face at the rack."""
        self.rack_shoulder = 0.25
        """Height of the shoulders above the palms while working at the shelf."""
        self.squeeze = 0.035
        """Depth inside the nominal carton face at which the palms found a firm grip."""
        self.held = Pose.from_yaw([0.5, 0.0, 0.9], math.pi)
        """Commanded chassis-frame pose of the carton in hand."""
        self.unreachable = ""
        """Description of the last arm step that inverse kinematics could not reach."""

    # -- bookkeeping -------------------------------------------------------------------------------------

    @property
    def time(self) -> float:
        return float(self.env.episode_length_buf[0]) * self.cfg.control_dt

    def note(self, message: str) -> None:
        self.log.add(self.time, message, self.verbose)

    def action(self, twist=(0.0, 0.0, 0.0)) -> Action:
        q = self.q
        # Keep the head upright as the trunk leans: it stays out of the shelf above and the head
        # camera keeps looking ahead.
        self.head[1] = self.ik.level_head_tilt(q)
        return np.concatenate([np.asarray(twist, dtype=float), q[:4], self.head, q[4:11], q[11:18], self.grip])

    # -- postures ----------------------------------------------------------------------------------------

    @staticmethod
    def _torso(palm_height: float, shoulder_above: float, forward: float) -> np.ndarray:
        """Torso angles putting the shoulders a set height above the working palms, hips set back."""
        standing = kinematics.STANDING_SHOULDER
        shoulder = float(np.clip(palm_height - BASE_HEIGHT + shoulder_above, 0.85, standing))
        knee = float(np.clip(0.8 * (standing - shoulder), 0.0, MAX_KNEE))
        return kinematics.torso_for_shoulder(shoulder, knee, forward)

    def stance(self, height: float) -> np.ndarray:
        """Natural full posture for palms working at this height in the open: hips back, shoulders above the palms."""
        return self.seed(self._torso(height, self.cfg.shoulder_above, 0.03))

    def shelf_posture(self, height: float, beam_distance: float, shoulder_above: float) -> np.ndarray:
        """Torso posture for working at a shelf: as far forward as keeps the head and chest clear of the beams.

        Args:
            height: Palm height above the floor.
            beam_distance: Distance from the chassis centre to the front face of the rack beams.
            shoulder_above: Height of the shoulders above the palms.
        """
        q = np.zeros(18)
        for forward in (0.15, 0.12, 0.09, 0.06, 0.03, 0.0, -0.03):
            q[:4] = self._torso(height, shoulder_above, forward)
            if self.ik.upper_body_front(q) <= beam_distance - self.cfg.beam_clearance:
                break
        return q[:4].copy()

    @staticmethod
    def seed(torso: np.ndarray) -> np.ndarray:
        """A full posture with the given torso and the arms in their natural forward-reaching shape."""
        return np.concatenate([torso, NATURAL_LEFT_ARM, NATURAL_RIGHT_ARM])

    @staticmethod
    def travel_posture() -> np.ndarray:
        """Standing with the forearms folded up in front of the chest, inside the chassis footprint."""
        q = np.zeros(18)
        q[4:11] = (0.0, -0.05, -1.4, 2.2, 0.0, 0.0, 0.0)
        q[11:18] = (0.0, -0.05, 1.4, 2.2, 0.0, 0.0, 0.0)
        return q

    # -- low-level motion --------------------------------------------------------------------------------

    def hold(self, duration: float, hold: PoseController | None = None) -> Skill:
        for _ in range(max(1, round(duration / self.cfg.control_dt))):
            yield self.action(self._hold_twist(hold))
        return True

    def _hold_twist(self, hold: PoseController | None) -> np.ndarray:
        if hold is None:
            return np.zeros(3)
        position, yaw = self.env.base_pose()
        return hold.command(position, yaw, self.cfg.control_dt)[0]

    def move_joints(self, goal: np.ndarray, duration: float, hold: PoseController | None = None) -> Skill:
        """Move the torso and arms to a joint goal along a smooth-step profile."""
        start = self.q.copy()
        steps = max(1, round(duration / self.cfg.control_dt))
        for index in range(steps):
            s = (index + 1) / steps
            s = s * s * (3 - 2 * s)
            self.q = start + (goal - start) * s
            yield self.action(self._hold_twist(hold))
        return True

    def solve(self, step: Step, seed: np.ndarray | None = None, continuous: bool = False) -> np.ndarray | None:
        """Joint angles for one step, or ``None`` if its palm targets are out of reach.

        Tries the given seed first, for continuity with the current posture, then the natural seed.
        With ``continuous`` only the given seed is tried: a solution found from the natural seed can
        lie on another branch of the arm's kinematics, and moving there would swing the hands apart.
        """
        left, right = step.targets
        if step.torso is not None:
            natural = self.seed(step.torso)
            options = dict(torso=step.torso)
        else:
            natural = self.stance(step.height)
            # The knee is pinned only while squatting, where it would otherwise creep toward a
            # fixture. Standing, the legs may lean a little to help the arms at the top of their range.
            options = dict(knee=natural[0] if natural[0] > 0.02 else None)
        starts = (seed,) if continuous and seed is not None else (seed, natural)
        for start in starts:
            if start is None:
                continue
            q, ok = self.ik.solve(start, left, right, posture=natural, posture_cost=0.005, iterations=300, **options)
            if ok:
                return q
        return None

    def follow(self, steps, duration: float, hold: PoseController | None = None, fresh: bool = False) -> Skill:
        """Move through a list of :class:`Step`, solving IK from one to the next.

        ``fresh`` starts the search from the natural arm shape instead of the current posture; use it
        for the first reach of a skill, since continuing from the tucked travel posture can land on an
        elbows-out solution. Wherever the torso would change quickly between two waypoints, extra
        waypoints are inserted that move it gradually with the palms already on the next target.
        Nothing moves, and the result is False, if any step is out of reach.
        """
        solutions = []
        previous = None if fresh else self.q
        for index, step in enumerate(steps):
            q = self.solve(step, previous, continuous=previous is not None)
            if q is None:
                self.unreachable = self._describe(step, index, len(steps))
                return False
            if previous is not None:
                pieces = math.ceil(float(np.abs(q[:4] - previous[:4]).max()) / self.cfg.torso_step)
                for k in range(1, pieces):
                    torso = previous[:4] + (q[:4] - previous[:4]) * k / pieces
                    between = self.solve(
                        Step(step.targets, torso), solutions[-1] if solutions else previous, continuous=True
                    )
                    if between is not None:
                        solutions.append(between)
            solutions.append(q)
            previous = q
        # The arms are soft and trail a fast torso; with a carton between the palms that lag is a
        # jolt, so no joint is asked to move faster than the configured speed.
        for q in solutions:
            slowest = float(np.abs(q - self.q).max()) / self.cfg.joint_speed
            yield from self.move_joints(q, max(duration / len(solutions), slowest), hold)
        return True

    @staticmethod
    def _describe(step: Step, index: int, count: int) -> str:
        """One-line description of a step that could not be reached, for the mission log."""
        palms = [
            None if target is None else np.round(target.position + [0.0, 0.0, BASE_HEIGHT], 3).tolist()
            for target in step.targets
        ]
        torso = "free" if step.torso is None else np.round(step.torso, 2).tolist()
        return f"step {index + 1}/{count}: palms {palms}, torso {torso}"

    def chain(self, steps) -> np.ndarray | None:
        """Final posture of a chain of steps solved from the natural arm shape, or ``None`` if one is out of reach."""
        seed = None
        for step in steps:
            seed = self.solve(step, seed)
            if seed is None:
                return None
        return seed

    def reachable(self, steps) -> bool:
        """Whether a chain of steps can be solved starting from the natural arm shape."""
        return self.chain(steps) is not None

    def shelf_headroom(self, item: Item) -> tuple[float, float]:
        """Height of the underside of the shelf above a rack item, and the free height over the item [m]."""
        tops = sorted(self.env.manifest.rack.shelf_tops)
        deck = tops[item.level - 1]
        above = tops[item.level] if item.level < len(tops) else deck + (tops[-1] - tops[-2])
        underside = above - self.cfg.deck_thickness
        return underside, underside - (deck + item.size[2])

    def shelf_lift(self, item: Item) -> float:
        """How far a rack item may be lifted without touching the shelf above."""
        return max(0.015, min(self.cfg.lift, self.shelf_headroom(item)[1] - 0.025))

    # -- driving -----------------------------------------------------------------------------------------

    def drive_to(self, goal_xy, final_yaw: float, loaded: bool) -> Skill:
        """Plan and follow a route to a floor position, ending at the given heading."""
        position, yaw = self.env.base_pose()
        try:
            path = self.planners[loaded].plan(position[:2], goal_xy)
        except NoRouteError as error:
            self.note(f"route planning failed: {error}")
            return False
        speed = self.cfg.carry_speed if loaded else self.cfg.travel_speed
        follower = PathFollower(path, speed=speed, acceleration=0.3 if loaded else 0.5, final_yaw=final_yaw)
        budget = follower.remaining / (0.3 * speed) + 30.0
        elapsed = 0.0
        while elapsed < budget:
            position, yaw = self.env.base_pose()
            twist, done = follower.command(position, yaw, self.cfg.control_dt)
            if done:
                break
            yield self.action(twist)
            elapsed += self.cfg.control_dt
        else:
            self.note("route following timed out")
            return False
        return (yield from self.dock(goal_xy, final_yaw, tolerance=0.02, yaw_tolerance=0.02))

    def dock(
        self, xy, yaw: float, *, tolerance: float = 0.015, yaw_tolerance: float = 0.02, speed: float = 0.3
    ) -> Skill:
        """Park the chassis on a pose and return once it has settled there."""
        controller = PoseController(xy, yaw, speed=speed, position_tolerance=tolerance, yaw_tolerance=yaw_tolerance)
        budget = float(np.linalg.norm(np.asarray(xy) - self.env.base_pose()[0][:2])) / (0.4 * speed) + 12.0
        elapsed = 0.0
        while elapsed < budget:
            position, heading = self.env.base_pose()
            twist, done = controller.command(position, heading, self.cfg.control_dt)
            if done:
                return True
            yield self.action(twist)
            elapsed += self.cfg.control_dt
        return False

    def retreat(self, xy, yaw: float) -> Skill:
        """Back straight away from a fixture on extended arms, then tuck them for travel."""
        self.skill = "retreat"
        yield from self.dock(xy, yaw, tolerance=0.05, yaw_tolerance=0.08, speed=0.2)
        yield from self.move_joints(self.travel_posture(), 2.5)
        return True

    # -- clamp geometry ----------------------------------------------------------------------------------

    def clamp_targets(self, box: Pose, item: Item, gap: float):
        """Chassis-frame palm targets clamping a carton whose box frame is ``box`` (also chassis-frame).

        ``gap`` is the clearance between each hand's bearing edge and the carton face: positive is
        open, negative squeezes. The carton's front (+X of its box frame) faces the robot, so the
        robot's right palm takes the box's +Y side.
        """
        half = item.size[1] / 2 + HOUSING_HALF_THICKNESS + gap
        toe = self.cfg.clamp_toe
        # Tall cartons are gripped higher: a clamped carton can turn about the line between the
        # palms, and the further its centre of mass hangs below that line the steadier it rides.
        up = min(max(self.cfg.clamp_above, self.cfg.clamp_above_fraction * item.size[2]), item.size[2] / 2 - 0.03)
        targets = []
        for side in (-1.0, 1.0):  # left palm on the box's -Y side, right palm on +Y
            # The front edge of each hand's rubber housing goes level with the carton's centre of
            # mass. Measured at the shelf, a carton gripped there hangs level to within a few degrees;
            # with the middle of the pad there instead it hangs nose-up.
            position = box.apply([CONTACT_LEAD - self.forward, side * (half + toe * CONTACT_LEAD), up])
            # Each palm's +Y axis faces the carton, which keeps the elbow just outside the hand. The
            # toe-in turns the fingertips toward the carton so the forearm, which is thicker than the
            # hand, clears the near corner as the hands close.
            inward = box.rotation @ np.array([toe, -side, 0.0])
            # One pitch from grasp to release: friction along the bearing edge makes the carton turn
            # with the hands, so a pitch that drifted as the arms rose would tip it.
            fingers = box.rotation @ np.array([-math.cos(self.pitch), 0.0, math.sin(self.pitch)])
            target = HandTarget(position, normal=inward / np.linalg.norm(inward), fingers=fingers, free_pitch=False)
            targets.append(target)
        return targets[0], targets[1]

    @staticmethod
    def line(start: Pose, end: Pose, step: float = 0.08) -> list[Pose]:
        """Chassis-frame carton poses in a straight line from ``start`` (excluded) to ``end``, front to the robot."""
        count = max(1, math.ceil(float(np.linalg.norm(end.position - start.position)) / step))
        return [
            Pose.from_yaw(start.position + (end.position - start.position) * k / count, math.pi)
            for k in range(1, count + 1)
        ]

    def open_steps(self, item: Item, poses, gap: float) -> list[Step]:
        """Clamp steps for carton poses in the open, where the torso is free to follow the carton."""
        return [
            Step(self.clamp_targets(pose, item, gap), height=float(pose.position[2]) + BASE_HEIGHT) for pose in poses
        ]

    def box_in_chassis(self, address: str) -> Pose:
        return self.env.base_frame().inverse() * self.env.object_pose(address)

    def carried_box_pose(self, item: Item, height: float | None = None, reach: float | None = None) -> Pose:
        """Chassis-frame pose at which a clamped carton is held: square, centred, front toward the robot."""
        height = self.cfg.carry_height if height is None else height
        distance = self.cfg.carry_gap + item.size[0] / 2 if reach is None else reach
        return Pose.from_yaw([distance, 0.0, height - BASE_HEIGHT], math.pi)

    def place_reach(self, item: Item, placement: Placement) -> float:
        """Distance from the chassis centre to the carton's centre when it is set down on its column."""
        zone = self.env.manifest.destination(item.address)
        normal = np.array(zone.normal, dtype=float)
        front = placement.pose.position[:2] + normal * item.size[0] / 2
        edge_distance = float((np.array(zone.edge) - front) @ normal)
        return max(self.cfg.carry_gap + item.size[0] / 2, edge_distance + self.cfg.pallet_clearance + item.size[0] / 2)

    def plan_grasp(self, item: Item, placement: Placement) -> dict | None:
        """Choose, before the pick, how the palms grip the carton and how the robot stands at the rack.

        Everything follows from inverse kinematics rather than a reach table. The palms grip at the
        centre of mass, at a finger pitch they can keep from the shelf to the stack, if the arms can
        then hold the carton out over its column and open there. A light item may be gripped a little
        short of centre when they cannot, as far as the friction moment of the two bearing edges
        allows. The rack dock is the farthest one that leaves :attr:`TeacherConfig.reach_margin`,
        with the torso held where the head and chest stay clear of the beam above.

        Returns:
            ``forward`` (grip offset ahead of the carton's centre), ``pitch`` (finger pitch),
            ``dock`` (distance from the carton's front face at the rack) and ``shoulder`` (shoulder
            height above the palms at the shelf); ``None`` if no grip serves both ends.
        """
        cfg = self.cfg
        rack_height = float(self.env.object_pose(item.address).position[2])
        place_height = float(placement.pose.position[2])
        out = Pose.from_yaw([self.place_reach(item, placement), 0.0, place_height - BASE_HEIGHT], math.pi)
        above = Pose.from_yaw(out.position + np.array([0.0, 0.0, cfg.place_clearance]), math.pi)
        carry = self.carried_box_pose(item)
        deepest = -self.squeeze_depth(item)
        limit = cfg.grip_moment / (item.mass * 9.81)
        nearest = cfg.rack_min_dock + item.size[0] / 2
        underside, _ = self.shelf_headroom(item)
        lift = self.shelf_lift(item)
        saved = self.forward, self.pitch
        try:
            for forward in -np.arange(0.0, limit + 1e-9, 0.02):
                for pitch in cfg.clamp_pitches:
                    self.forward, self.pitch = float(forward), float(pitch)
                    level = Pose.from_yaw([out.position[0], 0.0, carry.position[2]], math.pi)
                    in_the_open = (
                        self.open_steps(item, [carry] + self.line(carry, level) + self.line(level, above), deepest),
                        self.open_steps(item, [out], deepest) + self.open_steps(item, [out], cfg.release_gaps[-1]),
                    )
                    if not all(self.reachable(steps) for steps in in_the_open):
                        continue
                    best = None
                    for shoulder_above in cfg.rack_shoulder_heights:
                        for centre in np.arange(cfg.max_box_reach, nearest - 1e-9, -0.02):
                            dock = centre - item.size[0] / 2
                            if best is not None and dock <= best["dock"]:
                                break
                            torso = self.shelf_posture(
                                rack_height, dock - self.env.manifest.front_inset, shoulder_above
                            )
                            poses = [
                                Pose.from_yaw([centre + reach, 0.0, rack_height - BASE_HEIGHT + rise], math.pi)
                                for reach in (0.0, cfg.reach_margin)
                                for rise in (0.0, lift)
                            ]
                            far, *rest = poses[2], poses[0], poses[1], poses[3]
                            shelf = [
                                Step(self.clamp_targets(far, item, gap), torso) for gap in (cfg.clamp_open, deepest)
                            ]
                            reached = self.chain(shelf)
                            if reached is None or self.ik.elbow_height(reached) > underside - cfg.elbow_headroom:
                                continue
                            if not all(
                                self.reachable([Step(self.clamp_targets(pose, item, deepest), torso)]) for pose in rest
                            ):
                                continue
                            lifted = poses[1]
                            route = self.open_steps(item, [lifted] + self.line(lifted, carry, 0.05), deepest)
                            if self.reachable(route):
                                best = dict(
                                    forward=float(forward),
                                    pitch=float(pitch),
                                    dock=float(dock),
                                    shoulder=shoulder_above,
                                )
                                break
                    if best is not None:
                        return best
            return None
        finally:
            self.forward, self.pitch = saved

    # -- clamp pick --------------------------------------------------------------------------------------

    def clamp_pick(self, address: str) -> Skill:
        """Dock at the rack, clamp the carton between both palms, lift it clear and back away."""
        item = self.env.manifest.item(address)
        rack = self.env.manifest.rack
        normal = np.array(rack.normal, dtype=float)
        cfg = self.cfg
        for attempt in range(cfg.pick_attempts):
            self.skill, self.target = "pick", address
            box_world = self.env.object_pose(address)
            height = float(box_world.position[2])
            front = box_world.position[:2] + normal * item.size[0] / 2
            dock_distance = self.dock_distance
            torso = self.shelf_posture(height, dock_distance - self.env.manifest.front_inset, self.rack_shoulder)
            dock_xy = front + normal * dock_distance + self._dock_noise()
            dock_yaw = rack.robot_yaw + self._yaw_noise()
            approach_xy = dock_xy + normal * cfg.standoff
            self.note(f"pick {address}: attempt {attempt + 1}, dock {dock_distance:.2f} m from its front face")

            if not (yield from self.drive_to(approach_xy, dock_yaw, loaded=False)):
                continue
            self.grip[:] = GRIPPER_CLOSED
            nominal = Pose.from_yaw([dock_distance + item.size[0] / 2, 0.0, height - BASE_HEIGHT], math.pi)
            # Swinging both arms out turns the chassis a little, so it is held on the standoff pose
            # while they move and squared up again before it carries them in beside the carton.
            standoff = PoseController(approach_xy, dock_yaw, hold=True)
            opened = [Step(self.clamp_targets(nominal, item, cfg.clamp_open), torso)]
            if not (yield from self.follow(opened, cfg.reach_time, standoff, fresh=True)):
                self.note(f"pick {address}: carton out of reach at this shelf ({self.unreachable})")
                return False
            yield from self.dock(approach_xy, dock_yaw, tolerance=0.01, yaw_tolerance=0.01, speed=0.2)
            if not (yield from self.dock(dock_xy, dock_yaw, speed=0.2)):
                self.note(f"pick {address}: could not settle at the rack, backing out")
                yield from self.retreat(approach_xy, dock_yaw)
                continue
            hold = PoseController(dock_xy, dock_yaw, hold=True)

            # Close on the carton where it actually is, squeeze until the arms meet resistance and
            # lift. A carton that comes up nose-down or nose-up is held off its centre of mass: it
            # is set back down and gripped again further along, as a person shifts their hands.
            lifted = None
            for regrip in range(cfg.regrips + 1):
                box = self.box_in_chassis(address)
                box = Pose.from_yaw(box.position, box.yaw)
                around = [Step(self.clamp_targets(box, item, cfg.clamp_open), torso)]
                if not (yield from self.follow(around, 0.8, hold)):
                    self.note(f"pick {address}: carton out of reach from this dock ({self.unreachable})")
                    break
                if not (yield from self.squeeze_until_tight(item, box, torso, hold)):
                    self.note(f"pick {address}: no firm grip on the carton, releasing")
                    yield from self.follow(around, cfg.squeeze_time, hold)
                    break

                # Lift as far as the shelf above allows, then top the command up by whatever the
                # loaded arms gave way, so the carton really clears the deck before the chassis moves.
                lift = self.shelf_lift(item)
                start_height = float(self.env.object_pose(address).position[2])
                top = Pose.from_yaw(box.position + np.array([0.0, 0.0, lift]), box.yaw)
                raise_steps = [
                    Step(self.clamp_targets(pose, item, -self.squeeze), torso) for pose in self.line(box, top, 0.03)
                ]
                if not (yield from self.follow(raise_steps, cfg.lift_time, hold)):
                    self.note(f"pick {address}: lift out of reach ({self.unreachable})")
                yield from self.hold(cfg.settle_time, hold)
                rise = float(self.env.object_pose(address).position[2]) - start_height
                # The carton turns with the hands, and loaded wrists droop: level it by pitching
                # the hands back by what it hangs, before judging the grip.
                if rise > 0.005:
                    yield from self.level(
                        address,
                        lambda top=top, torso=torso: [Step(self.clamp_targets(top, item, -self.squeeze), torso)],
                        hold,
                    )
                    rise = float(self.env.object_pose(address).position[2]) - start_height
                nose = self.nose_down(address)
                if abs(nose) <= cfg.level_tolerance and 0.005 < rise < lift - 0.008:
                    higher = Pose.from_yaw(top.position + np.array([0.0, 0.0, lift - rise]), box.yaw)
                    if (
                        yield from self.follow(
                            [Step(self.clamp_targets(higher, item, -self.squeeze), torso)], 0.8, hold
                        )
                    ):
                        top = higher
                    yield from self.hold(cfg.settle_time, hold)
                    rise = float(self.env.object_pose(address).position[2]) - start_height
                    nose = self.nose_down(address)
                if abs(nose) <= cfg.level_tolerance and rise >= 0.4 * lift:
                    lifted = top
                    break
                reason = (
                    f"hangs {math.degrees(nose):+.1f} deg nose-down"
                    if abs(nose) > cfg.level_tolerance
                    else f"rose only {rise * 100:.1f} cm"
                )
                self.note(f"pick {address}: carton {reason}, setting it down")
                yield from self.follow(list(reversed(raise_steps)), cfg.lift_time, hold)
                yield from self.follow(around, cfg.squeeze_time, hold)
                yield from self.hold(cfg.settle_time, hold)
                if abs(nose) > cfg.level_tolerance and regrip < cfg.regrips:
                    shift = math.copysign(cfg.regrip_step, nose)
                    self.forward += shift
                    self.note(f"pick {address}: gripping {shift * 100:+.1f} cm further in")
                else:
                    break
            if lifted is None:
                yield from self.retreat(approach_xy, dock_yaw)
                continue

            # Back out on extended arms, then bring the carton straight from where it hangs to the
            # carry pose, the torso straightening gradually as it comes.
            if not (yield from self.dock(approach_xy, dock_yaw, tolerance=0.04, yaw_tolerance=0.04, speed=0.15)):
                return False
            standoff = PoseController(approach_xy, dock_yaw, hold=True)
            carried = self.carried_box_pose(item)
            route = self.open_steps(item, self.line(lifted, carried, 0.05), -self.squeeze)
            if not (yield from self.follow(route, 7.0, standoff)):
                self.note(f"pick {address}: could not bring the carton to the carry pose ({self.unreachable})")
                return False
            self.held = carried
            yield from self.hold(cfg.settle_time, standoff)
            yield from self.level_held(address, standoff)
            if self.is_held(address):
                self.note(f"pick {address}: carton secured")
                return True
            self.note(f"pick {address}: carton lost while backing out")
            return False
        return False

    def nose_down(self, address: str) -> float:
        """Angle by which a clamped carton's far end hangs below its near end [rad]."""
        return math.asin(float(np.clip(self.box_in_chassis(address).rotation[2, 0], -1.0, 1.0)))

    def level(self, address: str, steps, hold) -> Skill:
        """Pitch the hands until the clamped carton hangs level.

        A carton turns with the hands that clamp it, and loaded wrists droop by an amount that
        changes with the arms' posture. ``steps`` returns the clamp steps for the current hand pitch.
        """
        for _ in range(4):
            nose = self.nose_down(address)
            if abs(nose) < 0.02:
                return True
            pitch = self.pitch
            self.pitch = float(np.clip(pitch + np.clip(nose, -0.08, 0.08), -0.4, 0.4))
            if not (yield from self.follow(steps(), 1.0, hold)):
                self.pitch = pitch
                self.note(
                    f"{self.skill} {address}: cannot pitch the hands further to level the carton ({self.unreachable})"
                )
                return False
            yield from self.hold(self.cfg.settle_time, hold)
        return abs(self.nose_down(address)) < 0.04

    def level_held(self, address: str, hold) -> Skill:
        """Level a carton held in the open at :attr:`held`."""
        item = self.env.manifest.item(address)
        return (yield from self.level(address, lambda: self.open_steps(item, [self.held], -self.squeeze), hold))

    def grip_force(self, item: Item) -> float:
        """Clamping force per hand that holds this item with the configured margin [N]."""
        cfg = self.cfg
        return max(cfg.min_grip_force, cfg.grip_safety * item.mass * 9.81 / (2.0 * cfg.grip_friction))

    def squeeze_depth(self, item: Item) -> float:
        """Deepest commanded closing the planner must find reachable for this item.

        Sized for an arm at the soft end of the range seen in clamp postures (1500 N/m), plus 1.5 cm
        for a carton narrower at grip height than its bounding box.
        """
        needed = self.grip_force(item) / 1500.0 + 0.015
        return next((depth for depth in self.cfg.squeeze_steps if depth >= needed), self.cfg.squeeze_steps[-1])

    def squeeze_until_tight(self, item: Item, box: Pose, torso: np.ndarray, hold) -> Skill:
        """Close the palms on a carton in small steps until the arms press it firmly enough for its weight.

        The palms are position-controlled, so the distance by which they stand wider apart than
        commanded, times the arms' lateral stiffness in this posture, is the clamping force. Closing
        stops as soon as that force is reached: pressing harder would twist the wrists flat and bring
        the forearms against the carton, which moves the bearing line off its centre of mass. The
        steps also adapt to cartons narrower at grip height than their bounding box, such as tapered
        crates. Sets :attr:`squeeze` on success.
        """
        cfg = self.cfg
        needed = self.grip_force(item)
        force = 0.0
        for depth in cfg.squeeze_steps:
            targets = self.clamp_targets(box, item, -depth)
            if not (yield from self.follow([Step(targets, torso)], 0.3, hold)):
                break
            yield from self.hold(0.25, hold)
            left, right = self.env.palm_poses()
            commanded = float(np.linalg.norm(targets[0].position - targets[1].position))
            give = float(np.linalg.norm(left.position - right.position)) - commanded
            force = 0.5 * give * self.ik.lateral_stiffness(self.q, CONTACT_LEAD)
            self.squeeze = depth
            if force >= needed:
                return True
        # Out of reach or out of steps: a grip most of the way there is still worth a careful lift.
        return force >= 0.7 * needed

    def is_held(self, address: str) -> bool:
        """True while the item is off its supports, level and between the hands."""
        pose = self.box_in_chassis(address)
        return bool(
            0.1 < pose.position[0] < 1.0 and abs(pose.position[1]) < 0.25 and self.env.object_pose(address).tilt < 0.5
        )

    # -- carry and stack ---------------------------------------------------------------------------------

    def clamp_place(self, address: str, placement: Placement) -> Skill:
        """Carry a clamped carton to its column, set it down square and flush, release and back away."""
        item = self.env.manifest.item(address)
        zone = self.env.manifest.destination(address)
        normal = np.array(zone.normal, dtype=float)
        cfg = self.cfg
        self.skill, self.target = "place", address
        goal = placement.pose
        # The chassis cannot pass the pallet edge, so the carton is held out beyond it.
        reach = self.place_reach(item, placement)
        dock_xy = goal.position[:2] + normal * reach + self._dock_noise()
        dock_yaw = zone.robot_yaw + self._yaw_noise()
        approach_xy = dock_xy + normal * cfg.standoff
        self.note(f"place {address}: column {placement.column}, tier {placement.tier + 1}, reach {reach:.2f} m")

        if not (yield from self.drive_to(approach_xy, dock_yaw, loaded=True)):
            return False
        if not self.is_held(address):
            self.note(f"place {address}: carton lost in transit")
            return False
        # Hold the carton out first and raise it second, still in the open aisle. Raising it close to
        # the body would bring a tall carton up into the robot's own head.
        above = float(goal.position[2]) + cfg.place_clearance
        out = self.carried_box_pose(item, None, reach)
        extended = self.carried_box_pose(item, above, reach)
        standoff = PoseController(approach_xy, dock_yaw, hold=True)
        if not (yield from self.follow(self.open_steps(item, self.line(self.held, out), -self.squeeze), 2.0, standoff)):
            self.note(f"place {address}: column out of reach ({self.unreachable})")
            return False
        self.held = out
        yield from self.level_held(address, standoff)
        if not (yield from self.follow(self.open_steps(item, self.line(out, extended), -self.squeeze), 3.0, standoff)):
            self.note(f"place {address}: stack height out of reach ({self.unreachable})")
            return False
        self.held = extended
        yield from self.hold(cfg.settle_time, standoff)
        yield from self.level_held(address, standoff)
        # The loaded arms sag and give a little. Correct the command by what the carton actually
        # shows, so it is truly at the planned reach and height before the chassis moves in.
        for _ in range(3):
            live = self.box_in_chassis(address)
            error = extended.position - live.position
            error[1] = 0.0
            if float(np.linalg.norm(error)) < 0.006:
                break
            target = Pose.from_yaw(self.held.position + np.clip(error, -0.04, 0.04), math.pi)
            if not (yield from self.follow(self.open_steps(item, [target], -self.squeeze), 0.8, standoff)):
                break
            self.held = target
            yield from self.hold(0.3, standoff)
        yield from self.level_held(address, standoff)
        yield from self.dock(approach_xy, dock_yaw, tolerance=0.01, yaw_tolerance=0.01, speed=0.2)

        # Servo the chassis until the carton itself, not the plan, is over its resting pose.
        hold = None
        offset, twist = np.inf, np.inf
        for _ in range(5):
            position, yaw = self.env.base_pose()
            live = self.env.object_pose(address)
            yaw_fix = wrap_angle(goal.yaw - live.yaw)
            pivot = live.position[:2] - position[:2]
            c, s = math.cos(yaw_fix), math.sin(yaw_fix)
            turned = np.array([c * pivot[0] - s * pivot[1], s * pivot[0] + c * pivot[1]])
            target_xy = goal.position[:2] - turned
            target_yaw = wrap_angle(yaw + yaw_fix)
            if float((target_xy - np.array(zone.edge)) @ normal) < CHASSIS_RADIUS + 0.005:
                self.note(f"place {address}: carton sits too close to the chassis to reach its column")
                return False
            hold = PoseController(target_xy, target_yaw, hold=True)
            yield from self.dock(target_xy, target_yaw, tolerance=0.01, yaw_tolerance=0.012, speed=0.15)
            yield from self.hold(0.3, hold)
            live = self.env.object_pose(address)
            offset = float(np.linalg.norm(live.position[:2] - goal.position[:2]))
            twist = abs(wrap_angle(goal.yaw - live.yaw))
            if offset < 0.015 and twist < 0.03:
                break
        if offset > 0.035 or twist > 0.08:
            self.note(f"place {address}: could not line the carton up with its column ({offset * 100:.1f} cm off)")
            return False

        # Lower by the measured height error, let go, and back the chassis straight out.
        drop = max(float(self.env.object_pose(address).position[2] - goal.position[2]) - 0.004, 0.0)
        lowered = Pose.from_yaw(self.held.position - np.array([0.0, 0.0, drop]), math.pi)
        yield from self.follow(self.open_steps(item, self.line(self.held, lowered, 0.03), -self.squeeze), 1.8, hold)
        yield from self.hold(cfg.settle_time, hold)
        yield from self.release(address, hold)
        yield from self.retreat(approach_xy, dock_yaw)
        return True

    def release(self, address: str, hold) -> Skill:
        """Take the squeeze off a set-down carton and open the hands as far as the arms allow.

        Commanding the arms to where they already are removes the clamping force even at the edge of
        their range, where no wider opening is reachable. A carton that is merely touched does not
        follow the hands when the chassis backs away.
        """
        item = self.env.manifest.item(address)
        measured = self.env.ik_joint_positions()
        self.q = np.concatenate([self.q[:4], measured[4:]])
        yield from self.hold(0.5, hold)
        resting = self.box_in_chassis(address)
        resting = Pose.from_yaw(resting.position, resting.yaw)
        torso = self.q[:4].copy()
        for gap in self.cfg.release_gaps:
            if (yield from self.follow([Step(self.clamp_targets(resting, item, gap), torso)], 1.0, hold)):
                break
            if (yield from self.follow(self.open_steps(item, [resting], gap), 1.0, hold)):
                break
        yield from self.hold(self.cfg.settle_time, hold)
        return True

    # -- pinch pick and bin place (the bottle) -----------------------------------------------------------

    def pinch_target(self, point: np.ndarray, pitch: float = 0.0):
        """Target pair that brings the right gripper's pads, closing sideways, onto a chassis-frame point."""
        fingers = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        normal = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
        right = HandTarget(np.asarray(point) - fingers * PINCH_REACH, normal=normal, fingers=fingers, free_pitch=False)
        return None, right

    def pinch_pick(self, address: str) -> Skill:
        """Take a small upright item off the shelf between the right gripper's pads."""
        item = self.env.manifest.item(address)
        rack = self.env.manifest.rack
        normal = np.array(rack.normal, dtype=float)
        cfg = self.cfg
        for attempt in range(cfg.pick_attempts):
            self.skill, self.target = "pick", address
            box_world = self.env.object_pose(address)
            grasp_height = float(box_world.position[2]) - 0.25 * item.size[2]
            front = box_world.position[:2] + normal * item.size[0] / 2
            dock_distance = cfg.pinch_dock
            torso = self.shelf_posture(grasp_height, dock_distance - self.env.manifest.front_inset, 0.25)
            # Stand to the left of the bottle so it lines up with the right shoulder.
            lateral = np.array([normal[1], -normal[0]]) * 0.18
            dock_xy = front + normal * dock_distance + lateral + self._dock_noise()
            dock_yaw = rack.robot_yaw + self._yaw_noise()
            approach_xy = dock_xy + normal * cfg.standoff
            self.note(f"pick {address}: attempt {attempt + 1}, pinch at {grasp_height:.2f} m")
            if not (yield from self.drive_to(approach_xy, dock_yaw, loaded=False)):
                continue
            standoff = PoseController(approach_xy, dock_yaw, hold=True)
            self.grip[1] = GRIPPER_OPEN
            local = self.env.base_frame().inverse().apply(box_world.position)
            point = np.array([dock_distance + item.size[0] / 2, local[1], grasp_height - BASE_HEIGHT])
            staged = point - np.array([0.10, 0.0, 0.0])
            if not (
                yield from self.follow([Step(self.pinch_target(staged), torso)], cfg.reach_time, standoff, fresh=True)
            ):
                self.note(f"pick {address}: bottle out of reach on this shelf")
                return False
            yield from self.dock(approach_xy, dock_yaw, tolerance=0.01, yaw_tolerance=0.01, speed=0.2)
            if not (yield from self.dock(dock_xy, dock_yaw, speed=0.2)):
                yield from self.retreat(approach_xy, dock_yaw)
                continue
            hold = PoseController(dock_xy, dock_yaw, hold=True)
            # Line the open pads up with the bottle where it actually stands before sliding them
            # around it: they clear it by little more than a centimetre a side.
            point = self.box_in_chassis(address).position.copy()
            point[2] = grasp_height - BASE_HEIGHT
            aligned = point - np.array([0.10, 0.0, 0.0])
            if not (yield from self.follow([Step(self.pinch_target(aligned), torso)], 1.0, hold)):
                self.note(f"pick {address}: bottle out of reach from this dock")
                yield from self.retreat(approach_xy, dock_yaw)
                continue
            yield from self.hold(0.4, hold)
            right = self.env.base_frame().inverse().apply(self.env.palm_poses()[1].position)
            point[1] += float(np.clip(aligned[1] - right[1], -0.02, 0.02))
            if not (
                yield from self.follow(
                    [
                        Step(self.pinch_target(point - np.array([0.10, 0.0, 0.0])), torso),
                        Step(self.pinch_target(point), torso),
                    ],
                    2.0,
                    hold,
                )
            ):
                self.note(f"pick {address}: bottle out of reach from this dock")
                yield from self.retreat(approach_xy, dock_yaw)
                continue
            self.grip[1] = GRIPPER_CLOSED
            yield from self.hold(1.2, hold)
            start_height = float(self.env.object_pose(address).position[2])
            raised = point + np.array([0.0, 0.0, 0.04])
            yield from self.follow([Step(self.pinch_target(raised), torso)], 1.0, hold)
            yield from self.hold(cfg.settle_time, hold)
            if float(self.env.object_pose(address).position[2]) - start_height < 0.015:
                self.note(f"pick {address}: bottle did not come with the hand")
                self.grip[1] = GRIPPER_OPEN
                yield from self.hold(0.6, hold)
                yield from self.retreat(approach_xy, dock_yaw)
                continue
            if not (yield from self.dock(approach_xy, dock_yaw, tolerance=0.04, yaw_tolerance=0.04, speed=0.15)):
                return False
            standoff = PoseController(approach_xy, dock_yaw, hold=True)
            carry = np.array([0.42, -0.2, 1.1 - BASE_HEIGHT])
            waypoints = [raised + (carry - raised) * k / 6 for k in range(1, 7)]
            steps = [Step(self.pinch_target(p), height=float(p[2]) + BASE_HEIGHT) for p in waypoints]
            if not (yield from self.follow(steps, 4.0, standoff)):
                self.note(f"pick {address}: could not bring the bottle down to carry height")
                return False
            self.note(f"pick {address}: bottle secured")
            return True
        return False

    def bin_place(self, address: str, placement: Placement) -> Skill:
        """Lower a pinched bottle into the open tote at its drop spot and let go."""
        item = self.env.manifest.item(address)
        zone = self.env.manifest.destination(address)
        normal = np.array(zone.normal, dtype=float)
        cfg = self.cfg
        self.skill, self.target = "place", address
        goal = placement.pose
        tote = self.env.manifest.objects[placement.support]
        rim = float(self.env.object_pose(placement.support).position[2]) + tote.size[2] / 2
        dock_distance = float((np.array(zone.edge) - goal.position[:2]) @ normal) + cfg.pallet_clearance
        # Stand to the left of the tote so its centre lines up with the right hand.
        lateral = np.array([normal[1], -normal[0]]) * 0.2
        dock_xy = goal.position[:2] + normal * dock_distance + lateral + self._dock_noise()
        dock_yaw = zone.robot_yaw + self._yaw_noise()
        approach_xy = dock_xy + normal * cfg.standoff
        self.note(f"place {address}: into tote {placement.support}")
        if not (yield from self.drive_to(approach_xy, dock_yaw, loaded=True)):
            return False
        if self.env.object_pose(address).position[2] < 0.6:
            self.note(f"place {address}: bottle lost in transit")
            return False
        if not (yield from self.dock(dock_xy, dock_yaw)):
            return False
        hold = PoseController(dock_xy, dock_yaw, hold=True)
        # The pads hold the bottle a quarter of its height below centre.
        grip_offset = 0.25 * item.size[2]
        goal_local = self.env.base_frame().inverse().apply(goal.position)
        current = np.array([0.42, -0.2, 1.1 - BASE_HEIGHT])
        over = np.array([goal_local[0], goal_local[1], rim + 0.05 + item.size[2] / 2 - grip_offset - BASE_HEIGHT])
        inside = over.copy()
        inside[2] = float(goal.position[2]) + 0.03 - grip_offset - BASE_HEIGHT
        path = [current + (over - current) * k / 6 for k in range(1, 7)]
        path += [over + (inside - over) * k / 4 for k in range(1, 5)]
        steps = [Step(self.pinch_target(p), height=float(p[2]) + BASE_HEIGHT + 0.1) for p in path]
        if not (yield from self.follow(steps, 6.0, hold)):
            self.note(f"place {address}: tote out of reach")
            return False
        self.grip[1] = GRIPPER_OPEN
        yield from self.hold(1.0, hold)
        yield from self.follow(list(reversed(steps[5:-1])), 2.0, hold)
        self.grip[1] = GRIPPER_CLOSED
        yield from self.retreat(approach_xy, dock_yaw)
        return True

    # -- demonstration noise -----------------------------------------------------------------------------

    def _dock_noise(self) -> np.ndarray:
        if self.cfg.dock_noise <= 0:
            return np.zeros(2)
        return self.rng.uniform(-self.cfg.dock_noise, self.cfg.dock_noise, size=2)

    def _yaw_noise(self) -> float:
        if self.cfg.dock_yaw_noise <= 0:
            return 0.0
        return float(self.rng.uniform(-self.cfg.dock_yaw_noise, self.cfg.dock_yaw_noise))

    # -- executive ---------------------------------------------------------------------------------------

    def apply_plan(self, plan: dict) -> None:
        """Adopt a grip and torso schedule from :meth:`plan_grasp` for the item about to be picked."""
        self.forward, self.pitch, self.dock_distance = plan["forward"], plan["pitch"], plan["dock"]
        self.rack_shoulder = plan["shoulder"]

    def wait_until_placed(self, address: str, timeout: float = 3.0) -> Skill:
        for _ in range(round(timeout / self.cfg.control_dt)):
            if self.env.status[address] in ("placed", "dropped"):
                break
            yield self.action()
        return self.env.status[address] == "placed"

    def sort_item(self, address: str) -> Skill:
        """Pick one item from the rack and put it away. Returns True once the environment confirms it."""
        item = self.env.manifest.item(address)
        pinch = item.kind == "bottle"
        try:
            placement = self.env.placement(address)
        except NoPlacementError as error:
            self.note(f"{address}: no legal placement ({error})")
            return False
        if not pinch:
            plan = self.plan_grasp(item, placement)
            if plan is None:
                self.note(f"{address}: no grip reaches both its shelf and its column")
                return False
            self.apply_plan(plan)
            self.note(
                f"{address}: grip {self.forward * 100:+.0f} cm from centre, fingers pitched "
                f"{math.degrees(self.pitch):+.0f} deg, dock {self.dock_distance:.2f} m, shoulders "
                f"{self.rack_shoulder:.2f} m above the palms"
            )
        for attempt in range(self.cfg.place_attempts):
            if self.env.status[address] == "dropped":
                self.note(f"{address}: dropped, cannot recover from the floor")
                return False
            picked = yield from (self.pinch_pick(address) if pinch else self.clamp_pick(address))
            if not picked:
                self.note(f"{address}: pick failed")
                return False
            placed = yield from (self.bin_place(address, placement) if pinch else self.clamp_place(address, placement))
            if placed and (yield from self.wait_until_placed(address)):
                self.note(f"{address}: placed and verified")
                return True
            self.note(f"{address}: placement attempt {attempt + 1} not verified")
            if self.env.status[address] != "rack":
                return False
        return False

    def run(self) -> Skill:
        """Sort every item in strict alphabetical order. Stops at the first item that cannot be completed."""
        self.q = self.env.ik_joint_positions().copy()
        yield from self.move_joints(self.travel_posture(), 2.0)
        for address in self.env.manifest.order:
            if self.env.status[address] == "placed":
                continue
            success = yield from self.sort_item(address)
            self.log.item_results[address] = "placed" if success else "failed"
            if not success:
                self.note(f"mission stopped at {address}: a blocked item is never skipped")
                return False
        self.skill, self.target = "done", ""
        self.note("mission complete: all six items placed in order")
        return True


def clamp_pregrasp(size, height: float, dock: float, shoulder_above: float = 0.25, config: TeacherConfig | None = None):
    """Torso-and-arm posture with the hands open either side of a shelved carton, and its squeezed twin.

    A simulator-free helper for skill environments that start a learner at the teacher's pre-grasp.

    Args:
        size: Carton depth, width and height [m].
        height: Height of the carton's centre above the floor [m].
        dock: Distance from the chassis centre to the carton's front face [m].
        shoulder_above: Height of the shoulders above the palms [m].

    Returns:
        ``(open, closed)`` joint vectors in :data:`kinematics.IK_JOINTS` order; ``closed`` is ``None``
        if the squeezed pose is out of reach.
    """
    from types import SimpleNamespace

    teacher = Teacher.__new__(Teacher)
    teacher.cfg = config or TeacherConfig()
    teacher.ik = UpperBodyIK()
    teacher.forward = teacher.pitch = 0.0
    teacher.unreachable = ""
    item = SimpleNamespace(size=tuple(size))
    torso = teacher.shelf_posture(height, dock - 0.03, shoulder_above)
    box = Pose.from_yaw([dock + size[0] / 2, 0.0, height - BASE_HEIGHT], math.pi)
    opened = teacher.chain([Step(teacher.clamp_targets(box, item, teacher.cfg.clamp_open), torso)])
    if opened is None:
        raise ValueError("The carton is out of reach from this dock")
    closed = teacher.solve(Step(teacher.clamp_targets(box, item, -0.03), torso), opened, continuous=True)
    return opened, closed
