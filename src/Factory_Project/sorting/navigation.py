"""Route planning and chassis control over the warehouse occupancy grid. Simulator-free.

The planner inflates the static map by the robot's footprint radius and searches it with A*. The
followers turn a path or a dock pose into a chassis twist (forward m/s, left m/s, yaw rad/s) from the
measured pose, so they work the same whether the pose comes from the simulator or a localiser.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import wrap_angle

NAV_MAP_PATH = Path(__file__).resolve().parents[1] / "assets/data/environments/nav/occupancy.npz"
PLANNING_RESOLUTION = 0.1
"""Cell size used for the route search [m]."""


class NoRouteError(RuntimeError):
    """The goal cannot be reached at the requested clearance."""


class OccupancyMap:
    """Static obstacles of the warehouse on a regular grid. ``grid[row, column]`` maps to (y, x)."""

    def __init__(self, grid: np.ndarray, origin: np.ndarray, resolution: float):
        self.grid = (np.asarray(grid) > 0).astype(np.uint8)
        self.origin = np.asarray(origin, dtype=float)
        self.resolution = float(resolution)

    @classmethod
    def load(cls, path: Path | str = NAV_MAP_PATH, rectangles=()) -> OccupancyMap:
        """Load the saved map and mark extra axis-aligned (low, high) rectangles as obstacles."""
        data = np.load(path)
        result = cls(data["grid"], data["origin"], float(data["resolution"]))
        for low, high in rectangles:
            result.add_rectangle(low, high)
        return result

    def cell(self, point) -> tuple[int, int]:
        column, row = np.floor((np.asarray(point, dtype=float)[:2] - self.origin) / self.resolution).astype(int)
        return int(row), int(column)

    def point(self, row: int, column: int) -> np.ndarray:
        return self.origin + (np.array([column, row]) + 0.5) * self.resolution

    def add_rectangle(self, low, high) -> None:
        row0, column0 = self.cell(low)
        row1, column1 = self.cell(high)
        self.grid[max(row0, 0) : row1 + 1, max(column0, 0) : column1 + 1] = 1

    def clearance(self) -> np.ndarray:
        """Distance from every cell centre to the nearest obstacle [m]."""
        return cv2.distanceTransform(1 - self.grid, cv2.DIST_L2, 5) * self.resolution

    def resampled(self, resolution: float) -> OccupancyMap:
        """A coarser copy in which a cell is occupied if any finer cell inside it is."""
        factor = int(round(resolution / self.resolution))
        rows, columns = (np.array(self.grid.shape) // factor) * factor
        blocks = self.grid[:rows, :columns].reshape(rows // factor, factor, columns // factor, factor)
        return OccupancyMap(blocks.max(axis=(1, 3)), self.origin, self.resolution * factor)

    def is_free(self, point, radius: float = 0.0) -> bool:
        row, column = self.cell(point)
        reach = int(math.ceil(radius / self.resolution))
        window = self.grid[max(row - reach, 0) : row + reach + 1, max(column - reach, 0) : column + reach + 1]
        return not window.any()


class RoutePlanner:
    """A* over the inflated map, preferring the middle of aisles."""

    def __init__(self, occupancy: OccupancyMap, radius: float, preferred_clearance: float = 1.2):
        self.fine = occupancy
        self.map = occupancy.resampled(PLANNING_RESOLUTION)
        self.radius = radius
        distance = self.map.clearance()
        self.blocked = distance <= radius
        # Extra cost close to obstacles keeps routes centred without forbidding tight passages.
        shortfall = np.clip((preferred_clearance - distance) / max(preferred_clearance - radius, 1e-6), 0.0, 1.0)
        self.cost = 1.0 + 2.0 * shortfall**2
        self.distance = distance

    def _nearest_free(self, cell: tuple[int, int], limit: float = 1.5) -> tuple[int, int]:
        if not self.blocked[cell]:
            return cell
        reach = int(limit / self.map.resolution)
        row, column = cell
        best, best_distance = None, None
        for r in range(max(row - reach, 0), min(row + reach + 1, self.blocked.shape[0])):
            for c in range(max(column - reach, 0), min(column + reach + 1, self.blocked.shape[1])):
                if not self.blocked[r, c]:
                    d = (r - row) ** 2 + (c - column) ** 2
                    if best is None or d < best_distance:
                        best, best_distance = (r, c), d
        if best is None:
            raise NoRouteError(f"No free cell within {limit} m of {self.map.point(*cell)}")
        return best

    def _line_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        steps = max(int(np.linalg.norm(b - a) / (self.map.resolution * 0.5)), 1)
        for t in np.linspace(0.0, 1.0, steps + 1):
            row, column = self.map.cell(a + (b - a) * t)
            if self.blocked[row, column]:
                return False
        return True

    def plan(self, start, goal) -> np.ndarray:
        """Return waypoints [m] from ``start`` to ``goal``, both included.

        Endpoints inside the inflated obstacles (a dock pose beside a pallet, for example) are joined
        to the nearest free cell by a straight segment.

        Raises:
            NoRouteError: If no route exists at this planner's clearance.
        """
        start, goal = np.asarray(start, dtype=float)[:2], np.asarray(goal, dtype=float)[:2]
        first = self._nearest_free(self.map.cell(start))
        last = self._nearest_free(self.map.cell(goal))
        rows, columns = self.blocked.shape
        best = np.full((rows, columns), np.inf)
        parent = np.full((rows, columns, 2), -1, dtype=np.int32)
        best[first] = 0.0
        queue = [(0.0, first)]
        moves = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        while queue:
            _, (row, column) = heapq.heappop(queue)
            if (row, column) == last:
                break
            travelled = best[row, column]
            for dr, dc, length in moves:
                r, c = row + dr, column + dc
                if not (0 <= r < rows and 0 <= c < columns) or self.blocked[r, c]:
                    continue
                if dr and dc and (self.blocked[row + dr, column] or self.blocked[row, column + dc]):
                    continue  # no cutting corners
                candidate = travelled + length * self.cost[r, c]
                if candidate < best[r, c]:
                    best[r, c] = candidate
                    parent[r, c] = (row, column)
                    heuristic = math.hypot(r - last[0], c - last[1])
                    heapq.heappush(queue, (candidate + heuristic, (r, c)))
        if not np.isfinite(best[last]):
            raise NoRouteError(f"No route from {start} to {goal} at {self.radius:.2f} m clearance")
        cells = [last]
        while cells[-1] != first:
            cells.append(tuple(parent[cells[-1]]))
        points = [self.map.point(*cell) for cell in reversed(cells)]
        # String-pull: keep only the corners the straight segments cannot skip.
        route = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            reach = len(points) - 1
            while reach > anchor + 1 and not self._line_free(points[anchor], points[reach]):
                reach -= 1
            route.append(points[reach])
            anchor = reach
        if np.linalg.norm(route[0] - start) > 1e-3:
            route.insert(0, start)
        if np.linalg.norm(route[-1] - goal) > 1e-3:
            route.append(goal)
        return np.array(route)


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


class PathFollower:
    """Pure-pursuit follower for a holonomic chassis that drives nose-first.

    The chassis turns toward the path before it speeds up, so the head camera and a carried carton
    lead the way, and it eases off toward the last waypoint.
    """

    def __init__(
        self,
        path: np.ndarray,
        *,
        speed: float = 0.9,
        acceleration: float = 0.5,
        lookahead: float = 0.7,
        turn_rate: float = 1.0,
        tolerance: float = 0.12,
        final_yaw: float | None = None,
    ):
        self.path = np.asarray(path, dtype=float)
        self.speed, self.acceleration, self.lookahead = speed, acceleration, lookahead
        self.turn_rate, self.tolerance, self.final_yaw = turn_rate, tolerance, final_yaw
        segments = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(segments)])
        self._progress = 0.0
        self._command_speed = 0.0

    def _point_at(self, arc: float) -> np.ndarray:
        arc = float(np.clip(arc, 0.0, self._arc[-1]))
        index = min(int(np.searchsorted(self._arc, arc, side="right")) - 1, len(self.path) - 2)
        span = self._arc[index + 1] - self._arc[index]
        t = 0.0 if span < 1e-9 else (arc - self._arc[index]) / span
        return self.path[index] + (self.path[index + 1] - self.path[index]) * t

    def _project(self, position: np.ndarray) -> float:
        best, best_distance = self._progress, np.inf
        for index in range(len(self.path) - 1):
            a, b = self.path[index], self.path[index + 1]
            span = b - a
            length = float(span @ span)
            t = 0.0 if length < 1e-12 else float(np.clip((position - a) @ span / length, 0.0, 1.0))
            arc = self._arc[index] + t * math.sqrt(length)
            # Never jump backwards or far ahead onto another leg of a route that doubles back.
            if arc < self._progress - 0.05 or arc > self._progress + 2.0:
                continue
            distance = float(np.linalg.norm(position - (a + span * t)))
            if distance < best_distance:
                best, best_distance = arc, distance
        return best

    @property
    def remaining(self) -> float:
        return float(self._arc[-1] - self._progress)

    def command(self, position, yaw: float, dt: float) -> tuple[np.ndarray, bool]:
        """Return the chassis twist for the measured pose and whether the end has been reached."""
        position = np.asarray(position, dtype=float)[:2]
        if len(self.path) < 2:
            return np.zeros(3), True
        self._progress = self._project(position)
        to_end = float(np.linalg.norm(self.path[-1] - position))
        if self.remaining < self.tolerance and to_end < 2 * self.tolerance:
            return np.zeros(3), True
        target = self._point_at(self._progress + self.lookahead)
        offset = target - position
        distance = float(np.linalg.norm(offset))
        direction = offset / max(distance, 1e-6)
        heading = math.atan2(direction[1], direction[0])
        if self.final_yaw is not None and self.remaining < 1.0:
            blend = 1.0 - self.remaining / 1.0
            heading = yaw + wrap_angle(heading - yaw) * (1 - blend) + wrap_angle(self.final_yaw - yaw) * blend
        yaw_error = wrap_angle(heading - yaw)
        # Slow to a crawl while the nose is far off the path, and brake for the end.
        alignment = max(0.0, math.cos(yaw_error)) ** 2 if abs(yaw_error) < math.pi / 2 else 0.0
        limit = min(
            self.speed * max(alignment, 0.08), math.sqrt(2 * self.acceleration * max(self.remaining, 0.0)) + 0.05
        )
        self._command_speed = min(limit, self._command_speed + self.acceleration * dt)
        velocity = direction * self._command_speed
        c, s = math.cos(yaw), math.sin(yaw)
        forward, left = c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]
        turn = float(np.clip(2.5 * yaw_error, -self.turn_rate, self.turn_rate))
        return np.array([forward, left, turn]), False


class PoseController:
    """Proportional controller that parks the chassis on a pose and reports when it has settled.

    With ``hold=True`` it keeps a chassis that is already parked in place while the arms work: a
    gentle proportional correction with a small dead band and no minimum command, so it does not
    hunt around the pose and shake a clamped carton.
    """

    def __init__(
        self,
        target_xy,
        target_yaw: float,
        *,
        speed: float = 0.3,
        turn_rate: float = 0.8,
        position_tolerance: float = 0.015,
        yaw_tolerance: float = 0.02,
        settle_time: float = 0.3,
        gain: float = 2.0,
        min_speed: float = 0.04,
        min_turn_rate: float = 0.10,
        hold: bool = False,
    ):
        self.target_xy = np.asarray(target_xy, dtype=float)[:2]
        self.target_yaw = target_yaw
        self.speed, self.turn_rate, self.gain = speed, turn_rate, gain
        self.position_tolerance, self.yaw_tolerance, self.settle_time = position_tolerance, yaw_tolerance, settle_time
        self.min_speed, self.min_turn_rate = min_speed, min_turn_rate
        self.hold = hold
        self._settled = 0.0

    def error(self, position, yaw: float) -> tuple[float, float]:
        return float(np.linalg.norm(self.target_xy - np.asarray(position)[:2])), abs(wrap_angle(self.target_yaw - yaw))

    def command(self, position, yaw: float, dt: float) -> tuple[np.ndarray, bool]:
        offset = self.target_xy - np.asarray(position, dtype=float)[:2]
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * offset[0] + s * offset[1], -s * offset[0] + c * offset[1]])
        if self.hold:
            yaw_error = wrap_angle(self.target_yaw - yaw)
            velocity = np.zeros(2) if float(np.linalg.norm(offset)) < 0.004 else np.clip(1.5 * local, -0.08, 0.08)
            turn = 0.0 if abs(yaw_error) < 0.004 else float(np.clip(1.5 * yaw_error, -0.2, 0.2))
            return np.array([velocity[0], velocity[1], turn]), True
        velocity = self.gain * local
        norm = float(np.linalg.norm(velocity))
        distance = float(np.linalg.norm(offset))
        if norm > self.speed:
            velocity *= self.speed / norm
        elif distance > 0.5 * self.position_tolerance and norm < self.min_speed:
            # The roller wheels do not move below a small command; keep creeping until inside tolerance.
            velocity *= self.min_speed / max(norm, 1e-9)
        yaw_error = wrap_angle(self.target_yaw - yaw)
        turn = float(np.clip(self.gain * yaw_error, -self.turn_rate, self.turn_rate))
        if abs(yaw_error) > 0.5 * self.yaw_tolerance and abs(turn) < self.min_turn_rate:
            turn = math.copysign(self.min_turn_rate, yaw_error)
        inside = float(np.linalg.norm(offset)) < self.position_tolerance and abs(yaw_error) < self.yaw_tolerance
        self._settled = self._settled + dt if inside else 0.0
        return np.array([velocity[0], velocity[1], turn]), self._settled >= self.settle_time
