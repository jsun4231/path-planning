#!/usr/bin/env python3
"""
Plan a 2D drone path through the arena defined in "STEP MAP.STEP".

Map interpretation (read straight from the STEP B-rep, units = metres):
  * The model is a SolidWorks part extruded along +Y (Y = up).
  * The floor face (horizontal plane carrying the two marker discs) has an outer
    loop that is the boundary of free space; everything outside it is wall
    (1.65 m tall -> cannot be flown over).
  * The two circular inner loops of the floor face are the markers.
    Smaller disc = start, larger disc = goal.

Output frame (top-down, matches the SolidWorks "Top" view):
  x = CAD X,  y = -CAD Z,  origin at the inside corner of the arena, metres.

Method:
  1. Exact clearance (distance-to-wall) field from the boundary polygon.
  2. Drone footprint 0.50 m x 0.40 m -> circumscribed circle R = 0.320 m, so the
     path is safe for ANY yaw. margin = clearance - R (gap from drone to wall).
  3. A* on a 2 cm grid (16-connected) minimising
         cost = integral of [1 + W * ((T - margin)/T)^2]_(margin<T) ds
     i.e. path length, plus a penalty that grows as the margin drops below T.
  4. Elastic-band refinement of the grid path with exact distance gradients
     plus a small curvature term (smooth, fast-to-fly curve).
  5. Douglas-Peucker simplification into straight-line waypoints, re-verified.
"""
import csv
import heapq
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP_FILE = HERE / "STEP MAP.STEP"

# ---------------------------------------------------------------- parameters
DRONE_WIDTH = 0.500   # m (lateral)
DRONE_LENGTH = 0.400  # m (fore-aft)
DRONE_RADIUS = 0.5 * math.hypot(DRONE_WIDTH, DRONE_LENGTH)  # 0.320 m

MARGIN_TARGET = 0.40  # m: margins above this carry no penalty
MARGIN_WEIGHT = 3.0   # how strongly to trade length for margin below the target

GRID = 0.02             # m, A* grid resolution
BAND_SPACING = 0.05     # m, elastic-band point spacing
SMOOTH_WEIGHT = 0.02    # m, weight on integral(curvature^2) in the elastic band
WAYPOINT_TOL = 0.02     # m, max deviation of straight-line waypoints from the smooth path
DENSE_SPACING = 0.10    # m, spacing of the dense output path


# ---------------------------------------------------------------- STEP parsing
def _parse_list(s, i):
    """Parse a parenthesised STEP argument list starting at s[i] == '('."""
    assert s[i] == "("
    out, i = [], i + 1
    while True:
        while s[i].isspace():
            i += 1
        c = s[i]
        if c == ")":
            return out, i + 1
        if c == ",":
            i += 1
        elif c == "(":
            sub, i = _parse_list(s, i)
            out.append(sub)
        elif c == "'":
            j = i + 1
            while True:
                j = s.index("'", j)
                if j + 1 < len(s) and s[j + 1] == "'":
                    j += 2
                    continue
                break
            out.append(s[i + 1:j])
            i = j + 1
        elif c == "#":
            m = re.match(r"#(\d+)", s[i:])
            out.append(int(m.group(1)))
            i += m.end()
        elif c == ".":
            j = s.index(".", i + 1)
            tok = s[i + 1:j]
            out.append({"T": True, "F": False}.get(tok, tok))
            i = j + 1
        elif c in "*$":
            out.append(None)
            i += 1
        elif c.isalpha():  # typed parameter, e.g. LENGTH_MEASURE(1.0)
            m = re.match(r"[A-Z_0-9]+\s*", s[i:])
            i += m.end()
            sub, i = _parse_list(s, i)
            out.append(sub[0] if len(sub) == 1 else sub)
        else:
            m = re.match(r"[-+0-9.Ee]+", s[i:])
            out.append(float(m.group(0)))
            i += m.end()


def read_step(path):
    text = Path(path).read_text(errors="replace")
    data = text.split("DATA;", 1)[1].split("ENDSEC;", 1)[0]
    ents = {}
    for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?:\n|$)", data, re.S):
        body = m.group(2).strip()
        if body.startswith("("):  # complex entity (units etc.) - not needed
            continue
        name, rest = body.split("(", 1)
        args, _ = _parse_list("(" + rest, 0)
        ents[int(m.group(1))] = (name.strip(), args)
    return ents


class StepGeom:
    def __init__(self, ents):
        self.e = ents

    def point(self, ref):
        return np.array(self.e[ref][1][1], float)

    def vertex(self, ref):
        return self.point(self.e[ref][1][1])

    def direction(self, ref, default):
        if ref is None:
            return np.array(default, float)
        d = np.array(self.e[ref][1][1], float)
        return d / np.linalg.norm(d)

    def placement(self, ref):
        a = self.e[ref][1]
        loc = self.point(a[1])
        z = self.direction(a[2], (0, 0, 1))
        x = self.direction(a[3], (1, 0, 0))
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        return loc, z, x, np.cross(z, x)

    def loop_points(self, loop_ref, arc_step=0.01):
        """Discretise an EDGE_LOOP into an ordered list of 3D points."""
        pts = []
        for oe in self.e[loop_ref][1][1]:
            _, oa = self.e[oe]                  # ORIENTED_EDGE(name,*,*,edge,orient)
            edge, forward = oa[3], oa[4]
            _, ea = self.e[edge]                # EDGE_CURVE(name,v1,v2,curve,same_sense)
            p1, p2 = self.vertex(ea[1]), self.vertex(ea[2])
            cname, ca = self.e[ea[3]]
            if cname == "LINE":
                seg = [p1, p2]
            elif cname == "CIRCLE":
                c, _, xd, yd = self.placement(ca[1])
                r = float(ca[2])
                t1 = math.atan2((p1 - c) @ yd, (p1 - c) @ xd)
                t2 = math.atan2((p2 - c) @ yd, (p2 - c) @ xd)
                if ea[4]:
                    dt = (t2 - t1) % (2 * math.pi) or 2 * math.pi
                else:
                    dt = -((t1 - t2) % (2 * math.pi) or 2 * math.pi)
                n = max(2, int(math.ceil(abs(dt) * r / arc_step)) + 1)
                ts = t1 + dt * np.linspace(0, 1, n)
                seg = [c + r * (math.cos(t) * xd + math.sin(t) * yd) for t in ts]
            else:
                raise NotImplementedError(f"edge curve type {cname}")
            if not forward:
                seg = seg[::-1]
            pts.extend(seg[:-1])
        return np.array(pts)


def to2d(p3):
    """CAD (X, Y_up, Z) -> top view (x, y) = (X, -Z)."""
    p3 = np.atleast_2d(p3)
    return np.column_stack([p3[:, 0], -p3[:, 2]])


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))


def load_map(path):
    g = StepGeom(read_step(path))
    up = np.array([0.0, 1.0, 0.0])
    best = None
    for fid, (name, a) in g.e.items():
        if name != "ADVANCED_FACE":
            continue
        sname, sa = g.e[a[2]]
        if sname != "PLANE":
            continue
        loc, n, _, _ = g.placement(sa[1])
        if abs(abs(n @ up) - 1) > 1e-6:
            continue
        key = (len(a[1]), -loc @ up)          # most loops, then lowest
        if best is None or key > best[0]:
            best = (key, fid, a[1])
    _, floor_id, bounds = best
    loops = [to2d(g.loop_points(g.e[b][1][1])) for b in bounds]
    loops.sort(key=lambda p: -abs(polygon_area(p)))
    boundary, marker_loops = loops[0], loops[1:]
    markers = []
    for p in marker_loops:
        c = p.mean(axis=0)
        markers.append((c, float(np.mean(np.linalg.norm(p - c, axis=1)))))
    markers.sort(key=lambda m: m[1])
    return boundary, markers, floor_id


# ---------------------------------------------------------------- clearance
class Walls:
    def __init__(self, poly):
        self.poly = poly
        self.a = poly
        self.b = np.roll(poly, -1, axis=0)
        self.d = self.b - self.a
        self.dd = np.einsum("ij,ij->i", self.d, self.d)

    def nearest(self, P, chunk=4000):
        """Unsigned distance to the wall boundary and the nearest boundary point."""
        P = np.atleast_2d(P)
        dist = np.empty(len(P))
        near = np.empty_like(P)
        for s in range(0, len(P), chunk):
            Q = P[s:s + chunk]
            ap = Q[:, None, :] - self.a[None]
            t = np.clip(np.einsum("mij,ij->mi", ap, self.d) / self.dd, 0, 1)
            q = self.a[None] + t[..., None] * self.d[None]
            d2 = np.sum((Q[:, None, :] - q) ** 2, axis=2)
            k = np.argmin(d2, axis=1)
            dist[s:s + chunk] = np.sqrt(d2[np.arange(len(Q)), k])
            near[s:s + chunk] = q[np.arange(len(Q)), k]
        return dist, near

    def inside(self, P):
        P = np.atleast_2d(P)
        x, y = P[:, 0:1], P[:, 1:2]
        ax, ay, bx, by = self.a[:, 0], self.a[:, 1], self.b[:, 0], self.b[:, 1]
        crosses = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = ax + (y - ay) * (bx - ax) / (by - ay)
        return np.sum(crosses & (x < xint), axis=1) % 2 == 1

    def clearance(self, P):
        d, _ = self.nearest(P)
        return np.where(self.inside(P), d, -d)


def margin_cost(m, weight=MARGIN_WEIGHT, target=MARGIN_TARGET):
    """Cost per metre as a function of margin m, plus its derivative."""
    m = np.asarray(m, float)
    c = np.ones_like(m)
    dc = np.zeros_like(m)
    soft = (m >= 0) & (m < target)
    u = (target - m[soft]) / target
    c[soft] = 1 + weight * u ** 2
    dc[soft] = -2 * weight * u / target
    hard = m < 0                               # barrier: never let the drone touch
    c[hard] = 1 + weight + 1e5 * m[hard] ** 2
    dc[hard] = 2e5 * m[hard]
    return c, dc


# ---------------------------------------------------------------- A* on grid
def grid_astar(walls, start, goal, weight):
    lo = walls.poly.min(axis=0)
    hi = walls.poly.max(axis=0)
    xs = np.arange(lo[0], hi[0] + GRID / 2, GRID)
    ys = np.arange(lo[1], hi[1] + GRID / 2, GRID)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    m = (walls.clearance(P) - DRONE_RADIUS).reshape(X.shape)
    c, _ = margin_cost(m, weight=weight)
    c[m < 0] = np.inf
    ny, nx = c.shape
    cost = c.ravel().tolist()

    def idx(p):
        return int(round((p[1] - ys[0]) / GRID)) * nx + int(round((p[0] - xs[0]) / GRID))

    s, t = idx(start), idx(goal)
    tx, ty = t % nx, t // nx
    moves = []
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1)]:
        for sx in (1, -1):
            for sy in (1, -1):
                mv = (dx * sx, dy * sy)
                if mv in [mm[0] for mm in moves]:
                    continue
                # intermediate cells a knight move sweeps over
                via = []
                if abs(mv[0]) + abs(mv[1]) == 3:
                    hx, hy = mv[0] / 2, mv[1] / 2
                    via = {(int(math.floor(hx)), int(math.floor(hy))),
                           (int(math.ceil(hx)), int(math.ceil(hy)))}
                    via = [v for v in via if v != (0, 0) and v != mv]
                moves.append((mv, math.hypot(*mv) * GRID, via))

    INF = float("inf")
    gbest = [INF] * (nx * ny)
    parent = [-1] * (nx * ny)
    gbest[s] = 0.0
    heap = [(0.0, s)]
    while heap:
        f, u = heapq.heappop(heap)
        if u == t:
            break
        gu = gbest[u]
        ux, uy = u % nx, u // nx
        if f > gu + math.hypot(ux - tx, uy - ty) * GRID + 1e-9:
            continue
        cu = cost[u]
        for (mx, my), length, via in moves:
            vx, vy = ux + mx, uy + my
            if not (0 <= vx < nx and 0 <= vy < ny):
                continue
            v = vy * nx + vx
            cv = cost[v]
            if cv == INF or any(cost[(uy + b) * nx + ux + a] == INF for a, b in via):
                continue
            ng = gu + length * 0.5 * (cu + cv)
            if ng < gbest[v]:
                gbest[v] = ng
                parent[v] = u
                heapq.heappush(heap, (ng + math.hypot(vx - tx, vy - ty) * GRID, v))
    if parent[t] == -1:
        raise RuntimeError("no feasible path - drone does not fit")
    path, u = [], t
    while u != -1:
        path.append((xs[u % nx], ys[u // nx]))
        u = parent[u]
    path = np.array(path[::-1])
    path[0], path[-1] = start, goal
    return path


# ---------------------------------------------------------------- refinement
def resample(P, h):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / h)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])])


def elastic_band(walls, P, weight, iters=6000, step=1.0, max_move=0.003):
    P = resample(P, BAND_SPACING)
    for it in range(iters):
        seg = P[1:] - P[:-1]
        L = np.linalg.norm(seg, axis=1)
        u = seg / L[:, None]
        mid = 0.5 * (P[1:] + P[:-1])
        d, q = walls.nearest(mid)
        inside = walls.inside(mid)
        gm = (mid - q) / np.maximum(d, 1e-9)[:, None]
        gm[~inside] *= -1
        m = np.where(inside, d, -d) - DRONE_RADIUS
        c, dc = margin_cost(m, weight=weight)
        g = np.zeros_like(P)
        gc = 0.5 * (dc * L)[:, None] * gm
        g[:-1] += gc - c[:, None] * u
        g[1:] += gc + c[:, None] * u
        # bending energy  SMOOTH_WEIGHT * sum |D2|^2 / h^3  ~=  SMOOTH_WEIGHT * integral kappa^2 ds
        h = BAND_SPACING
        D2 = P[:-2] - 2 * P[1:-1] + P[2:]
        k = 2 * SMOOTH_WEIGHT / h ** 3
        g[:-2] += k * D2
        g[1:-1] += -2 * k * D2
        g[2:] += k * D2
        g[0] = g[-1] = 0
        alpha = step / (4 * c.max() / h + 16 * k)
        mv = -alpha * g
        n = np.linalg.norm(mv, axis=1)
        mv *= np.minimum(1, max_move / np.maximum(n, 1e-12))[:, None]
        P = P + mv
        if it % 20 == 19:
            P = resample(P, BAND_SPACING)
    return resample(P, BAND_SPACING)


def path_cost(walls, P, weight, h=0.01):
    P = resample(P, h)
    mid = 0.5 * (P[1:] + P[:-1])
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    m = walls.clearance(mid) - DRONE_RADIUS
    c, _ = margin_cost(m, weight=weight)
    return float(c @ L)


def douglas_peucker(P, tol, walls=None, slack_frac=0.05, slack_min=0.005):
    """Straight-line simplification. With `walls`, a chord is also rejected if it
    gives up more wall clearance than max(slack_min, slack_frac * local margin)
    versus the smooth path, so tight sections keep their centreline."""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    if walls is not None:
        clear = walls.clearance(P)
        slack = np.maximum(slack_min, slack_frac * (clear - DRONE_RADIUS))
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        t = np.clip(((P[i + 1:j] - a) @ ab) / (ab @ ab), 0, 1)
        chord = a + t[:, None] * ab
        bad = np.linalg.norm(P[i + 1:j] - chord, axis=1) / tol
        if walls is not None:
            loss = (clear[i + 1:j] - walls.clearance(chord)) / slack[i + 1:j]
            bad = np.maximum(bad, loss)
        k = int(np.argmax(bad))
        if bad[k] > 1:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return P[keep]


def length(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def rect_margin(walls, P):
    """Gap from the actual 0.50 x 0.40 rectangle to the walls when the drone's
    nose points along the direction of travel (informational)."""
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.column_stack([-T[:, 1], T[:, 0]])
    hl, hw = DRONE_LENGTH / 2, DRONE_WIDTH / 2
    out = np.empty(len(P))
    for i, (p, t, n) in enumerate(zip(P, T, N)):
        corners = np.array([p + hl * t + hw * n, p - hl * t + hw * n,
                            p - hl * t - hw * n, p + hl * t - hw * n])
        # sample rectangle perimeter densely, plus wall vertices against rectangle
        edges = np.concatenate([np.linspace(corners[k], corners[(k + 1) % 4], 26)
                                for k in range(4)])
        d1, _ = walls.nearest(edges)
        loc = walls.poly - p
        lx, ly = np.abs(loc @ t) - hl, np.abs(loc @ n) - hw
        d2 = np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0)
        out[i] = min(d1.min(), d2.min())
    return out


# ---------------------------------------------------------------- plotting
def plot(walls, markers, path, waypoints, shortest, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon, Rectangle

    poly = walls.poly
    lo, hi = poly.min(axis=0) - 0.05, poly.max(axis=0) + 0.05
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Rectangle(lo, *(hi - lo), color="#555"))
    ax.add_patch(Polygon(poly, closed=True, facecolor="white", edgecolor="black", lw=1))

    xs = np.arange(lo[0], hi[0], 0.01)
    ys = np.arange(lo[1], hi[1], 0.01)
    X, Y = np.meshgrid(xs, ys)
    C = walls.clearance(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    ax.contourf(X, Y, C, levels=[0, DRONE_RADIUS], colors=["#f6c9c9"])
    ax.contour(X, Y, C, levels=[DRONE_RADIUS], colors=["#d55"], linewidths=0.8,
               linestyles="--")

    ax.plot(*shortest.T, color="#999", lw=1, ls=":", label="shortest feasible (no margin)")

    m = walls.clearance(path) - DRONE_RADIUS
    segs = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=3)
    lc.set_array(0.5 * (m[:-1] + m[1:]))
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("margin: drone envelope to wall (m)")

    # drone footprint every ~0.6 m, nose along travel
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    step = max(1, int(round(0.6 / DENSE_SPACING)))
    for i in list(range(0, len(path), step)) + [len(path) - 1]:
        t = T[i]
        n = np.array([-t[1], t[0]])
        p = path[i]
        hl, hw = DRONE_LENGTH / 2, DRONE_WIDTH / 2
        corners = [p + hl * t + hw * n, p - hl * t + hw * n, p - hl * t - hw * n, p + hl * t - hw * n]
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="#1f77b4", lw=0.6,
                             alpha=0.7))
        ax.add_patch(Circle(p, DRONE_RADIUS, fill=False, edgecolor="#1f77b4", lw=0.3,
                            alpha=0.35))

    ax.plot(*waypoints.T, "o", color="black", ms=3.5, zorder=5)
    for k, w in enumerate(waypoints):
        ax.annotate(str(k), w, xytext=(4, 4), textcoords="offset points", fontsize=7)

    (s, rs), (g, rg) = markers[0], markers[-1]
    ax.add_patch(Circle(s, rs, color="green", zorder=6))
    ax.add_patch(Circle(g, rg, color="red", zorder=6))
    ax.annotate("START", s, xytext=(8, -12), textcoords="offset points", color="green",
                weight="bold")
    ax.annotate("GOAL", g, xytext=(8, -14), textcoords="offset points", color="red",
                weight="bold")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)  [= CAD X]")
    ax.set_ylabel("y (m)  [= -CAD Z]")
    ax.set_title("Drone path (pink = unreachable by drone centre)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


# ---------------------------------------------------------------- main
def plan(walls, start, goal, weight):
    coarse = grid_astar(walls, start, goal, weight)
    return elastic_band(walls, coarse, weight)


def main():
    boundary, markers, floor_id = load_map(STEP_FILE)
    (start, r_s), (goal, r_g) = markers[0], markers[-1]
    walls = Walls(boundary)
    print(f"floor face #{floor_id}: boundary with {len(boundary)} vertices")
    print(f"start marker (smaller, r={r_s*1000:.0f} mm): ({start[0]:.3f}, {start[1]:.3f})")
    print(f"goal  marker (larger,  r={r_g*1000:.0f} mm): ({goal[0]:.3f}, {goal[1]:.3f})")
    for name, p in [("start", start), ("goal", goal)]:
        cl = walls.clearance(p)[0]
        print(f"  {name}: clearance {cl:.3f} m -> margin {cl - DRONE_RADIUS:.3f} m")

    smooth = plan(walls, start, goal, MARGIN_WEIGHT)
    shortest = plan(walls, start, goal, 0.0)

    dense = resample(smooth, DENSE_SPACING)
    wps = douglas_peucker(resample(smooth, 0.01), WAYPOINT_TOL, walls)

    def stats(P, label):
        Pf = resample(P, 0.005)
        m = walls.clearance(Pf) - DRONE_RADIUS
        k = int(np.argmin(m))
        print(f"{label:<28} length {length(P):6.2f} m   min margin {m.min()*1000:5.0f} mm "
              f"at ({Pf[k,0]:.2f}, {Pf[k,1]:.2f})")
        return m

    print()
    stats(shortest, "shortest feasible path")
    stats(smooth, "planned path (smooth)")
    stats(wps, "planned path (waypoints)")
    rm = rect_margin(walls, resample(wps, 0.02))
    print(f"rectangle footprint, nose along travel: min margin {rm.min()*1000:.0f} mm")

    wm = walls.clearance(wps) - DRONE_RADIUS
    # worst margin along the straight leg that arrives at each waypoint
    leg = [float("nan")]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.005) + 1)
        seg = a + np.linspace(0, 1, n)[:, None] * (b - a)
        leg.append(float((walls.clearance(seg) - DRONE_RADIUS).min()))
    with open(HERE / "path_waypoints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x_m", "y_m", "margin_m", "leg_min_margin_m"])
        for i, (p, mm, lm) in enumerate(zip(wps, wm, leg)):
            w.writerow([i, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{mm:.3f}",
                        "" if i == 0 else f"{lm:.3f}"])
    dm = walls.clearance(dense) - DRONE_RADIUS
    with open(HERE / "path_dense.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x_m", "y_m", "margin_m"])
        for i, (p, mm) in enumerate(zip(dense, dm)):
            w.writerow([i, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{mm:.3f}"])

    print(f"\n{len(wps)} waypoints (x, y) in metres; margin at point / worst on leg into it:")
    for i, (p, mm, lm) in enumerate(zip(wps, wm, leg)):
        legs = "" if i == 0 else f" / {lm*1000:4.0f} mm"
        print(f"  {i:2d}: ({p[0]:.3f}, {p[1]:.3f})   {mm*1000:4.0f} mm{legs}")

    plot(walls, markers, dense, wps, shortest, HERE / "path.png")
    print("\nwrote path_waypoints.csv, path_dense.csv, path.png")


if __name__ == "__main__":
    sys.exit(main())
