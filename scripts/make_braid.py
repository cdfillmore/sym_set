"""Blender edge-only closed braid with a twisted annular / cylindrical embedding.

Inspired by Chambers, Fillmore, Stephenson & Wintraecken, "Braiding Vineyards",
Theorem 1.1, steps 1-4 and Figures 1, 2, 14-16.

Run in Blender's Scripting workspace (Run Script), or:
    blender --background --python blender_twisted_braid.py

Two edge-only mesh objects are created: the closed braid and an observation
loop following the OUTER EDGE OF THE TWISTED ANNULUS (including its cylinder).
The closed components are exactly those of the closure of BRAID_WORD.
Each component is resampled at approximately uniform arc-length intervals,
with the target TOTAL mesh vertex count shared across all components.

Conventions:
    +i = sigma_i: the strand currently in radial lane i (inner lane)
         passes OVER the strand in lane i+1.
    -i = sigma_i^-1: the strand in lane i passes UNDER lane i+1.
    The braid travels with increasing theta around the positive z axis.
    Lanes are numbered 1..STRAND_COUNT, from inner to outer.

The antipodal perturbations here move the OVER strand radially outward
and the UNDER strand inward at theta_crossing + pi. This matches the
qualitative 'push/pull' mechanism in the paper; it is not a complete
implementation of the proof's elder-rule surgery and inequalities (7),(8).
"""

import math
from bisect import bisect_right
from dataclasses import dataclass

try:
    import bpy
except ImportError:
    bpy = None  # Allows geometric validation with ordinary Python.


# =====================================================================
# USER SETTINGS
# =====================================================================

# Always add an additional reidemeister twist to fix elder strand
STRAND_COUNT = 4
BRAID_WORD = [1, -2, 1, -2, 1, -2, 3]
# Other examples:
#   STRAND_COUNT = 2; BRAID_WORD = [1, 1, 1]    # trefoil
#   STRAND_COUNT = 2; BRAID_WORD = [1, 1]       # Hopf link
#   STRAND_COUNT = 4; BRAID_WORD = [1, -2, 3, 1]

CORE_RADIUS = 20
LANE_SPACING = 1.0
CROSSING_HEIGHT = 0.5  # Each front participant rises/falls this far.
BACK_RADIAL_PUSH = 1.0  # Each paired back participant moves +/- this far.

# Four angular regions, with angle zero pointing along +x:
# front:      [-30, +30] degrees (pi/3 wide)
# intermediate A: [+30, +150] degrees
# back:       [+150, +210] degrees (pi/3 wide)
# intermediate B: [+210, +330] degrees (wraps to front at -30)
FRONT_HALF_ANGLE_DEG = 30.0

# In each intermediate region the 90-degree twist is concentrated near
# the front. Outside that short ramp the intermediate strands are
# vertically stacked at EXACTLY CORE_RADIUS.
TWIST_RAMP_FRACTION = 0.28

# Each crossing occupies this fraction of its evenly spaced front slot.
CROSSING_SLOT_FRACTION = 0.60

# Total vertex budget of the FINAL mesh (across all linked components).
# The allocated counts are proportional to component arc lengths, subject to
# MIN_VERTICES_PER_COMPONENT. Exactly this many vertices and edges are created.
TARGET_VERTEX_COUNT = 3000
MIN_VERTICES_PER_COMPONENT = 24

# Arc-length integration uses a separate, dense, temporary sampling grid.
# The multiplier applies to the average final vertices per original strand;
# independent crossing/twist samples below are included in the grid too.
ARC_LENGTH_OVERSAMPLE_FACTOR = 4
SAMPLES_PER_TURN = 1100
SAMPLES_PER_CROSSING = 32  # Per crossing window, both front and back.
SAMPLES_PER_TWIST = 50

OBJECT_NAME = "BraidingVineyards_EdgeOnly"
# Observation loop follows the outer edge of the rotating annular cross-section:
# front: radially outside; middle: rotates with the strands; back: ABOVE
# the vertically stacked strands. It remains exterior to the strand envelope.
CREATE_OBSERVATION_LOOP = True
OBSERVATION_CLEARANCE = 2.0
OBSERVATION_VERTEX_COUNT = 720
OBSERVATION_OBJECT_NAME = "BraidingVineyards_ObservationLoop"
# Soft-maximum sharpness (in inverse spatial units); guarantees that the
# observation curve remains outside all strands in its local outward direction.
# Constant outer-boundary offsets. No per-angle strand-envelope tracking.
# The front is a circle at a fixed radius; the sides/back are at fixed z.
OBSERVATION_ENVELOPE_SHARPNESS = 45.0  # Legacy setting, no longer used.
SAVE_BLEND_PATH = ""  # Optional: e.g. "//braiding_vineyards.blend"


# =====================================================================
# PARAMETRIC CONSTRUCTION
# =====================================================================

PI = math.pi
FRONT_BEGIN = -math.radians(FRONT_HALF_ANGLE_DEG)
FRONT_END = math.radians(FRONT_HALF_ANGLE_DEG)
BACK_BEGIN = FRONT_BEGIN + PI
BACK_END = FRONT_END + PI
FULL_END = FRONT_BEGIN + 2.0 * PI


def clamp01(t):
    return min(1.0, max(0.0, t))


def smooth5(t):
    """C2 transition: value, first and second derivatives match at ends."""
    t = clamp01(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def bump(t):
    """Smooth, compactly supported bump: 0 at ends, 1 at center."""
    if t <= 0.0 or t >= 1.0:
        return 0.0
    return math.sin(PI * t) ** 4


@dataclass(frozen=True)
class FrontEvent:
    a: float
    b: float
    lane_before: int   # Zero-based radial lane
    lane_after: int
    height_sign: int   # +1 means geometrically over (positive z)


@dataclass(frozen=True)
class BackEvent:
    a: float
    b: float
    radial_sign: int   # +1 = outward, -1 = inward


@dataclass
class BraidPlan:
    n: int
    word: list
    permutation: list   # permutation[start_strand] = ending_lane
    front: list         # front[strand_id] -> list[FrontEvent]
    front_ends: list
    back: list          # back[strand_id] -> list[BackEvent]
    back_ends: list
    crossings: list     # (theta_center, inner_id, outer_id, sign, halfwidth)


def check_settings():
    if type(STRAND_COUNT) is not int or STRAND_COUNT < 2:
        raise ValueError("STRAND_COUNT must be an integer >= 2")
    for g in BRAID_WORD:
        if type(g) is not int or g == 0 or abs(g) >= STRAND_COUNT:
            raise ValueError(
                "Each braid generator must be a nonzero integer in "
                "[-(STRAND_COUNT-1), +(STRAND_COUNT-1)]"
            )
    if CORE_RADIUS <= (STRAND_COUNT - 1) * LANE_SPACING * 0.5 + BACK_RADIAL_PUSH:
        raise ValueError("CORE_RADIUS is too small for the radial strand levels")
    if LANE_SPACING <= 0 or CROSSING_HEIGHT <= 0 or BACK_RADIAL_PUSH <= 0:
        raise ValueError("Spacing and both displacements must be positive")
    if not (5.0 < FRONT_HALF_ANGLE_DEG < 85.0):
        raise ValueError("FRONT_HALF_ANGLE_DEG must be between 5 and 85")
    if not (0.0 < TWIST_RAMP_FRACTION <= 1.0):
        raise ValueError("TWIST_RAMP_FRACTION must be in (0,1]")
    if not (0.0 < CROSSING_SLOT_FRACTION < 1.0):
        raise ValueError("CROSSING_SLOT_FRACTION must be in (0,1)")
    if SAMPLES_PER_TURN < 64 or SAMPLES_PER_CROSSING < 8:
        raise ValueError("Increase mesh sampling")
    if type(TARGET_VERTEX_COUNT) is not int or TARGET_VERTEX_COUNT < 3:
        raise ValueError("TARGET_VERTEX_COUNT must be an integer >= 3")
    if type(MIN_VERTICES_PER_COMPONENT) is not int or MIN_VERTICES_PER_COMPONENT < 3:
        raise ValueError("MIN_VERTICES_PER_COMPONENT must be an integer >= 3")
    if CREATE_OBSERVATION_LOOP:
        if OBSERVATION_CLEARANCE <= 0:
            raise ValueError("OBSERVATION_CLEARANCE must be positive")
        if OBSERVATION_ENVELOPE_SHARPNESS <= 0:
            raise ValueError("OBSERVATION_ENVELOPE_SHARPNESS must be positive")
        if type(OBSERVATION_VERTEX_COUNT) is not int or OBSERVATION_VERTEX_COUNT < 3:
            raise ValueError("OBSERVATION_VERTEX_COUNT must be >= 3")
    if type(ARC_LENGTH_OVERSAMPLE_FACTOR) is not int or ARC_LENGTH_OVERSAMPLE_FACTOR < 2:
        raise ValueError("ARC_LENGTH_OVERSAMPLE_FACTOR must be an integer >= 2")


def compile_braid(n, word):
    """Track braid strand identity as crossings exchange radial slots."""
    front = [[] for _ in range(n)]
    back = [[] for _ in range(n)]
    lane_to_strand = list(range(n))
    crossings = []

    angular_slot = (FRONT_END - FRONT_BEGIN) / (len(word) + 1)
    halfwidth = 0.5 * CROSSING_SLOT_FRACTION * angular_slot

    for j, g in enumerate(word):
        lane = abs(g) - 1
        sign = 1 if g > 0 else -1
        inner_id = lane_to_strand[lane]
        outer_id = lane_to_strand[lane + 1]
        center = FRONT_BEGIN + (j + 1) * angular_slot
        a = center - halfwidth
        b = center + halfwidth

        # Radial positions swap; only these two strands leave the xy plane.
        front[inner_id].append(FrontEvent(a, b, lane, lane + 1, sign))
        front[outer_id].append(FrontEvent(a, b, lane + 1, lane, -sign))

        # The antipodal pair acts on THESE SAME physical strands, although
        # their final back z-levels depend on the whole braid permutation.
        back[inner_id].append(BackEvent(a + PI, b + PI, sign))
        back[outer_id].append(BackEvent(a + PI, b + PI, -sign))
        crossings.append((center, inner_id, outer_id, sign, halfwidth))

        lane_to_strand[lane], lane_to_strand[lane + 1] = (
            lane_to_strand[lane + 1], lane_to_strand[lane]
        )

    # The strand starting at slot i ends at slot permutation[i].
    permutation = [None] * n
    for ending_lane, start_id in enumerate(lane_to_strand):
        permutation[start_id] = ending_lane

    return BraidPlan(
        n=n, word=list(word), permutation=permutation, front=front,
        front_ends=[[e.b for e in es] for es in front],
        back=back, back_ends=[[e.b for e in es] for es in back],
        crossings=crossings,
    )


def lane_offset(lane, n):
    return (lane - 0.5 * (n - 1)) * LANE_SPACING


def radius_and_height(plan, strand_id, theta):
    """Return (radius, z), with theta in [FRONT_BEGIN, FRONT_BEGIN+2*pi)."""
    final_level = lane_offset(plan.permutation[strand_id], plan.n)

    # Region 1: the front, planar except for z bumps at crossings.
    if theta <= FRONT_END:
        events = plan.front[strand_id]
        k = bisect_right(plan.front_ends[strand_id], theta)
        lane = strand_id if k == 0 else events[k - 1].lane_after
        z = 0.0
        if k < len(events) and theta >= events[k].a:
            event = events[k]
            t = (theta - event.a) / (event.b - event.a)
            lane = event.lane_before + (
                event.lane_after - event.lane_before
            ) * smooth5(t)
            z = event.height_sign * CROSSING_HEIGHT * bump(t)
        return CORE_RADIUS + lane_offset(lane, plan.n), z

    # Region 2: first intermediate. Rotate radial separation by 90 degrees.
    # Keep almost the entire intermediate at constant radius, stacked in z.
    if theta < BACK_BEGIN:
        ramp_length = (BACK_BEGIN - FRONT_END) * TWIST_RAMP_FRACTION
        t = (theta - FRONT_END) / ramp_length
        if t >= 1.0:
            return CORE_RADIUS, final_level
        phi = 0.5 * PI * smooth5(t)
        return CORE_RADIUS + final_level * math.cos(phi), final_level * math.sin(phi)

    # Region 3: back, vertically stacked. Radial antipodal push/pull.
    if theta <= BACK_END:
        events = plan.back[strand_id]
        k = bisect_right(plan.back_ends[strand_id], theta)
        r = CORE_RADIUS
        if k < len(events) and theta >= events[k].a:
            event = events[k]
            t = (theta - event.a) / (event.b - event.a)
            r += event.radial_sign * BACK_RADIAL_PUSH * bump(t)
        return r, final_level

    # Region 4: second intermediate. Remain cylindrical most of the way,
    # then rotate the z levels back to radial lanes as the loop closes.
    ramp_length = (FULL_END - BACK_END) * TWIST_RAMP_FRACTION
    ramp_begin = FULL_END - ramp_length
    if theta <= ramp_begin:
        return CORE_RADIUS, final_level
    t = (theta - ramp_begin) / ramp_length
    phi = 0.5 * PI * (1.0 - smooth5(t))
    return CORE_RADIUS + final_level * math.cos(phi), final_level * math.sin(phi)


def point_xyz(plan, strand_id, theta):
    r, z = radius_and_height(plan, strand_id, theta)
    return (r * math.cos(theta), r * math.sin(theta), z)


def link_cycles(permutation):
    """Each cycle is one connected component of the closed braid."""
    visited = set()
    cycles = []
    for start in range(len(permutation)):
        if start in visited:
            continue
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = permutation[current]
        if current != start:
            raise RuntimeError("Invalid closure permutation")
        cycles.append(cycle)
    return cycles


def sample_angles(plan, base_samples=None):
    """Fine integration grid: uniform angular samples plus event windows."""
    samples = {}

    def add_window(a, b, count):
        for k in range(count + 1):
            t = a + (b - a) * k / count
            if FRONT_BEGIN <= t < FULL_END - 1.0e-12:
                samples[round(t, 12)] = t

    if base_samples is None:
        base_samples = SAMPLES_PER_TURN
    add_window(FRONT_BEGIN, FULL_END, base_samples)
    for center, _inner, _outer, _sign, halfwidth in plan.crossings:
        add_window(center - halfwidth, center + halfwidth, SAMPLES_PER_CROSSING)
        add_window(center + PI - halfwidth, center + PI + halfwidth,
                   SAMPLES_PER_CROSSING)

    first_ramp_end = FRONT_END + (BACK_BEGIN - FRONT_END) * TWIST_RAMP_FRACTION
    last_ramp_begin = FULL_END - (FULL_END - BACK_END) * TWIST_RAMP_FRACTION
    add_window(FRONT_END, first_ramp_end, SAMPLES_PER_TWIST)
    add_window(last_ramp_begin, FULL_END, SAMPLES_PER_TWIST)
    for boundary in (FRONT_BEGIN, FRONT_END, BACK_BEGIN, BACK_END):
        samples[round(boundary, 12)] = boundary
    return [samples[k] for k in sorted(samples)]


def dense_component_curve(plan, component, angles):
    """Sample a closed link component and accumulate its 3D polygonal lengths.

    The intermediate samples are not Blender vertices. Adjacent turns are
    concatenated using the closure permutation (no duplicated seam points).
    """
    turns = len(component)
    parameters = []
    positions = []
    for turn, strand_id in enumerate(component):
        for theta in angles:
            parameters.append(2.0 * PI * turn + (theta - FRONT_BEGIN))
            positions.append(point_xyz(plan, strand_id, theta))

    # The final point closes the last turn to the first, including the closing
    # seam. Its position must agree with that on the next strand at FRONT_BEGIN.
    parameters.append(2.0 * PI * turns)
    positions.append(positions[0])

    lengths = [0.0]
    for previous, current in zip(positions, positions[1:]):
        ds = math.dist(previous, current)
        if ds <= 0.0:
            raise ValueError("Dense curve contains a zero-length segment")
        lengths.append(lengths[-1] + ds)
    return parameters, lengths


def allocate_vertices_by_length(component_lengths, target_count):
    """Distribute an exact total count by arc length with a per-loop floor."""
    k = len(component_lengths)
    floor = MIN_VERTICES_PER_COMPONENT
    if target_count < floor * k:
        raise ValueError(
            "TARGET_VERTEX_COUNT={} is insufficient for {} link components "
            "with MIN_VERTICES_PER_COMPONENT={}".format(target_count, k, floor)
        )

    counts = [0] * k
    active = list(range(k))
    remaining = target_count

    # Freeze any very short components whose proportional share is too small.
    while active:
        active_length = sum(component_lengths[i] for i in active)
        underfull = [i for i in active if
                     remaining * component_lengths[i] / active_length < floor]
        if not underfull:
            break
        for i in underfull:
            counts[i] = floor
            remaining -= floor
        active = [i for i in active if i not in underfull]

    if active:
        total_length = sum(component_lengths[i] for i in active)
        quotas = {i: remaining * component_lengths[i] / total_length for i in active}
        for i in active:
            counts[i] = int(math.floor(quotas[i]))
        extras = target_count - sum(counts)
        # Largest remainder method: deterministic even on tied fractions.
        order = sorted(active, key=lambda i: (-(quotas[i] - counts[i]), i))
        for i in order[:extras]:
            counts[i] += 1

    assert sum(counts) == target_count
    assert all(n >= floor for n in counts)
    return counts


def resample_component(plan, component, parameters, arc_lengths, count):
    """Invert cumulative arc length and evaluate the EXACT parametric curve.

    Parameter values are found by interpolation on a fine sampled s(theta)
    table. Final vertices are evaluated with point_xyz, not interpolated on
    the temporary polyline, preserving the authored crossing/twist geometry.
    """
    total_length = arc_lengths[-1]
    points = []
    for vertex_index in range(count):
        desired_s = total_length * vertex_index / count
        k = bisect_right(arc_lengths, desired_s) - 1
        k = min(k, len(arc_lengths) - 2)
        ds = arc_lengths[k + 1] - arc_lengths[k]
        alpha = (desired_s - arc_lengths[k]) / ds
        parameter = parameters[k] + alpha * (parameters[k + 1] - parameters[k])

        # Each full 2pi turn corresponds to one physical braid strand.
        turn = min(int(parameter / (2.0 * PI)), len(component) - 1)
        theta = FRONT_BEGIN + (parameter - turn * 2.0 * PI)
        points.append(point_xyz(plan, component[turn], theta))
    return points


def generate_edge_mesh(plan):
    """One uniformly spaced closed edge cycle per link component; exact budget."""
    components = link_cycles(plan.permutation)

    # Sampling is driven by the requested final budget, but the integration
    # remains significantly finer than the final mesh.
    samples_per_turn = max(
        SAMPLES_PER_TURN,
        math.ceil(ARC_LENGTH_OVERSAMPLE_FACTOR * TARGET_VERTEX_COUNT / plan.n),
    )
    angles = sample_angles(plan, base_samples=samples_per_turn)

    dense_data = [dense_component_curve(plan, c, angles) for c in components]
    component_lengths = [arcs[-1] for _params, arcs in dense_data]
    component_counts = allocate_vertices_by_length(component_lengths,
                                                   TARGET_VERTEX_COUNT)

    vertices = []
    edges = []
    vertex_spans = []
    for component, (parameters, arc_lengths), count in zip(
            components, dense_data, component_counts):
        first = len(vertices)
        vertices.extend(resample_component(plan, component, parameters,
                                           arc_lengths, count))
        last = len(vertices)
        edges.extend((i, i + 1) for i in range(first, last - 1))
        edges.append((last - 1, first))
        vertex_spans.append((first, last))

    assert len(vertices) == TARGET_VERTEX_COUNT
    return vertices, edges, components, vertex_spans, component_lengths


def validate_geometry(plan, vertices, edges, components, vertex_spans):
    # Front crossings have the prescribed positive/negative z separation;
    # their antipodal copies have corresponding opposite radial shifts.
    for center, inner, outer, sign, _halfwidth in plan.crossings:
        r1, z1 = radius_and_height(plan, inner, center)
        r2, z2 = radius_and_height(plan, outer, center)
        assert abs(r1 - r2) < 1e-8, "Front crossing radial alignment failed"
        assert abs(z1 - sign * CROSSING_HEIGHT) < 1e-8
        assert abs(z2 + sign * CROSSING_HEIGHT) < 1e-8
        br1, bz1 = radius_and_height(plan, inner, center + PI)
        br2, bz2 = radius_and_height(plan, outer, center + PI)
        assert abs(br1 - (CORE_RADIUS + sign * BACK_RADIAL_PUSH)) < 1e-8
        assert abs(br2 - (CORE_RADIUS - sign * BACK_RADIAL_PUSH)) < 1e-8
        assert abs(bz1 - bz2) >= LANE_SPACING - 1e-8

    assert len(vertices) == len(edges)
    assert len(components) == len(vertex_spans)
    degrees = [0] * len(vertices)
    for a, b in edges:
        assert a != b
        degrees[a] += 1
        degrees[b] += 1
        d2 = sum((vertices[a][k] - vertices[b][k]) ** 2 for k in range(3))
        assert d2 > 1e-18, "Degenerate edge detected"
    assert all(d == 2 for d in degrees), "Mesh is not a disjoint union of circles"
    assert all(math.isfinite(x) for point in vertices for x in point)
    assert all(math.hypot(x, y) > 0.0 for x, y, _ in vertices)


def outward_twist_angle(theta):
    """Cross-section rotation: 0 front, pi/2 on cylindrical back."""
    if theta <= FRONT_END:
        return 0.0
    if theta < BACK_BEGIN:
        ramp = (BACK_BEGIN - FRONT_END) * TWIST_RAMP_FRACTION
        return 0.5 * PI * smooth5((theta - FRONT_END) / ramp)
    if theta <= BACK_END:
        return 0.5 * PI
    ramp = (FULL_END - BACK_END) * TWIST_RAMP_FRACTION
    return 0.5 * PI * (1.0 - smooth5((theta - (FULL_END - ramp)) / ramp))


def observation_point(plan, theta):
    """Piecewise constant-radius / constant-height outer observation loop.

    FRONT: fixed radius and z=0.
    SIDES AND BACK: fixed z and radius CORE_RADIUS.
    The two short transition ramps adjacent to the front interpolate
    smoothly by following the outer edge of the twisting cross-section.
    """
    phi = outward_twist_angle(theta)
    outer_offset = 0.5 * (plan.n - 1) * LANE_SPACING
    distance = outer_offset + OBSERVATION_CLEARANCE
    r = CORE_RADIUS + distance * math.cos(phi)
    z = distance * math.sin(phi)
    return (r * math.cos(theta), r * math.sin(theta), z)


def create_observation_loop(plan):
    """Create the observation curve along the outside of the twisted annulus.

    This is NOT a planar circle: it rotates from the outer radial boundary
    to the top of the cylindrical section, as in Figures 1 and 2 of the paper.
    It is everywhere outside all strand positions in the local annular
    outward direction, with at least OBSERVATION_CLEARANCE of separation.
    """
    n = OBSERVATION_VERTEX_COUNT
    vertices = [observation_point(plan, FRONT_BEGIN + 2 * PI * i / n)
                for i in range(n)]
    edges = [(i, (i + 1) % n) for i in range(n)]
    if bpy is None:
        return None, vertices, edges

    existing = bpy.data.objects.get(OBSERVATION_OBJECT_NAME)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    mesh = bpy.data.meshes.new(OBSERVATION_OBJECT_NAME + "_Mesh")
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(OBSERVATION_OBJECT_NAME, mesh)
    bpy.context.collection.objects.link(obj)
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.color = (0.12, 0.45, 1.0, 1.0)
    obj["role"] = "Observation loop following the twisted annulus exterior"
    obj["outward_clearance"] = OBSERVATION_CLEARANCE
    obj["angle_direction"] = "increasing theta about +z"
    return obj, vertices, edges


def validate_observation(plan, vertices):
    assert len(vertices) == OBSERVATION_VERTEX_COUNT
    for i, observed in enumerate(vertices):
        theta = FRONT_BEGIN + 2 * PI * i / len(vertices)
        phi = outward_twist_angle(theta)
        obs_r = math.hypot(observed[0], observed[1])
        obs_z = observed[2]
        if theta <= FRONT_END:
            assert abs(obs_r - (CORE_RADIUS + 0.5 * (plan.n - 1) *
                                LANE_SPACING + OBSERVATION_CLEARANCE)) < 1e-8
        if FRONT_END + (BACK_BEGIN - FRONT_END)*TWIST_RAMP_FRACTION <= theta <= (FULL_END - (FULL_END - BACK_END)*TWIST_RAMP_FRACTION):
            assert abs(obs_z - (0.5 * (plan.n - 1)*LANE_SPACING + OBSERVATION_CLEARANCE)) < 1e-8
        for strand in range(plan.n):
            r, z = radius_and_height(plan, strand, theta)
            separation = math.hypot(obs_r - r, obs_z - z)
            assert separation >= OBSERVATION_CLEARANCE - 1e-8, (i, strand, separation)


def create_blender_object(vertices, edges, plan, components):
    if bpy is None:
        return None
    existing = bpy.data.objects.get(OBJECT_NAME)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    mesh = bpy.data.meshes.new(OBJECT_NAME + "_Mesh")
    mesh.from_pydata(vertices, edges, [])  # Explicitly NO faces.
    mesh.update()

    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.collection.objects.link(obj)
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj["braid_word"] = str(plan.word)
    obj["strand_count"] = plan.n
    obj["component_count"] = len(components)
    obj["target_vertex_count"] = TARGET_VERTEX_COUNT
    obj["arc_length_resampled"] = True
    obj["closure_permutation_zero_based"] = list(plan.permutation)
    obj["description"] = "Four-region twisted annular closed braid; vertices and edges only"

    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    return obj


def main():
    check_settings()
    plan = compile_braid(STRAND_COUNT, BRAID_WORD)
    verts, edges, components, spans, lengths = generate_edge_mesh(plan)
    validate_geometry(plan, verts, edges, components, spans)
    obj = create_blender_object(verts, edges, plan, components)
    if CREATE_OBSERVATION_LOOP:
        observation_obj, observation_vertices, observation_edges = create_observation_loop(plan)
        validate_observation(plan, observation_vertices)
        print("Observation loop: fixed front radius, fixed side/back height, smooth transition ramps")
        print("Observation loop:", len(observation_vertices), "vertices,",
              len(observation_edges), "edges, 0 faces")
    print("Braid word:", BRAID_WORD)
    print("Closure permutation (zero-based):", plan.permutation)
    print("Link components (strand IDs, zero-based):", components)
    print("Vertices per link component:", [b - a for a, b in spans])
    print("Estimated component arc lengths:", [round(length, 6) for length in lengths])
    print("Mesh:", len(verts), "vertices,", len(edges), "edges, 0 faces")
    if obj is None:
        print("Blender not available: geometry constructed and validated only.")
    else:
        print("Created Blender object:", obj.name)
    if bpy is not None and SAVE_BLEND_PATH:
        bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(SAVE_BLEND_PATH))
    return verts, edges, components


if __name__ == "__main__":
    main()