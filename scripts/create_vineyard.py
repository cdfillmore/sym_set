#!/usr/bin/env python3
"""Compute a persistence vineyard of a line-graph OBJ along an observation loop.

The output is an OBJ whose vertices are (birth, death, loop arc length) and
whose ``l`` records join matched persistence points in consecutive slices.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import gudhi


def read_line_obj(path):
    """Read OBJ vertices and line segments, supporting multi-index polylines."""
    vertices, edges = [], set()
    with open(path, "r", encoding="utf-8") as obj:
        for line_number, line in enumerate(obj, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"{path}:{line_number}: expected 3D vertex")
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "l":
                indices = [int(token.split("/")[0]) - 1 for token in fields[1:]]
                edges.update(tuple(sorted(pair)) for pair in zip(indices, indices[1:]))
    if not vertices or not edges:
        raise ValueError(f"{path} must contain vertices and OBJ line elements")
    if any(a < 0 or b >= len(vertices) for a, b in edges):
        raise ValueError(f"{path} contains a line index outside its vertex list")
    return np.asarray(vertices, dtype=float), sorted(edges)


def ordered_loop(path):
    """Order a single closed OBJ line loop by following its degree-two graph."""
    points, edges = read_line_obj(path)
    graph = [set() for _ in points]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    if any(len(neighbors) != 2 for neighbors in graph):
        raise ValueError("observation loop must be one closed loop with degree 2 at each vertex")
    order = [0]
    previous, current = 0, min(graph[0])
    while current != 0:
        if current in order:
            raise ValueError("observation loop is not a single simple cycle")
        order.append(current)
        next_vertices = graph[current] - {previous}
        previous, current = current, next_vertices.pop()
    if len(order) != len(points):
        raise ValueError("observation loop OBJ has multiple disconnected cycles")
    return points[order]


def persistence_slice(points, edges, center):
    """Return H0/H1 persistence points for the distance filtration at center."""
    tree = gudhi.SimplexTree()
    values = np.linalg.norm(points - center, axis=1)
    for i, value in enumerate(values):
        tree.insert([i], filtration=float(value))
    for a, b in edges:
        tree.insert([a, b], filtration=float(max(values[a], values[b])))
    tree.compute_persistence()
    cap = float(values.max())
    result = {0: [], 1: []}
    for birth_simplex, death_simplex in tree.persistence_pairs():
        dim = len(birth_simplex) - 1
        if dim not in result or not birth_simplex:
            continue
        birth = float(tree.filtration(birth_simplex))
        death = float(tree.filtration(death_simplex)) if death_simplex else cap
        result[dim].append([birth, death])
    return result


def write_vineyard(path, slices, loop_positions):
    """Match points between slices and write independent track segments."""
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(loop_positions, axis=0), axis=1))))
    vertices, lines = [], []
    for dim in (0, 1):
        previous = []
        previous_indices = []
        for step, diagram in enumerate(slices):
            current = np.asarray(diagram[dim], dtype=float).reshape((-1, 2))
            indices = []
            for birth, death in current:
                indices.append(len(vertices))
                vertices.append((birth, death, arc[step]))
            if len(previous) and len(current):
                cost = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(cost)
                for row, col in zip(rows, cols):
                    lines.append((previous_indices[row], indices[col]))
            previous, previous_indices = current, indices

    with open(path, "w", encoding="utf-8") as obj:
        obj.write("# Persistence vineyard: (birth, death, observation arc length)\n")
        for birth, death, position in vertices:
            obj.write(f"v {birth:.12g} {death:.12g} {position:.12g}\n")
        for a, b in lines:
            obj.write(f"l {a + 1} {b + 1}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("link_obj", help="OBJ containing the link as line elements")
    parser.add_argument("observation_loop_obj", help="closed observation-loop OBJ")
    parser.add_argument("output_obj", help="path for the generated vineyard OBJ")
    args = parser.parse_args()

    points, edges = read_line_obj(args.link_obj)
    loop = ordered_loop(args.observation_loop_obj)
    slices = [persistence_slice(points, edges, center) for center in loop]
    write_vineyard(args.output_obj, slices, loop)
    print(f"Wrote {args.output_obj}: {len(loop)} slices, {len(points)} link vertices, {len(edges)} edges")


if __name__ == "__main__":
    main()
