#!/usr/bin/env python3
"""
Create 3D evolute focal surfaces and 0th / 1st medial axis meshes from an OBJ.

Mirrors the workflow in examples.py ("Create 3d 0th + 1st medial axes + evolute obj").
"""

import argparse
import os

from sym_set import (
    approx_medial_axis,
    evolute_3d,
    gen_sample_from_obj,
    read_obj,
    write_obj,
)


def create_medial_axes_3d(
    input_path,
    output_prefix,
    *,
    alpha=1.0,
    lambda_val=0.35,
    num_samples=10000,
    evolute_radius=2.0,
):
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_prefix = output_prefix.rstrip(".obj")

    inputs = read_obj(input_path)
    focal1, focal2 = evolute_3d(*inputs, radius=evolute_radius)
    write_obj(
        f"{out_prefix}_focal1.obj",
        focal1,
        inputs[1],
        name=f"{name}_focal_1",
    )
    write_obj(
        f"{out_prefix}_focal2.obj",
        focal2,
        inputs[1],
        name=f"{name}_focal_2",
    )

    sample = gen_sample_from_obj(input_path, num_samples)
    _, _, v_pts, v_faces = approx_medial_axis(
        sample, False, False, alpha, lambda_val, True, False
    )
    write_obj(
        f"{out_prefix}_medial_0.obj",
        v_pts,
        v_faces,
        f"{name}_medial.0",
    )
    _, _, v_pts, v_faces = approx_medial_axis(
        sample, False, False, alpha, lambda_val, True, True
    )
    write_obj(
        f"{out_prefix}_medial_2.obj",
        v_pts,
        v_faces,
        f"{name}_medial.2",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build evolute focals and 0th/1st medial axes from a 3D OBJ mesh. "
            "Writes <output>_focal1.obj, _focal2.obj, _medial_0.obj, and _medial_2.obj."
        )
    )
    parser.add_argument("input", help="Path to input .obj mesh")
    parser.add_argument(
        "output",
        help="Output path prefix (extension optional; suffixes are appended)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha shape parameter (default: 1)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_val",
        type=float,
        default=0.35,
        help="Lambda ridge filter (default: 0.35)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=10000,
        help="Number of surface samples for medial axis (default: 10000)",
    )
    parser.add_argument(
        "--evolute-radius",
        type=float,
        default=2.0,
        help="Principal curvature neighborhood radius for evolute (default: 2)",
    )
    args = parser.parse_args()

    create_medial_axes_3d(
        args.input,
        args.output,
        alpha=args.alpha,
        lambda_val=args.lambda_val,
        num_samples=args.num_samples,
        evolute_radius=args.evolute_radius,
    )


if __name__ == "__main__":
    main()
