#!/usr/bin/env python3
"""Finite-size mixed-CDT connectivity scaling scan.

Extends the existing three-patch fixed-boundary connectivity toy by gluing a
sequence of independently flippable causal move patches.  At each depth m, all
2^m local-move configurations must:
- be causal connected 4D simplicial manifolds-with-boundary;
- share exactly one external tetrahedral boundary;
- form a connected m-dimensional move hypercube.

The scan is intentionally finite and constructive.  PASS means this patch
architecture survives the audited size increase; it is not a proof that the
full generic CDT/QCCG triangulation space is ergodic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

import run_cdt_mixed_connectivity_toy as base


PATCHES = (
    ("A", [base.A0, base.A1], base.A_TIMES),
    ("B", [base.B0, base.B1], base.B_TIMES),
    ("C", [base.C0, base.C1], base.C_TIMES),
    ("A2", [base.A0, base.A1], base.A_TIMES),
    ("B2", [base.B0, base.B1], base.B_TIMES),
)
MAX_PATCHES = len(PATCHES)


def cube_edges(m):
    n = 1 << m
    edges = []
    for i in range(n):
        for bit in range(m):
            j = i ^ (1 << bit)
            if i < j:
                edges.append((i, j, bit))
    return edges


def laplacian(n, edges):
    L = sp.zeros(n)
    for a,b,_ in edges:
        L[a,a] += 1
        L[b,b] += 1
        L[a,b] -= 1
        L[b,a] -= 1
    return L


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "mixed-cdt-connectivity-scaling-scan",
        "artifact": "qccg/run_cdt_scalable_connectivity_scan.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    name0, states0, times0 = PATCHES[0]
    states = list(states0)
    times = dict(times0)
    next_label = max(times) + 1

    ok0, boundary0, rows0 = base.audit_states(states, times)
    depth_rows = [{
        "patches": 1,
        "patch_sequence": [name0],
        "n_states": len(states),
        "n_boundary_tetrahedra": len(boundary0),
        "all_states_valid": ok0,
        "gluing": None,
    }]

    all_depths_valid = ok0
    reached = 1
    glue_failure = None

    for depth in range(2, MAX_PATCHES + 1):
        name, patch_states, patch_times = PATCHES[depth-1]
        glued = base.glue_second(states, times, patch_states, patch_times, next_label)
        if glued is None:
            all_depths_valid = False
            glue_failure = {
                "attempted_depth": depth,
                "patch": name,
                "reason": "no common-boundary causal fixed-boundary gluing found",
            }
            break

        states, times, next_label, meta = glued
        ok, bnd, rows = base.audit_states(states, times)
        m = depth
        edges = cube_edges(m)
        L = laplacian(1 << m, edges)
        nullity = len(L.nullspace())
        vals = []
        for val,mult in L.eigenvals().items():
            vals.extend([float(sp.N(val))] * int(mult))
        positives = [x for x in vals if x > 1e-12]
        gap = min(positives) if positives else 0.0
        graph_connected = base.graph_connected(1 << m, edges)

        row = {
            "patches": m,
            "patch_sequence": [x[0] for x in PATCHES[:m]],
            "n_states": len(states),
            "n_boundary_tetrahedra": len(bnd),
            "all_states_valid": ok,
            "move_graph_connected": graph_connected,
            "move_graph_nullity": nullity,
            "move_graph_gap": gap,
            "move_graph_eigenvalue_min": min(vals),
            "move_graph_eigenvalue_max": max(vals),
            "gluing": meta,
        }
        depth_rows.append(row)
        if not (ok and graph_connected and nullity == 1 and gap > 0):
            all_depths_valid = False
            glue_failure = {
                "attempted_depth": depth,
                "patch": name,
                "reason": "combined state/manifold or move-graph connectivity audit failed",
                "row": row,
            }
            break
        reached = depth

    # Negative control at the largest reached depth: remove the final move bit.
    m = reached
    n = 1 << m
    full_edges = cube_edges(m)
    reduced = [e for e in full_edges if e[2] != m-1]
    reduced_L = laplacian(n, reduced)
    reduced_nullity = len(reduced_L.nullspace())
    control_pass = reduced_nullity == 2 if m >= 2 else True

    scaling_pass = all_depths_valid and reached == MAX_PATCHES

    result = {
        "schema": 1,
        "scope": "finite constructive mixed-CDT fixed-boundary connectivity scaling scan",
        "evidence": [
            evidence(
                "qccg-cdt-connectivity-scaling-toy",
                "CDT_CONNECTIVITY_SCALING_TOY",
                "PASS" if scaling_pass else "FAIL",
                (
                    "The constructive mixed-move patch architecture remains one connected fixed-boundary causal sector "
                    f"through {MAX_PATCHES} independently flippable patches ({1<<MAX_PATCHES} states)."
                    if scaling_pass else
                    "The constructive mixed-move patch architecture fails before the preregistered maximum depth."
                ),
                max_patches=MAX_PATCHES,
                reached_patches=reached,
                depth_rows=depth_rows,
                failure=glue_failure,
            ),
            evidence(
                "qccg-cdt-connectivity-scaling-control",
                "CDT_CONNECTIVITY_SCALING_NEGATIVE_CONTROL",
                "PASS" if control_pass else "FAIL",
                "Removing the last local move family/bit splits the largest audited hypercube sector into two connected components.",
                patches=reached,
                n_states=n,
                reduced_nullity=reduced_nullity,
            ),
            evidence(
                "qccg-cdt-scalable-connectivity-open",
                "QCCG_CDT_SCALABLE_CONNECTIVITY",
                "OPEN",
                "Connectivity is demonstrated for a finite constructive family of glued independent local patches, not for generic causal triangulations at increasing volume.",
                next_step=(
                    "Generate generic fixed-boundary causal triangulations at increasing N4, apply the realized move templates "
                    "algorithmically, and measure connected-component counts and candidate conserved invariants."
                ),
            ),
        ],
    }

    p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("CDT connectivity scaling toy failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    main()
