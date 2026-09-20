#!/usr/bin/env python3
"""Causal/foliated 4D simplicial architecture audit for QCCG.

Start from the boundary of a 4-simplex, a closed triangulation of S^3.
For each spatial tetrahedron and each adjacent pair of time slices, triangulate
tetrahedron x interval by the standard staircase decomposition into four
4-simplices of types (1,4), (2,3), (3,2), (4,1).

The resulting finite simplicial cylinder has:
- a global discrete time label;
- only adjacent-slice 4-simplices;
- closed-manifold incidence in the interior;
- the expected spatial boundary tetrahedra at initial/final time.

A deliberately inserted simplex spanning two time steps is a causal negative
control.

This constructs a CDT-like kinematic architecture only.  It does not show that
QCCG dynamics selects a 4D extended phase.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


N_SLABS = 5


def spatial_s3():
    verts = range(5)
    return {tuple(sorted(set(verts) - {omit})) for omit in verts}


def staircase_prism(tet, t0, t1):
    vs = sorted(tet)
    out = []
    for j in range(4):
        simplex = []
        for i in range(j + 1):
            simplex.append((t0, vs[i]))
        for i in range(j, 4):
            simplex.append((t1, vs[i]))
        out.append(tuple(sorted(simplex)))
    return tuple(out)


def build():
    simplices = set()
    spatial = spatial_s3()
    for t in range(N_SLABS):
        for tet in spatial:
            simplices.update(staircase_prism(tet, t, t + 1))
    return simplices, spatial


def face_map(simplices):
    m = collections.defaultdict(list)
    for s in simplices:
        for f in itertools.combinations(s, 4):
            m[tuple(sorted(f))].append(s)
    return m


def type_of(simplex):
    times = collections.Counter(t for t, _v in simplex)
    if len(times) != 2:
        return tuple(sorted(times.values()))
    keys = sorted(times)
    if keys[1] - keys[0] != 1:
        return ("nonadjacent", tuple(sorted(times.items())))
    return (times[keys[0]], times[keys[1]])


def audit_complex(simplices, spatial):
    fmap = face_map(simplices)
    valid_types = {(1, 4), (2, 3), (3, 2), (4, 1)}
    types = [type_of(s) for s in simplices]
    adjacent = all(t in valid_types for t in types)

    boundary_faces = []
    interior_faces = []
    bad_faces = []
    for face, inc in fmap.items():
        times = {t for t, _v in face}
        on_initial = times == {0}
        on_final = times == {N_SLABS}
        if on_initial or on_final:
            boundary_faces.append((face, len(inc)))
            if len(inc) != 1:
                bad_faces.append((face, len(inc)))
        else:
            interior_faces.append((face, len(inc)))
            if len(inc) != 2:
                bad_faces.append((face, len(inc)))

    expected_boundary = 2 * len(spatial)
    return {
        "adjacent_slice_types_only": adjacent,
        "type_counts": {
            str(k): types.count(k)
            for k in valid_types
        },
        "n_4simplices": len(simplices),
        "n_faces": len(fmap),
        "n_boundary_tetrahedra": len(boundary_faces),
        "expected_boundary_tetrahedra": expected_boundary,
        "n_interior_tetrahedra": len(interior_faces),
        "bad_face_count": len(bad_faces),
        "bad_faces_preview": [
            {"face": [[t, v] for t, v in f], "incidence": n}
            for f, n in bad_faces[:5]
        ],
        "manifold_with_boundary_incidence": (
            adjacent
            and len(boundary_faces) == expected_boundary
            and not bad_faces
        ),
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "causal-simplicial-architecture-audit",
        "artifact": "qccg/run_causal_foliation_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    simplices, spatial = build()
    good = audit_complex(simplices, spatial)

    # Negative control: insert a 4-simplex that jumps from t=1 to t=3.
    bad_simplices = set(simplices)
    bad_simplices.add(tuple(sorted(((1, 0), (1, 1), (3, 2), (3, 3), (3, 4)))))
    bad = audit_complex(bad_simplices, spatial)
    causal_violation_detected = not bad["adjacent_slice_types_only"]

    result = {
        "schema": 1,
        "scope": "finite CDT-like foliated simplicial-cylinder architecture; not a critical-phase simulation",
        "evidence": [
            evidence(
                "qccg-causal-foliated-4d-architecture",
                "CAUSAL_FOLIATED_4D_ARCHITECTURE",
                "PASS" if good["manifold_with_boundary_incidence"] else "FAIL",
                "A finite 4D simplicial cylinder is constructed entirely from adjacent-time (4,1)/(3,2)/(2,3)/(1,4) building blocks with correct interior and spatial-boundary tetrahedral incidence.",
                n_slabs=N_SLABS,
                spatial_tetrahedra=len(spatial),
                audit=good,
            ),
            evidence(
                "qccg-causal-foliation-negative-control",
                "CAUSAL_FOLIATION_NEGATIVE_CONTROL",
                "PASS" if causal_violation_detected else "FAIL",
                "A simplex spanning nonadjacent time slices is detected as violating the causal/foliation building-block rule.",
                bad_audit=bad,
            ),
            evidence(
                "qccg-causal-move-dynamics-open",
                "QCCG_CAUSAL_MOVE_DYNAMICS",
                "OPEN",
                "No reversible ergodic set of local QCCG graph-changing moves and weights has yet been implemented within the foliated causal complex class.",
                next_step=(
                    "Construct foliation-preserving local refinement/coarsening moves between these "
                    "causal slabs, define a Hermitian/Metropolis-compatible geometric weight, and "
                    "scan increasing volumes for a stable extended four-dimensional phase."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("causal foliation architecture audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
