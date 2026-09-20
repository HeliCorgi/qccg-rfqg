#!/usr/bin/env python3
"""4D simplicial-cellulation negative control for QCCG manifold emergence.

We generate closed combinatorial 4-manifold triangulations from the boundary of
a 5-simplex using valid 1->5 Pachner refinements.  A second ensemble additionally
uses valid 2->4 Pachner moves.

Every tetrahedral 3-face is checked to have incidence two, so the complexes
remain closed combinatorial 4-manifold candidates.  We then measure the dual
4-simplex adjacency graph with:
- lazy-walk spectral dimension between t=4 and t=8;
- effective ball-volume growth dimension between radii 2 and 4.

Result expected from this negative control:
unweighted/refinement-only move dynamics does NOT produce a 4D metric scaling
phase.  Adding 2->4 moves alone is also insufficient.  Therefore QCCG cannot
promote a uniform cellulation superposition to a 4D continuum merely from
combinatorial dimension; a weighted critical geometric ensemble is required.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
from pathlib import Path


SEED = 20260920
N_REFINE = 100
N_24 = 200
ROOTS = 30
SPECTRAL_T1 = 4
SPECTRAL_T2 = 8
VOL_R1 = 2
VOL_R2 = 4
FAIL4_THRESHOLD = 3.0


def boundary_5_simplex():
    verts = range(6)
    return {tuple(sorted(set(verts) - {omit})) for omit in verts}


def refine_1_to_5(simplices, simplex, new_vertex):
    vertices = set(simplex)
    simplices.remove(simplex)
    for omit in simplex:
        simplices.add(tuple(sorted({new_vertex} | (vertices - {omit}))))


def edges_of(simplices):
    out = set()
    for s in simplices:
        out.update(tuple(sorted(e)) for e in itertools.combinations(s, 2))
    return out


def face_map(simplices):
    m = collections.defaultdict(list)
    for s in simplices:
        for face in itertools.combinations(s, 4):
            m[tuple(sorted(face))].append(s)
    return m


def candidates_2_to_4(simplices):
    fmap = face_map(simplices)
    edge_set = edges_of(simplices)
    out = []
    for face, pair in fmap.items():
        if len(pair) != 2:
            continue
        a = next(iter(set(pair[0]) - set(face)))
        b = next(iter(set(pair[1]) - set(face)))
        if tuple(sorted((a, b))) in edge_set:
            continue
        new = [
            tuple(sorted({a, b} | (set(face) - {omit})))
            for omit in face
        ]
        if all(s not in simplices for s in new):
            out.append((pair[0], pair[1], tuple(face), a, b, tuple(new)))
    return out


def move_2_to_4(simplices, candidate):
    s1, s2, _face, _a, _b, new = candidate
    simplices.remove(s1)
    simplices.remove(s2)
    simplices.update(new)


def generate(refine_moves, moves24=0, seed=SEED):
    rng = random.Random(seed)
    simplices = boundary_5_simplex()
    next_vertex = 6
    for _ in range(refine_moves):
        simplex = rng.choice(sorted(simplices))
        refine_1_to_5(simplices, simplex, next_vertex)
        next_vertex += 1

    performed24 = 0
    for _ in range(moves24):
        candidates = candidates_2_to_4(simplices)
        if not candidates:
            break
        move_2_to_4(simplices, rng.choice(candidates))
        performed24 += 1
    return simplices, performed24


def dual_adjacency(simplices):
    fmap = face_map(simplices)
    adj = {s: set() for s in simplices}
    for pair in fmap.values():
        if len(pair) == 2:
            a, b = pair
            adj[a].add(b)
            adj[b].add(a)
    return adj


def manifold_check(simplices):
    counts = [len(x) for x in face_map(simplices).values()]
    return {
        "all_tetrahedral_faces_incidence_two": all(x == 2 for x in counts),
        "min_face_incidence": min(counts),
        "max_face_incidence": max(counts),
    }


def lazy_return(adj, root, tmax):
    p = {root: 1.0}
    returns = [1.0]
    for _ in range(tmax):
        q = collections.defaultdict(float)
        for v, pv in p.items():
            q[v] += 0.5 * pv
            share = 0.5 * pv / len(adj[v])
            for w in adj[v]:
                q[w] += share
        p = q
        returns.append(p.get(root, 0.0))
    return returns


def ball_volumes(adj, root, maxr):
    seen = {root}
    frontier = {root}
    vols = [1]
    for _ in range(maxr):
        nxt = set()
        for v in frontier:
            nxt.update(adj[v])
        nxt.difference_update(seen)
        seen.update(nxt)
        frontier = nxt
        vols.append(len(seen))
    return vols


def measure(simplices):
    adj = dual_adjacency(simplices)
    roots = sorted(adj)[:min(ROOTS, len(adj))]

    return_rows = [lazy_return(adj, r, SPECTRAL_T2) for r in roots]
    p1 = sum(x[SPECTRAL_T1] for x in return_rows) / len(return_rows)
    p2 = sum(x[SPECTRAL_T2] for x in return_rows) / len(return_rows)
    ds = -2.0 * math.log(p2 / p1) / math.log(SPECTRAL_T2 / SPECTRAL_T1)

    volume_rows = [ball_volumes(adj, r, VOL_R2) for r in roots]
    v1 = sum(x[VOL_R1] for x in volume_rows) / len(volume_rows)
    v2 = sum(x[VOL_R2] for x in volume_rows) / len(volume_rows)
    dv = math.log(v2 / v1) / math.log(VOL_R2 / VOL_R1)

    degrees = [len(x) for x in adj.values()]
    return {
        "n_vertices": len({v for s in simplices for v in s}),
        "n_4simplices": len(simplices),
        "dual_degree_min": min(degrees),
        "dual_degree_max": max(degrees),
        "dual_degree_mean": sum(degrees) / len(degrees),
        "spectral_return_t1": p1,
        "spectral_return_t2": p2,
        "spectral_dimension": ds,
        "volume_ball_r1_mean": v1,
        "volume_ball_r2_mean": v2,
        "volume_growth_dimension": dv,
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "simplicial-pachner-ensemble-audit",
        "artifact": "qccg/run_pachner_ensemble_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    refine, n24a = generate(N_REFINE, 0, SEED)
    mixed, n24b = generate(N_REFINE, N_24, SEED)

    topo_refine = manifold_check(refine)
    topo_mixed = manifold_check(mixed)
    meas_refine = measure(refine)
    meas_mixed = measure(mixed)

    topology_pass = (
        topo_refine["all_tetrahedral_faces_incidence_two"]
        and topo_mixed["all_tetrahedral_faces_incidence_two"]
    )

    refine_fails4 = (
        meas_refine["spectral_dimension"] < FAIL4_THRESHOLD
        and meas_refine["volume_growth_dimension"] < FAIL4_THRESHOLD
    )
    mixed_fails4 = (
        meas_mixed["spectral_dimension"] < FAIL4_THRESHOLD
        and meas_mixed["volume_growth_dimension"] < FAIL4_THRESHOLD
    )

    result = {
        "schema": 1,
        "scope": "closed 4D simplicial Pachner-move negative-control ensembles; not a tuned QCCG continuum phase",
        "evidence": [
            evidence(
                "qccg-pachner-topology-control",
                "PACHNER_4D_MANIFOLD_TOPOLOGY_CONTROL",
                "PASS" if topology_pass else "FAIL",
                "The audited 1->5 and mixed 1->5 plus 2->4 ensembles preserve closed combinatorial 4-manifold face incidence.",
                refine_topology=topo_refine,
                mixed_topology=topo_mixed,
                refine_moves=N_REFINE,
                requested_2_to_4=N_24,
                performed_2_to_4=n24b,
            ),
            evidence(
                "qccg-unweighted-pachner-dimension-failure",
                "UNWEIGHTED_PACHNER_DIMENSION_FAILURE_DETECTED",
                "PASS" if refine_fails4 else "FAIL",
                "A topologically valid unweighted 4D refinement ensemble nevertheless has spectral and volume-growth dimensions far below four.",
                target_failure_threshold=FAIL4_THRESHOLD,
                measurement=meas_refine,
            ),
            evidence(
                "qccg-moveset-repair-insufficient",
                "PACHNER_MOVESET_REPAIR_INSUFFICIENT",
                "PASS" if mixed_fails4 else "FAIL",
                "Adding many valid 2->4 Pachner moves without a geometric action/critical weight still fails the broad four-dimensional metric-scaling target.",
                target_failure_threshold=FAIL4_THRESHOLD,
                performed_2_to_4=n24b,
                measurement=meas_mixed,
            ),
            evidence(
                "qccg-weighted-critical-cellulation-open",
                "QCCG_WEIGHTED_CRITICAL_CELLULATION_ENSEMBLE",
                "OPEN",
                "The naive uniform cellulation liquid is rejected. A QCCG geometric/Regge-like weight with inverse/mixing moves must be specified and tuned to a critical manifold-like phase before continuum promotion.",
                repair=(
                    "Replace a uniform H_move ground state over admissible cellulations by a weighted "
                    "graph-changing Hamiltonian/action that controls curvature/entropy competition; "
                    "include reversible Pachner moves and scan for a stable 4D critical phase."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("Pachner ensemble audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
