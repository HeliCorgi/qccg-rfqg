#!/usr/bin/env python3
"""Causal/foliated local Pachner-move dynamics toy for QCCG.

Starting from the audited finite foliated 4D simplicial cylinder, enumerate
ordinary 2->4 Pachner candidates and retain only those whose resulting complex:
- uses adjacent-time (4,1)/(3,2)/(2,3)/(1,4) simplices;
- preserves the expected initial/final spatial boundary;
- has incidence two on every interior tetrahedron.

For accepted moves we explicitly search for a 4->2 inverse that restores the
original complex exactly.  A finite symmetric state graph is then built from a
small set of accepted neighboring configurations and its graph-Laplacian move
Hamiltonian is checked to be Hermitian positive semidefinite with a uniform
zero mode.

This establishes a finite reversible causal move architecture only.  It does
not establish ergodicity or a critical 4D continuum phase.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import sympy as sp


N_SLABS = 3
MAX_NEIGHBORS = 12


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


def face_map(S):
    m = collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s, 4):
            m[tuple(sorted(f))].append(s)
    return m


def edge_map(S):
    m = collections.defaultdict(list)
    for s in S:
        for e in itertools.combinations(s, 2):
            m[tuple(sorted(e))].append(s)
    return m


def edges(S):
    return set(edge_map(S))


def type_of(simplex):
    times = collections.Counter(t for t, _v in simplex)
    if len(times) != 2:
        return ("bad-time-count", tuple(sorted(times.items())))
    keys = sorted(times)
    if keys[1] - keys[0] != 1:
        return ("nonadjacent", tuple(sorted(times.items())))
    return (times[keys[0]], times[keys[1]])


def audit(S, spatial):
    valid_types = {(1, 4), (2, 3), (3, 2), (4, 1)}
    adjacent = all(type_of(s) in valid_types for s in S)
    bad = []
    boundary = 0
    for f, inc in face_map(S).items():
        times = {t for t, _v in f}
        on_boundary = times == {0} or times == {N_SLABS}
        expected = 1 if on_boundary else 2
        if on_boundary:
            boundary += 1
        if len(inc) != expected:
            bad.append((f, len(inc), expected))
    return {
        "adjacent": adjacent,
        "bad_face_count": len(bad),
        "boundary_faces": boundary,
        "expected_boundary_faces": 2 * len(spatial),
        "valid": adjacent and not bad and boundary == 2 * len(spatial),
    }


def candidates_2_to_4(S):
    fmap = face_map(S)
    eset = edges(S)
    out = []
    for face, pair in fmap.items():
        if len(pair) != 2:
            continue
        a = next(iter(set(pair[0]) - set(face)))
        b = next(iter(set(pair[1]) - set(face)))
        if tuple(sorted((a, b))) in eset:
            continue
        new = tuple(
            tuple(sorted({a, b} | (set(face) - {omit})))
            for omit in face
        )
        if all(s not in S for s in new):
            out.append((pair[0], pair[1], tuple(face), a, b, new))
    return out


def apply_2_to_4(S, cand):
    S2 = set(S)
    s1, s2, _face, _a, _b, new = cand
    S2.remove(s1)
    S2.remove(s2)
    S2.update(new)
    return S2


def candidates_4_to_2(S):
    out = []
    for (a, b), star in edge_map(S).items():
        if len(star) != 4:
            continue
        other = set().union(*(set(s) - {a, b} for s in star))
        if len(other) != 4:
            continue
        expected = {
            tuple(sorted({a, b} | (other - {omit})))
            for omit in other
        }
        if set(star) != expected:
            continue
        new = (tuple(sorted({a} | other)), tuple(sorted({b} | other)))
        if new[0] in S or new[1] in S:
            continue
        out.append(((a, b), tuple(star), new))
    return out


def apply_4_to_2(S, cand):
    S2 = set(S)
    _edge, star, new = cand
    for s in star:
        S2.remove(s)
    S2.update(new)
    return S2


def canon(S):
    return tuple(sorted(S))


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "causal-pachner-move-toy",
        "artifact": "qccg/run_causal_move_dynamics_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    S0, spatial = build()
    base = audit(S0, spatial)

    all24 = candidates_2_to_4(S0)
    valid = []
    invalid = []
    inverse_rows = []

    for cand in all24:
        S1 = apply_2_to_4(S0, cand)
        a1 = audit(S1, spatial)
        if a1["valid"]:
            inverse_found = False
            inverse_count = 0
            for inv in candidates_4_to_2(S1):
                Sback = apply_4_to_2(S1, inv)
                if Sback == S0:
                    inverse_found = True
                    inverse_count += 1
            valid.append((cand, S1))
            inverse_rows.append({
                "opposite_vertices": [list(cand[3]), list(cand[4])],
                "inverse_found": inverse_found,
                "inverse_count": inverse_count,
                "n_simplices_after": len(S1),
            })
        else:
            invalid.append({
                "opposite_vertices": [list(cand[3]), list(cand[4])],
                "audit": a1,
            })

    move_exists = len(valid) > 0
    all_reversible = move_exists and all(r["inverse_found"] for r in inverse_rows)

    # Finite state graph: center + up to MAX_NEIGHBORS valid one-move neighbors.
    states = [S0] + [x[1] for x in valid[:MAX_NEIGHBORS]]
    keys = [canon(s) for s in states]
    key_to_i = {k: i for i, k in enumerate(keys)}
    edge_set = set()

    for i, S in enumerate(states):
        for cand in candidates_2_to_4(S):
            S2 = apply_2_to_4(S, cand)
            if not audit(S2, spatial)["valid"]:
                continue
            j = key_to_i.get(canon(S2))
            if j is not None and i != j:
                edge_set.add(tuple(sorted((i, j))))
        for cand in candidates_4_to_2(S):
            S2 = apply_4_to_2(S, cand)
            if not audit(S2, spatial)["valid"]:
                continue
            j = key_to_i.get(canon(S2))
            if j is not None and i != j:
                edge_set.add(tuple(sorted((i, j))))

    n = len(states)
    L = sp.zeros(n)
    for i, j in edge_set:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    hermitian = L.H == L
    uniform = sp.ones(n, 1)
    uniform_zero = L * uniform == sp.zeros(n, 1)
    evals = []
    for v, mult in L.eigenvals().items():
        evals.extend([float(sp.N(v))] * int(mult))
    positive_semidefinite = min(evals) >= -1.0e-12
    nullity = len(L.nullspace())
    finite_graph_connected = nullity == 1 if n > 1 else False

    # Negative control: explicitly time-skipping replacement is invalid.
    bad = set(S0)
    victim = next(iter(bad))
    bad.remove(victim)
    bad.add(tuple(sorted(((0, 0), (0, 1), (2, 2), (2, 3), (2, 4)))))
    bad_control = not audit(bad, spatial)["valid"]

    finite_pass = (
        base["valid"]
        and move_exists
        and all_reversible
        and hermitian
        and uniform_zero
        and positive_semidefinite
        and finite_graph_connected
    )

    result = {
        "schema": 1,
        "scope": "finite causal-foliated 2<->4 Pachner move architecture toy",
        "evidence": [
            evidence(
                "qccg-causal-local-moves",
                "CAUSAL_LOCAL_MOVESET_TOY",
                "PASS" if finite_pass else "FAIL",
                "A finite foliated 4D complex admits local 2<->4 Pachner moves that remain inside the causal adjacent-slice manifold class; accepted moves have exact inverse moves.",
                n_slabs=N_SLABS,
                n_base_simplices=len(S0),
                base_audit=base,
                n_raw_2_to_4_candidates=len(all24),
                n_valid_causal_moves=len(valid),
                n_invalid_candidates=len(invalid),
                inverse_rows=inverse_rows[:MAX_NEIGHBORS],
            ),
            evidence(
                "qccg-causal-move-hamiltonian",
                "CAUSAL_MOVE_HERMITIAN_TOY",
                "PASS" if finite_pass else "FAIL",
                "The finite accepted-move state graph has a Hermitian positive graph-Laplacian move Hamiltonian with a unique uniform zero mode.",
                n_states=n,
                edges=[list(x) for x in sorted(edge_set)],
                eigenvalues=sorted(evals),
                nullity=nullity,
                hermitian=hermitian,
                uniform_zero=uniform_zero,
                positive_semidefinite=positive_semidefinite,
            ),
            evidence(
                "qccg-causal-move-negative-control",
                "CAUSAL_MOVE_NEGATIVE_CONTROL",
                "PASS" if bad_control else "FAIL",
                "A time-skipping local replacement is rejected by the same causal/manifold audit.",
            ),
            evidence(
                "qccg-causal-move-dynamics-finite-toy-open",
                "QCCG_CAUSAL_MOVE_DYNAMICS",
                "OPEN",
                "A finite reversible causal 2<->4 move architecture is demonstrated, but ergodicity and scalable reversible dynamics over increasing foliated QCCG ensembles remain unproved.",
                next_step=(
                    "Extend the finite causal state graph through multiple move depths and inverse move families, "
                    "test connectivity across fixed-boundary sectors, then add geometric weights and scan "
                    "increasing spatial volume/time extent for an extended four-dimensional critical phase."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("causal local-move toy audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
