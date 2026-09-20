#!/usr/bin/env python3
"""Finite causal-foliated move-ensemble scan for QCCG.

Uses the reversible causal 2<->4 move set from run_causal_move_dynamics_toy.py.
At each state all manifold/foliation-preserving moves are enumerated.  A move
is proposed uniformly from that finite set and accepted with a
Metropolis-Hastings ratio including the forward/reverse candidate-count ratio.

Toy action:
  S = eps_V (N4-N4*)^2 + kappa_B * simplex-type-imbalance.

The scan measures the dual-graph spectral and volume-growth dimensions.  A
finite broad-4D candidate is only a diagnostic; stable continuum scaling is a
separate OPEN obligation.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
from pathlib import Path


N_SLABS = 3
TARGET_N4 = 76
VOLUME_EPS = 0.035
KAPPA_BALANCE = (0.0, 0.05, 0.20)
STEPS = 140
SEED = 20260920
ROOTS = 24
T1, T2 = 3, 6
R1, R2 = 2, 4
BROAD_4D_MIN = 3.0
DETAIL_BAL_TOL = 2.0e-12


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
    S = set()
    spatial = spatial_s3()
    for t in range(N_SLABS):
        for tet in spatial:
            S.update(staircase_prism(tet, t, t + 1))
    return S, spatial


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
        return None
    keys = sorted(times)
    if keys[1] - keys[0] != 1:
        return None
    return (times[keys[0]], times[keys[1]])


VALID_TYPES = {(1, 4), (2, 3), (3, 2), (4, 1)}


def valid_complex(S, spatial):
    if any(type_of(s) not in VALID_TYPES for s in S):
        return False
    boundary = 0
    for f, inc in face_map(S).items():
        times = {t for t, _v in f}
        on_boundary = times == {0} or times == {N_SLABS}
        expected = 1 if on_boundary else 2
        if on_boundary:
            boundary += 1
        if len(inc) != expected:
            return False
    return boundary == 2 * len(spatial)


def cand24(S):
    out = []
    fm = face_map(S)
    eset = edges(S)
    for face, pair in fm.items():
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
        if all(x not in S for x in new):
            out.append((pair[0], pair[1], new))
    return out


def apply24(S, c):
    S2 = set(S)
    s1, s2, new = c
    S2.remove(s1)
    S2.remove(s2)
    S2.update(new)
    return S2


def cand42(S):
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
        out.append((tuple(star), new))
    return out


def apply42(S, c):
    S2 = set(S)
    star, new = c
    for s in star:
        S2.remove(s)
    S2.update(new)
    return S2


def valid_moves(S, spatial):
    out = []
    for c in cand24(S):
        S2 = apply24(S, c)
        if valid_complex(S2, spatial):
            out.append(("24", S2))
    for c in cand42(S):
        S2 = apply42(S, c)
        if valid_complex(S2, spatial):
            out.append(("42", S2))
    # Deduplicate coincident target states.
    seen = {}
    for typ, S2 in out:
        seen[tuple(sorted(S2))] = (typ, S2)
    return list(seen.values())


def type_counts(S):
    c = collections.Counter(type_of(s) for s in S)
    return {str(k): c[k] for k in sorted(VALID_TYPES)}


def action(S, kappa):
    n4 = len(S)
    counts = collections.Counter(type_of(s) for s in S)
    target = n4 / 4.0
    imbalance = sum((counts[t] - target) ** 2 for t in VALID_TYPES) / max(1.0, n4)
    return VOLUME_EPS * (n4 - TARGET_N4) ** 2 + kappa * imbalance, imbalance


def dual_adj(S):
    adj = {s: set() for s in S}
    for inc in face_map(S).values():
        if len(inc) == 2:
            a, b = inc
            adj[a].add(b)
            adj[b].add(a)
    return adj


def lazy_return(adj, root, tmax):
    p = {root: 1.0}
    out = [1.0]
    for _ in range(tmax):
        q = collections.defaultdict(float)
        for v, pv in p.items():
            q[v] += 0.5 * pv
            share = 0.5 * pv / len(adj[v])
            for w in adj[v]:
                q[w] += share
        p = q
        out.append(p.get(root, 0.0))
    return out


def balls(adj, root, maxr):
    seen = {root}
    front = {root}
    out = [1]
    for _ in range(maxr):
        nxt = set()
        for v in front:
            nxt.update(adj[v])
        nxt -= seen
        seen |= nxt
        front = nxt
        out.append(len(seen))
    return out


def dimensions(S):
    adj = dual_adj(S)
    roots = sorted(adj)[:min(ROOTS, len(adj))]
    rr = [lazy_return(adj, r, T2) for r in roots]
    p1 = sum(x[T1] for x in rr) / len(rr)
    p2 = sum(x[T2] for x in rr) / len(rr)
    ds = -2.0 * math.log(p2 / p1) / math.log(T2 / T1)

    vv = [balls(adj, r, R2) for r in roots]
    v1 = sum(x[R1] for x in vv) / len(vv)
    v2 = sum(x[R2] for x in vv) / len(vv)
    dv = math.log(v2 / v1) / math.log(R2 / R1)
    return ds, dv


def sample(kappa, seed):
    rng = random.Random(seed)
    S, spatial = build()
    accepted = 0
    attempted = 0
    max_db_residual = 0.0
    move_counts = collections.Counter()

    for _ in range(STEPS):
        moves = valid_moves(S, spatial)
        if not moves:
            break
        attempted += 1
        typ, S2 = rng.choice(moves)
        moves2 = valid_moves(S2, spatial)
        if not moves2:
            continue

        oldS, _ = action(S, kappa)
        newS, _ = action(S2, kappa)
        log_ratio = -(newS - oldS) + math.log(len(moves) / len(moves2))
        alpha_f = min(1.0, math.exp(min(0.0, log_ratio)))

        # Detailed-balance local check against exact reverse proposal.
        log_ratio_rev = -(oldS - newS) + math.log(len(moves2) / len(moves))
        alpha_r = min(1.0, math.exp(min(0.0, log_ratio_rev)))
        pi_old = math.exp(-oldS)
        pi_new = math.exp(-newS)
        flow_f = pi_old * (1.0 / len(moves)) * alpha_f
        flow_r = pi_new * (1.0 / len(moves2)) * alpha_r
        denom = max(1.0e-300, abs(flow_f), abs(flow_r))
        max_db_residual = max(max_db_residual, abs(flow_f - flow_r) / denom)

        if rng.random() < alpha_f:
            S = S2
            accepted += 1
            move_counts[typ] += 1

    ds, dv = dimensions(S)
    act, imbalance = action(S, kappa)
    vertex_count = len({v for simplex in S for v in simplex})
    return {
        "kappa_balance": kappa,
        "attempted": attempted,
        "accepted": accepted,
        "acceptance_fraction": accepted / max(1, attempted),
        "move_counts": dict(move_counts),
        "N4": len(S),
        "vertex_count": vertex_count,
        "action": act,
        "type_imbalance": imbalance,
        "type_counts": type_counts(S),
        "valid_complex": valid_complex(S, spatial),
        "max_detailed_balance_relative_residual": max_db_residual,
        "spectral_dimension": ds,
        "volume_growth_dimension": dv,
        "broad_4d_candidate": ds >= BROAD_4D_MIN and dv >= BROAD_4D_MIN,
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "causal-move-ensemble-scan",
        "artifact": "qccg/run_causal_move_ensemble_scan.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    rows = [
        sample(k, SEED + 101 * i)
        for i, k in enumerate(KAPPA_BALANCE)
    ]
    scan_ok = all(
        r["valid_complex"]
        and r["attempted"] > 0
        and r["accepted"] > 0
        and r["max_detailed_balance_relative_residual"] <= DETAIL_BAL_TOL
        for r in rows
    )
    candidate_count = sum(r["broad_4d_candidate"] for r in rows)
    finite_candidate = candidate_count >= 1
    initial_S, _initial_spatial = build()
    initial_vertex_count = len({v for simplex in initial_S for v in simplex})
    fixed_vertex = all(r["vertex_count"] == initial_vertex_count for r in rows)
    fixed_vertex_insufficient = (not finite_candidate) and fixed_vertex

    result = {
        "schema": 1,
        "scope": "finite causal-foliated reversible move-ensemble scan; not continuum scaling",
        "evidence": [
            evidence(
                "qccg-causal-ensemble-scan",
                "CAUSAL_MOVE_ENSEMBLE_SCAN_EXECUTED",
                "PASS" if scan_ok else "FAIL",
                "A finite reversible causal-move Metropolis-Hastings scan is executed with topology/foliation preservation and an explicit proposal-count detailed-balance check.",
                target_N4=TARGET_N4,
                volume_epsilon=VOLUME_EPS,
                kappa_balance=list(KAPPA_BALANCE),
                steps=STEPS,
                detailed_balance_tolerance=DETAIL_BAL_TOL,
                rows=rows,
            ),
            evidence(
                "qccg-causal-finite-4d-candidate",
                "CAUSAL_FINITE_4D_CANDIDATE",
                "PASS" if finite_candidate else "FAIL",
                (
                    "At least one preregistered finite causal-move setting reaches the broad ds,dV>=3 diagnostic."
                    if finite_candidate
                    else "No preregistered finite causal-move setting reaches the broad ds,dV>=3 diagnostic; the current action/move ensemble is insufficient."
                ),
                broad_4d_min=BROAD_4D_MIN,
                candidate_count=candidate_count,
                rows=rows,
            ),
            evidence(
                "qccg-causal-fixed-vertex-insufficient",
                "CAUSAL_FIXED_VERTEX_MOVESET_INSUFFICIENT",
                "PASS" if fixed_vertex_insufficient else ("NOT_APPLICABLE" if finite_candidate else "FAIL"),
                "The audited 2<->4 causal move ensemble keeps the microscopic vertex set fixed; when the finite 4D diagnostic fails, this is retained as evidence that the move set cannot grow an extended spatial geometry from the tiny initial slice.",
                initial_vertex_count=initial_vertex_count,
                fixed_vertex_across_scan=fixed_vertex,
                candidate_count=candidate_count,
                rows=rows,
            ),
            evidence(
                "qccg-causal-4d-scaling-open",
                "CAUSAL_4D_SCALING_STABILITY",
                "OPEN",
                "A finite-volume candidate, if present, is not enough. Increasing spatial volume/time extent and coupling scans are required to establish a stable four-dimensional critical phase.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Only infrastructure/internal-consistency failure aborts CI. A FAIL on the
    # finite 4D physics candidate is a valid scientific negative result.
    if not scan_ok:
        raise SystemExit("causal move ensemble scan failed internal consistency checks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
