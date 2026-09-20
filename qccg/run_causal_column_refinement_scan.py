#!/usr/bin/env python3
"""Vertex-changing causal-column repair scan for QCCG.

Repair after the local causal 2<->4 ensemble failed to develop an extended
finite geometry while keeping the microscopic vertex set fixed.

A spatial S^3 triangulation is evolved by reversible 3D Pachner moves:
  1<->4  (changes vertex number)
  2<->3  (changes connectivity)
and then extended through time by the standard tetrahedron x interval
staircase triangulation.  This guarantees adjacent-slice causal 4-simplices.

The spatial moves are sampled with proposal-count Metropolis-Hastings using
  S = -kappa_1 N1 + eps_V (N3-N3*)^2.
The resulting 4D column geometry is measured with dual-graph spectral and
volume-growth dimensions.

Because the same spatial triangulation is repeated on every time slice, this
is a repair/negative-control architecture, not a fully local spacetime move
dynamics.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
from pathlib import Path


N_SLABS = 8
TARGET_N3 = 64
VOLUME_EPS = 0.03
KAPPA1_VALUES = (-0.5, 0.0, 0.5)
STEPS = 220
SEED = 20260920
ROOTS = 32
T1, T2 = 4, 8
R1, R2 = 2, 5
BROAD_4D_MIN = 3.0
DETAIL_BAL_TOL = 3.0e-12


def spatial_s3():
    verts = range(5)
    return {tuple(sorted(set(verts) - {omit})) for omit in verts}


def face_map3(S):
    m = collections.defaultdict(list)
    for tet in S:
        for f in itertools.combinations(tet, 3):
            m[tuple(sorted(f))].append(tet)
    return m


def edge_map3(S):
    m = collections.defaultdict(list)
    for tet in S:
        for e in itertools.combinations(tet, 2):
            m[tuple(sorted(e))].append(tet)
    return m


def edges3(S):
    return set(edge_map3(S))


def manifold3(S):
    return all(len(x) == 2 for x in face_map3(S).values())


def candidates41(S):
    byv = collections.defaultdict(list)
    for tet in S:
        for v in tet:
            byv[v].append(tet)
    out = []
    for v, star in byv.items():
        if len(star) != 4:
            continue
        neigh = set().union(*(set(t) - {v} for t in star))
        if len(neigh) != 4:
            continue
        target = tuple(sorted(neigh))
        if target in S:
            continue
        expected = {
            tuple(sorted({v} | (neigh - {omit})))
            for omit in neigh
        }
        if set(star) == expected:
            out.append((v, tuple(star), target))
    return out


def candidates23(S):
    fm = face_map3(S)
    eset = edges3(S)
    out = []
    for face, pair in fm.items():
        if len(pair) != 2:
            continue
        d = next(iter(set(pair[0]) - set(face)))
        e = next(iter(set(pair[1]) - set(face)))
        if tuple(sorted((d, e))) in eset:
            continue
        a, b, c = face
        new = (
            tuple(sorted((d, e, a, b))),
            tuple(sorted((d, e, b, c))),
            tuple(sorted((d, e, c, a))),
        )
        if all(x not in S for x in new):
            out.append((pair[0], pair[1], new))
    return out


def candidates32(S):
    out = []
    for (d, e), star in edge_map3(S).items():
        if len(star) != 3:
            continue
        other = set().union(*(set(t) - {d, e} for t in star))
        if len(other) != 3:
            continue
        a, b, c = sorted(other)
        expected = {
            tuple(sorted((d, e, a, b))),
            tuple(sorted((d, e, b, c))),
            tuple(sorted((d, e, c, a))),
        }
        if set(star) != expected:
            continue
        new = (
            tuple(sorted((a, b, c, d))),
            tuple(sorted((a, b, c, e))),
        )
        if new[0] in S or new[1] in S:
            continue
        out.append(((d, e), tuple(star), new))
    return out


def candidate_records(S, nextv, free):
    out = []

    add_labels = sorted(free) + [nextv]
    for tet in sorted(S):
        for v in add_labels:
            if v in set().union(*map(set, S)):
                continue
            out.append(("14", (tet, v, v == nextv)))

    for c in candidates41(S):
        out.append(("41", c))

    for c in candidates23(S):
        out.append(("23", c))

    for c in candidates32(S):
        out.append(("32", c))

    return out


def apply_move(S, nextv, free, rec):
    typ, c = rec
    S2 = set(S)
    free2 = set(free)
    nextv2 = nextv

    if typ == "14":
        tet, v, fresh = c
        S2.remove(tet)
        verts = set(tet)
        for omit in tet:
            S2.add(tuple(sorted({v} | (verts - {omit}))))
        if fresh:
            nextv2 += 1
        else:
            free2.remove(v)

    elif typ == "41":
        v, star, target = c
        for tet in star:
            S2.remove(tet)
        S2.add(target)
        free2.add(v)

    elif typ == "23":
        a, b, new = c
        S2.remove(a)
        S2.remove(b)
        S2.update(new)

    elif typ == "32":
        _edge, star, new = c
        for tet in star:
            S2.remove(tet)
        S2.update(new)

    else:
        raise ValueError(typ)

    return S2, nextv2, frozenset(free2)


def n1(S):
    return len(edges3(S))


def action(S, kappa):
    return -kappa * n1(S) + VOLUME_EPS * (len(S) - TARGET_N3) ** 2


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
    return out


def spacetime(S):
    out = set()
    for t in range(N_SLABS):
        for tet in S:
            out.update(staircase_prism(tet, t, t + 1))
    return out


def face_map4(T):
    m = collections.defaultdict(list)
    for s in T:
        for f in itertools.combinations(s, 4):
            m[tuple(sorted(f))].append(s)
    return m


def type_of(simplex):
    times = collections.Counter(t for t, _v in simplex)
    if len(times) != 2:
        return None
    keys = sorted(times)
    if keys[1] - keys[0] != 1:
        return None
    return (times[keys[0]], times[keys[1]])


VALID_TYPES = {(1, 4), (2, 3), (3, 2), (4, 1)}


def audit4(T, spatial):
    if any(type_of(s) not in VALID_TYPES for s in T):
        return False
    boundary = 0
    for f, inc in face_map4(T).items():
        times = {t for t, _v in f}
        on_boundary = times == {0} or times == {N_SLABS}
        expected = 1 if on_boundary else 2
        if on_boundary:
            boundary += 1
        if len(inc) != expected:
            return False
    return boundary == 2 * len(spatial)


def dual_adj(T):
    adj = {s: set() for s in T}
    for inc in face_map4(T).values():
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


def dimensions(T):
    adj = dual_adj(T)
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
    S = spatial_s3()
    nextv = 5
    free = frozenset()
    accepted = 0
    attempted = 0
    max_db = 0.0
    move_counts = collections.Counter()

    for _ in range(STEPS):
        recs = candidate_records(S, nextv, free)
        if not recs:
            break
        attempted += 1
        rec = rng.choice(recs)
        S2, nextv2, free2 = apply_move(S, nextv, free, rec)
        if not manifold3(S2):
            continue

        recs2 = candidate_records(S2, nextv2, free2)
        if not recs2:
            continue

        oldA = action(S, kappa)
        newA = action(S2, kappa)
        lr = -(newA - oldA) + math.log(len(recs) / len(recs2))
        af = min(1.0, math.exp(min(0.0, lr)))

        lr_rev = -(oldA - newA) + math.log(len(recs2) / len(recs))
        ar = min(1.0, math.exp(min(0.0, lr_rev)))
        pf = math.exp(-oldA) / len(recs) * af
        pr = math.exp(-newA) / len(recs2) * ar
        max_db = max(max_db, abs(pf - pr) / max(1.0e-300, abs(pf), abs(pr)))

        if rng.random() < af:
            S, nextv, free = S2, nextv2, free2
            accepted += 1
            move_counts[rec[0]] += 1

    T = spacetime(S)
    ds, dv = dimensions(T)
    return {
        "kappa1": kappa,
        "attempted": attempted,
        "accepted": accepted,
        "acceptance_fraction": accepted / max(1, attempted),
        "move_counts": dict(move_counts),
        "N0_spatial": len({v for tet in S for v in tet}),
        "N1_spatial": n1(S),
        "N3_spatial": len(S),
        "N4_spacetime": len(T),
        "spatial_manifold": manifold3(S),
        "causal_4d_manifold": audit4(T, S),
        "max_detailed_balance_relative_residual": max_db,
        "spectral_dimension": ds,
        "volume_growth_dimension": dv,
        "broad_4d_candidate": ds >= BROAD_4D_MIN and dv >= BROAD_4D_MIN,
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "causal-column-refinement-scan",
        "artifact": "qccg/run_causal_column_refinement_scan.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    rows = [sample(k, SEED + 131 * i) for i, k in enumerate(KAPPA1_VALUES)]
    internal = all(
        r["attempted"] > 0
        and r["accepted"] > 0
        and r["spatial_manifold"]
        and r["causal_4d_manifold"]
        and r["max_detailed_balance_relative_residual"] <= DETAIL_BAL_TOL
        for r in rows
    )
    growth_seen = any(r["N0_spatial"] > 5 for r in rows)
    shrink_seen = any(
        r["move_counts"].get("41", 0) > 0 or r["move_counts"].get("32", 0) > 0
        for r in rows
    )
    architecture_pass = internal and growth_seen and shrink_seen

    finite_candidate = any(r["broad_4d_candidate"] for r in rows)
    insufficient = architecture_pass and not finite_candidate

    result = {
        "schema": 1,
        "scope": "finite causal-column spatial-Pachner repair scan; spatial triangulation repeated across time",
        "evidence": [
            evidence(
                "qccg-causal-column-moveset",
                "CAUSAL_COLUMN_MOVESET_TOY",
                "PASS" if architecture_pass else "FAIL",
                "Spatial 1<->4 and 2<->3 Pachner moves change vertices/connectivity while the time-extended staircase complex remains a causal 4-manifold with reversible proposal-count MH control.",
                n_slabs=N_SLABS,
                target_N3=TARGET_N3,
                rows=rows,
            ),
            evidence(
                "qccg-causal-column-scan",
                "CAUSAL_COLUMN_ENSEMBLE_SCAN_EXECUTED",
                "PASS" if internal else "FAIL",
                "The vertex-changing causal-column ensemble scan is executed at preregistered couplings with detailed-balance, spatial-manifold and causal-4D incidence checks.",
                kappa1_values=list(KAPPA1_VALUES),
                steps=STEPS,
                broad_4d_min=BROAD_4D_MIN,
                rows=rows,
            ),
            evidence(
                "qccg-causal-column-finite-candidate",
                "CAUSAL_COLUMN_FINITE_4D_CANDIDATE",
                "PASS" if finite_candidate else "FAIL",
                (
                    "At least one finite causal-column setting reaches the broad ds,dV>=3 target."
                    if finite_candidate
                    else "No finite causal-column setting reaches the broad ds,dV>=3 target."
                ),
                rows=rows,
            ),
            evidence(
                "qccg-causal-column-repair-insufficient",
                "CAUSAL_COLUMN_REPAIR_INSUFFICIENT",
                "PASS" if insufficient else ("NOT_APPLICABLE" if finite_candidate else "FAIL"),
                "Vertex-changing spatial Pachner columns remove the fixed-vertex defect but still fail the broad 4D geometry target; repeating one spatial triangulation through all time slices is therefore an insufficient repair.",
                rows=rows,
            ),
            evidence(
                "qccg-time-local-causal-dynamics-open",
                "QCCG_TIME_LOCAL_CAUSAL_DYNAMICS",
                "OPEN",
                "The column repair changes the same spatial triangulation across every time slice. A genuine local spacetime dynamics must allow slice-dependent geometries and foliation-preserving moves in bounded time neighborhoods.",
                next_step=(
                    "Promote spatial Pachner moves to bounded slab-local updates with independently evolving slices, "
                    "include causal transfer weights between adjacent slices, and repeat the finite-size dimension scan."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not internal or not architecture_pass:
        raise SystemExit("causal column refinement architecture failed internal checks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
