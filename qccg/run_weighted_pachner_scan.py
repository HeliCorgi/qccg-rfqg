#!/usr/bin/env python3
"""Finite Regge-like weighted Pachner repair scan for QCCG.

This is the first repair attempt after the unweighted 4D Pachner ensemble failed
the metric-dimension diagnostics.

We sample closed 4D triangulations with reversible 1<->5 and 2<->4 Pachner
moves using a Metropolis-Hastings weight

    S_toy = -kappa2 * N2 + eps * (N4 - N4_target)^2

where N2 is the triangle count and N4 the number of 4-simplices.

The scan is finite-volume and deliberately modest.  A failure to find a 4D
plateau is a scoped negative result, not a theorem excluding a critical point.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
from pathlib import Path


KAPPA2_VALUES = (-0.5, 0.0, 0.5)
STEPS = 800
TARGET_N4 = 120
VOLUME_EPS = 0.2
INITIAL_1_TO_5 = 20
ROOTS = 24
T1, T2 = 4, 8
R1, R2 = 2, 4
BROAD_4D_MIN = 3.0
SEED = 20260920


def boundary_5_simplex():
    v = range(6)
    return {tuple(sorted(set(v) - {omit})) for omit in v}


def face_map(S):
    out = collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s, 4):
            out[tuple(sorted(f))].append(s)
    return out


def edge_map(S):
    out = collections.defaultdict(list)
    for s in S:
        for e in itertools.combinations(s, 2):
            out[tuple(sorted(e))].append(s)
    return out


def edges(S):
    return set(edge_map(S))


def refine_1_to_5(S, simplex, newv):
    verts = set(simplex)
    S.remove(simplex)
    for omit in simplex:
        S.add(tuple(sorted({newv} | (verts - {omit}))))


def candidates_5_to_1(S):
    by_v = collections.defaultdict(list)
    for s in S:
        for v in s:
            by_v[v].append(s)
    out = []
    for v, star in by_v.items():
        if len(star) != 5:
            continue
        neigh = set().union(*(set(s) - {v} for s in star))
        if len(neigh) != 5:
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


def apply_5_to_1(S, cand):
    _v, star, target = cand
    for s in star:
        S.remove(s)
    S.add(target)


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
    s1, s2, _face, _a, _b, new = cand
    S.remove(s1)
    S.remove(s2)
    S.update(new)


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
    _edge, star, new = cand
    for s in star:
        S.remove(s)
    S.update(new)


def n2_count(S):
    tris = set()
    for s in S:
        tris.update(tuple(sorted(t)) for t in itertools.combinations(s, 3))
    return len(tris)


def action(S, kappa2):
    n4 = len(S)
    n2 = n2_count(S)
    return -kappa2 * n2 + VOLUME_EPS * (n4 - TARGET_N4) ** 2, n2, n4


def candidate_count(S, typ):
    if typ == "15":
        return len(S)
    if typ == "51":
        return len(candidates_5_to_1(S))
    if typ == "24":
        return len(candidates_2_to_4(S))
    if typ == "42":
        return len(candidates_4_to_2(S))
    raise ValueError(typ)


def propose(S, typ, rng, nextv):
    S2 = set(S)
    if typ == "15":
        simplex = rng.choice(sorted(S2))
        refine_1_to_5(S2, simplex, nextv)
        return S2, nextv + 1, "51"

    if typ == "51":
        cs = candidates_5_to_1(S2)
        if not cs:
            return None, nextv, None
        apply_5_to_1(S2, rng.choice(cs))
        return S2, nextv, "15"

    if typ == "24":
        cs = candidates_2_to_4(S2)
        if not cs:
            return None, nextv, None
        apply_2_to_4(S2, rng.choice(cs))
        return S2, nextv, "42"

    if typ == "42":
        cs = candidates_4_to_2(S2)
        if not cs:
            return None, nextv, None
        apply_4_to_2(S2, rng.choice(cs))
        return S2, nextv, "24"

    raise ValueError(typ)


def manifold_ok(S):
    counts = [len(x) for x in face_map(S).values()]
    return all(x == 2 for x in counts)


def dual_adj(S):
    fm = face_map(S)
    adj = {s: set() for s in S}
    for pair in fm.values():
        if len(pair) == 2:
            a, b = pair
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


def initial(seed):
    rng = random.Random(seed)
    S = boundary_5_simplex()
    nextv = 6
    for _ in range(INITIAL_1_TO_5):
        refine_1_to_5(S, rng.choice(sorted(S)), nextv)
        nextv += 1
    return S, nextv


def sample(kappa2, seed):
    rng = random.Random(seed)
    S, nextv = initial(seed)
    types = ("15", "51", "24", "42")
    accepted = 0
    attempted = 0
    topology_rejected_proposals = 0

    for _ in range(STEPS):
        typ = rng.choice(types)
        nf = candidate_count(S, typ)
        if nf == 0:
            continue
        attempted += 1
        S2, nextv2, reverse = propose(S, typ, rng, nextv)
        if S2 is None:
            continue
        # Candidate enumeration is intentionally treated as permissive.
        # A purported inverse Pachner pattern is not trusted until the
        # proposed complex itself passes the closed-manifold incidence audit.
        # Invalid proposals become self-loops and never enter the MH ratio.
        if not manifold_ok(S2):
            topology_rejected_proposals += 1
            continue
        nr = candidate_count(S2, reverse)
        if nr == 0:
            continue

        oldS, _oldn2, _oldn4 = action(S, kappa2)
        newS, _newn2, _newn4 = action(S2, kappa2)
        log_alpha = -(newS - oldS) + math.log(nf / nr)
        if math.log(rng.random()) < min(0.0, log_alpha):
            S = S2
            nextv = nextv2
            accepted += 1

    act, n2, n4 = action(S, kappa2)
    ds, dv = dimensions(S)
    return {
        "kappa2": kappa2,
        "attempted": attempted,
        "accepted": accepted,
        "acceptance_fraction": accepted / max(1, attempted),
        "topology_rejected_proposals": topology_rejected_proposals,
        "N2": n2,
        "N4": n4,
        "action": act,
        "closed_manifold": manifold_ok(S),
        "spectral_dimension": ds,
        "volume_growth_dimension": dv,
        "broad_4d_candidate": ds >= BROAD_4D_MIN and dv >= BROAD_4D_MIN,
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "finite-regge-pachner-scan",
        "artifact": "qccg/run_weighted_pachner_scan.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    rows = [
        sample(k, SEED + i * 101)
        for i, k in enumerate(KAPPA2_VALUES)
    ]
    topology_ok = all(r["closed_manifold"] for r in rows)
    no_plateau = all(not r["broad_4d_candidate"] for r in rows)

    result = {
        "schema": 1,
        "scope": "finite-volume toy Regge-weighted reversible Pachner scan; not an equilibrium proof or CDT calculation",
        "evidence": [
            evidence(
                "qccg-regge-weighted-pachner-scan",
                "REGGE_WEIGHTED_PACHNER_SCAN_EXECUTED",
                "PASS" if topology_ok else "FAIL",
                "A reversible 1<->5 and 2<->4 Metropolis-Hastings scan with a simple Regge-like N2 weight and volume fixing is executed while preserving closed 4-manifold incidence.",
                kappa2_values=list(KAPPA2_VALUES),
                steps=STEPS,
                target_N4=TARGET_N4,
                volume_epsilon=VOLUME_EPS,
                rows=rows,
                topology_guard=(
                    "Every proposed move is revalidated by tetrahedral-face incidence "
                    "before the Metropolis-Hastings ratio; invalid inverse-move patterns "
                    "are rejected as self-loops."
                ),
            ),
            evidence(
                "qccg-regge-weight-no-4d-plateau",
                "SIMPLE_REGGE_WEIGHT_REPAIR_INSUFFICIENT",
                "PASS" if no_plateau else "FAIL",
                "Within the preregistered finite scan, no simple N2-weighted ensemble reaches even the broad ds,dV >= 3 four-dimensional target.",
                broad_4d_min=BROAD_4D_MIN,
                rows=rows,
            ),
            evidence(
                "qccg-causal-cellulation-repair-open",
                "QCCG_CAUSAL_CRITICAL_CELLULATION_DYNAMICS",
                "OPEN",
                "The simple Euclidean Regge-like weighting is insufficient at the audited finite scope. The next repair must add stronger geometric/causal structure (for example a causal/foliated move sector) and demonstrate a stable 4D scaling phase.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("weighted Pachner scan audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
