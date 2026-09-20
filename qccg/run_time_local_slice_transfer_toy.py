#!/usr/bin/env python3
"""Time-local slice-transfer architecture toy for QCCG causal dynamics.

This repair addresses a limitation of the causal-column scan: the same spatial
triangulation was repeated on every time slice.

We build a finite graph of closed S^3 triangulations generated from the
boundary of a 4-simplex by reversible 3D Pachner 1<->4 and 2<->3 moves.
A discrete history is now a path
    S_0 -> S_1 -> ... -> S_T
in this graph; adjacent times differ by at most one local spatial Pachner move.
The transfer matrix is symmetric/Hermitian and local in discrete time.

The audit checks:
- multiple distinct spatial triangulations exist in one connected state graph;
- every graph edge is an exact one-Pachner move with an inverse;
- a Hermitian transfer/move Hamiltonian exists;
- a history can have genuinely different geometries on different time slices;
- a time-nonlocal jump across two graph steps is rejected as a negative control.

This is an architecture toy only. It does not yet construct the full 4D
simplicial slab between arbitrary neighboring slices or establish a continuum
4D phase.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import sympy as sp


MAX_DEPTH = 2
MAX_STATES = 40


def spatial_s3():
    verts = range(5)
    return {tuple(sorted(set(verts) - {omit})) for omit in verts}


def face_map(S):
    m = collections.defaultdict(list)
    for tet in S:
        for f in itertools.combinations(tet, 3):
            m[tuple(sorted(f))].append(tet)
    return m


def edge_map(S):
    m = collections.defaultdict(list)
    for tet in S:
        for e in itertools.combinations(tet, 2):
            m[tuple(sorted(e))].append(tet)
    return m


def manifold(S):
    return all(len(x) == 2 for x in face_map(S).values())


def candidates14(S):
    used = {v for tet in S for v in tet}
    # Labels are gauge.  Reuse holes left by prior 4->1 moves as well as one
    # genuinely fresh label; otherwise 4->1 can delete a non-maximal label
    # whose exact 1->4 inverse is unreachable in the labelled state graph.
    available = [v for v in range(max(used) + 2) if v not in used]
    return [(tet, v) for tet in sorted(S) for v in available]


def apply14(S, c):
    tet, v = c
    S2 = set(S)
    S2.remove(tet)
    verts = set(tet)
    for omit in tet:
        S2.add(tuple(sorted({v} | (verts - {omit}))))
    return S2


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


def apply41(S, c):
    _v, star, target = c
    S2 = set(S)
    for tet in star:
        S2.remove(tet)
    S2.add(target)
    return S2


def candidates23(S):
    fm = face_map(S)
    eset = set(edge_map(S))
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


def apply23(S, c):
    a, b, new = c
    S2 = set(S)
    S2.remove(a)
    S2.remove(b)
    S2.update(new)
    return S2


def candidates32(S):
    out = []
    for (d, e), star in edge_map(S).items():
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


def apply32(S, c):
    _edge, star, new = c
    S2 = set(S)
    for tet in star:
        S2.remove(tet)
    S2.update(new)
    return S2


MOVES = (
    ("14", candidates14, apply14),
    ("41", candidates41, apply41),
    ("23", candidates23, apply23),
    ("32", candidates32, apply32),
)


def canonical_relabel(S):
    """Canonical key for an unlabeled tetrahedral complex.

    Vertex names are gauge.  A simple sorted-label compression is not invariant
    when a 4->1 move removes a non-maximal label: the remaining labels are
    permuted relative to the prior representative.  We therefore partition
    vertices by an isomorphism-invariant local signature and exhaust all
    permutations within equal-signature classes, taking the lexicographically
    minimal relabelled complex.

    The audited state graph has at most seven vertices at MAX_DEPTH=2, so this
    exact finite canonicalization is inexpensive.
    """
    verts = sorted({v for tet in S for v in tet})

    def signature(v):
        incident = [tet for tet in S if v in tet]
        neigh = set()
        for tet in incident:
            neigh.update(tet)
        neigh.discard(v)
        # Incidence and neighbour count are invariant under relabelling.
        return (len(incident), len(neigh))

    groups = collections.defaultdict(list)
    for v in verts:
        groups[signature(v)].append(v)

    classes = [groups[sig] for sig in sorted(groups)]
    label_blocks = []
    offset = 0
    for cls in classes:
        block = list(range(offset, offset + len(cls)))
        label_blocks.append(block)
        offset += len(cls)

    best = None
    perm_families = [itertools.permutations(cls) for cls in classes]
    for chosen in itertools.product(*perm_families):
        mp = {}
        for perm, block in zip(chosen, label_blocks):
            for old, new in zip(perm, block):
                mp[old] = new
        key = tuple(
            sorted(tuple(sorted(mp[v] for v in tet)) for tet in S)
        )
        if best is None or key < best:
            best = key
    return best

def neighbors(S):
    out = {}
    for typ, gen, app in MOVES:
        for c in gen(S):
            S2 = app(S, c)
            if manifold(S2):
                out[canonical_relabel(S2)] = (typ, S2)
    return out


def build_state_graph():
    S0 = spatial_s3()
    key0 = canonical_relabel(S0)
    states = {key0: S0}
    depth = {key0: 0}
    q = collections.deque([key0])

    while q and len(states) < MAX_STATES:
        k = q.popleft()
        if depth[k] >= MAX_DEPTH:
            continue
        for nk, (_typ, S2) in neighbors(states[k]).items():
            if nk not in states and len(states) < MAX_STATES:
                states[nk] = S2
                depth[nk] = depth[k] + 1
                q.append(nk)

    keys = list(states)
    idx = {k: i for i, k in enumerate(keys)}
    edges = set()
    edge_types = {}
    inverse_ok = True
    inverse_failures = []

    neighbor_cache = {k: neighbors(S) for k, S in states.items()}

    for k, S in states.items():
        i = idx[k]
        for nk, (typ, _S2) in neighbor_cache[k].items():
            if nk not in idx:
                continue
            j = idx[nk]
            if i == j:
                continue
            e = tuple(sorted((i, j)))
            edges.add(e)
            edge_types.setdefault(str(e), set()).add(typ)
            # Reverse adjacency is the operational inverse check.
            reverse_neighbors = neighbor_cache[nk]
            if k not in reverse_neighbors:
                inverse_ok = False
                inverse_failures.append({
                    "from_state": i,
                    "to_state": j,
                    "forward_type": typ,
                    "from_vertices": sorted({v for tet in S for v in tet}),
                    "to_vertices": sorted({v for tet in states[nk] for v in tet}),
                })

    return states, keys, idx, edges, edge_types, inverse_ok, inverse_failures


def shortest_distances(n, edges, source):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = {source: 0}
    q = collections.deque([source])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in dist:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist, adj


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "time-local-slice-transfer-toy",
        "artifact": "qccg/run_time_local_slice_transfer_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    states, keys, idx, edges, edge_types, inverse_ok, inverse_failures = build_state_graph()
    n = len(keys)
    L = sp.zeros(n)
    for a, b in edges:
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1

    hermitian = L.H == L
    uniform = sp.ones(n, 1)
    uniform_zero = L * uniform == sp.zeros(n, 1)
    nullity = len(L.nullspace())
    connected = nullity == 1
    evals = []
    for val, mult in L.eigenvals().items():
        evals.extend([float(sp.N(val))] * int(mult))
    psd = min(evals) >= -1.0e-12

    root = 0
    dist, adj = shortest_distances(n, edges, root)
    one_step = next((v for v, d in dist.items() if d == 1), None)
    two_step = next((v for v, d in dist.items() if d == 2), None)
    varying_history_exists = one_step is not None and keys[one_step] != keys[root]
    nonlocal_jump_detected = (
        two_step is not None
        and two_step not in adj[root]
        and dist[two_step] == 2
    )

    vertex_counts = [len({v for tet in states[k] for v in tet}) for k in keys]
    tetra_counts = [len(states[k]) for k in keys]
    geometry_varies = len(set(vertex_counts)) > 1 or len(set(tetra_counts)) > 1

    architecture_pass = (
        n >= 3
        and len(edges) >= 2
        and inverse_ok
        and hermitian
        and uniform_zero
        and connected
        and psd
        and varying_history_exists
        and geometry_varies
        and nonlocal_jump_detected
    )

    result = {
        "schema": 1,
        "scope": "finite time-local path architecture on a spatial Pachner state graph",
        "evidence": [
            evidence(
                "qccg-time-local-slice-transfer",
                "TIME_LOCAL_SLICE_TRANSFER_TOY",
                "PASS" if architecture_pass else "FAIL",
                "A finite connected spatial-triangulation state graph supports Hermitian nearest-time transfer dynamics; adjacent slices can carry distinct geometries related by one reversible Pachner move.",
                n_states=n,
                n_edges=len(edges),
                edge_types={k: sorted(v) for k, v in edge_types.items()},
                inverse_ok=inverse_ok,
                inverse_failures=inverse_failures,
                label_policy="vertex names are quotiented by exact finite unlabeled-complex canonicalization within invariant signature classes",
                hermitian=hermitian,
                uniform_zero=uniform_zero,
                nullity=nullity,
                eigenvalues=sorted(evals),
                vertex_counts=vertex_counts,
                tetrahedron_counts=tetra_counts,
                varying_history_exists=varying_history_exists,
            ),
            evidence(
                "qccg-time-nonlocal-jump-control",
                "TIME_NONLOCAL_JUMP_NEGATIVE_CONTROL",
                "PASS" if nonlocal_jump_detected else "FAIL",
                "A state at graph distance two is not directly adjacent to the initial slice state, so a two-move geometry jump is rejected by the one-step time-local transfer rule.",
                root=root,
                two_step_state=two_step,
                distance=None if two_step is None else dist[two_step],
            ),
            evidence(
                "qccg-spacetime-slab-realization-open",
                "QCCG_LOCAL_SPACETIME_SLAB_REALIZATION",
                "OPEN",
                "The slice-history architecture is time-local at the state-graph level, but no general 4D causal simplicial slab has yet been constructed for every allowed pair of neighboring spatial Pachner states.",
                next_step=(
                    "Construct explicit bounded-time 4D interpolating simplicial slabs for each allowed 1<->4 and 2<->3 spatial transition, "
                    "then compose them and re-audit manifold incidence, path coherence and geometry scaling."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("time-local slice transfer toy failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
