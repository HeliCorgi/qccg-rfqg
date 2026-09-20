#!/usr/bin/env python3
"""Composition toy for local 4-simplex Pachner cobordisms.

Start with one 4-simplex realizing a spatial 1->4 move.  Then attach a second
4-simplex along one boundary tetrahedron, realizing a second 1->4 refinement.
The union is audited as a 4D simplicial manifold-with-boundary:
- the glued tetrahedron has incidence two;
- every other boundary tetrahedron has incidence one;
- the dual graph of 4-simplices is a tree, giving a causal attachment order;
- the final spatial boundary differs from the initial one by two local
  refinements.

A negative control attaches a third simplex along an already-internal
tetrahedron, producing incidence three and being rejected.

This establishes finite cobordism composition only, not a strict equal-time
global foliation for arbitrary move histories.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


S0 = (0, 1, 2, 3, 4)
S1 = (1, 2, 3, 4, 5)
GLUE = (1, 2, 3, 4)


def facets(simplex):
    return {
        tuple(sorted(f))
        for f in itertools.combinations(simplex, 4)
    }


def incidence(simplices):
    m = collections.defaultdict(list)
    for i, s in enumerate(simplices):
        for f in facets(s):
            m[f].append(i)
    return m


def dual_edges(simplices):
    inc = incidence(simplices)
    return {
        tuple(sorted(pair))
        for pair in inc.values()
        if len(pair) == 2
    }


def audit(simplices):
    inc = incidence(simplices)
    max_inc = max(len(v) for v in inc.values())
    internal = {f: v for f, v in inc.items() if len(v) == 2}
    boundary = {f: v for f, v in inc.items() if len(v) == 1}
    bad = {f: v for f, v in inc.items() if len(v) not in (1, 2)}
    edges = dual_edges(simplices)

    # For n simplices, a connected dual graph with n-1 edges is a tree.
    n = len(simplices)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    seen = set()
    stack = [0] if n else []
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(adj[v] - seen)
    dual_tree = len(seen) == n and len(edges) == max(0, n - 1)

    return {
        "n_4simplices": n,
        "max_tetrahedron_incidence": max_inc,
        "internal_tetrahedra": [list(f) for f in sorted(internal)],
        "boundary_tetrahedra": [list(f) for f in sorted(boundary)],
        "bad_tetrahedra": [
            {"tetrahedron": list(f), "incidence": len(v)}
            for f, v in sorted(bad.items())
        ],
        "dual_edges": [list(x) for x in sorted(edges)],
        "dual_tree": dual_tree,
        "manifold_with_boundary": not bad,
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "pachner-cobordism-composition-audit",
        "artifact": "qccg/run_pachner_cobordism_composition.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    good = audit((S0, S1))
    composition_pass = (
        good["manifold_with_boundary"]
        and good["dual_tree"]
        and GLUE in {tuple(x) for x in good["internal_tetrahedra"]}
        and len(good["internal_tetrahedra"]) == 1
    )

    # Bad: add a distinct simplex sharing the already-internal GLUE face,
    # forcing incidence three on that tetrahedron.
    Sbad = (1, 2, 3, 4, 6)
    bad = audit((S0, S1, Sbad))
    overglue_detected = not bad["manifold_with_boundary"] and bad["max_tetrahedron_incidence"] >= 3

    result = {
        "schema": 1,
        "scope": "finite sequential 1->4 local 4-simplex cobordism composition",
        "evidence": [
            evidence(
                "qccg-pachner-cobordism-composition",
                "PACHNER_COBORDISM_COMPOSITION_TOY",
                "PASS" if composition_pass else "FAIL",
                "Two sequential local 1->4 Pachner cobordism blocks glue along one tetrahedron to form a 4D simplicial manifold-with-boundary with a tree-like causal attachment graph.",
                first_simplex=list(S0),
                second_simplex=list(S1),
                glued_tetrahedron=list(GLUE),
                audit=good,
            ),
            evidence(
                "qccg-pachner-overglue-control",
                "PACHNER_COBORDISM_OVERGLUE_NEGATIVE_CONTROL",
                "PASS" if overglue_detected else "FAIL",
                "Attaching a third 4-simplex along an already-internal tetrahedron creates incidence three and is rejected.",
                bad_simplex=list(Sbad),
                audit=bad,
            ),
            evidence(
                "qccg-strict-foliation-composition-open",
                "QCCG_STRICT_FOLIATION_COMPOSITION",
                "OPEN",
                "Sequential local cobordisms compose as a 4D manifold-with-boundary, but a single strict equal-time labeling compatible with arbitrary mixed Pachner histories has not been established.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("Pachner cobordism composition audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
