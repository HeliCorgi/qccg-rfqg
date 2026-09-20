#!/usr/bin/env python3
"""Cellulation-superposition vacuum audit for QCCG.

H_move = sum_(Gamma~Gamma') (|Gamma>-|Gamma'>)(<Gamma|-<Gamma'|)
is the graph Laplacian of the local-move graph.  For a connected move graph,
the uniform superposition is the unique zero-energy state.

This finite toy audit checks that algebra and a disconnected negative control.
It does not establish manifold-likeness or a Lorentzian continuum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


N = 6
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    (0, 3), (1, 4),
)
DISCONNECTED_EDGES = ((0, 1), (1, 2), (3, 4), (4, 5))


def laplacian(n, edges):
    L = sp.zeros(n)
    for a, b in edges:
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1
    return L


def spectrum_info(L):
    eig = L.eigenvals()
    vals = []
    for value, multiplicity in eig.items():
        vals.extend([sp.simplify(value)] * int(multiplicity))
    vals = sorted(vals, key=lambda x: float(sp.N(x)))
    nullity = len(L.nullspace())
    positives = [float(sp.N(v)) for v in vals if float(sp.N(v)) > 1e-12]
    gap = min(positives) if positives else 0.0
    return vals, nullity, gap


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-cellulation-audit",
        "artifact": "qccg/run_cellulation_liquid_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    L = laplacian(N, EDGES)
    vals, nullity, gap = spectrum_info(L)
    uniform = sp.ones(N, 1)
    uniform_residual = L * uniform
    main_pass = nullity == 1 and uniform_residual == sp.zeros(N, 1) and gap > 0

    L_bad = laplacian(N, DISCONNECTED_EDGES)
    _, bad_nullity, bad_gap = spectrum_info(L_bad)
    negative_detected = bad_nullity > 1

    result = {
        "schema": 1,
        "scope": "finite cellulation-move graph toy",
        "evidence": [
            evidence(
                "qccg-cellulation-liquid-zero-mode",
                "CELLULATION_LIQUID_ZERO_MODE",
                "PASS" if main_pass else "FAIL",
                "For the connected audited move graph, H_move is positive semidefinite with a unique uniform zero-energy cellulation superposition.",
                n_cellulations=N,
                edges=[list(x) for x in EDGES],
                eigenvalues=[str(x) for x in vals],
                nullity=nullity,
                spectral_gap=gap,
                uniform_residual=[str(x) for x in uniform_residual],
            ),
            evidence(
                "qccg-cellulation-connectivity-negative-control",
                "CELLULATION_CONNECTIVITY_CONTROL",
                "PASS" if negative_detected else "FAIL",
                "Disconnecting the move graph produces multiple zero modes, so the audit detects failure of a unique liquid vacuum.",
                disconnected_edges=[list(x) for x in DISCONNECTED_EDGES],
                nullity=bad_nullity,
                spectral_gap=bad_gap,
            ),
            evidence(
                "qccg-manifold-likeness-scaling",
                "MANIFOLD_LIKENESS_SCALING",
                "OPEN",
                "No scaling study yet establishes that the cellulation liquid concentrates on smooth 3+1 Lorentzian manifold-like geometries.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("cellulation-liquid audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
