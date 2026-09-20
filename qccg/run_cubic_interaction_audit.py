#!/usr/bin/env python3
"""Cubic-interaction repair audit for the QCCG finite-Weyl parent.

The original cosine-only parent has vanishing cubic derivatives around its flat
vacuum.  This script records that obstruction and tests a Hermitian three-body
finite-Weyl interaction whose linear and quadratic derivatives vanish while a
mixed cubic derivative is nonzero.

This is only a generic nonlinear-interaction repair.  It does not establish the
GR three-graviton Ward identities or asymptotic-safety matching.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import sympy as sp


K_VALUES = (7, 9, 11, 15)
LAMBDA_GOOD = 0.10
LAMBDA_BAD = 0.50


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-weyl-interaction-audit",
        "artifact": "qccg/run_cubic_interaction_audit.py",
        "note": note,
        "metadata": metadata,
    }


def symbolic():
    d, l = sp.symbols("d l", positive=True, real=True)
    q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)

    old = sum((2 - 2 * sp.cos(d * q)) / (2 * d**2) for q in (q1, q2, q3))
    h3 = l / d**3 * (
        sp.sin(d * (q1 + q2 + q3))
        - sp.sin(d * q1)
        - sp.sin(d * q2)
        - sp.sin(d * q3)
    )
    new = old + h3
    zero = {q1: 0, q2: 0, q3: 0}

    old_mixed3 = sp.simplify(sp.diff(old, q1, q2, q3).subs(zero))
    new_grad = [sp.simplify(sp.diff(new, q).subs(zero)) for q in (q1, q2, q3)]
    new_hessian = sp.Matrix(
        [[sp.simplify(sp.diff(new, a, b).subs(zero)) for b in (q1, q2, q3)]
         for a in (q1, q2, q3)]
    )
    h3_hessian = sp.Matrix(
        [[sp.simplify(sp.diff(h3, a, b).subs(zero)) for b in (q1, q2, q3)]
         for a in (q1, q2, q3)]
    )
    new_mixed3 = sp.simplify(sp.diff(new, q1, q2, q3).subs(zero))

    return {
        "old_mixed_third_derivative": str(old_mixed3),
        "new_gradient": [str(x) for x in new_grad],
        "new_hessian": [[str(x) for x in row] for row in new_hessian.tolist()],
        "interaction_hessian": [[str(x) for x in row] for row in h3_hessian.tolist()],
        "new_mixed_third_derivative": str(new_mixed3),
        "old_parent_cubic_absent": old_mixed3 == 0,
        "repair_preserves_stationary_point": all(x == 0 for x in new_grad),
        "repair_preserves_quadratic_hessian": h3_hessian == sp.zeros(3),
        "repair_has_cubic_vertex": sp.simplify(new_mixed3 + l) == 0,
    }


def energy(K, lam, ns):
    d = 2.0 * math.pi / K
    qs = [d * n for n in ns]
    h2 = sum((2.0 - 2.0 * math.cos(q)) / (2.0 * d * d) for q in qs)
    h3 = lam / d**3 * (
        math.sin(sum(qs)) - sum(math.sin(q) for q in qs)
    )
    return h2 + h3


def scan(K, lam):
    best_e = float("inf")
    minima = []
    for ns in itertools.product(range(K), repeat=3):
        e = energy(K, lam, ns)
        if e < best_e - 1.0e-12:
            best_e = e
            minima = [ns]
        elif abs(e - best_e) <= 1.0e-12:
            minima.append(ns)
    return {
        "K": K,
        "lambda": lam,
        "minimum_energy": best_e,
        "minima": [list(x) for x in minima],
        "flat_vacuum_unique": minima == [(0, 0, 0)],
    }


def run():
    sym = symbolic()
    old_obstruction = sym["old_parent_cubic_absent"]
    repair_symbolic = (
        sym["repair_preserves_stationary_point"]
        and sym["repair_preserves_quadratic_hessian"]
        and sym["repair_has_cubic_vertex"]
    )

    good = [scan(K, LAMBDA_GOOD) for K in K_VALUES]
    good_stable = all(r["flat_vacuum_unique"] for r in good)

    bad = scan(11, LAMBDA_BAD)
    instability_detected = not bad["flat_vacuum_unique"]

    return {
        "schema": 1,
        "scope": "finite-Weyl cubic nonlinearity repair, not GR vertex matching",
        "evidence": [
            evidence(
                "qccg-cosine-parent-cubic-obstruction",
                "CUBIC_VERTEX_OBSTRUCTION_DETECTED",
                "PASS" if old_obstruction else "FAIL",
                "The cosine-only parent has zero mixed third derivative at the flat vacuum and cannot by itself supply a cubic collective vertex there.",
                symbolic=sym,
            ),
            evidence(
                "qccg-cubic-weyl-repair",
                "GENERIC_CUBIC_INTERACTION_REPAIR",
                "PASS" if repair_symbolic and good_stable else "FAIL",
                "A Hermitian finite-Weyl three-body term supplies a nonzero cubic derivative while leaving the stationary point and quadratic Hessian unchanged; the preregistered weak coupling preserves the unique flat vacuum on the audited K values.",
                symbolic=sym,
                lambda_value=LAMBDA_GOOD,
                scans=good,
            ),
            evidence(
                "qccg-cubic-strong-coupling-negative-control",
                "CUBIC_VACUUM_STABILITY_CONTROL",
                "PASS" if instability_detected else "FAIL",
                "A stronger cubic coupling shifts the global minimum away from the flat vacuum, demonstrating that the nonlinear repair has a genuine stability bound rather than being automatically harmless.",
                scan=bad,
            ),
            evidence(
                "qccg-gr-cubic-ward-identity",
                "GR_CUBIC_VERTEX_MATCH",
                "OPEN",
                "The repaired generic cubic interaction has not been shown to satisfy the nonlinear diffeomorphism/soft-graviton Ward identities of the GR three-graviton vertex.",
            ),
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = run()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [e for e in result["evidence"] if e["status"] == "FAIL"]
    if failed:
        raise SystemExit("cubic-interaction audit failed: " + ", ".join(e["obligation"] for e in failed))


if __name__ == "__main__":
    main()
