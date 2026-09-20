#!/usr/bin/env python3
"""Derivative-cubic refinement for the QCCG finite-Weyl parent.

The earlier generic three-body repair proves that a cubic interaction can be
added without changing the quadratic Hessian, but its leading continuum term is
non-derivative and is not an acceptable proxy for the GR three-graviton
structure.

This audit constructs a link-based Hermitian finite-Weyl candidate:
  S_avg(q1,q2) * L(q1-q2)
where S is an odd Weyl phase function and L is the positive link cosine energy.
Its leading continuum term is proportional to
  (q1+q2) (q1-q2)^2,
i.e. a field times two lattice derivatives.  This is structurally closer to
h (partial h)^2 while still not constituting the full tensor GR vertex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


d, lam = sp.symbols("d lam", positive=True, real=True)
q1, q2 = sp.symbols("q1 q2", real=True)
zero = {q1: 0, q2: 0}

S1 = sp.sin(d * q1) / d
S2 = sp.sin(d * q2) / d
Savg = (S1 + S2) / 2
L12 = (2 - 2 * sp.cos(d * (q1 - q2))) / (2 * d**2)
H3d = sp.simplify(lam * Savg * L12)

# Earlier local/non-derivative cubic repair restricted to a uniform 3-field mode.
q = sp.symbols("q", real=True)
H3local_uniform = lam / d**3 * (sp.sin(3 * d * q) - 3 * sp.sin(d * q))


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-derivative-cubic-audit",
        "artifact": "qccg/run_derivative_cubic_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    grad = [sp.simplify(sp.diff(H3d, z).subs(zero)) for z in (q1, q2)]
    hessian = sp.Matrix(
        [[sp.simplify(sp.diff(H3d, x, y).subs(zero)) for y in (q1, q2)]
         for x in (q1, q2)]
    )
    third = {
        "111": sp.simplify(sp.diff(H3d, q1, q1, q1).subs(zero)),
        "112": sp.simplify(sp.diff(H3d, q1, q1, q2).subs(zero)),
        "122": sp.simplify(sp.diff(H3d, q1, q2, q2).subs(zero)),
        "222": sp.simplify(sp.diff(H3d, q2, q2, q2).subs(zero)),
    }
    series_eps = sp.symbols("eps", real=True)
    continuum_series = sp.series(
        H3d.subs({q1: series_eps * q1, q2: series_eps * q2}),
        series_eps, 0, 5
    )
    uniform_derivative = sp.simplify(H3d.subs(q2, q1))
    uniform_local = sp.simplify(H3local_uniform)
    local_nonzero_generic = uniform_local != 0

    preserves_quadratic = all(x == 0 for x in grad) and hessian == sp.zeros(2)
    has_cubic = any(v != 0 for v in third.values())
    uniform_mode_decouples = uniform_derivative == 0

    leading_expected = sp.expand(lam * (q1 + q2) * (q1 - q2)**2 / 4)
    cubic_coeff = sp.expand(
        sp.limit(H3d.subs(d, sp.symbols("dd", positive=True)), sp.symbols("dd", positive=True), 0)
    )
    # Direct d->0 limit should equal the derivative-cubic expression.
    limit_expr = sp.simplify(sp.limit(H3d, d, 0, dir="+"))
    continuum_match = sp.simplify(limit_expr - leading_expected) == 0

    result = {
        "schema": 1,
        "scope": "derivative-cubic finite-Weyl structural repair; not full GR vertex",
        "evidence": [
            evidence(
                "qccg-nonderivative-cubic-mismatch",
                "NONDERIVATIVE_CUBIC_MISMATCH_DETECTED",
                "PASS" if local_nonzero_generic else "FAIL",
                "The earlier local cubic repair acts on a spatially uniform collective mode, so it is not a two-derivative GR-like cubic structure.",
                uniform_local_expression=str(uniform_local),
            ),
            evidence(
                "qccg-derivative-cubic-repair",
                "DERIVATIVE_CUBIC_REPAIR",
                "PASS" if preserves_quadratic and has_cubic and uniform_mode_decouples and continuum_match else "FAIL",
                "A Hermitian link-based Weyl interaction has zero linear/quadratic variation, nonzero cubic variation, vanishes on uniform modes, and tends to a field-times-two-derivatives cubic structure.",
                expression=str(H3d),
                gradient=[str(x) for x in grad],
                hessian=[[str(x) for x in row] for row in hessian.tolist()],
                third_derivatives={k: str(v) for k, v in third.items()},
                d_to_zero_limit=str(limit_expr),
                expected_limit=str(leading_expected),
                uniform_mode_value=str(uniform_derivative),
                epsilon_series=str(continuum_series),
            ),
            evidence(
                "qccg-full-gr-three-graviton-ward",
                "GR_CUBIC_VERTEX_MATCH",
                "OPEN",
                "The derivative-cubic repair has not been lifted to the full tensor index structure or shown to satisfy nonlinear diffeomorphism/soft-graviton Ward identities.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("derivative cubic audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
