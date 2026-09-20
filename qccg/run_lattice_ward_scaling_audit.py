#!/usr/bin/env python3
"""Generic lattice Ward-residual scaling audit for the QCCG tensor cubic.

For a local symmetric lattice derivative
    qhat(k) = sin(a k)/a,
continuum momentum conservation k1+k2+k3=0 does not imply
qhat(k1)+qhat(k2)+qhat(k3)=0 at finite a.

This is the kinematic source of finite-spacing gauge/Ward violations in the
engineered lattice-EH cubic.  The audit proves symbolically that the mismatch is
O(a^2), while a forward difference has an O(a) mismatch.

The repair criterion is therefore continuum restoration with a controlled
power law, not exact finite-a diffeomorphism invariance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


a, k1, k2 = sp.symbols("a k1 k2", real=True)
k3 = -k1 - k2

q_sym = lambda k: sp.sin(a * k) / a
q_fwd = lambda k: (sp.exp(sp.I * a * k) - 1) / (sp.I * a)

r_sym = sp.simplify(q_sym(k1) + q_sym(k2) + q_sym(k3))
r_fwd = sp.simplify(q_fwd(k1) + q_fwd(k2) + q_fwd(k3))

series_sym = sp.series(r_sym, a, 0, 6)
series_fwd = sp.series(r_fwd, a, 0, 4)

lead_sym_a2 = sp.simplify(sp.limit(r_sym / a**2, a, 0))
lead_sym_a1 = sp.simplify(sp.limit(r_sym / a, a, 0))
lead_fwd_a1 = sp.simplify(sp.limit(r_fwd / a, a, 0))

expected_sym = sp.simplify(k1 * k2 * (k1 + k2) / 2)
expected_fwd = sp.simplify(sp.I * (k1**2 + k2**2 + k3**2) / 2)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-lattice-ward-scaling",
        "artifact": "qccg/run_lattice_ward_scaling_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    sym_pass = (
        lead_sym_a1 == 0
        and sp.simplify(lead_sym_a2 - expected_sym) == 0
        and lead_sym_a2 != 0
    )
    fwd_pass = (
        lead_fwd_a1 != 0
        and sp.simplify(lead_fwd_a1 - expected_fwd) == 0
    )

    # Numerical scaling cross-check at generic conserved momenta.
    x1, x2 = 0.7, -1.1
    vals = []
    for av in (0.4, 0.2, 0.1, 0.05):
        x3 = -x1 - x2
        rs = (
            sp.N(r_sym.subs({a: av, k1: x1, k2: x2}))
        )
        rf = (
            sp.N(r_fwd.subs({a: av, k1: x1, k2: x2}))
        )
        vals.append({
            "a": av,
            "symmetric_abs": float(abs(complex(rs))),
            "forward_abs": float(abs(complex(rf))),
        })

    sym_ratios = [
        vals[i]["symmetric_abs"] / vals[i + 1]["symmetric_abs"]
        for i in range(len(vals) - 1)
    ]
    fwd_ratios = [
        vals[i]["forward_abs"] / vals[i + 1]["forward_abs"]
        for i in range(len(vals) - 1)
    ]
    # Halving a gives factor ~4 for O(a^2), ~2 for O(a).
    numeric_sym = all(3.6 <= x <= 4.4 for x in sym_ratios[-2:])
    numeric_fwd = all(1.8 <= x <= 2.2 for x in fwd_ratios[-2:])

    result = {
        "schema": 1,
        "scope": "generic componentwise momentum-conservation defect of lattice derivative symbols",
        "evidence": [
            evidence(
                "qccg-symmetric-ward-residual-a2",
                "LATTICE_WARD_RESIDUAL_OA2",
                "PASS" if sym_pass and numeric_sym else "FAIL",
                "For conserved continuum momenta, the symmetric lattice derivative violates additive momentum conservation only at O(a^2); this calibrates the expected finite-spacing Ward residual.",
                exact_residual=str(r_sym),
                series=str(series_sym),
                leading_over_a=str(lead_sym_a1),
                leading_over_a2=str(lead_sym_a2),
                expected_leading=str(expected_sym),
                numeric_rows=vals,
                halving_ratios=sym_ratios,
            ),
            evidence(
                "qccg-forward-difference-ward-control",
                "FORWARD_DIFFERENCE_WARD_OA_NEGATIVE_CONTROL",
                "PASS" if fwd_pass and numeric_fwd else "FAIL",
                "A forward-difference derivative has an O(a) momentum-additivity/Ward defect, demonstrating that the symmetric derivative's O(a^2) restoration is a nontrivial improvement.",
                exact_residual=str(r_fwd),
                series=str(series_fwd),
                leading_over_a=str(lead_fwd_a1),
                expected_leading=str(expected_fwd),
                numeric_rows=vals,
                halving_ratios=fwd_ratios,
            ),
            evidence(
                "qccg-finite-spacing-exact-diffeo-open",
                "FINITE_LATTICE_EXACT_DIFFEO_WARD",
                "OPEN",
                "The local finite-spacing lattice derivative does not satisfy exact continuum momentum additivity/Leibniz identities. QCCG therefore requires emergent continuum Ward restoration rather than exact microscopic diffeomorphism symmetry in this discretization.",
                repair=(
                    "Use O(a^2)-improved symmetric/local operators and require Ward residuals "
                    "to vanish under scaling; do not promote finite-a selected-kinematics "
                    "agreement to exact microscopic diffeomorphism invariance."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("lattice Ward scaling audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
