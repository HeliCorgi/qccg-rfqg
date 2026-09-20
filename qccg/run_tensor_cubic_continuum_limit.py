#!/usr/bin/env python3
"""Operator-level continuum-limit audit for the engineered QCCG tensor cubic.

At the microscopic level the collective field coordinate is represented by
  S_delta(eps h) = sin(delta eps h)/delta,
and a symmetric local derivative by
  D_a(k) = sin(a k)/a.

For a generic cubic monomial of the EH type h (partial h)(partial h), this
audit extracts the coefficient cubic in the field-amplitude bookkeeping
parameter eps.  It shows:

1. finite-Weyl sine coordinates do not alter the cubic coefficient at all;
   their first correction enters at quintic field order;
2. the symmetric derivative changes the generic cubic momentum kernel only at
   O(a^2);
3. a forward derivative gives an O(a) correction as a negative control.

This is an operator-structure continuum-limit statement.  It does not derive
the EH tensor coefficients from QCCG RG and does not prove the full nonlinear
Ward identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


eps, delta, a = sp.symbols("eps delta a", real=True)
h1, h2, h3 = sp.symbols("h1 h2 h3", real=True)
k2, k3 = sp.symbols("k2 k3", real=True)

S = lambda h: sp.sin(delta * eps * h) / delta
Dsym = lambda k: sp.sin(a * k) / a
Dfwd = lambda k: (sp.exp(sp.I * a * k) - 1) / (sp.I * a)

mono_sym = sp.expand(S(h1) * Dsym(k2) * S(h2) * Dsym(k3) * S(h3))
mono_fwd = sp.expand(S(h1) * Dfwd(k2) * S(h2) * Dfwd(k3) * S(h3))

# Extract the cubic field-amplitude coefficient exactly by differentiation.
cub_sym = sp.simplify(sp.diff(mono_sym, eps, 3).subs(eps, 0) / sp.factorial(3))
cub_fwd = sp.simplify(sp.diff(mono_fwd, eps, 3).subs(eps, 0) / sp.factorial(3))
continuum = sp.simplify(h1 * h2 * h3 * k2 * k3)

sym_series = sp.series(cub_sym - continuum, a, 0, 5)
fwd_series = sp.series(cub_fwd - continuum, a, 0, 3)

sym_over_a = sp.simplify(sp.limit((cub_sym - continuum) / a, a, 0))
sym_over_a2 = sp.simplify(sp.limit((cub_sym - continuum) / a**2, a, 0))
fwd_over_a = sp.simplify(sp.limit((cub_fwd - continuum) / a, a, 0))

expected_sym = sp.simplify(
    -h1 * h2 * h3 * (k2**3 * k3 + k2 * k3**3) / 6
)
expected_fwd = sp.simplify(
    sp.I * h1 * h2 * h3 * (k2**2 * k3 + k2 * k3**2) / 2
)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-tensor-cubic-continuum-limit",
        "artifact": "qccg/run_tensor_cubic_continuum_limit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    delta_independent = not cub_sym.has(delta)
    sym_pass = (
        delta_independent
        and sym_over_a == 0
        and sp.simplify(sym_over_a2 - expected_sym) == 0
    )
    fwd_pass = (
        fwd_over_a != 0
        and sp.simplify(fwd_over_a - expected_fwd) == 0
    )

    # Explicit field-order check: one sine-coordinate correction starts at eps^3,
    # hence replacing one of three linear fields first modifies eps^5 overall.
    s_series = sp.series(S(h1), eps, 0, 6)
    triple_series = sp.series(S(h1) * S(h2) * S(h3), eps, 0, 6)
    cubic_field_coeff = sp.simplify(
        sp.diff(S(h1) * S(h2) * S(h3), eps, 3).subs(eps, 0) / sp.factorial(3)
    )
    quintic_has_delta = sp.expand(triple_series.removeO()).coeff(eps, 5).has(delta)
    field_order_pass = (
        cubic_field_coeff == h1 * h2 * h3
        and quintic_has_delta
    )

    result = {
        "schema": 1,
        "scope": "generic h(dh)(dh) cubic monomial continuum structure; not full tensor Ward/RG derivation",
        "evidence": [
            evidence(
                "qccg-weyl-cubic-delta-independence",
                "FINITE_WEYL_CUBIC_COEFFICIENT_EXACT",
                "PASS" if delta_independent and field_order_pass else "FAIL",
                "The finite-Weyl sine coordinate leaves the cubic field coefficient exactly equal to the continuum field product; finite-delta corrections first enter at quintic field order.",
                sine_series=str(s_series),
                triple_field_series=str(triple_series),
                cubic_field_coefficient=str(cubic_field_coeff),
                cubic_kernel=str(cub_sym),
                delta_independent=delta_independent,
            ),
            evidence(
                "qccg-tensor-cubic-operator-a2",
                "TENSOR_CUBIC_OPERATOR_CONTINUUM_OA2",
                "PASS" if sym_pass else "FAIL",
                "For a generic h(dh)(dh) cubic monomial, symmetric local derivatives reproduce the continuum cubic momentum kernel with leading O(a^2) correction.",
                lattice_cubic_kernel=str(cub_sym),
                continuum_kernel=str(continuum),
                difference_series=str(sym_series),
                leading_over_a=str(sym_over_a),
                leading_over_a2=str(sym_over_a2),
                expected_leading=str(expected_sym),
            ),
            evidence(
                "qccg-tensor-cubic-forward-control",
                "TENSOR_CUBIC_FORWARD_OA_NEGATIVE_CONTROL",
                "PASS" if fwd_pass else "FAIL",
                "Replacing the symmetric derivative by a forward derivative worsens the generic cubic operator error to O(a), providing an operator-level discretization negative control.",
                forward_kernel=str(cub_fwd),
                difference_series=str(fwd_series),
                leading_over_a=str(fwd_over_a),
                expected_leading=str(expected_fwd),
            ),
            evidence(
                "qccg-cubic-rg-selection-open",
                "QCCG_CUBIC_RG_DERIVATION",
                "OPEN",
                "The engineered EH tensor coefficients are still inserted as the continuum target. No microscopic QCCG coarse-graining calculation has selected these cubic coefficients from a generic finite-Weyl interaction basis.",
            ),
        ],
    }

    p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("tensor cubic continuum-limit audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
