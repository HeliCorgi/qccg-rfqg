#!/usr/bin/env python3
"""QCCG -> RFQG bridge-observable extraction audit.

This does not run a gravitational FRG.  It extracts quantities that a future
QCCG/RFQG universality comparison must match and demonstrates a negative
control: identical infrared two-point data do not identify a unique UV
universality class or nonlinear theory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


k, a, b, lam = sp.symbols("k a b lam", positive=True, real=True)
dispersion = (2 / a * sp.sin(a * k / 2)) ** 2
series = sp.series(dispersion, k, 0, 8).removeO().expand()
k2_coeff = sp.expand(series).coeff(k, 2)
k4_coeff = sp.expand(series).coeff(k, 4)
k6_coeff = sp.expand(series).coeff(k, 6)

# Relative k^4 correction to the k^2 term.
relative4 = sp.simplify((k4_coeff * k**4) / (k2_coeff * k**2))
relative4_rescaled = sp.simplify(relative4.subs(k, k / b) / relative4)

# Cubic repair from the parent audit.
d, q1, q2, q3 = sp.symbols("d q1 q2 q3", positive=True, real=True)
h3 = lam / d**3 * (
    sp.sin(d * (q1 + q2 + q3))
    - sp.sin(d * q1)
    - sp.sin(d * q2)
    - sp.sin(d * q3)
)
mixed3 = sp.simplify(sp.diff(h3, q1, q2, q3).subs({q1: 0, q2: 0, q3: 0}))


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "symbolic-rg-bridge-audit",
        "artifact": "qccg/run_rg_bridge_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    pole_pass = k2_coeff == 1
    exponent_pass = sp.simplify(relative4_rescaled - b**-2) == 0
    cubic_pass = sp.simplify(mixed3 + lam) == 0

    # Two parent points with identical quadratic sector but different cubic data.
    parent_a = {"lambda": 0.05, "k2_coeff": 1.0, "k4_over_a2": -1.0 / 12.0, "mixed_cubic": -0.05}
    parent_b = {"lambda": 0.10, "k2_coeff": 1.0, "k4_over_a2": -1.0 / 12.0, "mixed_cubic": -0.10}
    same_two_point = (
        parent_a["k2_coeff"] == parent_b["k2_coeff"]
        and parent_a["k4_over_a2"] == parent_b["k4_over_a2"]
    )
    nonlinear_different = parent_a["mixed_cubic"] != parent_b["mixed_cubic"]
    insufficiency_detected = same_two_point and nonlinear_different

    result = {
        "schema": 1,
        "scope": "IR bridge-observable extraction; no nonperturbative gravitational FRG",
        "evidence": [
            evidence(
                "qccg-rg-bridge-observables",
                "RG_BRIDGE_OBSERVABLES_EXTRACTED",
                "PASS" if pole_pass and exponent_pass and cubic_pass else "FAIL",
                "The audited parent supplies explicit IR pole normalization, leading lattice correction exponent and a nonlinear cubic derivative for future universality matching.",
                dispersion_series=str(series),
                k2_coefficient=str(k2_coeff),
                k4_coefficient=str(k4_coeff),
                k6_coefficient=str(k6_coeff),
                relative_k4_scaling_under_k_to_k_over_b=str(relative4_rescaled),
                irrelevant_exponent=2,
                mixed_cubic_derivative=str(mixed3),
            ),
            evidence(
                "qccg-two-point-universality-negative-control",
                "IR_TWO_POINT_MATCH_INSUFFICIENT",
                "PASS" if insufficiency_detected else "FAIL",
                "Two parent models can share the same audited quadratic IR pole/correction data while carrying different cubic interactions, so two-point IR agreement cannot by itself establish RFQG/asymptotic-safety universality.",
                parent_a=parent_a,
                parent_b=parent_b,
                same_two_point_data=same_two_point,
                nonlinear_data_different=nonlinear_different,
            ),
            evidence(
                "qccg-nonperturbative-rg-flow",
                "QCCG_NONPERTURBATIVE_RG_FLOW",
                "OPEN",
                "No nonperturbative coarse-graining calculation yet produces beta functions for the essential QCCG continuum couplings.",
            ),
            evidence(
                "qccg-essential-critical-exponents",
                "ESSENTIAL_CRITICAL_EXPONENT_MATCH",
                "OPEN",
                "No QCCG UV critical exponents or essential relevant-direction spectrum have yet been computed for comparison with the RFQG-5/asymptotic-safety target.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("RG bridge audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
