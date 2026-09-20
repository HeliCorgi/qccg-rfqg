#!/usr/bin/env python3
"""Soft-graviton universality target audit.

At leading soft order the gauge variation is proportional to
  sum_i eta_i g_i p_i^mu.
Momentum conservation gives sum_i eta_i p_i^mu = 0.  A universal coupling
therefore cancels the soft gauge variation, while generic species-dependent
couplings do not.

This is a conditional consistency theorem/negative control.  It does not prove
that QCCG has already produced the required physical massless spin-2 S-matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


# 2 -> 2 massless kinematics; incoming eta=-1, outgoing eta=+1.
momenta = [
    sp.Matrix([1, 1, 0, 0]),
    sp.Matrix([1, -1, 0, 0]),
    sp.Matrix([1, 0, 1, 0]),
    sp.Matrix([1, 0, -1, 0]),
]
eta = [-1, -1, +1, +1]


def weighted_sum(couplings):
    out = sp.zeros(4, 1)
    for s, g, p in zip(eta, couplings, momenta):
        out += s * g * p
    return sp.simplify(out)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-soft-graviton-audit",
        "artifact": "qccg/run_soft_graviton_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    momentum_conservation = weighted_sum([1, 1, 1, 1])
    universal_pass = momentum_conservation == sp.zeros(4, 1)

    nonuniversal = [1, 2, 1, 1]
    bad = weighted_sum(nonuniversal)
    negative_detected = bad != sp.zeros(4, 1)

    # Symbolic two-species statement: if independent conserved momentum exchange
    # is allowed, species-dependent coefficients leave uncancelled weighted
    # momentum unless the couplings coincide.  The explicit kinematics above is
    # the concrete negative control.
    result = {
        "schema": 1,
        "scope": "conditional leading soft-graviton gauge-consistency target",
        "evidence": [
            evidence(
                "qccg-soft-universal-coupling",
                "SOFT_GRAVITON_UNIVERSALITY_TARGET",
                "PASS" if universal_pass else "FAIL",
                "For universal coupling, the leading soft-graviton gauge variation cancels by momentum conservation in the audited scattering kinematics.",
                momenta=[[str(x) for x in p] for p in momenta],
                eta=eta,
                universal_weighted_sum=[str(x) for x in momentum_conservation],
            ),
            evidence(
                "qccg-soft-nonuniversal-negative-control",
                "SOFT_NONUNIVERSAL_COUPLING_REJECTED",
                "PASS" if negative_detected else "FAIL",
                "A species-dependent coupling assignment leaves a nonzero weighted momentum and fails the leading soft gauge-consistency condition.",
                couplings=nonuniversal,
                weighted_sum=[str(x) for x in bad],
            ),
            evidence(
                "qccg-universal-coupling-physical-promotion",
                "UNIVERSAL_COUPLING_PHYSICAL",
                "OPEN",
                "Promotion to a QCCG physical prediction requires an actual massless spin-2 continuum S-matrix/soft limit; the present audit only verifies the conditional consistency target.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("soft-graviton audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
