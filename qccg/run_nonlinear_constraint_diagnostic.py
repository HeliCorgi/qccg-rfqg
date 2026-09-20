#!/usr/bin/env python3
"""Diagnostic rejecting a naive commuting-projector nonlinear GR completion.

The GR hypersurface-deformation algebra has
  {H[N1], H[N2]} = D[q^{ij}(N1 d_j N2 - N2 d_j N1)]
for generic fields/smearings.  A nonlinear completion that insists all
Hamiltonian-constraint projectors commute would force the left side to zero and
cannot reproduce the generic target.

This audit establishes only a design constraint: nonlinear QCCG constraints
must be allowed to be noncommuting and to close with operator/field-dependent
structure functions.  It does not construct that algebra.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


x, q = sp.symbols("x q", real=True, nonzero=True)
N1 = x
N2 = x**2
rhs_shift = sp.simplify(q * (N1 * sp.diff(N2, x) - N2 * sp.diff(N1, x)))
naive_commuting_lhs = sp.Integer(0)
mismatch = sp.simplify(rhs_shift - naive_commuting_lhs)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-hda-diagnostic",
        "artifact": "qccg/run_nonlinear_constraint_diagnostic.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    target_nonabelian = sp.simplify(rhs_shift) != 0
    naive_rejected = mismatch != 0

    result = {
        "schema": 1,
        "scope": "nonlinear HDA design negative control",
        "diagnosis": {
            "lapse_1": str(N1),
            "lapse_2": str(N2),
            "q_xx": str(q),
            "target_shift_smearing": str(rhs_shift),
            "naive_commuting_projector_lhs": str(naive_commuting_lhs),
            "mismatch": str(mismatch),
            "repair_rule": (
                "Do not require mutually commuting nonlinear Hamiltonian-constraint "
                "projectors.  Use constraint generators C_A whose commutators may "
                "close as [C_A,C_B]=F_AB^C(Q) C_C with operator-valued structure "
                "functions, and audit anomaly-free preservation of the physical kernel."
            ),
        },
        "evidence": [
            evidence(
                "qccg-hda-nonabelian-target",
                "HDA_NONABELIAN_TARGET",
                "PASS" if target_nonabelian else "FAIL",
                "A generic pair of lapse smearings gives a nonzero hypersurface-deformation shift target.",
                rhs_shift=str(rhs_shift),
            ),
            evidence(
                "qccg-reject-commuting-projector-hda",
                "COMMUTING_PROJECTOR_HDA_REJECTED",
                "PASS" if naive_rejected else "FAIL",
                "The naive mutually commuting Hamiltonian-projector completion fails the generic HDA target and is rejected.",
                mismatch=str(mismatch),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("nonlinear constraint diagnostic did not detect the expected obstruction")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
