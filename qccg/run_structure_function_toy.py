#!/usr/bin/env python3
"""Finite-dimensional structure-function constraint toy for QCCG.

This audit demonstrates that a finite Hilbert space can support:
- noncommuting constraints;
- operator-valued (here central) structure functions;
- exact Jacobi closure;
- a nontrivial common physical kernel;
- a positive constraint-penalty Hamiltonian with a gap;
- exact preservation of the physical kernel.

It is NOT the GR hypersurface-deformation algebra: the structure operator Q is
central in this toy and is not a dynamical inverse metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


I = sp.I
sx = sp.Matrix([[0, 1], [1, 0]]) / 2
sy = sp.Matrix([[0, -I], [I, 0]]) / 2
sz = sp.Matrix([[1, 0], [0, -1]]) / 2
Z1 = sp.zeros(1)


def block_diag(*blocks):
    return sp.diag(*blocks)


def comm(A, B):
    return sp.simplify(A * B - B * A)


s1 = sp.Integer(1)
s2 = sp.Integer(2)

Cx = block_diag(sp.zeros(1), s1 * sx, s2 * sx)
Cy = block_diag(sp.zeros(1), s1 * sy, s2 * sy)
Cz = block_diag(sp.zeros(1), s1 * sz, s2 * sz)
Q = block_diag(sp.zeros(1), s1 * sp.eye(2), s2 * sp.eye(2))

constraints = (Cx, Cy, Cz)


def null_intersection_dimension():
    stacked = sp.Matrix.vstack(*constraints)
    return len(stacked.nullspace()), stacked.nullspace()


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-structure-function-toy",
        "artifact": "qccg/run_structure_function_toy.py",
        "note": note,
        "metadata": metadata,
    }


def matrix_zero(A):
    return A == sp.zeros(*A.shape)


def main():
    closure = {
        "xy": matrix_zero(comm(Cx, Cy) - I * Q * Cz),
        "yz": matrix_zero(comm(Cy, Cz) - I * Q * Cx),
        "zx": matrix_zero(comm(Cz, Cx) - I * Q * Cy),
    }
    q_central = all(matrix_zero(comm(Q, C)) for C in constraints)

    jacobi = sp.simplify(
        comm(Cx, comm(Cy, Cz))
        + comm(Cy, comm(Cz, Cx))
        + comm(Cz, comm(Cx, Cy))
    )
    jacobi_pass = matrix_zero(jacobi)

    kernel_dim, kernel = null_intersection_dimension()
    kernel_pass = kernel_dim == 1 and kernel[0] == sp.Matrix([1, 0, 0, 0, 0])

    Hc = sp.simplify(Cx.H * Cx + Cy.H * Cy + Cz.H * Cz)
    h_commutes = all(matrix_zero(comm(Hc, C)) for C in constraints)
    h_kernel = Hc * kernel[0]
    spectrum = Hc.eigenvals()
    eigenvalues = []
    for value, multiplicity in spectrum.items():
        eigenvalues.extend([sp.simplify(value)] * int(multiplicity))
    eigenvalues = sorted(eigenvalues, key=lambda x: float(sp.N(x)))
    positive = [float(sp.N(x)) for x in eigenvalues if float(sp.N(x)) > 1e-12]
    gap = min(positive)
    gap_pass = gap > 0 and h_kernel == sp.zeros(5, 1)

    # Negative control: force a single sector-independent structure function.
    Qbad = block_diag(sp.zeros(1), sp.eye(2), sp.eye(2))
    bad_residual = sp.simplify(comm(Cx, Cy) - I * Qbad * Cz)
    negative_detected = not matrix_zero(bad_residual)

    toy_pass = all(closure.values()) and q_central and jacobi_pass and kernel_pass and h_commutes and gap_pass
    metadata = {
        "dimension": 5,
        "sector_scales": [str(s1), str(s2)],
        "closure": closure,
        "Q_commutes_with_constraints": q_central,
        "jacobi_zero": jacobi_pass,
        "common_kernel_dimension": kernel_dim,
        "common_kernel_basis": [[str(x) for x in v] for v in kernel],
        "constraint_penalty_matrix": [[str(x) for x in row] for row in Hc.tolist()],
        "constraint_penalty_eigenvalues": [str(x) for x in eigenvalues],
        "constraint_gap": gap,
        "penalty_commutes_with_constraints": h_commutes,
    }

    result = {
        "schema": 1,
        "scope": "finite-dimensional central structure-function closure toy",
        "evidence": [
            evidence(
                "qccg-structure-function-closure-toy",
                "STRUCTURE_FUNCTION_CLOSURE_TOY",
                "PASS" if toy_pass else "FAIL",
                "A five-dimensional finite Hilbert space realizes exact noncommuting constraint closure with an operator-valued central structure function, Jacobi identity, and a one-dimensional physical kernel.",
                **metadata,
            ),
            evidence(
                "qccg-constraint-kernel-gap-toy",
                "CONSTRAINT_KERNEL_GAP_TOY",
                "PASS" if gap_pass and h_commutes else "FAIL",
                "The positive constraint-penalty Hamiltonian has a nonzero gap above the common physical kernel and exactly preserves the constraint algebra in this toy.",
                **metadata,
            ),
            evidence(
                "qccg-wrong-structure-function-negative-control",
                "STRUCTURE_FUNCTION_NEGATIVE_CONTROL",
                "PASS" if negative_detected else "FAIL",
                "Replacing the operator-valued sector-dependent structure function by a single constant operator breaks the exact closure relation.",
                residual_nonzero=negative_detected,
                residual=[[str(x) for x in row] for row in bad_residual.tolist()],
            ),
            evidence(
                "qccg-gr-hda-structure-function-match",
                "GR_HDA_STRUCTURE_FUNCTION_MATCH",
                "OPEN",
                "The toy structure function is central. No finite-Weyl construction yet reproduces the dynamical inverse-metric structure functions and smearing dependence of the GR hypersurface-deformation algebra.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("structure-function toy audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
