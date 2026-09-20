#!/usr/bin/env python3
"""Cellulation-sector exact coarse-graining toy for QCCG.

Six fine cellulation states carry:
- a local-move graph Laplacian H_move;
- a curvature-like diagonal observable;
- a gap for pair-antisymmetric intra-block defect modes.

Three coarse states are uniform pair blocks.  The fine defect complement is
eliminated exactly by a Feshbach/Schur kernel.

The audit asks whether a narrow coarse ansatz
  I + t L_coarse + u C_coarse
is closed.  It is expected not to be: exact cellulation coarse graining
generates additional off-diagonal/diagonal structures and energy dependence.

This is still a finite cellulation toy, not the actual infinite QCCG RG flow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


SQ2 = sp.sqrt(2)
E = sp.symbols("E", real=True)
N_FINE = 6
N_COARSE = 3
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    (0, 3), (1, 4),
)
CURV_FINE = (0, 1, 2, 0, 1, 3)
T_MOVE = sp.Integer(1)
U_CURV = sp.Rational(1, 5)
DELTA_DEFECT = sp.Integer(4)


def laplacian(n, edges):
    L = sp.zeros(n)
    for a, b in edges:
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1
    return L


# Pair blocking: (0,1), (2,3), (4,5).
V = sp.zeros(N_FINE, N_COARSE)
W = sp.zeros(N_FINE, N_COARSE)
for c, (a, b) in enumerate(((0, 1), (2, 3), (4, 5))):
    V[a, c] = 1 / SQ2
    V[b, c] = 1 / SQ2
    W[a, c] = 1 / SQ2
    W[b, c] = -1 / SQ2

P = sp.simplify(V * V.T)
Q = sp.simplify(W * W.T)
T = V.row_join(W)

L_FINE = laplacian(N_FINE, EDGES)
C_FINE = sp.diag(*CURV_FINE)
H_FINE = sp.simplify(T_MOVE * L_FINE + U_CURV * C_FINE + DELTA_DEFECT * Q)

HT = sp.simplify(T.T * H_FINE * T)
A = HT[:N_COARSE, :N_COARSE]
B = HT[:N_COARSE, N_COARSE:]
D = HT[N_COARSE:, N_COARSE:]

# Narrow coarse ansatz: identity, 3-cycle Laplacian, simple coarse curvature.
L_COARSE = laplacian(3, ((0, 1), (1, 2), (2, 0)))
C_COARSE = sp.diag(0, 1, 2)
BASIS_NARROW = (sp.eye(3), L_COARSE, C_COARSE)


def heff(energy):
    return sp.simplify(A + B * (energy * sp.eye(3) - D).inv() * B.T)


def flatten(M):
    return sp.Matrix([M[i, j] for i in range(M.rows) for j in range(M.cols)])


def least_squares_fit(M, basis):
    X = sp.Matrix.hstack(*[flatten(Bi) for Bi in basis])
    y = flatten(M)
    coeff = sp.simplify((X.T * X).inv() * X.T * y)
    rec_vec = sp.simplify(X * coeff)
    rec = sp.Matrix(M.rows, M.cols, list(rec_vec))
    residual = sp.simplify(M - rec)
    return coeff, rec, residual


def full_symmetric_basis_3():
    out = []
    for i in range(3):
        M = sp.zeros(3)
        M[i, i] = 1
        out.append(M)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        M = sp.zeros(3)
        M[i, j] = 1
        M[j, i] = 1
        out.append(M)
    return tuple(out)


def max_abs_numeric(M):
    return max(abs(float(sp.N(x))) for x in M)


def zero(M):
    return M == sp.zeros(*M.shape)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-cellulation-rg-toy",
        "artifact": "qccg/run_cellulation_rg_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    orthonormal = sp.simplify(T.T * T) == sp.eye(6)
    rows = []
    narrow_rejected = True
    full_closed = True

    for energy in (sp.Rational(0), sp.Rational(1, 2)):
        M = heff(energy)
        coeff_n, rec_n, res_n = least_squares_fit(M, BASIS_NARROW)
        coeff_f, rec_f, res_f = least_squares_fit(M, full_symmetric_basis_3())
        narrow_err = max_abs_numeric(res_n)
        full_err = max_abs_numeric(res_f)
        narrow_rejected = narrow_rejected and narrow_err > 1.0e-5
        full_closed = full_closed and full_err < 1.0e-12
        rows.append({
            "E": str(energy),
            "H_eff": [[str(x) for x in row] for row in M.tolist()],
            "narrow_coefficients": [str(x) for x in coeff_n],
            "narrow_residual": [[str(x) for x in row] for row in res_n.tolist()],
            "narrow_max_abs_residual": narrow_err,
            "full_basis_coefficients": [str(x) for x in coeff_f],
            "full_basis_max_abs_residual": full_err,
        })

    M0 = heff(sp.Rational(0))
    M1 = heff(sp.Rational(1, 2))
    energy_diff = sp.simplify(M1 - M0)
    energy_dependent = not zero(energy_diff)

    # Exact determinant factorization in the block basis.
    lhs = sp.factor((E * sp.eye(6) - HT).det())
    rhs = sp.factor((E * sp.eye(3) - D).det() * (E * sp.eye(3) - heff(E)).det())
    feshbach_exact = sp.simplify(lhs - rhs) == 0

    defect_eigs = []
    defect_eig_imag_max = 0.0
    for x in D.eigenvals().keys():
        z = complex(sp.N(x, 30))
        defect_eig_imag_max = max(defect_eig_imag_max, abs(z.imag))
        defect_eigs.append(z.real)
    defect_eigs = sorted(defect_eigs)
    defect_gap = min(defect_eigs)
    gap_positive = defect_gap > 0 and defect_eig_imag_max < 1.0e-12

    result = {
        "schema": 1,
        "scope": "finite six-state cellulation-move/curvature coarse-graining toy",
        "evidence": [
            evidence(
                "qccg-cellulation-feshbach-rg",
                "CELLULATION_FESHBACH_RG_TOY",
                "PASS" if orthonormal and feshbach_exact and gap_positive else "FAIL",
                "Pair-block cellulation coarse graining is implemented with an orthonormal coarse/defect split and exact Feshbach determinant identity.",
                V=[[str(x) for x in row] for row in V.tolist()],
                W=[[str(x) for x in row] for row in W.tolist()],
                orthonormal_split=orthonormal,
                fine_edges=[list(x) for x in EDGES],
                curvature=list(CURV_FINE),
                defect_block_eigenvalues=defect_eigs,
                defect_eigenvalue_max_abs_imag=defect_eig_imag_max,
                defect_gap=defect_gap,
                feshbach_identity=feshbach_exact,
            ),
            evidence(
                "qccg-cellulation-narrow-basis-rejected",
                "CELLULATION_RESTRICTED_RG_BASIS_REJECTED",
                "PASS" if narrow_rejected else "FAIL",
                "The narrow I + move-Laplacian + simple-curvature coarse ansatz does not close under exact defect elimination.",
                rows=rows,
            ),
            evidence(
                "qccg-cellulation-full-basis-closure",
                "CELLULATION_FULL_OPERATOR_BASIS_CLOSURE_TOY",
                "PASS" if full_closed else "FAIL",
                "The complete real-symmetric three-state operator basis reconstructs the exact finite coarse kernel at the audited energies.",
                rows=rows,
            ),
            evidence(
                "qccg-cellulation-energy-dependent-kernel",
                "CELLULATION_ENERGY_DEPENDENT_KERNEL",
                "PASS" if energy_dependent else "FAIL",
                "Exact cellulation defect elimination generates an energy-dependent coarse kernel, reinforcing the need for full form-factor-like dependence.",
                difference_Ehalf_minus_E0=[[str(x) for x in row] for row in energy_diff.tolist()],
            ),
            evidence(
                "qccg-actual-cellulation-rg-flow",
                "QCCG_CELLULATION_NONPERTURBATIVE_RG",
                "OPEN",
                "This finite blocking toy does not yet define a compatible RG transformation across increasing QCCG cellulation ensembles or extract continuum essential beta functions.",
                next_step=(
                    "Construct nested QCCG-generated cellulation ensembles with compatible block maps, "
                    "track the generated full operator kernel across at least three scales, and test "
                    "for fixed-point/critical-exponent stability."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("cellulation RG toy audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
