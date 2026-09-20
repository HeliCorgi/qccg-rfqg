#!/usr/bin/env python3
"""Exact Feshbach coarse-graining toy for the QCCG -> RFQG bridge.

A low-energy two-state physical sector P is coupled to a two-state high-energy
constraint-defect sector Q.  Eliminating Q exactly gives

    H_eff(E) = A + B (E-D)^(-1) B^dagger.

The audit checks that:
- the Schur/Feshbach identity is exact;
- eliminating defects generates operator structures absent from the bare
  low-energy ansatz;
- the full Hermitian 2x2 operator basis closes the effective kernel;
- H_eff depends on E, so exact coarse graining generates a frequency-dependent
  kernel rather than one energy-independent local Hamiltonian.

This is a finite architecture toy, not the actual QCCG functional RG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


E = sp.symbols("E", real=True)
I = sp.I
I2 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])

# Bare low-energy ansatz deliberately contains only sigma_z.
A = sp.Rational(1, 3) * sz
# Generic coupling to two gapped defect channels.
B = sp.Matrix([
    [sp.Rational(1, 2), sp.Rational(1, 3)],
    [sp.Rational(1, 4), sp.Rational(2, 5)],
])
D = sp.diag(5, 7)
H = A.row_join(B).col_join(B.T.row_join(D))


def zero(M):
    return M == sp.zeros(*M.shape)


def feshbach(energy):
    return sp.simplify(A + B * (energy * I2 - D).inv() * B.T)


def pauli_coeffs(M):
    return {
        "I": sp.simplify(sp.trace(M) / 2),
        "X": sp.simplify(sp.trace(M * sx) / 2),
        "Y": sp.simplify(sp.trace(M * sy) / 2),
        "Z": sp.simplify(sp.trace(M * sz) / 2),
    }


def reconstruct(c):
    return sp.simplify(c["I"] * I2 + c["X"] * sx + c["Y"] * sy + c["Z"] * sz)


def max_abs_numeric(M):
    return max(abs(float(sp.N(x))) for x in M)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-feshbach-rg-toy",
        "artifact": "qccg/run_feshbach_rg_toy.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    Heff = feshbach(E)

    # Exact block determinant identity:
    # det(E-H) = det(E-D) det(E-H_eff(E)).
    lhs = sp.factor((E * sp.eye(4) - H).det())
    rhs = sp.factor((E * I2 - D).det() * (E * I2 - Heff).det())
    identity_pass = sp.simplify(lhs - rhs) == 0

    energies = (sp.Rational(0), sp.Rational(1))
    rows = []
    generated_offdiag = False
    full_basis_pass = True
    single_truncation_errors = []

    for e in energies:
        M = sp.simplify(Heff.subs(E, e))
        coeff = pauli_coeffs(M)
        rec = reconstruct(coeff)
        full_ok = zero(sp.simplify(M - rec))
        full_basis_pass = full_basis_pass and full_ok

        # Bare truncation permits I and sigma_z only.
        rec_single = sp.simplify(coeff["I"] * I2 + coeff["Z"] * sz)
        residual_single = sp.simplify(M - rec_single)
        single_err = max_abs_numeric(residual_single)
        single_truncation_errors.append(single_err)
        generated_offdiag = generated_offdiag or coeff["X"] != 0 or coeff["Y"] != 0

        rows.append({
            "E": str(e),
            "H_eff": [[str(x) for x in row] for row in M.tolist()],
            "pauli_coefficients": {k: str(v) for k, v in coeff.items()},
            "full_basis_exact": full_ok,
            "bare_I_plus_Z_residual": [[str(x) for x in row] for row in residual_single.tolist()],
            "bare_I_plus_Z_max_abs_residual": single_err,
        })

    truncation_rejected = generated_offdiag and min(single_truncation_errors) > 1.0e-6

    H0 = sp.simplify(Heff.subs(E, 0))
    H1 = sp.simplify(Heff.subs(E, 1))
    energy_difference = sp.simplify(H1 - H0)
    energy_dependence = not zero(energy_difference)

    # Leading low-energy expansion around E=0.
    series_entries = [
        [sp.series(Heff[i, j], E, 0, 3).removeO() for j in range(2)]
        for i in range(2)
    ]
    derivative0 = sp.simplify(sp.diff(Heff, E).subs(E, 0))
    derivative_nonzero = not zero(derivative0)

    result = {
        "schema": 1,
        "scope": "finite P/Q Feshbach coarse-graining architecture toy",
        "evidence": [
            evidence(
                "qccg-feshbach-identity",
                "FESHBACH_COARSE_GRAINING_TOY",
                "PASS" if identity_pass else "FAIL",
                "The exact block determinant equals the defect determinant times the determinant of the energy-dependent Feshbach effective kernel.",
                full_H=[[str(x) for x in row] for row in H.tolist()],
                determinant_identity=identity_pass,
                determinant_full=str(lhs),
                determinant_factorized=str(rhs),
            ),
            evidence(
                "qccg-local-truncation-proliferation",
                "COARSE_OPERATOR_PROLIFERATION_DETECTED",
                "PASS" if truncation_rejected else "FAIL",
                "Eliminating the gapped defect sector generates off-diagonal low-energy operators absent from the bare I+sigma_z truncation.",
                rows=rows,
                generated_offdiagonal_operator=generated_offdiag,
            ),
            evidence(
                "qccg-full-pauli-basis-closure",
                "FINITE_EFFECTIVE_BASIS_CLOSURE_TOY",
                "PASS" if full_basis_pass else "FAIL",
                "The complete Hermitian two-state operator basis reconstructs the exact effective kernel at the audited energies.",
                rows=rows,
            ),
            evidence(
                "qccg-energy-dependent-kernel",
                "ENERGY_DEPENDENT_EFFECTIVE_KERNEL_DETECTED",
                "PASS" if energy_dependence and derivative_nonzero else "FAIL",
                "Exact defect elimination produces a genuinely energy-dependent effective kernel, a finite toy analogue of momentum/frequency-dependent form factors.",
                H_eff_E0=[[str(x) for x in row] for row in H0.tolist()],
                H_eff_E1=[[str(x) for x in row] for row in H1.tolist()],
                difference=[[str(x) for x in row] for row in energy_difference.tolist()],
                dH_eff_dE_at_0=[[str(x) for x in row] for row in derivative0.tolist()],
                low_energy_series=[[str(x) for x in row] for row in series_entries],
            ),
            evidence(
                "qccg-actual-feshbach-rg-flow",
                "QCCG_GRAPH_CHANGING_EFFECTIVE_KERNEL",
                "OPEN",
                "The exact Feshbach mechanism is demonstrated only in a finite P/Q toy. The actual graph-changing constrained QCCG ensemble has not yet been reduced to a controlled momentum-dependent effective kernel.",
                required_next_step=(
                    "Apply a compatible coarse/defect split to increasing QCCG cellulation sectors, "
                    "track generated operators and frequency dependence, and compare the resulting "
                    "kernel with RFQG-5 vertex/form-factor data."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("Feshbach RG toy audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
