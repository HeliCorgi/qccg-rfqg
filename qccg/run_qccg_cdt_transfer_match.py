#!/usr/bin/env python3
"""QCCG -> CDT effective transfer-matrix matching audit.

Primary literature anchors:
- Ambjorn, Gizbert-Studnicki, Goerlich, Jurkiewicz,
  "The transfer matrix in four-dimensional CDT", JHEP 09 (2012) 017,
  arXiv:1205.3791, DOI:10.1007/JHEP09(2012)017.
- Ambjorn, Gizbert-Studnicki, Goerlich, Jurkiewicz,
  "The effective action in 4-dim CDT. The transfer matrix approach",
  JHEP 06 (2014) 034, arXiv:1403.5940,
  DOI:10.1007/JHEP06(2014)034.
- Ambjorn, Gizbert-Studnicki, Goerlich, Nemeth,
  "Is lattice quantum gravity asymptotically safe? Making contact between
  causal dynamical triangulations and the functional renormalization group",
  Phys. Rev. D 110, 126006 (2024),
  DOI:10.1103/PhysRevD.110.126006.

CDT comparison coordinate:
  effective states are labelled by spatial three-volume n=N3(t), and in the
  de Sitter phase the measured minisuperspace effective action has the form

    S_eff[n] = (1/Gamma) sum_t [
        (n_{t+1}-n_t)^2/(n_{t+1}+n_t)
        + mu n_t^(1/3) - lambda n_t
    ]

up to conventions / discretization choices.

This audit constructs a QCCG microscopic Euclidean transfer kernel exp(-tau H)
on the finite spatial-Pachner state graph, reduces it to volume sectors using a
two-slice probability coarse graining analogous to the CDT transfer-matrix
extraction, and tests only the kinetic shape.

A failure of the finite baseline is a scientific negative result, not a CI
failure. It means the tiny unweighted graph-Laplacian parent does not yet
reproduce the CDT de Sitter transfer law.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sympy as sp

import run_time_local_slice_transfer_toy as qslice


TAUS = (0.15, 0.30, 0.60, 1.00)
TAYLOR_TERMS = 80
KINETIC_R2_TARGET = 0.90
MIN_VOLUME_CLASSES = 4
TINY = 1.0e-300

SOURCES = [
    {
        "title": "The transfer matrix in four-dimensional CDT",
        "journal": "JHEP 09 (2012) 017",
        "arxiv": "1205.3791",
        "doi": "10.1007/JHEP09(2012)017",
    },
    {
        "title": "The effective action in 4-dim CDT. The transfer matrix approach",
        "journal": "JHEP 06 (2014) 034",
        "arxiv": "1403.5940",
        "doi": "10.1007/JHEP06(2014)034",
    },
    {
        "title": "Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group",
        "journal": "Phys. Rev. D 110, 126006 (2024)",
        "doi": "10.1103/PhysRevD.110.126006",
    },
]


def mat_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(A, B):
    n, p, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)]
        for i in range(n)
    ]


def matadd_scaled(A, B, s):
    return [
        [A[i][j] + s * B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def graph_laplacian(n, edges):
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    for a, b in edges:
        L[a][a] += 1.0
        L[b][b] += 1.0
        L[a][b] -= 1.0
        L[b][a] -= 1.0
    return L


def heat_kernel_uniformization(L, tau):
    """exp(-tau L) for a graph Laplacian using uniformization.

    P=I-L/Lambda is nonnegative stochastic for Lambda=max degree.
    """
    n = len(L)
    lam = max(L[i][i] for i in range(n))
    if lam <= 0:
        return mat_identity(n)

    I = mat_identity(n)
    P = [[I[i][j] - L[i][j] / lam for j in range(n)] for i in range(n)]

    out = [[0.0 for _ in range(n)] for _ in range(n)]
    power = mat_identity(n)
    coeff = math.exp(-lam * tau)
    out = matadd_scaled(out, power, coeff)
    poisson = coeff
    x = lam * tau

    for k in range(1, TAYLOR_TERMS + 1):
        power = matmul(power, P)
        poisson *= x / k
        out = matadd_scaled(out, power, poisson)
        if poisson < 1.0e-16 and k > x + 12:
            break
    return out


def volume_sectors(states, keys):
    sectors = {}
    for i, k in enumerate(keys):
        n3 = len(states[k])
        sectors.setdefault(n3, []).append(i)
    return dict(sorted(sectors.items()))


def reduced_transfer_from_two_slice_probability(K, sectors):
    """CDT-compatible reduced matrix from coarse two-slice probability.

    For symmetric microscopic K, coarse P2(n,m) is proportional to the sum of
    K_ij K_ji=K_ij^2 over microstates in the two volume sectors. The effective
    transfer element is sqrt(P2) up to one common normalization.
    """
    vols = list(sectors)
    M = {}
    for n in vols:
        for m in vols:
            p2 = 0.0
            for i in sectors[n]:
                for j in sectors[m]:
                    p2 += K[i][j] * K[j][i]
            M[(n, m)] = math.sqrt(max(0.0, p2))

    # Remove one irrelevant common normalization by max entry.
    z = max(M.values())
    if z > 0:
        M = {k: v / z for k, v in M.items()}
    return vols, M


def kinetic_fit(vols, M):
    rows = []
    xvals = []
    yvals = []
    for ai, n in enumerate(vols):
        for m in vols[ai + 1:]:
            mn = M[(n, m)]
            nn = M[(n, n)]
            mm = M[(m, m)]
            if min(mn, nn, mm) <= TINY:
                continue
            # Cancel volume-local potential / entropy factors.
            y = -math.log(mn / math.sqrt(nn * mm))
            x = (n - m) ** 2 / (n + m)
            if x <= 0:
                continue
            rows.append({"n": n, "m": m, "x_d2_over_s": x, "kinetic_log_ratio": y})
            xvals.append(x)
            yvals.append(y)

    if len(xvals) < 2:
        return {"ok": False, "reason": "too few off-diagonal volume pairs", "rows": rows}

    xx = sum(x * x for x in xvals)
    slope = sum(x * y for x, y in zip(xvals, yvals)) / xx if xx > 0 else 0.0
    pred = [slope * x for x in xvals]
    sse = sum((y - p) ** 2 for y, p in zip(yvals, pred))
    mean = sum(yvals) / len(yvals)
    sst = sum((y - mean) ** 2 for y in yvals)
    r2 = 1.0 - sse / sst if sst > 1.0e-18 else (1.0 if sse < 1.0e-18 else -1.0)
    return {
        "ok": True,
        "slope_inverse_Gamma": slope,
        "Gamma_eff": None if slope <= 0 else 1.0 / slope,
        "r2": r2,
        "rows": rows,
    }


def potential_fit(vols, M):
    # Diagnostic only: -log M_nn = c + mu n^(1/3) - lambda n.
    X = []
    y = []
    rows = []
    for n in vols:
        v = M[(n, n)]
        if v <= TINY:
            continue
        yn = -math.log(v)
        X.append([1.0, n ** (1.0 / 3.0), -float(n)])
        y.append(yn)
        rows.append({"n": n, "minus_log_diag": yn})

    if len(X) < 3:
        return {"ok": False, "reason": "too few diagonal volume classes", "rows": rows}

    MX = sp.Matrix(X)
    vy = sp.Matrix(y)
    if MX.rank() < 3:
        return {"ok": False, "reason": "rank-deficient potential design", "rows": rows}

    coeff = (MX.T * MX).inv() * MX.T * vy
    vals = [float(sp.N(z)) for z in coeff]
    pred = [sum(row[j] * vals[j] for j in range(3)) for row in X]
    mean = sum(y) / len(y)
    sse = sum((a - b) ** 2 for a, b in zip(y, pred))
    sst = sum((a - mean) ** 2 for a in y)
    r2 = 1.0 - sse / sst if sst > 1.0e-18 else (1.0 if sse < 1.0e-18 else -1.0)
    return {
        "ok": True,
        "constant": vals[0],
        "mu": vals[1],
        "lambda": vals[2],
        "r2": r2,
        "rows": rows,
    }


def synthetic_reference_control(vols):
    if len(vols) < 3:
        return {"ok": False}
    A = 0.7
    mu = 0.9
    lam = 0.08
    M = {}
    for n in vols:
        for m in vols:
            s = n + m
            kinetic = A * (n - m) ** 2 / s
            # Symmetric one-step discretization of a local potential.
            pot = 0.5 * (
                mu * n ** (1.0 / 3.0) - lam * n
                + mu * m ** (1.0 / 3.0) - lam * m
            )
            M[(n, m)] = math.exp(-(kinetic + pot))
    z = max(M.values())
    M = {k: v / z for k, v in M.items()}
    fit = kinetic_fit(vols, M)
    return {
        "ok": fit.get("ok", False) and fit["r2"] > 0.999999 and abs(fit["slope_inverse_Gamma"] - A) < 1e-10,
        "injected_inverse_Gamma": A,
        "fit": fit,
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "qccg-cdt-transfer-match",
        "artifact": "qccg/run_qccg_cdt_transfer_match.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    states, keys, _idx, edges, _edge_types, inverse_ok, _failures = qslice.build_state_graph()
    n_micro = len(keys)
    sectors = volume_sectors(states, keys)
    volumes = list(sectors)
    L = graph_laplacian(n_micro, edges)

    internal_ok = (
        inverse_ok
        and len(volumes) >= MIN_VOLUME_CLASSES
        and len(edges) > 0
    )

    reference = synthetic_reference_control(volumes)
    scan = []
    best = None
    for tau in TAUS:
        K = heat_kernel_uniformization(L, tau)
        vols, M = reduced_transfer_from_two_slice_probability(K, sectors)
        kin = kinetic_fit(vols, M)
        pot = potential_fit(vols, M)
        row = {
            "tau": tau,
            "kinetic": kin,
            "potential": pot,
        }
        scan.append(row)
        if kin.get("ok") and (best is None or kin["r2"] > best["kinetic"]["r2"]):
            best = row

    baseline_match = bool(
        best
        and best["kinetic"]["slope_inverse_Gamma"] > 0
        and best["kinetic"]["r2"] >= KINETIC_R2_TARGET
    )
    mismatch_detected = internal_ok and reference["ok"] and not baseline_match

    result = {
        "schema": 1,
        "scope": "finite QCCG spatial-Pachner state graph -> volume-reduced Euclidean transfer matrix; not a continuum CDT equivalence proof",
        "sources": SOURCES,
        "cdt_reference_form": {
            "kinetic": "(1/Gamma)*(n-m)^2/(n+m)",
            "potential": "mu*n^(1/3)-lambda*n",
            "coarse_variable": "n=N3(t), spatial three-volume",
            "two_slice_extraction": "P2(m,n) proportional to M_mn M_nm; symmetric M_mn proportional to sqrt(P2)",
        },
        "evidence": [
            evidence(
                "qccg-cdt-transfer-reference-calibration",
                "CDT_TRANSFER_FITTER_CALIBRATED",
                "PASS" if reference["ok"] else "FAIL",
                "The audit recovers the injected CDT kinetic coefficient exactly on a synthetic transfer matrix with the literature reference form.",
                sources=SOURCES,
                calibration=reference,
            ),
            evidence(
                "qccg-volume-reduced-transfer",
                "QCCG_VOLUME_REDUCED_TRANSFER_CONSTRUCTED",
                "PASS" if internal_ok else "FAIL",
                "A symmetric Euclidean transfer kernel exp(-tau H_move) is constructed on the finite unlabeled spatial-Pachner state graph and reduced to N3 volume sectors using a two-slice probability coarse graining analogous to the CDT effective-transfer extraction.",
                n_microstates=n_micro,
                n_edges=len(edges),
                volume_sector_sizes={str(k): len(v) for k, v in sectors.items()},
                volumes=volumes,
                tau_values=list(TAUS),
                inverse_ok=inverse_ok,
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-baseline-kinetic-match",
                "QCCG_CDT_BASELINE_KINETIC_MATCH",
                "PASS" if baseline_match else "OPEN",
                (
                    "The unweighted finite QCCG move-Laplacian transfer kernel reaches the preregistered CDT kinetic-shape target."
                    if baseline_match
                    else "The unweighted finite QCCG move-Laplacian transfer kernel does not yet reach the preregistered CDT de Sitter kinetic-shape target; this is retained as an unresolved matching gate."
                ),
                r2_target=KINETIC_R2_TARGET,
                best=best,
                scan=scan,
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-baseline-mismatch",
                "QCCG_CDT_BASELINE_MISMATCH_DIAGNOSTIC",
                "PASS" if mismatch_detected else "NOT_APPLICABLE",
                "If the baseline misses the CDT kinetic form while the reference fitter is calibrated, the mismatch is recorded rather than tuned away.",
                mismatch_detected=mismatch_detected,
                best=best,
                sources=SOURCES,
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Only internal/calibration failures fail CI. A physical baseline mismatch is
    # evidence, not infrastructure failure.
    if not internal_ok or not reference["ok"]:
        raise SystemExit("QCCG-CDT transfer construction/calibration failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
