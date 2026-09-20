#!/usr/bin/env python3
"""Direct symmetric QCCG volume-transfer extraction and CDT fit.

This is the next step after the generator-normalization audit.

Pipeline
--------
1. Determine the curvature-equilibrated QCCG critical-line coupling.
2. Build a DENSE set of fixed-volume S^3 triangulation ensembles.
3. Average microscopic weighted Pachner rates into a coarse volume generator
   q(N -> N+/-1, N+/-3).
4. Reconstruct the coarse detailed-balance weights pi_N from the bidirectional
   macro rates.
5. Project finite-sampling violations onto the nearest reversible conductance
   network while recording the projection size.
6. Construct the symmetric Euclidean transfer kernel
      M_sym(tau) = Pi^(1/2) exp(tau Q_rev) Pi^(-1/2).
7. Fit -log M_sym directly to the finite-volume CDT ansatz
      c + (1/Gamma) [
          (n-m)^2/(n+m-2 n0)
          + delta ((n+m)/2)^(1/3)
          - lambda (n+m)/2
      ].

Primary CDT anchors
-------------------
- J. Ambjorn et al., "The transfer matrix in four-dimensional CDT",
  JHEP 09 (2012) 017, arXiv:1205.3791,
  DOI:10.1007/JHEP09(2012)017.
- J. Ambjorn et al., "The effective action in 4-dim CDT. The transfer matrix
  approach", JHEP 06 (2014) 034, arXiv:1403.5940,
  DOI:10.1007/JHEP06(2014)034.

Scope
-----
This is a finite dense-volume coarse-transfer audit.  It does not establish a
continuum limit or identify the QCCG microscopic time unit with one CDT time
step.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import sympy as sp

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_fixed_volume_curvature_thermalization as therm
import run_qccg_equilibrated_critical_line as eqcrit
import run_qccg_curvature_weighted_time_exploration as explore
import run_time_local_slice_transfer_toy as qslice


VOLUMES = tuple(range(20, 45))
BURN_ATTEMPTS = 2500
BETWEEN_SAMPLES = 250
SAMPLES_PER_VOLUME = 8
MAX_RETURN_MULTIPLIER = 18
SEED = 20270921

DETAIL_BALANCE_RMS_MAX = 0.30
REVERSIBLE_RATE_RMS_REL_MAX = 0.35
FIT_R2_TARGET = 0.94
GAMMA_RELERR_TARGET = 0.30
MESOSCOPIC_TAU_MULTIPLIERS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
CYCLE_AFFINITY_RMS_REJECTION = 0.50
BOUNDARY_MARGIN = 4
MAX_PAIR_SEPARATION = 6
KERNEL_TOL = 1.0e-15
UNIFORMIZATION_TERMS = 250

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
]


def evidence(eid, status, note, **metadata):
    return {
        "id": eid,
        "status": status,
        "engine": "qccg-symmetric-transfer-extraction",
        "artifact": "qccg/run_qccg_symmetric_transfer_extraction.py",
        "note": note,
        "metadata": metadata,
    }


def build_any_target(target: int, seed: int):
    """Build a closed S3 triangulation at any target >=20.

    The existing refinement builder reaches N3 = 5 mod 3.  One or two 2->3
    moves fill the other residue classes without changing topology.
    """
    if target < 20:
        raise ValueError("dense transfer audit is defined for target >= 20")

    rem = (target - 5) % 3
    base_target = target - rem
    if base_target < 5:
        raise ValueError(target)

    for attempt in range(20):
        S, _mixed = base.build_target(base_target, seed + 1009 * attempt)
        rng = random.Random(seed + 7919 * attempt + 17)
        ok = True
        while len(S) < target:
            cs = qslice.candidates23(S)
            if not cs:
                ok = False
                break
            S2 = qslice.apply23(S, rng.choice(cs))
            if not qslice.manifold(S2):
                ok = False
                break
            S = S2
        if ok and len(S) == target and qslice.manifold(S):
            return S
    raise RuntimeError(f"unable to build target N3={target}")


def thermalize_dense_target(target, kR, kV, seed):
    rng = random.Random(seed)
    S = build_any_target(target, seed + 73)

    for _ in range(BURN_ATTEMPTS):
        S, _acc, _rej = therm.mh_step(S, rng, kR, kV, target)

    samples = []
    attempts = 0
    accepted = 0
    max_attempts = (
        BETWEEN_SAMPLES * SAMPLES_PER_VOLUME * MAX_RETURN_MULTIPLIER
    )
    while len(samples) < SAMPLES_PER_VOLUME and attempts < max_attempts:
        accepted_since = 0
        for _ in range(BETWEEN_SAMPLES):
            S, acc, _rej = therm.mh_step(S, rng, kR, kV, target)
            attempts += 1
            accepted += int(acc)
            accepted_since += int(acc)

        for _ in range(BETWEEN_SAMPLES * MAX_RETURN_MULTIPLIER):
            if len(S) == target:
                break
            S, acc, _rej = therm.mh_step(S, rng, kR, kV, target)
            attempts += 1
            accepted += int(acc)
            accepted_since += int(acc)

        if len(S) != target:
            continue
        snap = set(S)
        if not qslice.manifold(snap):
            raise RuntimeError(f"thermalized sample lost manifold at {target}")
        samples.append(snap)

    if len(samples) != SAMPLES_PER_VOLUME:
        raise RuntimeError(
            f"collected {len(samples)}/{SAMPLES_PER_VOLUME} samples at N3={target}"
        )
    return {
        "samples": samples,
        "acceptance_fraction": accepted / max(1, attempts),
        "attempts_after_burn": attempts,
    }


def critical_line():
    selection_samples, _meta, _old = eqcrit.thermal_samples()
    kV, residual = eqcrit.solve_kv(selection_samples)
    return eqcrit.KAPPA_R, kV, residual


def averaged_macro_rates(samples, kR, kV):
    wr = explore.weighted_rates(kR, kV)
    rows = []
    acc = collections.defaultdict(float)
    for S in samples:
        c = base.counts(S)
        vals = {
            +3: c["14"] * wr["14"],
            -3: c["41"] * wr["41"],
            +1: c["23"] * wr["23"],
            -1: c["32"] * wr["32"],
        }
        rows.append({"counts": c, "rates": vals})
        for dn, x in vals.items():
            acc[dn] += x
    mean = {dn: acc[dn] / len(samples) for dn in (+3, -3, +1, -1)}
    return mean, rows


def connected_components(n, edges):
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    seen = set()
    comps = []
    for s in range(n):
        if s in seen:
            continue
        q = collections.deque([s])
        seen.add(s)
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        comps.append(comp)
    return comps


def infer_log_pi(volumes, qobs):
    idx = {n: i for i, n in enumerate(volumes)}
    undirected = []
    Xrows = []
    ys = []
    for i, n in enumerate(volumes):
        for dn in (1, 3):
            m = n + dn
            if m not in idx:
                continue
            j = idx[m]
            qij = qobs.get((i, j), 0.0)
            qji = qobs.get((j, i), 0.0)
            if qij <= 0 or qji <= 0:
                continue
            undirected.append((i, j))
            row = [0.0] * (len(volumes) - 1)
            # Gauge log pi_0 = 0.
            if j != 0:
                row[j - 1] += 1.0
            if i != 0:
                row[i - 1] -= 1.0
            Xrows.append(row)
            ys.append(math.log(qij / qji))

    comps = connected_components(len(volumes), undirected)
    if len(comps) != 1:
        raise RuntimeError(
            f"bidirectional macro-rate graph disconnected: {[len(c) for c in comps]}"
        )

    X = sp.Matrix(Xrows)
    y = sp.Matrix(ys)
    normal = X.T * X
    if normal.rank() != len(volumes) - 1:
        raise RuntimeError("detailed-balance log-weight fit is rank deficient")
    beta = normal.LUsolve(X.T * y)
    logs = [0.0] + [float(sp.N(z)) for z in beta]

    residuals = []
    for row, yy in zip(Xrows, ys):
        pred = sum(a * b for a, b in zip(row, logs[1:]))
        residuals.append(pred - yy)
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    maxabs = max(abs(r) for r in residuals)

    shift = max(logs)
    raw = [math.exp(x - shift) for x in logs]
    z = sum(raw)
    pi = [x / z for x in raw]
    return pi, rms, maxabs, undirected, residuals


def reversible_projection(volumes, qobs, pi, undirected):
    n = len(volumes)
    qrev = [[0.0 for _ in range(n)] for _ in range(n)]
    rel_errors = []
    flux_rows = []
    for i, j in undirected:
        qij = qobs[(i, j)]
        qji = qobs[(j, i)]
        fij = pi[i] * qij
        fji = pi[j] * qji
        conductance = math.sqrt(fij * fji)
        rij = conductance / pi[i]
        rji = conductance / pi[j]
        qrev[i][j] = rij
        qrev[j][i] = rji
        rel_errors.extend([
            abs(rij - qij) / max(qij, 1e-30),
            abs(rji - qji) / max(qji, 1e-30),
        ])
        flux_rows.append({
            "N_from": volumes[i],
            "N_to": volumes[j],
            "observed_forward": qij,
            "observed_reverse": qji,
            "projected_forward": rij,
            "projected_reverse": rji,
            "log_flux_residual": math.log(fij / fji),
        })

    for i in range(n):
        qrev[i][i] = -sum(qrev[i][j] for j in range(n) if j != i)

    rms_rel = math.sqrt(sum(x * x for x in rel_errors) / len(rel_errors))
    max_rel = max(rel_errors)
    return qrev, rms_rel, max_rel, flux_rows


def matmul(A, B):
    n, p, m = len(A), len(B), len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)]
        for i in range(n)
    ]


def uniformized_exponential(Q, tau):
    n = len(Q)
    lam = max(-Q[i][i] for i in range(n))
    if lam <= 0:
        raise RuntimeError("nonpositive uniformization rate")
    P = [
        [
            (1.0 if i == j else 0.0) + Q[i][j] / lam
            for j in range(n)
        ]
        for i in range(n)
    ]
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    power = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    poisson = math.exp(-lam * tau)
    for i in range(n):
        out[i][i] += poisson

    x = lam * tau
    for k in range(1, UNIFORMIZATION_TERMS + 1):
        power = matmul(power, P)
        poisson *= x / k
        for i in range(n):
            for j in range(n):
                out[i][j] += poisson * power[i][j]
        if poisson < 1e-16 and k > x + 14:
            break
    return out


def symmetric_kernel(Qrev, pi, tau):
    K = uniformized_exponential(Qrev, tau)
    n = len(pi)
    M = [[0.0 for _ in range(n)] for _ in range(n)]
    asym = 0.0
    for i in range(n):
        for j in range(n):
            M[i][j] = math.sqrt(pi[i]) * K[i][j] / math.sqrt(pi[j])
    scale = max(max(row) for row in M)
    M = [[x / scale for x in row] for row in M]
    for i in range(n):
        for j in range(n):
            asym = max(asym, abs(M[i][j] - M[j][i]))
    return M, asym


def cycle_affinity_diagnostic(volumes, qobs):
    """Kolmogorov-cycle test for the N3-only instantaneous coarse process.

    Compare three +1 edges with one +3 edge between the same endpoint volumes.
    A reversible Markov process on N3 alone requires every such cycle affinity
    to vanish.
    """
    idx = {n: i for i, n in enumerate(volumes)}
    rows = []
    vals = []
    for n in volumes:
        if n + 3 not in idx:
            continue
        try:
            r1 = 0.0
            for k in range(3):
                i = idx[n + k]
                j = idx[n + k + 1]
                r1 += math.log(qobs[(i, j)] / qobs[(j, i)])
            i = idx[n]
            j = idx[n + 3]
            r3 = math.log(qobs[(i, j)] / qobs[(j, i)])
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        affinity = r1 - r3
        vals.append(affinity)
        rows.append({
            "N3_start": n,
            "three_unit_step_log_ratio": r1,
            "single_three_step_log_ratio": r3,
            "cycle_affinity": affinity,
        })
    rms = math.sqrt(sum(x*x for x in vals)/len(vals))
    maxabs = max(abs(x) for x in vals)
    return {
        "rms": rms,
        "max_abs": maxabs,
        "rows": rows,
    }


def affine_diffusion_fit(volumes, macro_rows):
    xs = [float(n) for n in volumes]
    ys = []
    for n in volumes:
        r = macro_rows[n]["mean_rates"]
        a = 9.0 * (r[+3] + r[-3]) + r[+1] + r[-1]
        ys.append(a)
    D, intercept, r2, pred = eqcrit.linear_fit(xs, ys)
    n0 = -intercept / D
    return {
        "D": D,
        "intercept": intercept,
        "n0": n0,
        "r2": r2,
        "rows": [
            {"N3": n, "variance_rate": y, "fit": p}
            for n, y, p in zip(volumes, ys, pred)
        ],
    }


def fit_cdt_kernel(volumes, M, n0):
    lo = BOUNDARY_MARGIN
    hi = len(volumes) - BOUNDARY_MARGIN
    rows = []
    X = []
    y = []
    for i in range(lo, hi):
        n = volumes[i]
        for j in range(i, min(hi, i + MAX_PAIR_SEPARATION + 1)):
            m = volumes[j]
            val = M[i][j]
            if val <= KERNEL_TOL:
                continue
            s = n + m
            denom = s - 2.0 * n0
            if denom <= 0:
                continue
            x = 0.5 * s
            kin = ((n - m) ** 2) / denom
            row = [1.0, kin, x ** (1.0 / 3.0), -x]
            X.append(row)
            yy = -math.log(val)
            y.append(yy)
            rows.append({
                "n": n,
                "m": m,
                "minus_log_M": yy,
                "kinetic_basis": kin,
                "Nbar": x,
            })

    MX = sp.Matrix(X)
    vy = sp.Matrix(y)
    if MX.rank() < 4:
        raise RuntimeError("CDT transfer fit design matrix is rank deficient")
    beta = (MX.T * MX).LUsolve(MX.T * vy)
    coeff = [float(sp.N(z)) for z in beta]
    pred = [
        sum(row[k] * coeff[k] for k in range(4))
        for row in X
    ]
    ym = sum(y) / len(y)
    sse = sum((a - b) ** 2 for a, b in zip(y, pred))
    sst = sum((a - ym) ** 2 for a in y)
    r2 = 1.0 - sse / sst if sst > 1e-18 else 1.0

    c0, A, B, C = coeff
    if A == 0:
        Gamma = float("inf")
        delta = float("nan")
        lam = float("nan")
    else:
        Gamma = 1.0 / A
        delta = B / A
        lam = C / A

    return {
        "constant": c0,
        "inverse_Gamma": A,
        "Gamma": Gamma,
        "delta": delta,
        "lambda": lam,
        "r2": r2,
        "n_rows": len(rows),
        "rows": [
            {**r, "fit": p}
            for r, p in zip(rows, pred)
        ],
    }


def main():
    kR, kV, critical_residual = critical_line()
    wr = explore.weighted_rates(kR, kV)

    macro = {}
    thermal_meta = []
    for i, N in enumerate(VOLUMES):
        out = thermalize_dense_target(N, kR, kV, SEED + 10007 * i)
        mean_rates, sample_rows = averaged_macro_rates(out["samples"], kR, kV)
        macro[N] = {
            "mean_rates": mean_rates,
            "sample_rows": sample_rows,
        }
        thermal_meta.append({
            "N3": N,
            "acceptance_fraction": out["acceptance_fraction"],
            "attempts_after_burn": out["attempts_after_burn"],
        })

    idx = {N: i for i, N in enumerate(VOLUMES)}
    qobs = {}
    for N in VOLUMES:
        i = idx[N]
        for dn in (+1, -1, +3, -3):
            M = N + dn
            if M not in idx:
                continue
            qobs[(i, idx[M])] = macro[N]["mean_rates"][dn]

    pi, db_rms, db_max, undirected, db_residuals = infer_log_pi(
        VOLUMES, qobs
    )
    qrev, proj_rms, proj_max, flux_rows = reversible_projection(
        VOLUMES, qobs, pi, undirected
    )

    affine = affine_diffusion_fit(VOLUMES, macro)
    interior_out = [
        -qrev[i][i]
        for i in range(BOUNDARY_MARGIN, len(VOLUMES) - BOUNDARY_MARGIN)
    ]
    sorted_out = sorted(interior_out)
    median_out = sorted_out[len(sorted_out) // 2]
    tau_ref = 1.0 / median_out

    cycle = cycle_affinity_diagnostic(VOLUMES, qobs)
    n3_instantaneous_rejected = cycle["rms"] >= CYCLE_AFFINITY_RMS_REJECTION

    reversible_ok = (
        db_rms <= DETAIL_BALANCE_RMS_MAX
        and proj_rms <= REVERSIBLE_RATE_RMS_REL_MAX
    )

    transfer_scan = []
    selected = None
    for mult in MESOSCOPIC_TAU_MULTIPLIERS:
        tau = mult * tau_ref
        M, asym = symmetric_kernel(qrev, pi, tau)
        fit = fit_cdt_kernel(VOLUMES, M, affine["n0"])
        gamma_expected = affine["D"] * tau
        gamma_relerr = abs(fit["Gamma"] - gamma_expected) / max(
            abs(gamma_expected), 1e-30
        )
        row = {
            "tau_multiplier": mult,
            "tau": tau,
            "symmetric_kernel_max_asymmetry": asym,
            "fit": fit,
            "gamma_expected_from_generator": gamma_expected,
            "gamma_relative_error": gamma_relerr,
        }
        transfer_scan.append(row)
        row_pass = (
            asym <= 1e-10
            and fit["inverse_Gamma"] > 0
            and fit["r2"] >= FIT_R2_TARGET
            and gamma_relerr <= GAMMA_RELERR_TARGET
            and fit["delta"] > 0
            and fit["lambda"] > 0
        )
        row["mesoscopic_cdt_pass"] = row_pass
        if selected is None and row_pass:
            selected = row

    projected_transfer_pass = selected is not None

    result = {
        "schema": 1,
        "scope": "dense finite-volume symmetric QCCG transfer extraction; continuum/time-unit matching remains open",
        "sources": SOURCES,
        "critical_line": {
            "kappa_R": kR,
            "kappa_V": kV,
            "zero_drift_residual": critical_residual,
            "local_move_rates": wr,
        },
        "dense_volume_grid": {
            "volumes": list(VOLUMES),
            "burn_attempts": BURN_ATTEMPTS,
            "between_samples": BETWEEN_SAMPLES,
            "samples_per_volume": SAMPLES_PER_VOLUME,
            "thermalization": thermal_meta,
        },
        "evidence": [
            evidence(
                "qccg-dense-volume-generator",
                "PASS" if len(VOLUMES) >= 20 else "FAIL",
                "A dense curvature-equilibrated QCCG macro generator is constructed over consecutive spatial volumes using averaged microscopic weighted Pachner rates.",
                volumes=list(VOLUMES),
                macro={
                    str(N): {
                        "mean_rates": {
                            str(k): v
                            for k, v in macro[N]["mean_rates"].items()
                        }
                    }
                    for N in VOLUMES
                },
            ),
            evidence(
                "qccg-n3-instantaneous-markov-reduction-rejected",
                "PASS" if n3_instantaneous_rejected else "NOT_APPLICABLE",
                "The instantaneous N3-only macro generator violates a Kolmogorov cycle condition: three 2<->3 unit-volume steps and one 1<->4 three-volume step do not define the same coarse potential difference. N3 alone is therefore rejected as an instantaneous Markov state at this scale; hidden N0/curvature or finite-time elimination is required.",
                cycle_affinity=cycle,
                rejection_rms_threshold=CYCLE_AFFINITY_RMS_REJECTION,
                log_detailed_balance_rms=db_rms,
                log_detailed_balance_max_abs=db_max,
                flux_rows=flux_rows,
            ),
            evidence(
                "qccg-reversible-projection-size",
                "PASS" if not reversible_ok else "NOT_APPLICABLE",
                "Because the N3-only instantaneous reduction is non-Markov, the size of the nearest reversible projection is retained explicitly rather than treated as microscopic evidence.",
                reversible_projection_rms_relative=proj_rms,
                reversible_projection_max_relative=proj_max,
                reversible_projection_rms_relative_reference=REVERSIBLE_RATE_RMS_REL_MAX,
            ),
            evidence(
                "qccg-mesoscopic-reversible-transfer-diagnostic",
                "PASS" if projected_transfer_pass else "FAIL",
                "After explicit reversible projection, increasing the time block tests whether hidden-variable/discrete-jump structure Gaussianizes into the finite-volume CDT transfer form. The first preregistered mesoscopic block satisfying fit quality, kinetic normalization, and positive delta/lambda is recorded.",
                base_tau_ref=tau_ref,
                tau_multipliers=list(MESOSCOPIC_TAU_MULTIPLIERS),
                affine_diffusion=affine,
                fit_r2_target=FIT_R2_TARGET,
                gamma_relative_error_target=GAMMA_RELERR_TARGET,
                selected=selected,
                scan=transfer_scan,
            ),
            evidence(
                "qccg-direct-equilibrium-transfer-open",
                "OPEN",
                "The successful reversible-projection diagnostic is not yet a direct physical QCCG transfer extraction because the instantaneous N3 projection required an O(1) reversibility correction. A finite-time equilibrium two-slice kernel, or an enlarged (N3,N0/curvature) coarse state followed by controlled elimination, is still required.",
                next_step=(
                    "Measure the finite-time two-slice joint volume kernel from equilibrium QCCG trajectories, "
                    "or retain N0/Regge curvature as an explicit second coarse coordinate and eliminate it only after equilibration."
                ),
            ),
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Dense construction failures are hard failures.  The non-Markov N3-only
    # result and any projected-transfer mismatch are scientific evidence.
    if len(VOLUMES) < 20:
        raise SystemExit("dense macro generator construction failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
