#!/usr/bin/env python3
"""Direct finite-time equilibrium QCCG two-slice transfer extraction.

This audit avoids the instantaneous N3-only generator that failed the
Kolmogorov cycle test.

Method
------
1. Select the curvature-equilibrated QCCG critical-line couplings.
2. Run one long weighted continuous-time Pachner trajectory inside a hard
   volume window.  The hard wall only reflects moves leaving the window.
3. After every accepted local move, randomly relabel all vertices.  Vertex
   names are gauge, so this deliberately erases birth-order label memory.
4. After burn-in, sample N3(t) and N0(t) at uniform microscopic-time intervals.
5. For several finite lags, form the empirical two-time joint distribution
      J_tau(n,m) = Prob[N3(t)=n, N3(t+tau)=m].
6. Test time-reversal symmetry J(n,m)=J(m,n) directly.
7. When the asymmetry is compatible with a direct reversible finite-time
   reduction, construct
      K_sym(n,m) = J_sym(n,m) / sqrt(pi_n pi_m)
   and fit -log K_sym to the finite-volume CDT transfer action.

No reversible projection of an instantaneous N3 generator is used.

Primary CDT anchors:
- Ambjorn et al., JHEP 09 (2012) 017, arXiv:1205.3791.
- Ambjorn et al., JHEP 06 (2014) 034, arXiv:1403.5940.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import run_qccg_symmetric_transfer_extraction as sym
import run_qccg_curvature_weighted_time_exploration as explore
import run_qccg_time_evolved_volume_kernel as tev
import run_time_local_slice_transfer_toy as qslice


VOLUME_MIN = 20
VOLUME_MAX = 44
CENTER_VOLUME = 32
BURN_TIME = 35.0
SAMPLE_DT = 0.02
N_SAMPLES = 12000
LAG_STEPS = (4, 7, 10)  # tau = 0.08, 0.14, 0.20
SEED = 20271007

JOINT_ASYMMETRY_MAX = 0.10
MIN_OCCUPANCY = 0.002
FIT_R2_TARGET = 0.88
N0_SCAN = tuple(-5.0 + 0.25 * i for i in range(41))
MIN_FIT_ROWS = 45

SOURCES = sym.SOURCES


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "qccg-finite-time-equilibrium-transfer",
        "artifact": "qccg/run_qccg_finite_time_equilibrium_transfer.py",
        "note": note,
        "metadata": metadata,
    }


def random_relabel(S, rng):
    """Uniformly randomize the compact vertex labels of one geometry."""
    verts = sorted({v for tet in S for v in tet})
    new_labels = list(range(len(verts)))
    rng.shuffle(new_labels)
    mp = dict(zip(verts, new_labels))
    return {
        tuple(sorted(mp[v] for v in tet))
        for tet in S
    }


def n0(S):
    return len({v for tet in S for v in tet})


def allowed_event_lists(S):
    n = len(S)
    c14, c41, c23, c32 = tev.event_lists(S)
    if n + 3 > VOLUME_MAX:
        c14 = []
    if n - 3 < VOLUME_MIN:
        c41 = []
    if n + 1 > VOLUME_MAX:
        c23 = []
    if n - 1 < VOLUME_MIN:
        c32 = []
    return {
        "14": c14,
        "41": c41,
        "23": c23,
        "32": c32,
    }


def apply_event(S, typ, candidate, rng):
    if typ == "14":
        used = {v for tet in S for v in tet}
        S2 = qslice.apply14(S, (candidate, max(used) + 1))
    elif typ == "41":
        S2 = tev.apply41(S, candidate)
    elif typ == "23":
        S2 = qslice.apply23(S, candidate)
    elif typ == "32":
        S2 = qslice.apply32(S, candidate)
    else:
        raise ValueError(typ)
    if not qslice.manifold(S2):
        raise RuntimeError(f"{typ} broke manifold")
    return random_relabel(S2, rng)


def step_event(S, rng, rates):
    cs = allowed_event_lists(S)
    weights = {
        typ: len(cs[typ]) * rates[typ]
        for typ in ("14", "41", "23", "32")
    }
    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError("trajectory has no allowed moves")

    dt = rng.expovariate(total)
    u = rng.random() * total
    acc = 0.0
    chosen = None
    for typ in ("14", "41", "23", "32"):
        acc += weights[typ]
        if u <= acc:
            chosen = typ
            break
    if chosen is None:
        chosen = "32"

    if not cs[chosen]:
        raise RuntimeError("selected empty move family")
    candidate = rng.choice(cs[chosen])
    return apply_event(S, chosen, candidate, rng), dt, chosen


def burn(S, rng, rates):
    t = 0.0
    moves = collections.Counter()
    while t < BURN_TIME:
        S, dt, typ = step_event(S, rng, rates)
        t += dt
        moves[typ] += 1
    return S, moves


def sample_series(S, rng, rates):
    series_n3 = []
    series_n0 = []
    move_counts = collections.Counter()
    t = 0.0
    next_sample = 0.0

    while len(series_n3) < N_SAMPLES:
        S2, dt, typ = step_event(S, rng, rates)
        event_time = t + dt

        while (
            next_sample <= event_time
            and len(series_n3) < N_SAMPLES
        ):
            series_n3.append(len(S))
            series_n0.append(n0(S))
            next_sample += SAMPLE_DT

        S = S2
        t = event_time
        move_counts[typ] += 1

    return series_n3, series_n0, move_counts, t


def joint_counts(series, lag):
    vols = list(range(VOLUME_MIN, VOLUME_MAX + 1))
    idx = {n: i for i, n in enumerate(vols)}
    J = [[0 for _ in vols] for _ in vols]
    for a, b in zip(series[:-lag], series[lag:]):
        J[idx[a]][idx[b]] += 1
    return vols, J


def joint_asymmetry(J):
    num = 0.0
    den = 0.0
    for i in range(len(J)):
        for j in range(i + 1, len(J)):
            num += abs(J[i][j] - J[j][i])
            den += J[i][j] + J[j][i]
    return num / den if den > 0 else float("inf")


def direct_symmetric_kernel(vols, J, series):
    counts = collections.Counter(series)
    total = len(series)
    pi = [counts[n] / total for n in vols]

    pair_total = sum(sum(row) for row in J)
    M = [[0.0 for _ in vols] for _ in vols]
    for i in range(len(vols)):
        for j in range(len(vols)):
            jsym = 0.5 * (J[i][j] + J[j][i]) / pair_total
            if pi[i] > 0 and pi[j] > 0:
                M[i][j] = jsym / math.sqrt(pi[i] * pi[j])

    scale = max(max(row) for row in M)
    if scale <= 0:
        raise RuntimeError("empty symmetric finite-time kernel")
    M = [[x / scale for x in row] for row in M]
    return M, pi


def fit_scan_n0(vols, M):
    fits = []
    for shift in N0_SCAN:
        try:
            fit = sym.fit_cdt_kernel(vols, M, shift)
        except Exception:
            continue
        fit = {**fit, "n0": shift}
        if (
            fit["inverse_Gamma"] > 0
            and math.isfinite(fit["delta"])
            and math.isfinite(fit["lambda"])
            and fit["n_rows"] >= MIN_FIT_ROWS
        ):
            fits.append(fit)
    if not fits:
        return None, []
    best = max(fits, key=lambda x: x["r2"])
    return best, fits


def occupancy_summary(n3_series, n0_series):
    n3c = collections.Counter(n3_series)
    by_n3 = collections.defaultdict(list)
    for n3v, n0v in zip(n3_series, n0_series):
        by_n3[n3v].append(n0v)
    rows = []
    for n in range(VOLUME_MIN, VOLUME_MAX + 1):
        vals = by_n3.get(n, [])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        rows.append({
            "N3": n,
            "occupancy": n3c[n] / len(n3_series),
            "N0_mean": mean,
            "N0_std": math.sqrt(var),
            "N0_min": min(vals),
            "N0_max": max(vals),
        })
    return rows


def main():
    kR, kV, critical_residual = sym.critical_line()
    rates = explore.weighted_rates(kR, kV)

    rng = random.Random(SEED)
    S = sym.build_any_target(CENTER_VOLUME, SEED + 101)
    S = random_relabel(S, rng)

    S, burn_moves = burn(S, rng, rates)
    n3_series, n0_series, moves, sampled_time = sample_series(
        S, rng, rates
    )

    occ = occupancy_summary(n3_series, n0_series)
    occupied = [r for r in occ if r["occupancy"] >= MIN_OCCUPANCY]

    lag_rows = []
    selected = None
    for lag in LAG_STEPS:
        vols, J = joint_counts(n3_series, lag)
        asym = joint_asymmetry(J)
        M, pi = direct_symmetric_kernel(vols, J, n3_series)
        best, fits = fit_scan_n0(vols, M)

        row = {
            "lag_steps": lag,
            "tau": lag * SAMPLE_DT,
            "joint_asymmetry": asym,
            "best_fit": best,
            "n_fits": len(fits),
            "occupied_volume_count": len(occupied),
        }
        row_pass = bool(
            best
            and asym <= JOINT_ASYMMETRY_MAX
            and best["r2"] >= FIT_R2_TARGET
            and best["delta"] > 0
            and best["lambda"] > 0
        )
        row["direct_transfer_pass"] = row_pass
        lag_rows.append(row)
        if selected is None and row_pass:
            selected = row

    direct_pass = selected is not None
    hidden_variable_visible = any(
        r["N0_max"] > r["N0_min"]
        for r in occ
    )

    result = {
        "schema": 1,
        "scope": "direct finite-time equilibrium N3 two-slice kernel from one long gauge-randomized QCCG trajectory in a reflecting volume window",
        "sources": SOURCES,
        "critical_line": {
            "kappa_R": kR,
            "kappa_V": kV,
            "zero_drift_residual": critical_residual,
            "rates": rates,
        },
        "trajectory": {
            "volume_window": [VOLUME_MIN, VOLUME_MAX],
            "center_volume": CENTER_VOLUME,
            "burn_time": BURN_TIME,
            "sample_dt": SAMPLE_DT,
            "n_samples": N_SAMPLES,
            "sampled_time": sampled_time,
            "burn_moves": dict(burn_moves),
            "sample_moves": dict(moves),
            "occupancy": occ,
        },
        "evidence": [
            evidence(
                "qccg-finite-time-gauge-randomized-trajectory",
                "QCCG_FINITE_TIME_EQUILIBRIUM_TRAJECTORY",
                "PASS" if len(occupied) >= 8 else "FAIL",
                "A long curvature-weighted QCCG trajectory is sampled after burn-in inside a reflecting volume window, with a random gauge relabelling after every local move to erase vertex birth-order memory.",
                occupied_volume_count=len(occupied),
                minimum_occupancy=MIN_OCCUPANCY,
                occupancy=occ,
                burn_moves=dict(burn_moves),
                sample_moves=dict(moves),
            ),
            evidence(
                "qccg-hidden-n0-observed",
                "QCCG_HIDDEN_N0_COARSE_VARIABLE_DIAGNOSTIC",
                "PASS" if hidden_variable_visible else "NOT_APPLICABLE",
                "At fixed N3 the equilibrium trajectory visits multiple N0 values in at least one sector, directly exhibiting a hidden geometric coordinate that is absent from the instantaneous N3-only reduction.",
                occupancy=occ,
            ),
            evidence(
                "qccg-direct-finite-time-two-slice-kernel",
                "QCCG_FINITE_TIME_EQUILIBRIUM_TRANSFER_EXTRACTION",
                "PASS" if direct_pass else "FAIL",
                "The empirical finite-time two-slice joint volume distribution is tested for time-reversal symmetry and converted directly into a symmetric kernel without projecting an instantaneous N3 generator. The first preregistered lag with CDT fit quality and positive delta/lambda is selected.",
                joint_asymmetry_max=JOINT_ASYMMETRY_MAX,
                fit_r2_target=FIT_R2_TARGET,
                n0_scan=[min(N0_SCAN), max(N0_SCAN), 0.25],
                selected=selected,
                lag_scan=lag_rows,
            ),
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # A direct physical mismatch is retained as evidence; only trajectory
    # construction failure is an infrastructure failure.
    if len(occupied) < 8:
        raise SystemExit("finite-time equilibrium trajectory failed to explore enough volumes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
