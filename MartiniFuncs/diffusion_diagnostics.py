"""Molecular diffusion and scalar stationarity screens with block uncertainty.

These diagnostics can reject unsupported estimates; they cannot prove equilibrium.
Bootstrap intervals assume approximately independent temporal blocks. All molecules
stay together within each block, preserving their instantaneous cross-correlations.
No uncertainty is inferred by treating overlapping MSD lags as independent data.
"""
import numpy as np

from .md_statistics import lag_msd, validate_times
from .stability import scalar_autocorrelation, statistical_inefficiency


def _times(times, count):
    """Validate uniformly spaced times in ps (shared validator) and return their spacing."""
    times = validate_times(times, count, minimum=3, uniform=True)
    return float(times[1] - times[0])


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _mean_interval(values, rng, resamples):
    """Percentile 95% CI of a mean from resampled whole temporal blocks."""
    means = np.mean(rng.choice(values, size=(resamples, len(values)), replace=True), axis=1)
    return np.quantile(means, [.025, .975])


def _msd(positions, steps):
    """MSD[k]=mean_(t,m) sum_xyz (R[t+k,m]-R[t,m])**2 at the requested lags, Angstrom^2."""
    steps = np.asarray(steps, dtype=int)
    return lag_msd(positions, int(steps.max()) + 1)[steps]


def diffusion_diagnostics(positions, times_ps, *, fit_intervals_ps, block_frames,
                          min_blocks=8, resamples=2000, seed=1234,
                          alpha_bounds=(.8, 1.2), relative_tolerance=.25,
                          max_relative_ci=.5, lag_points=100):
    """Screen one species' TOR COM trajectory; never certify equilibrium.

    positions is (T, molecules, 3), in Angstrom; times_ps is uniform. For each
    explicit lag interval, fit MSD=a+6*D*t with a free intercept. Also fit
    log(MSD)=c+alpha*log(t); alpha near one screens for diffusive (Fickian)
    scaling. A sizeable intercept (cage rattling faster than the frame spacing)
    keeps alpha below one until 6*D*t dominates it; that is intended, because
    such a curve is not yet in the diffusive regime. For information only,
    alpha_plateau_corrected fits log(MSD-max(a,0)) and residual_rms_angstrom2
    is the linear-fit residual; neither enters the screens (a flat, noisy
    confined MSD can give alpha_plateau_corrected near one). At most
    lag_points evenly distributed integer lags bound the work.

    Split samples into consecutive, nonoverlapping block_frames blocks, discarding
    the incomplete tail for block statistics only. Each block supplies its own D.
    Report their mean and percentile bootstrap 95% interval (resample whole blocks,
    including all molecules together). D is reported in Angstrom^2/ps and in
    cm^2/s (x1e-4), the same units as md_statistics.get_molecular_diffusion_coefficient.
    No displacement crosses a block boundary. Block span must be >=5*maximum lag.

    Screens require min_blocks, positive CI, CI half-width/mean<=max_relative_ci,
    alpha within alpha_bounds, and earlier/later block means differing by no more
    than relative_tolerance*abs(mean). Fit sensitivity is (max(D)-min(D))/abs(mean(D))
    across >=3 intervals; pooled-vs-block D must also agree within that tolerance.
    These are explicit heuristic thresholds, not formal hypothesis tests. Repeat
    with longer blocks, different source windows, and independent simulations.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 3 or p.shape[2] != 3 or p.shape[1] == 0 or not np.isfinite(p).all():
        raise ValueError("Require finite positions (frames, molecules, 3)")
    dt = _times(times_ps, len(p))
    _integer(block_frames, "block_frames", 3)
    _integer(min_blocks, "min_blocks", 4)
    _integer(resamples, "resamples", 100)
    _integer(lag_points, "lag_points", 10)
    bounds = np.asarray(alpha_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("alpha_bounds must be positive increasing bounds")
    screens = [relative_tolerance, max_relative_ci]
    if not np.isfinite(screens).all() or min(screens) <= 0:
        raise ValueError("Screen tolerances must be positive and finite")
    intervals = np.asarray(fit_intervals_ps, dtype=float)
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals)
            or not np.isfinite(intervals).all() or np.any(intervals[:, 0] <= 0)
            or np.any(intervals[:, 1] <= intervals[:, 0])):
        raise ValueError("Supply positive increasing (start, end) lag intervals in ps")
    maximum = float(intervals[:, 1].max())
    largest = min(len(p)-1, int(np.floor(maximum/dt)))
    steps = np.unique(np.r_[0, np.linspace(1, max(1, largest), min(lag_points, max(1, largest))).astype(int)])
    lag = steps*dt
    msd = _msd(p, steps)
    count = len(p)//block_frames
    # Skip block curves when they cannot cover the requested lag range.
    block_msd = np.array([_msd(p[i*block_frames:(i+1)*block_frames], steps) for i in range(count)]) if steps[-1] < block_frames else np.empty((0, len(steps)))
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in intervals:
        mask = (lag >= lo) & (lag <= hi) & (lag > 0)
        row = {"fit_start_ps": float(lo), "fit_end_ps": float(hi), "reasons": []}
        reasons = row["reasons"]
        if mask.sum() < 4 or hi > (len(p)-1)*dt:
            reasons.append("Fit interval unavailable or fewer than four sampled lags")
            row["status"] = "insufficient sampling"
            rows.append(row)
            continue
        slope, intercept = np.polyfit(lag[mask], msd[mask], 1)
        values = msd[mask]
        alpha = float(np.polyfit(np.log(lag[mask]), np.log(values), 1)[0]) if np.all(values > 0) else None
        corrected = values - max(intercept, 0.)
        corrected_alpha = float(np.polyfit(np.log(lag[mask]), np.log(corrected), 1)[0]) if np.all(corrected > 0) else None
        row.update(D_pooled_angstrom2_ps=float(slope/6), D_pooled_cm2_s=float(slope/6*1e-4),
                   intercept_angstrom2=float(intercept), alpha=alpha, alpha_plateau_corrected=corrected_alpha,
                   residual_rms_angstrom2=float(np.sqrt(np.mean((values - (intercept + slope*lag[mask]))**2))))
        if alpha is None or not bounds[0] <= alpha <= bounds[1]:
            reasons.append("MSD scaling is not close to linear in lag")
        if count < min_blocks or len(block_msd) < min_blocks:
            reasons.append(f"Need at least {min_blocks} complete blocks covering the fit lags")
        if (block_frames-1)*dt < 5*hi:
            reasons.append("Block span is shorter than five times the maximum fit lag")
        if len(block_msd) >= 2:
            ds = np.array([np.polyfit(lag[mask], curve[mask], 1)[0]/6 for curve in block_msd])
            mean = float(ds.mean())
            ci = _mean_interval(ds, rng, resamples)
            split = len(ds)//2
            shift = abs(ds[:split].mean()-ds[split:].mean())/max(abs(mean), 1e-30)
            row.update(D_block_angstrom2_ps=mean, D_cm2_s=mean*1e-4,
                       ci95_cm2_s=(ci*1e-4).tolist(), block_D_angstrom2_ps=ds.tolist(),
                       temporal_relative_shift=float(shift))
            if mean <= 0 or ci[0] <= 0:
                reasons.append("Positive diffusion is not resolved by the block interval")
            if (ci[1]-ci[0])/2 > max_relative_ci*abs(mean):
                reasons.append("Block uncertainty exceeds the requested relative precision")
            if shift > relative_tolerance:
                reasons.append("Earlier and later block estimates disagree")
            if abs(slope/6-mean) > relative_tolerance*max(abs(mean), 1e-30):
                reasons.append("Pooled and block estimates disagree")
        row["status"] = "candidate" if not reasons else "not supported"
        rows.append(row)
    estimates = [r["D_pooled_angstrom2_ps"] for r in rows if "D_pooled_angstrom2_ps" in r]
    distinct_intervals = len(np.unique(intervals, axis=0))
    sensitivity = float(np.ptp(estimates)/max(abs(np.mean(estimates)), 1e-30)) if len(estimates) >= 3 and distinct_intervals >= 3 else None
    stable = sensitivity is not None and sensitivity <= relative_tolerance
    if not stable:
        for row in rows:
            row["reasons"].append("Fit-interval stability not established (need >=3 intervals and consistent D)")
            if row["status"] == "candidate":
                row["status"] = "not supported"
    return {"lag_ps": lag, "msd_angstrom2": msd, "n_origins": len(p)-steps,
            "block_frames": int(block_frames), "blocks": count,
            "discarded_block_samples": len(p) % block_frames,
            "fit_relative_spread": sensitivity, "fits": rows}


def scalar_stationarity(values, times_ps, *, block_frames, tolerance=None,
                         resamples=2000, seed=1234):
    """Screen scalar stationarity with a user-defined absolute tolerance.

    Use complete consecutive blocks of b samples. Difference = mean(later block
    means)-mean(earlier block means). Bootstrap both groups of block means to get
    a percentile 95% interval; require the entire interval within [-tolerance,
    tolerance], in the observable's units. At least four blocks per group are
    required. Constant traces are inconclusive, not evidence of sampling.

    FFT autocovariance is sum_i(x_i-mu)(x_(i+k)-mu)/(N-k). Normalize by C[0],
    truncate before its first nonpositive lag, and estimate g=1+2*sum(rho[k]),
    tau=dt*g/2, N_eff=N/g. Require block span>=10*tau. This first-zero heuristic
    can miss slow modes; no zero crossing is inconclusive. Density, energy,
    pressure, temperature and structural observables should be checked separately.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Require a finite one-dimensional scalar trace")
    dt = _times(times_ps, len(x))
    _integer(block_frames, "block_frames", 3)
    _integer(resamples, "resamples", 100)
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be a positive absolute value")
    count = len(x)//block_frames
    means = x[:count*block_frames].reshape(count, block_frames).mean(axis=1) if count else np.array([])
    reasons = []
    result = {"mean": float(x.mean()), "blocks": count, "block_means": means.tolist(),
              "discarded_samples": len(x) % block_frames, "tolerance": tolerance, "reasons": reasons}
    acf = scalar_autocorrelation(x)
    if acf is None:
        reasons.append("Constant observable cannot establish independent sampling")
    else:
        g, found_zero = statistical_inefficiency(acf)
        tau = dt*g/2
        result.update(correlation_time_ps=tau, effective_samples_heuristic=len(x)/g)
        if not found_zero or (block_frames-1)*dt < 10*tau:
            reasons.append("Blocks do not resolve estimated correlation times")
    if count < 8:
        reasons.append("Need at least eight complete temporal blocks")
    else:
        a, b = np.array_split(means, 2)
        rng = np.random.default_rng(seed)
        diffs = (rng.choice(b, (resamples, len(b))).mean(axis=1)
                 - rng.choice(a, (resamples, len(a))).mean(axis=1))
        ci = np.quantile(diffs, [.025, .975])
        result.update(late_minus_early=float(b.mean()-a.mean()), difference_ci95=ci.tolist())
        if tolerance is not None and (ci[0] < -tolerance or ci[1] > tolerance):
            reasons.append("Early/late equivalence is not established within tolerance")
    if tolerance is None:
        reasons.append("Set a physically meaningful absolute tolerance to assess equivalence")
    result["status"] = "supports stationarity" if not reasons else "inconclusive or unstable"
    return result
