"""Molecular dynamics diagnostics beyond the MSD.

Caging and sampling diagnostics, non-affine (flow-subtracted) displacement,
Green-Kubo diffusion, the Yeh-Hummer finite-size correction, NEMD shear
viscosity, rotational correlation functions and potentials of mean force.
Units: positions Angstrom, times ps, velocities Angstrom/ps, temperatures K,
energies eV (kJ/mol also reported). The equations, their assumptions and how
to read the results are in docs/dynamics_diagnostics.md.
"""
import numpy as np
from ase import units

from . import md_statistics as md

KB_EV = units.kB                      # Boltzmann constant, eV/K
EV_TO_KJ_MOL = 96.48533212331002      # 1 eV per particle in kJ/mol
YEH_HUMMER_XI = 2.837297              # cubic-lattice constant (Yeh & Hummer 2004)


def _positions(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or x.shape[2] != 3 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite (frames, objects, 3) arrays")
    return x


def default_lags(samples, count=60):
    """About count unique integer lags 1..samples-1, evenly spaced in log(lag)."""
    if samples < 2:
        raise ValueError("Need at least two samples")
    return np.unique(np.geomspace(1, samples - 1, num=min(count, samples - 1)).astype(int))


def _displacements(x, lag, block=8192):
    """Yield blocks of x[t+lag]-x[t] over all origins, bounding memory."""
    for start in range(0, len(x) - lag, block):
        stop = min(start + block, len(x) - lag)
        yield x[start + lag:stop + lag] - x[start:stop]


# ----------------------------------------------------------------------------- non-affine displacement

def unwrap_nonaffine(wrapped_com, cells, origins, centers=None):
    """TOR trajectory with the cell's affine motion removed at every step.

    For each step, the TOR displacement mic(W_t - W_(t-1), H_t) minus
    md.affine_step: (o_t - o_(t-1)) + s_mid (H_t - H_(t-1)), the motion of a
    point that keeps its fractional coordinates s. Under fix deform this removes
    the imposed streaming flow u(s) = s dH/dt (with s from the box lo corner,
    the SLLOD/remap-v convention); under NPT it removes barostat dilation.
    Inputs as stored by analyze_run: wrapped_com_angstrom (relative to the cell
    center), cell_angstrom and cell_origin_angstrom; centers default to
    origin + (a+b+c)/2. Prefer analyze_run's com_nonaffine_angstrom, which
    uses every source frame; on strided data this is less accurate.
    """
    w = _positions(wrapped_com)
    cells = np.asarray(cells, dtype=float)
    origins = np.asarray(origins, dtype=float)
    if cells.shape != (len(w), 3, 3) or origins.shape != (len(w), 3):
        raise ValueError("cells must be (frames, 3, 3) and origins (frames, 3)")
    centers = origins + cells.sum(axis=1) / 2 if centers is None else np.asarray(centers, dtype=float)
    points = w + centers[:, None, :]
    out = np.empty_like(w)
    out[0] = w[0]
    for t in range(1, len(w)):
        step = md.minimum_image(points[t] - points[t - 1], cells[t])
        out[t] = out[t - 1] + step - md.affine_step(points[t - 1], step, cells[t - 1], cells[t],
                                                    origins[t - 1], origins[t])
    return out


# ----------------------------------------------------------------------------- caging diagnostics

def displacement_moments(positions, times_ps, *, lags=None):
    """MSD, fourth moment and non-Gaussian parameter versus lag (all origins).

    For d = R(t0+tau) - R(t0) over all origins t0 and objects:
    msd = <|d|^2>, mean_r4 = <|d|^4>, and in 3D
    alpha2 = 3 <|d|^4> / (5 <|d|^2>^2) - 1,
    which is zero for Gaussian (diffusive or ballistic) displacements and
    positive when a few objects move far while most stay caged.
    """
    x = _positions(positions)
    times = md.validate_times(times_ps, len(x), uniform=True)
    lags = default_lags(len(x)) if lags is None else np.asarray(lags, dtype=int)
    if np.any(lags < 1) or np.any(lags >= len(x)):
        raise ValueError("lags must lie in 1..frames-1")
    m2, m4 = [], []
    for lag in lags:
        s2 = s4 = 0.
        for d in _displacements(x, lag):
            r2 = np.einsum("tna,tna->tn", d, d)
            s2 += r2.sum()
            s4 += (r2 ** 2).sum()
        n = (len(x) - lag) * x.shape[1]
        m2.append(s2 / n)
        m4.append(s4 / n)
    m2, m4 = np.array(m2), np.array(m4)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha2 = 3 * m4 / (5 * m2 ** 2) - 1
    return {"lag_ps": lags * (times[1] - times[0]), "lag_frames": lags, "msd_angstrom2": m2,
            "mean_r4_angstrom4": m4, "alpha2": alpha2, "n_origins": len(x) - lags, "n_objects": x.shape[1]}


def van_hove_self(positions, times_ps, *, lags_ps, edges):
    """Self part of the van Hove function as a radial probability density.

    G_s(r, tau) = <delta(r - |d|)> / (4 pi r^2), d = R(t0+tau) - R(t0).
    Returned as p(r) = 4 pi r^2 G_s(r, tau) per Angstrom (integrates to the
    fraction of displacements below edges[-1]), together with the Gaussian
    with the same MSD: p_G(r) = 4 pi r^2 (3/(2 pi msd))^(3/2) exp(-3 r^2 / (2 msd)).
    A second peak or a tail above p_G marks hopping out of cages.
    """
    x = _positions(positions)
    times = md.validate_times(times_ps, len(x), uniform=True)
    dt = times[1] - times[0]
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be increasing and nonnegative")
    centers = (edges[1:] + edges[:-1]) / 2
    rows = []
    for tau in np.atleast_1d(lags_ps):
        lag = int(round(tau / dt))
        if not 1 <= lag < len(x):
            raise ValueError(f"lag {tau} ps is outside the trajectory")
        counts, total, sum_r2 = np.zeros(len(centers)), 0, 0.
        for d in _displacements(x, lag):
            r = np.sqrt(np.einsum("tna,tna->tn", d, d)).ravel()
            counts += np.histogram(r, bins=edges)[0]
            total += r.size
            sum_r2 += np.sum(r ** 2)
        msd = sum_r2 / total
        gaussian = 4 * np.pi * centers ** 2 * (3 / (2 * np.pi * msd)) ** 1.5 * np.exp(-3 * centers ** 2 / (2 * msd))
        rows.append({"lag_ps": lag * dt, "msd_angstrom2": msd, "r_angstrom": centers,
                     "p_r_per_angstrom": counts / (total * np.diff(edges)), "gaussian_p_r_per_angstrom": gaussian})
    return rows


def step_correlation(positions, times_ps, *, lags=None):
    """Correlation of consecutive displacements over a window of k frames.

    chi(k) = <d1 . d2> / <|d1|^2>, d1 = R(t+k)-R(t), d2 = R(t+2k)-R(t+k),
    averaged over origins and objects. Free diffusion gives 0, persistent
    (ballistic) motion > 0, and caged motion < 0; a slowly moving center plus
    fast motion that decorrelates within k frames gives -1/2.
    """
    x = _positions(positions)
    times = md.validate_times(times_ps, len(x), uniform=True)
    limit = (len(x) - 1) // 2
    if limit < 1:
        raise ValueError("Need at least three frames")
    lags = default_lags(limit + 1) if lags is None else np.asarray(lags, dtype=int)
    if np.any(lags < 1) or np.any(2 * lags >= len(x)):
        raise ValueError("lags must satisfy 1 <= k and 2k < frames")
    values = []
    for k in lags:
        d1 = x[k:len(x) - k] - x[:len(x) - 2 * k]
        d2 = x[2 * k:] - x[k:len(x) - k]
        values.append(np.sum(d1 * d2) / np.sum(d1 * d1))
    return {"lag_ps": lags * (times[1] - times[0]), "lag_frames": lags, "chi": np.array(values)}


def sampling_diagnostics(positions, velocities, times_ps):
    """Is motion faster than the frame spacing resolved?

    With C(0) = <|V|^2> (3 k_B T / M per molecule at equilibrium) and frame
    spacing dt, free flight gives MSD(dt) = C(0) dt^2. The ratio
    ballistic_ratio = MSD(dt) / (C(0) dt^2) is ~1 when the first saved lag is
    still ballistic (resolved) and << 1 when collisions/caging already
    happened within one frame. vacf_ratio = C(dt)/C(0) near zero also means
    the velocity memory is shorter than the frame spacing, so Green-Kubo
    integrals and short-time dynamics are not resolved. The ballistic
    crossover time sqrt(MSD(dt) / C(0)) estimates when free flight ends.
    """
    x, v = _positions(positions), _positions(velocities)
    if x.shape != v.shape:
        raise ValueError("positions and velocities must have the same shape")
    times = md.validate_times(times_ps, len(x), uniform=True)
    dt = float(times[1] - times[0])
    msd1 = float(md.lag_msd(x, 2)[1])
    c = md.lag_correlation(v, 2)
    ratio, vacf_ratio = msd1 / (c[0] * dt ** 2), float(c[1] / c[0])
    notes = []
    if ratio < .5:
        notes.append("Motion faster than the frame spacing is not resolved: MSD at the first lag is below half "
                     "the free-flight value. Apparent jumps at the first lag are sub-frame (ballistic/cage) motion.")
    if abs(vacf_ratio) < .2:
        notes.append("Velocity memory is shorter than the frame spacing: save frames more often for VACF/Green-Kubo.")
    return {"frame_spacing_ps": dt, "mean_square_velocity_angstrom2_ps2": float(c[0]),
            "msd_first_lag_angstrom2": msd1, "ballistic_ratio": ratio, "vacf_ratio_first_lag": vacf_ratio,
            "ballistic_crossover_ps": float(np.sqrt(msd1 / c[0])), "notes": notes}


# ----------------------------------------------------------------------------- transport

def green_kubo_diffusion(lag_ps, vacf, *, upper_ps=None):
    """Self-diffusion from the velocity autocorrelation, D = (1/3) int_0^inf <V(0).V(t)> dt.

    vacf is the unnormalized dot-product VACF in (Angstrom/ps)^2 (e.g. from
    molecular_window_statistics). Returns the running integral D(tau) by the
    trapezoidal rule and, with upper_ps, its value there. The plateau of D(tau)
    is the estimate; it must agree with the MSD slope (same quantity).
    """
    lag = np.asarray(lag_ps, dtype=float)
    c = np.asarray(vacf, dtype=float)
    if lag.shape != c.shape or len(lag) < 2 or lag[0] != 0 or np.any(np.diff(lag) <= 0):
        raise ValueError("lag_ps must start at 0, increase and match vacf")
    running = np.concatenate(([0.], np.cumsum(np.diff(lag) * (c[1:] + c[:-1]) / 2))) / 3
    result = {"lag_ps": lag, "running_D_angstrom2_per_ps": running, "running_D_cm2_per_s": running * 1e-4}
    if upper_ps is not None:
        if not lag[0] < upper_ps <= lag[-1]:
            raise ValueError("upper_ps must lie within the lag range")
        value = float(np.interp(upper_ps, lag, running))
        result.update(upper_ps=float(upper_ps), D_angstrom2_per_ps=value, D_cm2_per_s=value * 1e-4)
    return result


def yeh_hummer_correction(d_pbc_cm2_s, *, temperature_K, viscosity_pa_s, box_length_angstrom):
    """Infinite-system diffusion from a periodic cubic box (Yeh & Hummer, JPCB 108, 15873, 2004).

    D_inf = D_PBC + k_B T xi / (6 pi eta L), xi = 2.837297 for a cubic lattice
    of periodic images. SI units internally; D in cm^2/s. Only valid for cubic
    (or near-cubic) boxes; eta is the shear viscosity of the same model.
    """
    for name, value in (("temperature_K", temperature_K), ("viscosity_pa_s", viscosity_pa_s),
                        ("box_length_angstrom", box_length_angstrom)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    correction_m2_s = units._k * temperature_K * YEH_HUMMER_XI / (6 * np.pi * viscosity_pa_s * box_length_angstrom * 1e-10)
    correction = correction_m2_s * 1e4
    return {"D_pbc_cm2_s": float(d_pbc_cm2_s), "correction_cm2_s": float(correction),
            "D_infinite_cm2_s": float(d_pbc_cm2_s + correction), "xi": YEH_HUMMER_XI}


def cubic_box_length(box_lengths_angstrom, *, tolerance=.01):
    """Mean edge of a (near-)cubic box; refuses anisotropy above tolerance."""
    lengths = np.asarray(box_lengths_angstrom, dtype=float).reshape(-1, 3)
    mean = lengths.mean(axis=0)
    if np.ptp(mean) / mean.mean() > tolerance:
        raise ValueError("Box is not cubic; the Yeh-Hummer constant 2.837297 does not apply")
    return float(mean.mean())


def nemd_shear_viscosity(pxy_bar, *, shear_rate_per_ps, blocks=10):
    """Shear viscosity from a steady-shear (fix deform + nvt/sllod) run.

    eta = -<P_xy> / gamma_dot, with LAMMPS's pressure tensor P_xy (bar) and
    gamma_dot = d v_x / d y = the xy erate (1/ps). Units: 1 bar = 1e5 Pa,
    1/ps = 1e12 1/s. The uncertainty is the standard error of block means.
    Valid only in the linear-response regime and after the stress reaches
    steady state; at low shear rates the signal may be smaller than the noise.
    """
    p = np.asarray(pxy_bar, dtype=float)
    if p.ndim != 1 or len(p) < 2 * blocks or not np.isfinite(p).all():
        raise ValueError("Need a finite 1D P_xy series with at least two samples per block")
    if not np.isfinite(shear_rate_per_ps) or shear_rate_per_ps == 0:
        raise ValueError("shear_rate_per_ps must be finite and nonzero")
    scale = -1e5 / (shear_rate_per_ps * 1e12)
    means = p[:len(p) // blocks * blocks].reshape(blocks, -1).mean(axis=1) * scale
    return {"viscosity_pa_s": float(p.mean() * scale), "standard_error_pa_s": float(means.std(ddof=1) / np.sqrt(blocks)),
            "block_viscosities_pa_s": means.tolist(), "shear_rate_per_ps": float(shear_rate_per_ps)}


# ----------------------------------------------------------------------------- rotation

def orientation_correlation(axes, times_ps, *, order=2, nlag=None):
    """Rotational correlation C_l(tau) = <P_l(u(t0) . u(t0+tau))> over origins and molecules.

    axes is (frames, molecules, 3) unit vectors (e.g. orientation_axes[:, :, 0]
    for long axes, [:, :, 1] for plane normals). P1(x) = x needs sign-continuous
    vectors; P2(x) = (3x^2 - 1)/2 does not depend on the sign. Both are
    computed by FFT: P1 is a vector correlation; for P2, <(u.u')^2> is the
    correlation of the tensor q_ab = u_a u_b summed over its 9 components.
    For small-step rotational diffusion, C_l = exp(-l(l+1) D_r tau).
    """
    u = _positions(axes)
    norms = np.linalg.norm(u, axis=-1)
    if not np.allclose(norms, 1, atol=1e-6):
        raise ValueError("axes must be unit vectors")
    times = md.validate_times(times_ps, len(u), uniform=True)
    nlag = len(u) if nlag is None else int(nlag)
    if order == 1:
        c = md.lag_correlation(u, nlag)
    elif order == 2:
        q = np.einsum("tna,tnb->tnab", u, u).reshape(len(u), -1, 3)   # 9 components as 3 groups of 3
        c = (3 * (3 * md.lag_correlation(q, nlag)) - 1) / 2
    else:
        raise ValueError("order must be 1 or 2")
    return {"lag_ps": np.arange(nlag) * (times[1] - times[0]), "correlation": c, "order": order,
            "n_molecules": u.shape[1]}


def correlation_time(lag_ps, correlation, *, upper_ps=None):
    """Integral correlation time tau = int_0 C(t)/C(0) dt, up to upper_ps or the first zero.

    With C_l = exp(-l(l+1) D_r t): tau_1 = 1/(2 D_r), tau_2 = 1/(6 D_r); the
    returned rotational_diffusion_per_ps assumes that form (needs the order).
    """
    lag, c = np.asarray(lag_ps, dtype=float), np.asarray(correlation, dtype=float)
    if c[0] <= 0:
        raise ValueError("Correlation must be positive at lag 0")
    rho = c / c[0]
    stop = len(rho) if upper_ps is None else int(np.searchsorted(lag, upper_ps, side="right"))
    zero = np.flatnonzero(rho[1:stop] <= 0)
    stop = int(zero[0]) + 1 if len(zero) else stop
    tau = float(np.sum(np.diff(lag[:stop]) * (rho[1:stop] + rho[:stop - 1]) / 2))
    return {"tau_ps": tau, "integrated_to_ps": float(lag[stop - 1]), "reached_zero": bool(len(zero))}


# ----------------------------------------------------------------------------- free energies

def pmf_from_rdf(r_angstrom, g_r, *, temperature_K):
    """Potential of mean force W(r) = -k_B T ln g(r) (zero where g = 1, the bulk)."""
    g = np.asarray(g_r, dtype=float)
    kt = KB_EV * temperature_K
    with np.errstate(divide="ignore"):
        w = -kt * np.log(g)
    return {"r_angstrom": np.asarray(r_angstrom, dtype=float), "pmf_eV": w, "pmf_kJ_mol": w * EV_TO_KJ_MOL,
            "kT_eV": kt}


def pmf_1d(values, *, temperature_K, edges, jacobian=None):
    """W(x) = -k_B T ln[P(x) / J(x)], shifted so min W = 0, from samples x.

    jacobian="sin" for an angle between two directions in 3D (degrees), whose
    uniform reference density is proportional to sin(theta); "r2" for a
    radial distance (reference 4 pi r^2); None for no reference. Bins with no
    samples are +inf. This is a free energy only for the sampled ensemble:
    pair geometries exist only inside the 3.25 Angstrom screen, so it is a
    conditional PMF there.
    """
    x = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    counts = np.histogram(x[np.isfinite(x)], bins=edges)[0].astype(float)
    centers = (edges[1:] + edges[:-1]) / 2
    density = counts / max(counts.sum(), 1) / np.diff(edges)
    if jacobian == "sin":
        reference = np.sin(np.radians(centers))
    elif jacobian == "r2":
        reference = centers ** 2
    elif jacobian is None:
        reference = np.ones_like(centers)
    else:
        raise ValueError("jacobian must be None, 'sin' or 'r2'")
    kt = KB_EV * temperature_K
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -kt * np.log(density / reference)
    w[~np.isfinite(w)] = np.inf
    if np.isfinite(w).any():
        w -= w[np.isfinite(w)].min()
    return {"x": centers, "counts": counts, "pmf_eV": w, "pmf_kJ_mol": w * EV_TO_KJ_MOL, "kT_eV": kt}


def pmf_2d(x, y, *, temperature_K, x_edges, y_edges, y_jacobian=None):
    """Two-dimensional W(x, y) = -k_B T ln[P(x, y) / J(y)], min 0 (e.g. distance vs. angle)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    counts = np.histogram2d(x[keep], y[keep], bins=[x_edges, y_edges])[0]
    yc = (np.asarray(y_edges)[1:] + np.asarray(y_edges)[:-1]) / 2
    reference = np.sin(np.radians(yc)) if y_jacobian == "sin" else np.ones_like(yc)
    kt = KB_EV * temperature_K
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -kt * np.log(counts / counts.sum() / reference[None, :])
    w[~np.isfinite(w)] = np.inf
    if np.isfinite(w).any():
        w -= w[np.isfinite(w)].min()
    return {"x": (np.asarray(x_edges)[1:] + np.asarray(x_edges)[:-1]) / 2, "y": yc, "counts": counts,
            "pmf_eV": w, "pmf_kJ_mol": w * EV_TO_KJ_MOL}


def load_pair_geometries(index_jsonl):
    """Per-candidate geometry from an export's index.jsonl as arrays.

    Returns frame, time_ps, max_forming_distance and distance_imbalance
    (Angstrom), axis/plane/approach angles (degrees); NaN where the geometry
    could not be measured.
    """
    import json
    from pathlib import Path
    fields = ("frame", "time_ps", "max_forming_distance", "distance_imbalance",
              "axis_angle", "plane_angle", "approach_angle")
    rows = {name: [] for name in fields}
    for line in Path(index_jsonl).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        g = record.get("geometry")
        rows["frame"].append(record["frame"])
        rows["time_ps"].append(record["time_ps"])
        distances = g["forming_distances"] if g else [np.nan, np.nan]
        rows["max_forming_distance"].append(max(distances))
        rows["distance_imbalance"].append(abs(distances[0] - distances[1]))
        for name in ("axis_angle", "plane_angle", "approach_angle"):
            rows[name].append(g[name] if g else np.nan)
    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}
