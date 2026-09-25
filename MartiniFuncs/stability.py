"""Descriptive trajectory stability diagnostics, not equilibrium pass/fail tests."""
import numpy as np


def get_series_statistics(values, *, sample_interval_ps, batch_sizes=(20, 50, 100)):
    """Scalar time-series summary and uncertainty sensitivity to batch duration.

    Batch-mean standard errors assume approximately independent batches. They
    are diagnostics until a plateau with increasing batch size is demonstrated.
    Only complete batches contribute to each SE; discarded samples are reported.
    ACF uses the full-series mean and lag-dependent pair counts. Its integral is
    truncated at the first nonpositive lag: a heuristic, not a convergence test.

    For N samples x, mu=mean(x), s=sqrt(sum((x-mu)^2)/(N-1)).
    The drift slope is an unweighted free-intercept fit at t_i=i*sample_interval_ps.
    C[k]=sum_i (x_i-mu)(x_(i+k)-mu)/((N-k)*mean((x-mu)^2)).
    With K the first k>=1 having C[k]<=0 (or N if absent),
    g=1+2*sum_{k=1}^{K-1} C[k], N_eff=N/g, and tau=dt*g/2.
    For batch size b, B=floor(N/b); average each consecutive b samples and
    report SE=std(batch_means, ddof=1)/sqrt(B), dropping the final N-B*b samples.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Require at least three finite scalar samples")
    if not np.isfinite(sample_interval_ps) or sample_interval_ps <= 0:
        raise ValueError("sample_interval_ps must be positive")
    centered = x - x.mean()
    variance = np.mean(centered**2)
    result = {"samples": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
              "quantiles_05_50_95": np.quantile(x, [.05, .5, .95]).tolist(),
              "slope_per_ps": float(np.polyfit(np.arange(len(x))*sample_interval_ps, x, 1)[0]),
              "batch_mean_errors": []}
    if variance > 0:
        acf = np.correlate(centered, centered, mode="full")[len(x)-1:]
        acf /= np.arange(len(x), 0, -1) * variance
        nonpositive = np.flatnonzero(acf[1:] <= 0)
        stop = int(nonpositive[0]+1) if len(nonpositive) else len(x)
        g = float(1 + 2*np.sum(acf[1:stop]))
        result.update(statistical_inefficiency_first_zero=g,
                      effective_samples_heuristic=float(len(x)/g),
                      correlation_integral_ps=float(.5*g*sample_interval_ps),
                      acf_lag_ps=(np.arange(min(len(x), 101))*sample_interval_ps).tolist(),
                      acf=acf[:101].tolist())
    else:
        result.update(statistical_inefficiency_first_zero=None,
                      effective_samples_heuristic=None, correlation_integral_ps=None,
                      acf_lag_ps=[], acf=[])
    for size in batch_sizes:
        if not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError("Batch sizes must be integer sample counts >= 2")
        count = len(x)//size
        if count < 2:
            continue
        means = x[:count*size].reshape(count, size).mean(axis=1)
        result["batch_mean_errors"].append({"batch_duration_ps": size*sample_interval_ps,
            "batches": count, "discarded_samples": len(x)-count*size,
            "standard_error": float(means.std(ddof=1)/np.sqrt(count))})
    return result


def get_molecular_com_rdf(positions, box_lengths, species, *, species_a, species_b, edges):
    """Species COM RDF for fully periodic orthorhombic frames (Angstrom).

    Positions must represent each frame's actual geometry, not TOR trajectories.
    Uses unique unordered pairs for identical species, all cross pairs otherwise.
    Normalize each frame by its volume and available pair count, then average;
    the finite-population ideal-gas baseline is one. Also returns nearest-neighbor
    distances per central molecule per frame (A -> B, excluding self).

    For each frame, d_ij=r_j-r_i-L*floor((r_j-r_i)/L+1/2), componentwise.
    Bin |d_ij| using edges; shell volume V_s=4*pi/3*(edge_hi^3-edge_lo^3).
    g_s=mean_frames(count_s*prod(L)/(P*V_s)), where P=N_A*N_B for
    unlike species and P=N_A*(N_A-1)/2 for identical species (unique pairs).
    Nearest distances are min_j |d_ij| for each A molecule, excluding itself;
    they are not truncated at edges[-1].
    """
    p = np.asarray(positions, dtype=float)
    lengths = np.asarray(box_lengths, dtype=float)
    labels = np.asarray(species)
    edges = np.asarray(edges, dtype=float)
    if p.ndim != 3 or p.shape[-1] != 3 or len(p) == 0 or lengths.shape != (len(p), 3):
        raise ValueError("Require positions (frames, molecules, 3) and lengths (frames, 3)")
    if labels.shape != (p.shape[1],) or not np.isfinite(p).all():
        raise ValueError("Invalid species or positions")
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Box lengths must be finite and positive")
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0) or edges[-1] > lengths.min()/2):
        raise ValueError("RDF edges must increase within half the shortest box length")
    a, b = np.flatnonzero(labels == species_a), np.flatnonzero(labels == species_b)
    same = species_a == species_b
    if not len(a) or not len(b) or (same and len(a) < 2):
        raise ValueError("Insufficient molecules for selected species pair")
    pair_count = len(a)*(len(a)-1)/2 if same else len(a)*len(b)
    shell = 4*np.pi/3 * np.diff(edges**3)
    histograms, nearest = [], []
    for frame, box in zip(p, lengths):
        delta = frame[b][None, :, :] - frame[a][:, None, :]
        delta -= box*np.floor(delta/box + .5)
        distances = np.linalg.norm(delta, axis=-1)
        if same:
            pair_distances = distances[np.triu_indices(len(a), 1)]
            np.fill_diagonal(distances, np.inf)
        else:
            pair_distances = distances.ravel()
        counts = np.histogram(pair_distances, bins=edges)[0]
        histograms.append(counts*np.prod(box)/(pair_count*shell))
        nearest.append(distances.min(axis=1))
    return {"r_angstrom": .5*(edges[1:]+edges[:-1]), "g_r": np.mean(histograms, axis=0),
            "nearest_distances_angstrom": np.asarray(nearest)}
