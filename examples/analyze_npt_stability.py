"""Build late-time NPT stability report: python -m examples.analyze_npt_stability."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MartiniFuncs.statistics.stability import get_molecular_com_rdf, get_series_statistics


def main():
    """Build the historical fixed-window scalar and COM-structure report.

    Delegate scalar means, SE=std(batch_means,ddof=1)/sqrt(n_batches),
    and RDF g=mean_frames(count*V/(pair_count*shell_volume)) to stability.
    Nearest-neighbor ECDF at sorted distance d_j is (j+1)/N, j=0,...,N-1.
    Windows and 0.5 ps sampling are fixture-specific; use workflow 05 for new runs.
    """
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/npt_statistics"
    output = source / "stability"
    output.mkdir(exist_ok=True)
    with (source / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(row["time_ps"]) for row in rows])
    with np.load(source / "diffusion/com_trajectory.npz") as cache:
        np.testing.assert_allclose(times, cache["time_ps"], rtol=0, atol=1e-10)
        if "wrapped_com_angstrom" not in cache:
            raise ValueError("Regenerate COM cache with python -m examples.analyze_npt_diffusion")
        positions = cache["wrapped_com_angstrom"]
        boxes = cache["box_lengths_angstrom"]
        species = cache["species"]
    np.testing.assert_allclose(np.diff(times), .5)
    # Four equally sized, disjoint intervals: final endpoint excluded throughout.
    boundaries = np.array([500., 625., 750., 875., 1000.])
    masks = [(times >= a) & (times < b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    late = (times >= 500) & (times < 1000)
    if late.sum() != 1000 or any(mask.sum() != 250 for mask in masks):
        raise ValueError("Expected 0.5 ps sampling throughout 500 <= t < 1000 ps")
    if any(row["pair_error"] for row, keep in zip(rows, late) if keep):
        raise ValueError("Pair detection failed in the selected interval")
    report = {"trajectory_interval_ps": [500, 1000], "end_excluded": True,
              "block_boundaries_ps": boundaries.tolist(), "scalar_statistics": {},
              "structure": {}, "sources": ["../frames.csv", "../diffusion/com_trajectory.npz"],
              "limitations": ["No potential-energy or pressure series supplied",
                  "COM structure does not test internal conformation or reactive orientation",
                  "Fixed molecular identities inherited from COM extraction",
                  "No automatic equilibrium verdict; no independent replicas",
                  "Batch SE assumes decorrelation between batches; ACF estimates are heuristic"]}
    quantities = {"temperature_K": "Temperature (K)", "density_g_cm3": "Density (g/cm^3)",
                  "candidate_pairs": "Distance-screen candidate count"}
    colors = ["#177E89", "#C65B28", "#7851A9", "#4B843D"]
    labels = [f"{a:g}-{b:g} ps" for a, b in zip(boundaries[:-1], boundaries[1:])]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout="constrained")
    table = ["# Late-time NPT stability report", "",
             "Four disjoint 125 ps blocks; 500 <= time < 1000 ps. The frame at 1000 ps is excluded",
             "to give exactly 250 samples per block. All 80 molecules are retained (40 per species).", "",
             "## Scalar summaries", "",
             "Means +/- illustrative batch-mean standard errors using 25 ps batches (5 per block).",
             "These errors are conditional on independent batches, not established confidence intervals.", "",
             "| Trajectory block | Temperature (K) | Density (g/cm^3) | Candidate count |",
             "|---|---:|---:|---:|"]
    summaries = {}
    for row_index, (key, title) in enumerate(quantities.items()):
        values = np.array([float(row[key]) for row in rows])
        blocks = [get_series_statistics(values[m], sample_interval_ps=.5) for m in masks]
        total = get_series_statistics(values[late], sample_interval_ps=.5,
                                      batch_sizes=(10, 20, 50, 100, 250))
        report["scalar_statistics"][key] = {"late_interval": total, "blocks": blocks}
        summaries[key] = blocks
        bins = np.arange(values[late].min()-.5, values[late].max()+1.5) if key == "candidate_pairs" else np.linspace(values[late].min(), values[late].max(), 25)
        for mask, color, label, stats in zip(masks, colors, labels, blocks):
            axes[row_index, 0].plot(times[mask], values[mask], color=color, alpha=.45, lw=.6)
            axes[row_index, 0].plot([times[mask][0], times[mask][-1]], [stats["mean"]]*2, color=color, lw=2)
            axes[row_index, 1].hist(values[mask], bins=bins, density=True, histtype="step", color=color, label=label)
        batches = total["batch_mean_errors"]
        axes[row_index, 2].plot([b["batch_duration_ps"] for b in batches],
                                 [b["standard_error"] for b in batches], "o-")
        axes[row_index, 0].set(xlabel="Trajectory time (ps)", ylabel=title)
        axes[row_index, 1].set(xlabel=title, ylabel="Probability density")
        axes[row_index, 2].set(xlabel="Batch duration (ps)", ylabel="SE of 500-1000 ps mean")
    axes[0, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.savefig(output / "scalar_stability.png", dpi=160)
    fig.savefig(output / "scalar_stability.pdf")
    plt.close(fig)
    for i, label in enumerate(labels):
        cells = []
        for key in quantities:
            s = summaries[key][i]
            se = s["batch_mean_errors"][1]["standard_error"]
            cells.append(f"{s['mean']:.6g} +/- {se:.3g}")
        table.append("| " + " | ".join([label, *cells]) + " |")
    first_pairs, last_pairs = summaries["candidate_pairs"][0], summaries["candidate_pairs"][-1]
    pair_change = 100*(last_pairs["mean"]/first_pairs["mean"]-1)
    table.extend(["", "### Reading these diagnostics", "",
                  f"Candidate-count block means change by {pair_change:.1f}% from the first to last block.",
                  "Inspect the SE-versus-batch-duration curve before using the illustrative errors.",
                  "A rising curve can reflect slow correlations or drift; it does not identify the cause.",
                  "The four largest batches give a particularly noisy uncertainty estimate.",
                  "Autocorrelation summaries are saved in stability.json. First-zero truncation can",
                  "miss oscillatory correlations and slow tails, so its effective sample count must not",
                  "override the batch-duration diagnostic or be taken as proof of independent frames."])
    table.extend(["", "![Scalar diagnostics](scalar_stability.png)", "",
                  "## Species-resolved COM packing", "",
                  "RDFs use minimum-image distances in each actual NPT box, normalized per frame",
                  "by volume and available pairs (same-species self pairs excluded).",
                  "All RDF bins lie below half the shortest box length. Nearest-neighbor distances",
                  "are pooled over central molecules and frames; they are correlated observations.", "",
                  "| Central -> neighbor species | " + " | ".join(labels) + " |",
                  "|---|" + "---:|"*4])
    edges = np.linspace(0, float(boxes[late].min()/2)*.999, 81)
    pairs = [("CPD", "CPD"), ("CPD", "dienophile"), ("dienophile", "dienophile")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for row_index, (a, b) in enumerate(pairs):
        key = f"{a} -> {b}"
        entries, means = [], []
        for mask, color, label in zip(masks, colors, labels):
            result = get_molecular_com_rdf(positions[mask], boxes[mask], species,
                                           species_a=a, species_b=b, edges=edges)
            nearest = result["nearest_distances_angstrom"].ravel()
            sorted_nn = np.sort(nearest)
            axes[row_index, 0].plot(result["r_angstrom"], result["g_r"], color=color, label=label)
            axes[row_index, 1].plot(sorted_nn, np.arange(1,len(nearest)+1)/len(nearest), color=color)
            entries.append({"block": label, "r_angstrom": result["r_angstrom"].tolist(),
                            "g_r": result["g_r"].tolist(), "nearest_mean_angstrom": float(nearest.mean()),
                            "nearest_quantiles_05_50_95": np.quantile(nearest, [.05,.5,.95]).tolist()})
            means.append(f"{nearest.mean():.4f}")
        report["structure"][key] = entries
        axes[row_index, 0].set(title=key, xlabel="COM separation (angstrom)", ylabel="g(r)")
        axes[row_index, 1].set(title=key, xlabel="Nearest-neighbor COM distance (angstrom)", ylabel="Cumulative probability")
        axes[row_index, 0].axhline(1, color="gray", ls=":", lw=.7)
        table.append("| " + " | ".join([key, *means]) + " |")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.savefig(output / "structural_stability.png", dpi=160)
    fig.savefig(output / "structural_stability.pdf")
    plt.close(fig)
    table.extend(["", "Table: mean nearest-neighbor distances in angstrom.", "",
                  "![Structural diagnostics](structural_stability.png)", "", "## Limits", "",
                  *[f"- {item}." for item in report["limitations"]], "",
                  "Methods background: https://doi.org/10.33011/livecoms.1.1.5067", ""])
    (output / "stability.json").write_text(json.dumps(report, indent=2)+"\n")
    (output / "README.md").write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table[:19]))
    print(f"Saved report and figures to {output}")


if __name__ == "__main__":
    main()
