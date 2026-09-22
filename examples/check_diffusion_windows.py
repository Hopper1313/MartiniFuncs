"""Compare late NPT windows using the COM cache from analyze_npt_diffusion.

Run with ``python -m examples.check_diffusion_windows``.
Trajectory intervals and fitted lag intervals are deliberately distinct.
"""
import json
from pathlib import Path

import numpy as np

from MartiniFuncs import md_statistics as md


def main():
    directory = Path(__file__).resolve().parents[1] / "reports/npt_statistics/diffusion"
    windows = [(500, 1000), (750, 1000)]
    fit_intervals = [(10, 50), (25, 75), (50, 100)]
    with np.load(directory / "com_trajectory.npz") as cache:
        times = cache["time_ps"]
        positions = cache["com_angstrom"]
        species = cache["species"]
    report = {
        "source": "com_trajectory.npz",
        "method": "Whole-molecule TOR COM; no drift subtraction or finite-size correction",
        "note": "Nested trajectory windows are not independent replicates; fit spread is not uncertainty.",
        "species": {},
    }
    for name in sorted(set(species)):
        entries = []
        for start, end in windows:
            keep = (times >= start) & (times <= end)
            if not keep.any() or times[keep][0] != start or times[keep][-1] != end:
                raise ValueError(f"Cache does not cover requested window {start}-{end} ps")
            selected = positions[keep][:, species == name]
            msd = md.get_molecular_msd(selected, times[keep], unwrapped=True, selection=name)
            # Independent reference at a shared lag, averaged over every origin.
            k = int(round(100 / (times[1] - times[0])))
            reference = np.mean(np.sum((selected[k:] - selected[:-k])**2, axis=-1))
            np.testing.assert_allclose(msd["msd_angstrom2"][k], reference)
            fits = [md.get_molecular_diffusion_coefficient(msd, fit_start_ps=a, fit_end_ps=b)
                    for a, b in fit_intervals]
            entries.append({"trajectory_window_ps": [start, end], "frames": int(keep.sum()),
                            "molecules": int((species == name).sum()),
                            "lag_ps": msd["lag_ps"].tolist(),
                            "msd_angstrom2": msd["msd_angstrom2"].tolist(),
                            "n_origins": msd["n_origins"].tolist(), "fits": fits})
            for fit in fits:
                print(f"{name:10s} trajectory {start}-{end} ps; lag {fit['fit_interval_ps']}: "
                      f"D = {fit['diffusion_cm2_per_s']:.6e} cm^2/s")
        report["species"][name] = entries
    (directory / "late_windows.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
