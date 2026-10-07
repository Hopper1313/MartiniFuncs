"""Diffusion coefficients from a report's per-species lag files (species_NNN_lags.csv).

Run: python examples/diffusion_from_report.py REPORT_DIR --fit-ps 50 100

For each species: the Einstein fit MSD = a + 6 D t over the chosen lag window
(statistics.md.get_molecular_diffusion_coefficient), the log-log slope alpha
over the same window (alpha ~ 1 is required for D to mean diffusion), and the
Green-Kubo estimate D = (1/3) int VACF dt up to the window end, when the report
has a vacf column.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from MartiniFuncs.statistics import dynamics, md


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="report directory containing manifest.json")
    parser.add_argument("--fit-ps", type=float, nargs=2, required=True, metavar=("START", "END"),
                        help="lag window in ps for the MSD fit (within the report's max lag)")
    args = parser.parse_args(argv)
    start, end = args.fit_ps

    manifest = json.loads((args.report / "manifest.json").read_text(encoding="utf-8"))
    print(f"Report time window {manifest['time_window_ps']} ps; fit lags {start:g}-{end:g} ps")
    for name, entry in manifest["species"].items():
        table = pd.read_csv(args.report / entry["file"])
        lag, msd = table["lag_ps"].to_numpy(), table["msd_angstrom2"].to_numpy()
        fit = md.get_molecular_diffusion_coefficient(dict(lag_ps=lag, msd_angstrom2=msd, selection=name),
                                                     fit_start_ps=start, fit_end_ps=end)
        window = (lag >= start) & (lag <= end) & (lag > 0)
        alpha = np.polyfit(np.log(lag[window]), np.log(msd[window]), 1)[0] if np.all(msd[window] > 0) else np.nan
        print(f"\n{name} ({entry['n_molecules']} molecules)")
        print(f"  Einstein: D = {fit['diffusion_cm2_per_s']:.3e} cm^2/s ({fit['diffusion_angstrom2_per_ps']:.3e} A^2/ps), "
              f"intercept {fit['intercept_angstrom2']:.3f} A^2, residual {fit['residual_rms_angstrom2']:.3f} A^2")
        print(f"  MSD scaling over the window: alpha = {alpha:.2f}  (diffusive only if ~1)")
        if "vacf" in table:
            gk = dynamics.green_kubo_diffusion(lag, table["vacf"].to_numpy(), upper_ps=end)
            print(f"  Green-Kubo to {end:g} ps: D = {gk['D_cm2_per_s']:.3e} cm^2/s")


if __name__ == "__main__":
    main()
