# NVT_noshear motion diagnosis

Source: `C:/Users/nicho/Documents/Martini collab/uma_data/NVT_noshear`.
The local copy contains a molecular cache and `uma_4gpu_590047.out`, but not
the original dump, final atomic data file, or external driver source. This
diagnosis therefore checks cached motion and printed thermodynamics. It does
not independently validate atomic reconstruction or identify chemical reactions.

## Confirmed setup

The input echoed in the output actually executes
`fix integrate_nvt all nvt temp 400.0 400.0 0.05`, at a fixed cubic box of
approximately 22.713552 Angstrom per side. The comments and `nve_2p5` filenames
still say NVE; those labels do not change the executed NVT calculation.
The checkpoint loaded is `equilibrated_2p5_cont.restart`. The model-loading
record specifies `uma-s-1p1.pt`.

There are 300,001 cached frames, covering timesteps 4,000,000--7,000,000,
absolute time 2000--3500 ps. The integration timestep is 0.0005 ps, the dump
interval is 10 steps (0.005 ps), and thermo is printed every 1000 steps (0.5 ps).
The input comment equating 3,000,000 steps to 10,000 ps is incorrect: this run
is 1500 ps long, as both the cached timestamps and printed clock show.

## Motion and temperature

All cached COM coordinates are finite. Each temporal COM increment agrees
exactly with the nearest-image increment from the cached wrapped COMs. The box
lengths are constant. The largest molecular COM step is 0.062867 Angstrom,
with no box-sized discontinuity. These are cache-consistency checks, not an
independent raw-dump check.

The mass-weighted system displacement was computed as

$$d_{sys}(t)=\frac{\sum_m M_m[R_m(t)-R_m(0)]}{\sum_m M_m}.$$

Using standard elemental masses, the largest system displacement is 0.013372
Angstrom. Recomputing squared molecular displacements after subtracting
\(d_{sys}(t)\) leaves the observed features essentially unchanged.

| Elapsed-time window (ps) | MAL reference MSD (Angstrom2) | CPD reference MSD (Angstrom2) | Mean temperature (K) |
|---|---:|---:|---:|
| 850--880 | 0.168 | 0.310 | 398.9 |
| 900--950 | 0.298 | 0.534 | 400.6 |
| 1000--1200 | 0.197 | 0.402 | 400.1 |
| 1400--1500 | 0.333 | 0.649 | 399.4 |

The windows average every cached sample; these values are not slopes or
diffusion coefficients. The 880--950 ps MAL peak occurs at 907.985 ps. At that
instant the five largest contributions account for 39.7% of MAL squared
displacement and 42.8% of CPD squared displacement. The feature involves
multiple molecules, rather than a single trajectory discontinuity. The mean
subsequently decreases, so the peak should not be interpreted as irreversible
distance traveled.

The full-run temperature mean is 400.206 K, with printed samples from
367.75 to 429.44 K. No sustained heating accompanies the 900 ps feature or
the late increase. Potential energy fluctuates without a large abrupt jump;
its means in these windows differ by less than an eV for the whole system.
This does not exclude a reaction. The cache records `reaction_screen: null`
and fixed molecular memberships.

Interpretation: the observations are consistent with bounded molecular motion
plus intermittent rearrangements/excursions affecting multiple molecules.
They do not establish cage escape, structural relaxation, a particular
reaction, or equilibrium diffusion. Establishing those requires atomic
coordinates around the event and/or longer stationary trajectories.

## Separate pressure issue

For every one of the 3,001 thermo samples, calculate

$$P_{kin}=\frac{2K}{3V}\,(1.6021766208\times10^6)\quad\text{bar},$$

where printed `KinEng` is total kinetic energy in eV and `Volume` is in
Angstrom cubed. The maximum absolute difference from printed `Press` is
0.0007333 bar, while pressure itself is approximately 6400 bar. This strongly
indicates that the reported pressure lacks an interaction-virial contribution.

The logged setup uses `fix ext_fc all external pf/callback 1 1`.
[LAMMPS pressure documentation](https://docs.lammps.org/compute_pressure.html)
includes both kinetic and virial terms. Its
[fix external documentation](https://docs.lammps.org/fix_external.html)
requires the external code to provide the virial for variable-cell operations.
Review the driver's virial calculation and `fix_external_set_virial_global`
handling, units, and inclusion in the pressure compute. Supplying forces or
potential energy does not establish that the virial was supplied correctly.

The driver is unavailable here, so the exact cause is unconfirmed. This
finding does not prove that forces are wrong or cause the MSD feature:
the current NVT run does not barostat the cell. It does prevent interpreting
the printed pressure as a verified total pressure, and would matter directly
if the same missing contribution occurred during NPT. The preceding NPT run
must be checked separately, not presumed affected.

Follow-up inspection found that the published fairchem callback forwards
virial only when `box_change` is true, matching the fixed-cell symptom here.
See [the proposed driver correction and validation](uma_pressure_fix.md).
The cluster's installed callback still needs to be checked against that source.

## Reproduce and inspect

Run from the repository root:

```powershell
python -m tools.diagnose_cached_motion "C:/Users/nicho/Documents/Martini collab/uma_data/NVT_noshear" --output reports/nvt_noshear_diagnosis
```

Outputs:

- `summary.json`: cache consistency, motion, temperature and pressure checks.
- `window_summary.csv`: full-sampling window averages and drift comparison.
- `event_molecules.json`: ranked molecules at the 907.985 ps peak, with
  zero-based cache molecule indices and original one-based LAMMPS atom IDs.
- `motion_thermo.png`: reference MSD, temperature, potential energy, and system
  displacement on the same elapsed-time axis. Gray shading is 880--950 ps.
  Displayed COM curves use every 100th frame (0.5 ps); numerical motion metrics
  use every frame. The black energy line is a centered 100-sample (50 ps)
  moving mean, not additional observations.

The script avoids the 667 MB pair-history JSON and uses memory-mapped COM
arrays. Its system-drift mass mapping is specific to standard-mass CPD/MAL.
It assumes one thermo block in metal units and does not modify input files,
the existing cache, or simulation results.
