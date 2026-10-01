# LAMMPS thermo output

`MartiniFuncs.io.thermo` reads standard one-line `Step ...` tables from `.out` or
LAMMPS logs. Notebook 08 discovers a unique file automatically with
`LOAD_THERMO=True`. Set `THERMO_FILE` if multiple outputs are present;
`THERMO_BLOCK` selects a zero-based header occurrence for repeated run segments.

Notebook 08's loading cell refreshes the stateless `io.thermo` module before using
it and prints its loaded file path. This avoids keeping an old function signature
in an already-open kernel after updating the package. It also checks for the
required API before starting analysis. Reopen the updated notebook from disk if
your editor is still displaying an older loading cell; rerunning unchanged old
cell text does not load notebook edits.

```python
from MartiniFuncs.io.thermo import discover_thermo_file, read_lammps_thermo, attach_thermo

path = discover_thermo_file(run_dir)
thermo = read_lammps_thermo(path, units="metal")
frames, coverage = attach_thermo(result["frames"], thermo, dump_file=result["files"]["dump"])
```

New `analyze_run` results include exact integer `timestep` in each frame record.
For older results lacking it, supply `dump_file`: exact timestep integers are
recovered from dump headers at the recorded source frame indices. Notebook 08
does this automatically. It does not reconstruct step IDs from floating-point
time. Integral float/string columns are accepted only after exact validation;
fractional, missing, boolean or precision-unsafe floating IDs are rejected. Matching is exactly `dump timestep == thermo Step`.
No interpolation or forward filling occurs. Repeated steps require an explicit
segment; inconsistent printed `Time` and frame time are rejected.

Original headings remain in `thermo`. Joined aliases include `temperature_K`,
`pressure_bar`, `potential_energy_eV`, `kinetic_energy_eV`, `total_energy_eV`,
pressure tensor components, `thermo_volume_angstrom3`, `thermo_density_g_cm3`
and `thermo_time_ps`. Unknown/custom headings retain their original names in
the raw table; no units are guessed. Aliases currently require explicit metal
units. Pressure is bar (divide by 10,000 for GPa), temperature K, time ps,
energy eV, volume Å³ and density g/cm³. Energies retain `thermo_modify norm`
normalization as printed; inspect the run settings before interpreting totals.
Temperature retains LAMMPS's compute, bias and DOF conventions.

The supplied continuation writes thermo every 2,000 steps and dumps every 1,000
steps, with 0.0005 ps/step. Verified: 1,001 thermo rows match 1,001 of 2,001 dump
frames from steps 2,000,000–4,000,000. The other 1,000 frames have missing thermo
values. Notebook 08 evaluates scalar stability on observed samples with 1 ps
spacing, while molecular diffusion retains 0.5 ps spacing. It rejects irregular
observed spacing for scalar correlation estimates rather than filling gaps.

The parser supports multiple one-line tables, interleaved text warnings and
Fortran D exponents, with source line and table provenance. Malformed numeric
rows raise. YAML/multiline thermo formats are unsupported. File discovery does
not prove the output belongs to the dump; check timestep coverage and provenance.
The parser can read incomplete output, so row coverage is not a completion test.
