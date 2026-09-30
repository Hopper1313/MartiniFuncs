# COM tracking validation: continued_run

The audit of `equilibrate_2p5_cont.lammpstrj` in
`C:/Users/nicho/Documents/Martini collab/uma_data/smaller_system/continued_run`
found no discrepancy in molecular COM tracking, periodic unwrapping, or time
conversion. It reproduces the reported 75--100 ps diffusion fits. This validates
those numerical calculations for this trajectory; it does not establish that
the fitted slopes are converged diffusion coefficients.

## What was compared

The audit reads the dump text independently, without ASE or the package's dump
reader. It sorts raw atom IDs and checks that IDs, elements, and masses remain
unchanged in all 2,001 frames (1,360 atoms).

An independent all-pairs distance graph at frames 0, 500, 1000, 1500, and 2000
gives exactly the package's molecular memberships: 40 C5H6 and 40 C11H9NO2.
The graph uses the same declared 1.3 covalent-radius scale; it independently
checks implementation, not the chemical validity of that distance criterion.

Each molecule is reconstructed relative to one anchor atom using minimum-image
displacements, instead of the package's bond-tree traversal. Coordinates remain
in Angstroms. Its COM is computed directly from the dumped masses:

$$R_m=\frac{\sum_{i\in m}m_i r_i}{\sum_{i\in m}m_i}.$$

An additional reconstruction uses the raw LAMMPS image flags:
\(r_i^{image}=r_i+L(n_i+c_i)\), where the integer offsets \(c_i\) are fixed
once in frame zero to make molecules whole. All subsequent image counters come
from the dump. Relative atomic coordinates agree with the independent anchor
reconstruction throughout the trajectory (maximum component difference
\(3.6\times10^{-15}\) Angstrom). No subsequent frame is realigned.

For centered wrapped COMs \(w_t\), TOR accumulates nearest-image increments
using the current box lengths:

$$u_t=u_{t-1}+(w_t-w_{t-1})-L_t\operatorname{nint}[(w_t-w_{t-1})/L_t].$$

The independent image-flag trajectory supplies a separate temporal check. If
\(q_t=w_t+I_tL_t\) is its centered COM, the equivalent TOR increment is

$$\Delta u_t=(q_t-q_{t-1})-I_{t-1}(L_t-L_{t-1}).$$

This correction matters for NPT: the raw image-unwrapped trajectory and TOR
are not assumed identical when the box fluctuates. All operations above are
componentwise for this orthorhombic, fixed-center cell. The audit rejects tilted
cells and moving centers.

## Results

| Check | Result |
|---|---|
| Independent wrapped COM vs package | Maximum component difference 2.31e-14 Angstrom |
| Independent TOR COM vs package | Maximum component difference 6.39e-14 Angstrom |
| Image-flag-derived TOR vs independent TOR | Maximum component difference 8.35e-14 Angstrom |
| COM boundary crossings | 4,946 nonzero frame/molecule/axis crossing events |
| Largest saved COM displacement | 1.992 Angstrom |
| Largest displacement component / current box length | 0.0841, below the half-box ambiguity threshold |
| Dump timesteps | 2,000,000 through 4,000,000; spacing 1,000 |
| Integration timestep | 0.0005 ps, explicitly present in the input |
| Saved-frame times | 1000 through 2000 ps; spacing 0.5 ps |
| Printed thermo clock | All 1,001 matching Step/Time entries agree exactly |

The independent COMs reproduce the user's fitted values:

| Species | D from 75--100 ps slope (cm2/s) | Fitted intercept (Angstrom2) |
|---|---:|---:|
| C11H9NO2 | 3.195617726e-10 | 0.180631906 |
| C5H6 | 2.205158388e-10 | 0.359493810 |

Thus the nearly flat MSD is also present in the independently reconstructed
coordinates. These checks do not support blaming it on COM ordering, missed
net boundary crossings, or an incorrect conversion of dump intervals to ps.
They do not distinguish physical caging from simulation/model issues, prove
equilibration, or screen every frame for transient reactions. Bond membership
was rebuilt only at the five stated checkpoints.

## Reproduce and inspect

From the repository root in an environment with MartiniFuncs dependencies:

```powershell
python -m tools.verify_molecular_tracking "C:/Users/nicho/Documents/Martini collab/uma_data/smaller_system/continued_run" --output reports/com_tracking_validation --timestep-ps 0.0005
```

This standalone audit requires metal units, orthorhombic periodic bounds,
`id element mass x y z ix iy iz`, and matching printed `Step Time` thermo
columns in a `.out` file. It uses whole-trajectory molecular arrays in memory;
it is a verification tool, not a replacement for the production CLI. The
75--100 ps comparison is included only when the trajectory sampling permits it.
No model weights, pair extraction, or trajectory cache are needed. It writes:

- `reports/com_tracking_validation/summary.json`: numerical checks and fits.
- `reports/com_tracking_validation/com_comparison.npz`: reconstructed COMs,
  times, species, and atom-to-molecule mapping.

```python
import numpy as np
with np.load("reports/com_tracking_validation/com_comparison.npz") as audit:
    time_ps = audit["time_ps"]
    com = audit["independent_tor_com"]  # (frame, molecule, xyz), Angstrom
    species = audit["species"]         # same molecule order
    atom_ids = audit["atom_ids"]       # original, one-based LAMMPS IDs
    membership = audit["atom_molecule_index"]  # zero-based COM column per atom
```

`image_com` contains the uncorrected image-flag reconstruction; use
`image_tor_com` for comparison with the package's NPT TOR convention.
