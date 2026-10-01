# Molecular displacement from the first frame

`get_molecular_reference_msd` computes one squared-displacement history per
molecule and their mean from a single reference, the first supplied frame.
For molecular COM positions R with shape (frames, molecules, 3):

\[
s_m(t)=|R_m(t)-R_m(t_0)|^2,\qquad
\mathrm{MSD}_0(t)=\frac{1}{N}\sum_m s_m(t).
\]

Positions must already be temporally unwrapped in Angstrom with fixed molecule
ordering. Times are ps and may be irregular. No COM-drift subtraction, alignment
or flow correction is performed. This is not cumulative path length or a sum
of squared displacements between consecutive samples: a return to the reference
position gives zero squared displacement. It does not imply equilibrium.

```python
import numpy as np
from MartiniFuncs.postprocessing import load_analysis_cache
from MartiniFuncs.md_statistics import get_molecular_reference_msd

result, manifest = load_analysis_cache('postprocess/cache')
if result['com_angstrom'] is None:
    raise ValueError('MSD unavailable for this cache/cell convention')
times = np.array([frame['time_ps'] for frame in result['frames']])
chosen = np.asarray(result['species']) == 'C5H6'
measured = get_molecular_reference_msd(
    result['com_angstrom'][:, chosen], times,
    unwrapped=True, selection='C5H6',
)
# Shapes: (frames, molecules), and (frames,), respectively.
squared = measured['squared_displacement_angstrom2']
mean = measured['msd_angstrom2']
elapsed = measured['elapsed_time_ps']
```

The mean averages molecules equally (COM construction already uses atomic
masses). The original times and reference time are retained as `time_ps` and
`reference_time_ps`. `n_origins=1`; unlike `get_molecular_msd`, there is no
averaging over different starting times. The result stores all T*N squared
displacements (8*T*N bytes). Temporary displacement arrays are processed in
blocks; input coordinates and notebook selections also consume memory.

Notebook 04 (both the directory and workflow versions) plots the species means
and all individual molecular histories immediately after the MSD-versus-lag
plots, as part of normal execution. Notebook 08 includes these plots after loading.
Their `REFERENCE_PLOT_POINTS` setting limits rendering only; results
retain every selected frame. Original atom-ID lists are in
`reference_molecule_ids[species]`, in squared-displacement column order.
No trajectory rereading or cache regeneration is required if `com_angstrom`
already exists. Sheared/triclinic runs are supported (lab-frame displacements,
no shear-flow subtraction); caches written before 2026-10-01 have no
`com_angstrom` for such runs and must be regenerated. `unwrapped=True` does not
perform or validate unwrapping itself.

Single-reference squared displacements do not subtract when the reference moves:
|R(t)-R(1)|^2 = |R(t)-R(0)|^2 - |R(1)-R(0)|^2 - 2 (R(1)-R(0)).(R(t)-R(1)).
For caged molecules the cross term is large (successive steps anticorrelate),
so a curve restarted at a later reference has the same shape, not differences
of the earlier values. Motion faster than the frame spacing (ballistic/cage
rattling) appears as an apparent jump at the first saved lag.

`examples/02_plot_md_statistics.ipynb` now also has an independent cache-based
reference-MSD section at the top: set `CACHE_DIR`, then run its load and plot cells.
The historical report examples later in that notebook are optional and separate.
`examples/04_plot_md_statistics_from_directory.ipynb` accepts `CACHE_DIR` too;
leave it `None` to analyze `SIMULATION_DIR`, or set it to skip dump parsing.
