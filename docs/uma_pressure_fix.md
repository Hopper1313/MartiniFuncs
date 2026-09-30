# Proposed correction for UMA/LAMMPS pressure

The NVT output matches kinetic pressure, 2 KE/(3 V), within 0.001 bar. The
published [fairchem driver](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/lammps/lammps_fc.py)
inspected on 2026-09-29 forwards stress-derived virial only inside
`if box_change:` in `FixExternalCallback.__call__`. Fixed-cell NVT therefore
can receive forces and energy but no virial. This closely matches our evidence.
It does not establish the cluster version or imply the earlier NPT was wrong.

The local environment has fairchem-core 2.22.0 but no fairchem-lammps driver.
No installed dependency or cluster file has been modified. In the MD job's
cluster environment, identify the actual callback:

```bash
python -c "import inspect; import fairchem.lammps.lammps_fc as m; print(inspect.getfile(m)); print(inspect.getsource(m.FixExternalCallback))"
python -m pip show fairchem-core fairchem-lammps
```

## Change to validate

If the callback matches upstream, replace its `if box_change:` block with
unconditional virial forwarding below. Keep its force and energy assignments.
`results`, `cell`, `torch`, `np`, and `FIX_EXT_ID` exist in that upstream code.
This is a proposed patch for that tensor API, not an applied cluster fix.

```python
if "stress" not in results:
    raise RuntimeError("Pressure requires stress prediction for this UMA task")
stress = results["stress"].detach().cpu().numpy()
if stress.size != 9:
    raise ValueError(f"Expected one 3x3 stress tensor, got {stress.shape}")
stress = stress.reshape(3, 3)
volume = float(torch.det(cell).abs().item())
if not np.isfinite(stress).all() or not np.isfinite(volume) or volume <= 0:
    raise ValueError("Invalid stress or cell volume")
virial = -volume * stress
lmp.fix_external_set_virial_global(
    FIX_EXT_ID,
    [float(virial[i, j]) for i, j in
     [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]],
)
```

For this API, stress is tensile-positive in eV/Angstrom cubed and volume in
Angstrom cubed, giving virial in eV. The order is xx, yy, zz, xy, xz, yz;
it is not ASE's six-component order. LAMMPS's
[22Jul2025 implementation](https://raw.githubusercontent.com/lammps/lammps/stable_22Jul2025/src/fix_external.cpp)
stores the supplied virial directly; pressure computation handles conversion
to bar. Do not apply that conversion twice. Verify against the installed build.

Explicitly request `predict_untrained_stress={"omol"}` in the predictor's
`InferenceSettings`, retaining uma-s-1p1 and all other simulation settings.
Configure all workers before constructing a parallel predictor. Some versions
automatically add derivative stress tasks, so inspect returned results rather
than assuming an empty explicit setting means no stress was computed.
OMol stress is an energy derivative without stress-label training; this needs
validation for the target system. See [fairchem stress settings](https://github.com/facebookresearch/fairchem/blob/main/docs/core/common_tasks/ase_calculator.md).

`fix_modify ext_fc virial yes` alone cannot supply missing virial. It is already
the documented default, and this driver creates `ext_fc` after reading input
setup commands, so placing that command before creation would fail.

## Required checks before production

1. At the same full periodic frame, confirm enabling stress leaves energy and
   forces unchanged, using the same checkpoint, task, charge and spin.
2. Validate isotropic stress by scaling cell and positions together at fixed
   fractional coordinates. Compare -(E_plus-E_minus)/(V_plus-V_minus) with
   -trace(stress)/3. Sweep strain sizes: tiny energy differences may be noisy
   with float32/TF32 and large absolute energies. Check individual normal and
   shear strains to validate the full tensor.
3. In a short fixed-cell LAMMPS run, check that printed pressure agrees with
   [2 KE/(3 V) - trace(stress)/3] * 1.6021766208e6 bar at the same frames.
   Agreement, rather than merely nonzero pressure, tests units and sign.
4. Compare one-GPU and production multi-GPU predictions on fixed frames to
   exclude missing shards or double-counting the global virial.
5. Check the NPT callback separately before any long production run.

These are validation steps, not completed model tests: neither the cluster
driver nor model weights are available here. Fixing reporting alone need not
change an NVT trajectory. If NPT used an incorrect virial, its box evolution
cannot be repaired by relabeling output. Recovering pressure for saved NVT
frames requires atomic snapshots and the model; the molecular cache alone
does not contain enough information. A naive sum of wrapped positions times
forces is not a replacement for periodic many-body stress.

References: [fix external](https://docs.lammps.org/fix_external.html),
[LAMMPS pressure](https://docs.lammps.org/compute_pressure.html).
