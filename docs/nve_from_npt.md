# NVE continuation from an NPT restart

Use [in.nve_from_npt](../examples/lammps/in.nve_from_npt) with the existing external
fairchem-LAMMPS launch command. The template replaces the NPT integration fix
with `fix integrate_nve all nve`. It keeps restart coordinates, velocities,
box dimensions/tilts and step numbering. Temperature and pressure now fluctuate
freely; neither 400 K nor 25,000 bar is enforced. Volume remains the instantaneous
volume of the selected restart, not the NPT mean volume.

Edit these literals before launching:

| Setting | Meaning |
|---|---|
| `read_restart` filename | Chosen final NPT binary restart; e.g. `equilibrated_2p5_cont.restart` for the continuation output |
| `nve_2p5` output prefix | New dump/checkpoint/final-data names; use a fresh run directory |
| `dump_modify ... element` | Actual LAMMPS type ordering |
| `timestep 0.0005` | Integration interval h in ps (metal units) |
| Both `2000000` values in `run` | Number of additional steps N; keep equal so final writes occur at the end |
| `thermo` and `dump` intervals | Both 1000 here to give temperature/pressure at every saved frame |

The selected restart must be compatible with your LAMMPS executable. A fresh
`read_restart` does not recreate the NPT fix; there is no old integrator to
`unfix` in this standalone script. Switching ensembles inside an already-running
input instead requires removing the existing NPT integration fix first.
[LAMMPS restart documentation](https://docs.lammps.org/read_restart.html).

Use the same UMA model and electronic/inference settings as the source run.
The header uses the previously confirmed uma-s-1p1 / omol configuration; the
older pasted s-1p2 comment does not identify the model actually loaded. Changing
the Hamiltonian at the switch would change the interpretation of energy continuity.
The external driver must provide conservative forces and their consistent UMA
potential energy to LAMMPS; the input does not install or validate that coupling.

## Quantities to check

Run duration is N*h = 1000 ps; output spacing is 1000*h = 0.5 ps; checkpoint
spacing is 100000*h = 50 ps. The final `run ... every ...` remains one physical
line, preserving the driver convention in the supplied input.

For NVE, inspect physical total energy E(t)=K(t)+U(t), not constancy of K or T
separately. `thermo_modify norm no` requests total energies in eV. A useful
diagnostic is delta_e(t)=[E(t)-E(t0)]/N_atoms (eV/atom), and its fitted drift
slope (eV/atom/ps). Large absolute UMA energy offsets make fractional drift
relative to |E| potentially misleading. Confirm that logged `pe` includes the
external UMA energy before trusting `etotal`. Compare short runs at h and h/2
over the same physical duration if energy drift is concerning.

`fix nve` uses velocity-Verlet integration; it does not guarantee energy
conservation for an inconsistent external force/energy implementation.
[LAMMPS NVE documentation](https://docs.lammps.org/fix_nve.html).
Check molecular identity/reactions and stationarity as well as energy drift
before treating a segment as equilibrated production data. No simulation has
been executed or validated by creating this template.
