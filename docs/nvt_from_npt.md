# NVT continuation without shear

Use [in.nvt_from_npt](../examples/lammps/in.nvt_from_npt) with your existing
fairchem-LAMMPS launch command. It keeps the selected NPT restart's coordinates,
velocities, step number and instantaneous box (including tilts), and controls
temperature at 400 K. Pressure is measured, not held at 2.5 GPa. The output
prefix identifies the source NPT condition, not a pressure constraint.

`fix integrate_nvt all nvt temp 400.0 400.0 0.05` provides both time integration
and a Nose-Hoover thermostat. Do not also apply `fix nve`, `fix langevin` or
`fix npt` to these atoms. The standalone restart script does not need to unfix
the former NPT integrator; switching within an existing simulation does.

## Settings to edit

| Setting | Meaning |
|---|---|
| `read_restart` path | Compatible binary restart from the NPT run |
| Both `400.0` values | Start/end target temperature in K; keep equal for constant target temperature |
| `0.05` in the NVT fix | Thermostat damping time in ps, not an integration timestep |
| `timestep 0.0005` | Integration timestep in ps; assumes metal units |
| Both `2000000` values in `run` | Additional integration steps; keep equal for final writes at the end |
| `nvt_2p5` prefix | Change in all dump, checkpoint and final output names |
| `thermo 1000`, dump interval `1000` | Steps between outputs; matching intervals simplify analysis |
| `restart 100000` | Steps between binary checkpoints |
| `element C H N O` | Element mapping for types 1, 2, 3, 4 |

For integration timestep \(h\), run length \(N\), and output interval \(s\),
\[
t_{\mathrm{run}}=Nh,\qquad \Delta t_{\mathrm{output}}=sh.
\]
The example runs 1000 ps, saves frames every 0.5 ps, and checkpoints every 50 ps.
The damping time satisfies \(T_{\mathrm{damp}}/h=100\) steps here, a starting
choice consistent with the original NPT settings, not a validated optimum.

## Checks and analysis

Temperature fluctuates around its target. Physical total energy
\(E(t)=K(t)+U(t)\) also fluctuates because the thermostat exchanges heat;
NVE-style conservation of `etotal` is not the NVT acceptance criterion.
Inspect temperature, potential energy and pressure for stationary statistics,
verify the box remains constant, and check molecular identity before selecting
production frames. Fixed volume alone does not demonstrate equilibration.
For diffusion, check sensitivity to thermostat damping and the fit window.

Keep the same actual UMA model and electronic/inference settings as the source
run. Confirm that the external driver supplies consistent energies, forces and
stress; this template does not configure or validate the coupling. No UMA or
LAMMPS simulation was executed to validate this example.

The dump retains forces, velocities and image flags for MartiniFuncs workflows;
thermo output remains aligned with saved frames. See the
[LAMMPS NVT documentation](https://docs.lammps.org/fix_nh.html) for thermostat
semantics and the [NVE alternative](nve_from_npt.md) for unthermostatted runs.
