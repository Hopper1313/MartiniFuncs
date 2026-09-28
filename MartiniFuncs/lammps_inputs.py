"""LAMMPS metal-unit run templates for the collaboration's external UMA driver.

These files do not configure a potential. Launch with the existing fairchem
driver and a locally licensed uma-s-1p1 model using task omol. Plain LAMMPS alone
does not supply UMA forces. The final executable command is always run.
"""
import re

import numpy as np
from ase.data import atomic_numbers

from .initial_configurations import _integer, _positive


def _filename(value, name):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError(f"{name} must be a simple LAMMPS path without spaces or substitutions")
    return value


def render_lammps_input(*, data_file, type_symbols, ensemble, steps,
                        force_source, output_prefix="simulation", timestep_ps=0.0005,
                        temperature_K=None, pressure_bar=None,
                        temperature_damping_ps=0.05, pressure_damping_ps=0.5,
                        initialize_velocities=True, velocity_seed=4928459,
                        thermo_every=100, dump_every=1000, restart_every=100000):
    """Return an NPT, NVT, or NVE input, with explicit external force ownership.

    ``force_source`` must be 'external_uma'; no guessed pair_style is inserted.
    Temperature is K, pressure bar, timestep/damping times ps (LAMMPS metal).
    NVE has no thermostat/barostat: temperature_K sets initial velocities only.
    If initialize_velocities=False, the data file must already contain suitable
    velocities. This is a new run from data, not an exact read_restart continuation.
    NPT uses isotropic pressure coupling and fully periodic boundaries.
    Defaults are editable examples, not a validated equilibration protocol.

    Run duration is steps*timestep_ps; dump spacing is dump_every*timestep_ps.
    The emitted v_netforce variable is each atom's sqrt(fx^2+fy^2+fz^2),
    not the frame net force |sum_i F_i|. Integration and thermostat/barostat
    equations are executed by LAMMPS, not evaluated by this renderer.
    """
    if force_source != "external_uma":
        raise ValueError("Only force_source='external_uma' is implemented")
    ensemble = ensemble.lower()
    if ensemble not in {"npt", "nvt", "nve"}:
        raise ValueError("ensemble must be npt, nvt, or nve")
    steps = _integer(steps, "steps")
    dt = _positive(timestep_ps, "timestep_ps")
    if not isinstance(initialize_velocities, bool):
        raise ValueError("initialize_velocities must be boolean")
    if initialize_velocities or ensemble != "nve":
        temperature_K = _positive(temperature_K if temperature_K is not None else np.nan, "temperature_K")
    if ensemble != "nve":
        tdamp = _positive(temperature_damping_ps, "temperature_damping_ps")
    if ensemble == "npt":
        if pressure_bar is None or not np.isfinite(pressure_bar):
            raise ValueError("NPT requires finite pressure_bar")
        pdamp = _positive(pressure_damping_ps, "pressure_damping_ps")
    elif pressure_bar is not None:
        raise ValueError("pressure_bar applies only to NPT")
    seed = _integer(velocity_seed, "velocity_seed")
    if seed >= 900000000:
        raise ValueError("LAMMPS velocity seed must be < 900000000")
    thermo_every = _integer(thermo_every, "thermo_every")
    dump_every = _integer(dump_every, "dump_every")
    restart_every = _integer(restart_every, "restart_every")
    data_file = _filename(data_file, "data_file")
    prefix = _filename(output_prefix, "output_prefix")
    types = list(type_symbols)
    if not types or len(set(types)) != len(types) or any(s not in atomic_numbers or s == "X" for s in types):
        raise ValueError("type_symbols must be distinct chemical symbols in data-file type order")
    lines = ["# Requires external fairchem driver: uma-s-1p1, task omol.",
             "# This input does not install a LAMMPS potential or load model weights.",
             "# Keep run as the final executable command for the external driver.",
             "units metal", "atom_style atomic", "boundary p p p", "atom_modify map array",
             f"read_data {data_file}", "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes",
             f"timestep {dt:g}"]
    if initialize_velocities:
        lines.append(f"velocity all create {temperature_K:g} {seed} mom yes rot yes dist gaussian")
    else:
        lines.append("# Preserve velocities read from data; ensure they are present.")
    if ensemble == "nve":
        lines.append("fix integrate all nve")
    else:
        fix = f"fix integrate all {ensemble} temp {temperature_K:g} {temperature_K:g} {tdamp:g}"
        if ensemble == "npt":
            fix += f" iso {pressure_bar:g} {pressure_bar:g} {pdamp:g}"
        lines.append(fix)
    lines += [f"thermo {thermo_every}",
              "thermo_style custom step time temp pe ke etotal press vol density lx ly lz",
              "variable netforce atom sqrt(fx*fx+fy*fy+fz*fz)",
              f"dump trajectory all custom {dump_every} {prefix}.lammpstrj id type mass element x y z ix iy iz vx vy vz fx fy fz v_netforce",
              "dump_modify trajectory element " + " ".join(types) + " sort id",
              f"restart {restart_every} {prefix}.restart.*",
              f'run {steps} every {steps} "write_data {prefix}.final.data" "write_restart {prefix}.final.restart"']
    return "\n".join(lines)+"\n"
