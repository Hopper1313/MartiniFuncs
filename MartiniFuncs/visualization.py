"""Calculator-free Psience views of extracted ASE structures.

Psience is imported only when constructing a view. Source forces stay in
eV/Angstrom; arrow lengths are display displacements, not physical coordinates.
"""
import numpy as np


def to_display_molecule(atoms):
    """Convert geometry only (Angstrom to Bohr), without attaching an evaluator.

    Use the original ASE object for identity, metadata, and calculations. This
    display object does not replace it or preserve its calculator/constraints.

    r_bohr=r_angstrom*UnitsData.convert('Angstroms','BohrRadius').
    Only coordinates and elements are transferred; no force-unit conversion occurs.
    """
    from McUtils.Data import UnitsData
    from Psience.Molecools import Molecule

    angstrom_to_bohr = UnitsData.convert("Angstroms", "BohrRadius")
    coords_bohr = atoms.get_positions() * angstrom_to_bohr
    return Molecule(atoms=atoms.get_chemical_symbols(), coords=coords_bohr)


def force_arrow_displacements(atoms, *, array="lammps_forces", scale=0.2):
    """Return arrow displacements in Angstrom, with scale in Angstrom^2/eV.

    Input must be Cartesian forces in eV/Angstrom. No normalization or sign
    change is performed. Zero forces produce zero displacements.

    Arrow displacement d_i=scale*F_i, where F_i is eV/Angstrom and
    scale is Angstrom^2/eV, giving d_i in Angstrom.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    if array not in atoms.arrays:
        raise ValueError(f"Missing force array {array!r}")
    values = np.asarray(atoms.arrays[array], dtype=float)
    if values.shape != (len(atoms), 3) or not np.isfinite(values).all():
        raise ValueError("Forces must be a finite (N, 3) array")
    if atoms.info.get("force_units", "eV/Angstrom") != "eV/Angstrom":
        raise ValueError("Convert forces to eV/Angstrom before plotting")
    return values * scale


def plot_cluster(atoms, *, forces=False, force_array="lammps_forces", force_scale=0.2, **plot_kwargs):
    """Return a Psience X3D figure with CPD/MAL/shell colors and optional arrows.

    Green = CPD, orange = MAL, gray = shell; red arrows show the named stored
    force array. No energy/force calculation is triggered. Atom labels in the
    viewer are local rows; persistent IDs remain in the supplied ASE arrays.
    Bonds use Psience Molecule inference by default. Use bonds=False for a
    full simulation-cell view. To add guessed forming bonds, convert with
    to_display_molecule, append local-index bonds to mol.bonds, and plot mol.
    An explicit bonds list is passed through only when requested.
    X3D display requires a trusted notebook and the viewer's JavaScript assets.

    When forces=True, d_angstrom=force_scale*F; mode_vectors=d_angstrom
    *UnitsData.convert('Angstroms','BohrRadius'). These are display displacements,
    not physical forces converted to Hartree/Bohr.
    """
    from McUtils.Data import UnitsData

    roles = atoms.arrays.get("pair_role", np.full(len(atoms), "shell"))
    colors = {"CPD": "green", "MAL": "orange", "shell": "gray"}
    styles = {i: {"color": colors.get(str(role), "gray")} for i, role in enumerate(roles)}
    options = {}
    if forces:
        arrow_displacements_angstrom = force_arrow_displacements(
            atoms, array=force_array, scale=force_scale
        )
        # mode_vectors are display displacements in the Molecule's Bohr frame,
        # NOT forces in Hartree/Bohr. The source force array is unchanged.
        arrow_displacements_bohr = (
            arrow_displacements_angstrom * UnitsData.convert("Angstroms", "BohrRadius")
        )
        options["mode_vectors"] = arrow_displacements_bohr.reshape(-1)
        options["vector_style"] = {"color": "red"}

    return to_display_molecule(atoms).plot(
        backend="x3d", background="white",
        atom_style=styles, include_save_buttons=True, **options, **plot_kwargs,
    )


def plot_pair_geometry(atoms, *, disk_radius=2.0, transparency=0.5,
                       cpd_color="green", dienophile_color="orange",
                       line_color="blue", scale=1.3, **plot_kwargs):
    """Plot reactive-plane disks and the line joining the two site midpoints.

    Supply an extracted, whole, nonperiodic pair or pair-plus-shell with
    ``pair_role`` and ``global_index`` arrays. The site identification and fitted
    plane normals are the same as in pair_selection.measure_pair_geometry.
    Disk centers are the CPD termini midpoint and dienophile C=C midpoint;
    their orientations are fits to the four reactive carbons in each molecule.
    For nonplanar sites, these are orientation guides centered on the sites,
    not necessarily the exact least-squares plane offsets.

    disk_radius is in Angstrom; transparency ranges from 0 (opaque) to 1.
    Molecule construction converts ASE Angstrom coordinates to Bohr, while
    Molecule.plot explicitly converts back to Angstrom for the figure. The
    McUtils primitives use these figure coordinates directly. Plot units are
    fixed to Angstroms so disks and molecule cannot silently use different units.

    Disk centers are c_CPD=(r_end1+r_end2)/2 and c_CC=(r_CC1+r_CC2)/2.
    Each unit normal is the smallest right-singular vector of its centered
    four-carbon coordinates. Draw the line c_CPD to c_CC and disks of
    radius disk_radius in the Angstrom display frame.
    """
    import McUtils.Plots as plt
    from . import graph as mfg
    from .pair_selection import _sites, _normal

    if atoms.pbc.any():
        raise ValueError("Supply an extracted, whole, nonperiodic pair or cluster")
    if not np.isfinite(disk_radius) or disk_radius <= 0:
        raise ValueError("disk_radius must be finite and positive (Angstrom)")
    if not np.isfinite(transparency) or not 0 <= transparency <= 1:
        raise ValueError("transparency must be between 0 and 1")
    if plot_kwargs.pop("units", "Angstroms") != "Angstroms":
        raise ValueError("Pair geometry overlays require plot units='Angstroms'")
    if "pair_role" not in atoms.arrays or "global_index" not in atoms.arrays:
        raise ValueError("Require pair_role and global_index arrays from pair extraction")
    ids = mfg.atom_ids(atoms)
    roles = atoms.arrays["pair_role"]
    graph = mfg.build_bond_graph(atoms, scale=scale)
    ends, diene, cc, dienophile_plane = _sites(
        atoms, graph, ids[roles == "CPD"], ids[roles == "MAL"]
    )
    xyz = dict(zip(ids, atoms.positions))
    cpd_center = np.mean([xyz[i] for i in ends], axis=0)
    dienophile_center = np.mean([xyz[i] for i in cc], axis=0)
    cpd_normal = _normal(np.array([xyz[i] for i in diene]))
    dienophile_normal = _normal(np.array([xyz[i] for i in dienophile_plane]))
    fig = plot_cluster(atoms, units="Angstroms", **plot_kwargs)
    plt.Disk(cpd_center, disk_radius, normal=cpd_normal, color=cpd_color,
             transparency=transparency).plot(fig)
    plt.Disk(dienophile_center, disk_radius, normal=dienophile_normal,
             color=dienophile_color, transparency=transparency).plot(fig)
    plt.Line([cpd_center, dienophile_center], line_style="dashed", color=line_color).plot(fig)
    return fig
