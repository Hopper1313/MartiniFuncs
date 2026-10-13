"""Build finite CPD + N4T pair structures with a requested pair geometry.

The N4T (dienophile) template is never moved. The CPD template is moved as a
rigid body so the pair has the requested reaction-center distance, ring
dihedral and plane tilt; the result is then measured with exactly the same
equations as the trajectory analysis (selection.measure_pair_geometry).

Construction, all in the fixed N4T frame (P2 = CPD termini midpoint, P3 = N4T
C=C midpoint, P1 = CH2 carbon, P4 = imide N, as in PairGeometry.ring_dihedral):

1. u is the unit approach axis, perpendicular to the N4T vector P4 - P3 and
   to its C=C axis (the N4T ring normal); the CPD sits on side ``face`` (+1/-1).
2. The CPD is turned so P1 - P2 points along P4 - P3 rotated by -ring_dihedral
   about u, with its ring plane perpendicular to u (parallel stacking).
3. It is tilted by plane_angle about the axis through P2 perpendicular to both
   u and P1 - P2 (essentially the termini axis). Positive values open the CH2
   side away from the N4T, as in the Diels-Alder transition state; negative
   values tilt it toward the N4T. The tilt leaves the ring dihedral unchanged.
4. P2 = P3 - d u, with d chosen so the requested distance is met.

ring_dihedral and the distance are met to numerical precision. plane_angle and
approach_angle use least-squares ring normals, so they equal |plane_angle| up to
the templates' deviation from planarity (a fraction of a degree for MMFF CPD
and N4T); axis_angle follows from the dihedral for stacked rings. Always check
the returned geometry and minimum intermolecular distance; nothing here is a
transition-state or energy calculation.

product_template builds the endo or exo adduct itself from a stereo SMILES,
with the same row layout (CPD-derived atoms first).

Example::

    from MartiniFuncs.chemistry.cpd_n4t.build import build_pair, product_template, rdkit_templates
    cpd, n4t = rdkit_templates()
    atoms, geometry = build_pair(cpd, n4t, distance=3.0, ring_dihedral=180, plane_angle=15)
    atoms.write("endo_3A.xyz")
    print(geometry.forming_distances, geometry.approach_type, atoms.info["min_intermolecular_distance"])
    product, geometry = product_template("exo")
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from MartiniFuncs.chemistry.cpd_n4t.model import CPD_SIZE, DIENOPHILE_SIZE
from MartiniFuncs.chemistry.cpd_n4t.selection import _cpd_sites, _dienophile_sites, _measure_placed_pair
from MartiniFuncs.pairs.model import get_pair_model
from MartiniFuncs.structure import graph as mfg

# Cyclopentadiene and N-(p-tolyl)maleimide (N4T, C11H9NO2).
SMILES = {"C5H6": "C1C=CC=C1", "C11H9NO2": "Cc1ccc(cc1)N1C(=O)C=CC1=O"}
# The two Diels-Alder adducts (both meso, cis-fused at the former C=C):
# 4-(p-tolyl)-4-azatricyclo[5.2.1.0^2,6]dec-8-ene-3,5-dione.
PRODUCT_SMILES = {"endo": "Cc1ccc(N2C(=O)[C@@H]3[C@H](C2=O)[C@@H]2C=C[C@H]3C2)cc1",
                  "exo": "Cc1ccc(N2C(=O)[C@@H]3[C@H](C2=O)[C@H]2C=C[C@@H]3C2)cc1"}
_PRODUCT_CPD = "[CH1]1=[CH1][CH1][CH2][CH1]1"     # the CPD-derived norbornene carbons
DISTANCE_KINDS = ("centers", "max_forming", "screen")


def _embed(smiles, seed):
    """RDKit molecule with hydrogens, embedded (ETKDG) and relaxed with MMFF94."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(molecule, randomSeed=seed) != 0:
        raise RuntimeError(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMolecule(molecule, maxIters=2000)
    return molecule


def rdkit_templates(seed=61453):
    """(CPD, N4T) templates: RDKit 3D embedding (ETKDG) relaxed with MMFF94, in Angstrom.

    Deterministic for a given seed. A geometry relaxed with your production
    model (e.g. UMA) can be passed to build_pair instead.
    """
    templates = []
    for formula, smiles in SMILES.items():
        molecule = _embed(smiles, seed)
        atoms = Atoms([a.GetSymbol() for a in molecule.GetAtoms()],
                      positions=molecule.GetConformer().GetPositions())
        if atoms.get_chemical_formula() != formula:
            raise RuntimeError(f"RDKit template has formula {atoms.get_chemical_formula()}, expected {formula}")
        templates.append(atoms)
    return tuple(templates)


def product_template(approach, seed=61453, *, scale=1.3):
    """The endo or exo Diels-Alder adduct (RDKit ETKDG + MMFF94, Angstrom) as (Atoms, measured PairGeometry).

    Rows are laid out like build_pair: the CPD-derived atoms first (five ring
    carbons, then their six hydrogens), then the N4T-derived atoms, with
    global_index and pair_role arrays. Measuring the two parts with the pair
    equations (forming bonds = the two new C-C bonds) must give the requested
    approach type; info records the measured metrics.
    """
    from rdkit import Chem
    if approach not in PRODUCT_SMILES:
        raise ValueError(f"approach must be one of {sorted(PRODUCT_SMILES)}")
    molecule = _embed(PRODUCT_SMILES[approach], seed)
    ring = molecule.GetSubstructMatch(Chem.MolFromSmarts(_PRODUCT_CPD))
    cpd = list(ring) + [n.GetIdx() for i in ring for n in molecule.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetSymbol() == "H"]
    order = cpd + [i for i in range(molecule.GetNumAtoms()) if i not in set(cpd)]
    symbols = [molecule.GetAtomWithIdx(i).GetSymbol() for i in order]
    xyz = molecule.GetConformer().GetPositions()[order]
    if len(cpd) != CPD_SIZE or len(order) != CPD_SIZE + DIENOPHILE_SIZE:
        raise RuntimeError("Product does not split into a C5H6 and a C11H9NO2 part")

    # The pair equations on the two parts; the bonding of each part is its own, so the forming bonds are excluded.
    ends, diene, bridge = _template_sites(Atoms(symbols[:CPD_SIZE], xyz[:CPD_SIZE]), CPD_SIZE, _cpd_sites, scale)
    cc, plane, nitrogen = _template_sites(Atoms(symbols[CPD_SIZE:], xyz[CPD_SIZE:]), DIENOPHILE_SIZE,
                                          _dienophile_sites, scale)
    sites = (tuple(ends), tuple(diene), bridge, tuple(int(i) + CPD_SIZE for i in cc),
             tuple(int(i) + CPD_SIZE for i in plane), int(nitrogen) + CPD_SIZE)
    ci, mi = np.arange(CPD_SIZE), np.arange(CPD_SIZE, len(order))
    geometry = _measure_placed_pair(ci, xyz[ci], mi, xyz[mi], sites, {i: i for i in range(len(order))})
    if geometry.approach_type != approach:
        raise RuntimeError(f"Embedded {approach} product measures as {geometry.approach_type}")

    atoms = Atoms(symbols, positions=xyz)
    atoms.new_array("global_index", np.arange(len(atoms)))
    atoms.new_array("pair_role", np.array(["CPD"] * CPD_SIZE + ["MAL"] * DIENOPHILE_SIZE))
    metrics = get_pair_model("cpd_n4t").metrics(geometry)
    atoms.info.update(structure="product", smiles=PRODUCT_SMILES[approach], approach_type=geometry.approach_type,
                      **{f"measured_{name}": float(value) for name, value in metrics.items()})
    return atoms, geometry


def templates_from_frame(atoms, *, scale=1.3):
    """(CPD, N4T) templates: the first whole C5H6 and C11H9NO2 molecules of a (periodic) frame."""
    bonds = mfg.build_bond_graph(atoms, scale=scale)
    rows = {int(i): row for row, i in enumerate(mfg.atom_ids(atoms))}
    found = {}
    for component in mfg.connected_components(bonds, atom_ids=mfg.atom_ids(atoms)):
        ids, xyz, _ = mfg.place_component(atoms, bonds, component, strict=True)
        selection = atoms[[rows[int(i)] for i in ids]]
        formula = selection.get_chemical_formula()
        if formula in SMILES and formula not in found:
            found[formula] = Atoms(selection.get_chemical_symbols(), positions=xyz, masses=selection.get_masses())
    missing = set(SMILES) - set(found)
    if missing:
        raise ValueError(f"Frame has no whole molecule of {sorted(missing)}")
    return found["C5H6"], found["C11H9NO2"]


def _template_sites(template, expected_size, sites_function, scale):
    """Sites of one finite template, keyed by its row indices."""
    if len(template) != expected_size or np.any(template.pbc):
        raise ValueError(f"Template must be one finite, nonperiodic molecule of {expected_size} atoms")
    molecule = Atoms(template.get_chemical_symbols(), positions=template.positions)
    graph = mfg.build_bond_graph(molecule, scale=scale, index_array=None)
    elements = dict(enumerate(molecule.get_chemical_symbols()))
    return sites_function(graph, elements, list(range(len(molecule))))


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Degenerate template geometry")
    return vector / norm


def build_pair(cpd, n4t, *, distance, ring_dihedral=0.0, plane_angle=0.0, distance_kind="centers",
               face=1, min_contact=None, scale=1.3):
    """Place a rigid CPD next to a fixed N4T; return (Atoms, measured PairGeometry).

    distance (Angstrom) is, by distance_kind:
      "centers"      P2-P3, CPD termini midpoint to N4T C=C midpoint (default);
      "max_forming"  the longer forming C...C distance, as the max_forming_distance metric;
      "screen"       whole-CPD COM to C=C COM, the analysis's reactive_radius screen distance.
    ring_dihedral (degrees) is the signed CH2 - P2 - P3 - N dihedral: near 0 exo,
    near +/-180 endo. plane_angle (degrees, -90 < plane_angle < 90) tilts the CPD
    ring: positive opens the CH2 side away from the N4T, negative toward it.
    face (+1 or -1) picks the N4T ring face that the CPD approaches.
    min_contact (Angstrom), if given, rejects structures with a shorter
    intermolecular atom-atom distance.

    The returned Atoms are finite (no cell), CPD rows first then N4T rows, with
    global_index 0..N-1 and pair_role "CPD"/"MAL", and info recording the
    request, the measured metrics and min_intermolecular_distance. The N4T
    coordinates equal the template's exactly.
    """
    if distance_kind not in DISTANCE_KINDS:
        raise ValueError(f"distance_kind must be one of {DISTANCE_KINDS}")
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance must be positive and finite")
    if not np.isfinite(plane_angle) or not -90 < plane_angle < 90:
        raise ValueError("plane_angle must lie strictly between -90 and 90 degrees")
    if not np.isfinite(ring_dihedral):
        raise ValueError("ring_dihedral must be finite")
    if face not in (1, -1):
        raise ValueError("face must be +1 or -1")

    ends, diene, bridge = _template_sites(cpd, CPD_SIZE, _cpd_sites, scale)
    cc, plane, nitrogen = _template_sites(n4t, DIENOPHILE_SIZE, _dienophile_sites, scale)
    body = np.asarray(cpd.positions, dtype=float)
    fixed = np.asarray(n4t.positions, dtype=float)

    # Fixed N4T frame: P3, the N direction b, and the approach axis u perpendicular to b and the C=C.
    p3 = fixed[list(cc)].mean(axis=0)
    b_hat = _unit(fixed[nitrogen] - p3)
    cc_axis = fixed[cc[1]] - fixed[cc[0]]
    u = _unit(np.cross(_unit(cc_axis - np.dot(cc_axis, b_hat) * b_hat), b_hat)) * -face   # CPD -> N4T

    # CPD body frame: P2, the CH2 direction a, and the termini axis made perpendicular to a.
    p2_body = body[list(ends)].mean(axis=0)
    a_hat = _unit(body[bridge] - p2_body)
    termini = body[ends[1]] - body[ends[0]]
    t_hat = _unit(termini - np.dot(termini, a_hat) * a_hat)
    body_frame = np.column_stack([a_hat, t_hat, np.cross(a_hat, t_hat)])

    # Target: CH2 direction = b rotated by -ring_dihedral about u; ring plane perpendicular to u.
    target_a = Rotation.from_rotvec(np.radians(-ring_dihedral) * u).apply(b_hat)
    target_t = _unit(np.cross(u, target_a))
    target_frame = np.column_stack([target_a, target_t, np.cross(target_a, target_t)])
    stacked = target_frame @ body_frame.T
    tilt = Rotation.from_rotvec(np.radians(plane_angle) * target_t).as_matrix()   # +: CH2 away from N4T
    rotation = tilt @ stacked
    shape = (body - p2_body) @ rotation.T            # rigid CPD, termini midpoint at the origin

    ci = np.arange(len(cpd))
    mi = np.arange(len(cpd), len(cpd) + len(n4t))
    sites = (tuple(ends), tuple(diene), bridge,
             tuple(int(i) + len(cpd) for i in cc), tuple(int(i) + len(cpd) for i in plane),
             int(nitrogen) + len(cpd))
    rows = {int(i): int(i) for i in np.concatenate((ci, mi))}
    cpd_masses, n4t_masses = cpd.get_masses(), n4t.get_masses()   # template masses, else standard
    cc_masses = n4t_masses[list(cc)]
    cc_com = cc_masses @ fixed[list(cc)] / cc_masses.sum()

    def placed(d):
        return shape + (p3 - d * u)

    def measured(d):
        xyz = placed(d)
        if distance_kind == "centers":
            return d
        if distance_kind == "screen":
            return float(np.linalg.norm(cpd_masses @ xyz / cpd_masses.sum() - cc_com))
        # Same matching rule as the max_forming_distance metric: the matching with the smaller maximum.
        return min(max(np.linalg.norm(xyz[e] - fixed[c]) for e, c in zip(ends, partner))
                   for partner in (cc, cc[::-1]))

    if distance_kind == "centers":
        d = float(distance)
    else:
        grid = np.linspace(0., 40., 801)
        values = np.array([measured(x) - distance for x in grid])
        crossing = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
        if not len(crossing):
            raise ValueError(f"No placement along the approach axis gives {distance_kind} = {distance:g} Angstrom")
        k = int(crossing[0])
        d = float(brentq(lambda x: measured(x) - distance, grid[k], grid[k + 1], xtol=1e-12))

    xyz = placed(d)
    geometry = _measure_placed_pair(ci, xyz, mi, fixed, sites, rows)
    separations = np.linalg.norm(xyz[:, None, :] - fixed[None, :, :], axis=-1)
    contact = float(separations.min())
    if min_contact is not None and contact < min_contact:
        raise ValueError(f"Closest intermolecular atoms are {contact:.3f} Angstrom apart (min_contact {min_contact:g})")

    atoms = Atoms(list(cpd.get_chemical_symbols()) + list(n4t.get_chemical_symbols()),
                  positions=np.vstack((xyz, fixed)), masses=np.concatenate((cpd_masses, n4t_masses)))
    atoms.new_array("global_index", np.arange(len(atoms)))
    atoms.new_array("pair_role", np.array(["CPD"] * len(cpd) + ["MAL"] * len(n4t)))
    metrics = get_pair_model("cpd_n4t").metrics(geometry)
    # Flat scalar keys, so the structure writes to Extended XYZ unchanged.
    atoms.info.update(requested_distance=float(distance), requested_distance_kind=distance_kind,
                      requested_ring_dihedral=float(ring_dihedral), requested_plane_angle=float(plane_angle),
                      face=int(face), center_separation=d, approach_type=geometry.approach_type,
                      min_intermolecular_distance=contact,
                      **{f"measured_{name}": float(value) for name, value in metrics.items()})
    return atoms, geometry
