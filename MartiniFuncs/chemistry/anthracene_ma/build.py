"""Build finite anthracene + maleic anhydride pairs with a chosen distance and orientation.

Everything is expressed in the anthracene frame, so a placement reads directly:

    origin  centroid of the anthracene central ring (C9, C10, four fusion carbons)
    y       C9 -> C10 (the termini axis), projected into the ring plane
    z       the central-ring normal, pointing away from the outer-ring carbons when
            the anthracene is bent (as in a TS or adduct: the benzo rings fold away
            from the dienophile, which then lies on +z); either face if it is flat
    x       y cross z: the long axis, through the outer benzo rings

The anthracene is moved rigidly into this frame once. Maleic anhydride (MA) is
placed as a rigid body: its C=C midpoint goes to (offset_x, offset_y, height)
and its orientation is three rotations of its reference pose. In the reference
pose (all angles 0) the MA C=C is parallel to C9-C10, its ring plane is
parallel to the anthracene plane and the anhydride ring points along +x; with
height > 0 that is the stacked, reactive arrangement. Then

    fold    rotation about the y axis (the C=C in the reference pose); positive
            lifts the anhydride ring away from the anthracene
    roll    rotation about the x axis; tilts the C=C out of the anthracene plane
    twist   rotation about the z axis; turns the C=C in the plane (0: parallel
            to C9-C10, 90: along the long axis)

applied in that order: R = Rz(twist) Rx(roll) Ry(-fold) (placement_rotation).
The distance is set by distance_kind:

    "height"       z of the MA C=C midpoint above the ring centroid (default)
    "centers"      ring centroid to C=C midpoint, |(offset_x, offset_y, height)|:
                   the screen distance of the anthracene + MA pair
    "max_forming"  the longer forming C...C distance (minimax matching), found by
                   moving the MA along z at fixed offsets and orientation

The returned pair is measured with the shared Diels-Alder equations
(chemistry.diels_alder.measure_diels_alder), the same as a trajectory analysis.
measure_placement reads these parameters back off any anthracene + MA
structure (e.g. a reference TS), so builder coordinates and real structures
can be compared directly. Nothing here is a TS or energy calculation.

Example::

    from MartiniFuncs.chemistry.anthracene_ma.build import build_pair, measure_placement, reference_structure
    atoms, geometry = build_pair(distance=3.2)                         # stacked, reactive pose
    atoms, geometry = build_pair(distance=3.0, distance_kind="max_forming", twist=30, fold=15)
    print(measure_placement(reference_structure("transition_state")))
"""
from __future__ import annotations

from importlib import resources

import numpy as np
from ase import Atoms
from ase.io import read
from scipy.optimize import brentq

from MartiniFuncs.chemistry.anthracene_ma.sites import (ANHYDRIDE_SIZE, ANTHRACENE_SIZE, anhydride_sites,
                                                        anthracene_sites)
from MartiniFuncs.chemistry.diels_alder import measure_diels_alder, plane_normal, SHARED_METRICS
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.structure.molecules import atoms_from_smiles

SMILES = {"anthracene": "c1ccc2cc3ccccc3cc2c1", "maleic_anhydride": "O=C1OC(=O)C=C1"}
ROLE_LABELS = ("ANT", "MA")
DISTANCE_KINDS = ("height", "centers", "max_forming")
REFERENCES = ("anthracene", "maleic_anhydride", "reactant_complex", "transition_state", "product")
# MA body axes (C=C, ring direction, normal) -> anthracene axes in the reference pose: a->y, b->x, c->-z.
REFERENCE_POSE = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])


def reference_structure(name):
    """A UMA-s-1p2p1 reference structure (charge 0, spin 1) from the anthracene + MA hand-off, as ASE Atoms.

    name: "anthracene", "maleic_anhydride" (relaxed isolated molecules),
    "reactant_complex", "transition_state" or "product" (33-atom structures).
    """
    if name not in REFERENCES:
        raise ValueError(f"name must be one of {REFERENCES}")
    with resources.as_file(resources.files(__package__) / "data" / f"uma_{name}.xyz") as path:
        structure = read(path)
    # Geometry only; standard masses (the files carry the reference calculation's isotope masses).
    return Atoms(structure.get_chemical_symbols(), positions=structure.positions)


def uma_templates():
    """(anthracene, maleic anhydride) templates relaxed with UMA-s-1p2p1: the builder's default."""
    return reference_structure("anthracene"), reference_structure("maleic_anhydride")


def rdkit_templates(seed=61453):
    """(anthracene, maleic anhydride) templates from SMILES (RDKit ETKDG + MMFF94)."""
    return atoms_from_smiles(SMILES["anthracene"], seed), atoms_from_smiles(SMILES["maleic_anhydride"], seed)


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Degenerate template geometry")
    return vector / norm


def _graph_sites(molecule, sites_function, scale):
    """Sites of one finite molecule, keyed by its row indices."""
    plain = Atoms(molecule.get_chemical_symbols(), positions=molecule.positions)
    graph = mfg.build_bond_graph(plain, scale=scale, index_array=None)
    return sites_function(graph, dict(enumerate(plain.get_chemical_symbols())), list(range(len(plain))))


# Outer-ring carbons this far (Angstrom, mean) off the central-ring plane make the anthracene "bent".
BENT_TOLERANCE = 0.02


def anthracene_frame(xyz, termini, ring, outer):
    """(origin, axes, bent) of the anthracene frame; axes rows are x, y, z unit vectors (module docstring).

    outer are the outer-ring carbon rows. bent is True when their mean offset
    from the central-ring plane exceeds BENT_TOLERANCE; z then points away from them.
    """
    origin = xyz[list(ring)].mean(axis=0)
    z = plane_normal(xyz[list(ring)])
    wing = float(np.mean((xyz[list(outer)] - origin) @ z))
    bent = abs(wing) > BENT_TOLERANCE
    if bent and wing > 0:
        z = -z
    y = xyz[termini[1]] - xyz[termini[0]]
    y = _unit(y - np.dot(y, z) * z)
    return origin, np.vstack([np.cross(y, z), y, z]), bent


def _outer_carbons(symbols, rows, ring):
    """Anthracene carbon rows that are not in the central ring."""
    return [i for i in rows if symbols[i] == "C" and i not in set(ring)]


def anhydride_frame(xyz, alkene, ring, oxygen, hydrogens):
    """(origin, axes, bent) of the MA body frame; rows a, b, c.

    c is the ring-carbon normal, pointing away from the two vinyl H atoms when
    they are bent out of the ring plane (mean offset above BENT_TOLERANCE, as in
    a TS, where they bend away from the diene: c then points at the diene).
    b is in plane, toward the ring O; a = b x c lies along the C=C. For a flat
    MA, c is the fitted normal's sign and a follows; the C2 axis b relates both.
    """
    origin = xyz[list(alkene)].mean(axis=0)
    c = plane_normal(xyz[list(ring)])
    offset = float(np.mean((xyz[list(hydrogens)] - origin) @ c))
    bent = abs(offset) > BENT_TOLERANCE
    if bent and offset > 0:
        c = -c
    b = xyz[oxygen] - origin
    b = _unit(b - np.dot(b, c) * c)
    return origin, np.vstack([np.cross(b, c), b, c]), bent


def _hydrogens(symbols, rows):
    """Hydrogen rows of one molecule."""
    return [i for i in rows if symbols[i] == "H"]


def placement_rotation(twist=0.0, roll=0.0, fold=0.0):
    """R = Rz(twist) Rx(roll) Ry(-fold), angles in degrees, acting on reference-pose MA coordinates."""
    t, r, f = np.radians([twist, roll, -fold])
    rz = np.array([[np.cos(t), -np.sin(t), 0.], [np.sin(t), np.cos(t), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., np.cos(r), -np.sin(r)], [0., np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(f), 0., np.sin(f)], [0., 1., 0.], [-np.sin(f), 0., np.cos(f)]])
    return rz @ rx @ ry


class PairFrames:
    """Two templates in the builder's frames: anthracene coordinates and the MA reference pose.

    anthracene_xyz is the anthracene in the anthracene frame; anhydride_xyz the
    MA in its reference pose with the C=C midpoint at the origin. termini,
    central_ring, alkene, ring_carbons and bridging_oxygen index each template's rows.
    """

    def __init__(self, anthracene=None, anhydride=None, *, scale=1.3):
        if anthracene is None or anhydride is None:
            default = uma_templates()
            anthracene = default[0] if anthracene is None else anthracene
            anhydride = default[1] if anhydride is None else anhydride
        for molecule, size, name in ((anthracene, ANTHRACENE_SIZE, "anthracene"),
                                     (anhydride, ANHYDRIDE_SIZE, "maleic anhydride")):
            if len(molecule) != size or np.any(molecule.pbc):
                raise ValueError(f"The {name} template must be one finite molecule of {size} atoms")
        self.anthracene, self.anhydride = anthracene, anhydride
        self.termini, self.central_ring = _graph_sites(anthracene, anthracene_sites, scale)
        self.alkene, self.ring_carbons, self.bridging_oxygen = _graph_sites(anhydride, anhydride_sites, scale)
        outer = _outer_carbons(anthracene.get_chemical_symbols(), range(len(anthracene)), self.central_ring)
        origin, axes, _ = anthracene_frame(anthracene.positions, self.termini, self.central_ring, outer)
        self.anthracene_xyz = (anthracene.positions - origin) @ axes.T
        hydrogens = _hydrogens(anhydride.get_chemical_symbols(), range(len(anhydride)))
        origin, axes, _ = anhydride_frame(anhydride.positions, self.alkene, self.ring_carbons, self.bridging_oxygen,
                                          hydrogens)
        self.anhydride_xyz = ((anhydride.positions - origin) @ axes.T) @ REFERENCE_POSE.T

    def place(self, *, height, offset=(0.0, 0.0), twist=0.0, roll=0.0, fold=0.0):
        """MA coordinates (anthracene frame) for one placement; height and offsets in Angstrom."""
        position = np.array([offset[0], offset[1], height], dtype=float)
        return self.anhydride_xyz @ placement_rotation(twist, roll, fold).T + position

    def max_forming_distance(self, anhydride_xyz):
        """Longer forming C...C distance of the minimax matching (as the max_forming_distance metric)."""
        ant = self.anthracene_xyz
        return min(max(np.linalg.norm(ant[t] - anhydride_xyz[c]) for t, c in zip(self.termini, partner))
                   for partner in (self.alkene, self.alkene[::-1]))


def build_pair(anthracene=None, anhydride=None, *, distance=3.2, distance_kind="height", offset=(0.0, 0.0),
               twist=0.0, roll=0.0, fold=0.0, min_contact=None, scale=1.3, frames=None):
    """Place a rigid MA relative to an anthracene; return (Atoms, measured DielsAlderGeometry).

    Templates default to the UMA-relaxed molecules (uma_templates). distance and
    offset are Angstrom, twist/roll/fold degrees (module docstring). min_contact
    (Angstrom), if given, rejects structures with a shorter intermolecular
    atom-atom distance. frames (a PairFrames) skips re-deriving the template frames.

    The returned Atoms are finite, in the anthracene frame, anthracene rows first
    then MA rows, with global_index 0..N-1 and pair_role "ANT"/"MA"; info records
    the placement (height, offsets, angles), screen_distance (ring centroid to
    C=C midpoint), min_intermolecular_distance and the measured shared metrics.
    """
    if distance_kind not in DISTANCE_KINDS:
        raise ValueError(f"distance_kind must be one of {DISTANCE_KINDS}")
    if not np.isfinite(distance) or (distance_kind != "height" and distance <= 0):
        raise ValueError("distance must be finite (and positive for centers / max_forming)")
    offset = tuple(float(v) for v in offset)
    if len(offset) != 2 or not np.all(np.isfinite(offset)) or not np.all(np.isfinite([twist, roll, fold])):
        raise ValueError("offset must be two finite numbers and the angles finite")
    frames = frames or PairFrames(anthracene, anhydride, scale=scale)

    def placed(height):
        return frames.place(height=height, offset=offset, twist=twist, roll=roll, fold=fold)

    if distance_kind == "height":
        height = float(distance)
    elif distance_kind == "centers":
        squared = distance ** 2 - offset[0] ** 2 - offset[1] ** 2
        if squared <= 0:
            raise ValueError("centers distance must exceed the in-plane offset")
        height = float(np.sqrt(squared))
    else:
        # The lowest height (above the plane) at which the longer forming distance reaches the request.
        grid = np.linspace(0., 30., 601)
        values = np.array([frames.max_forming_distance(placed(h)) - distance for h in grid])
        crossing = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
        if not len(crossing):
            raise ValueError(f"No height at these offsets and angles gives max_forming = {distance:g} Angstrom")
        k = int(crossing[0])
        height = float(brentq(lambda h: frames.max_forming_distance(placed(h)) - distance, grid[k], grid[k + 1],
                              xtol=1e-12))

    ant, ma = frames.anthracene_xyz, placed(height)
    n = len(ant)
    ai, mi = np.arange(n), np.arange(n, n + len(ma))
    geometry = measure_diels_alder(ai, ant, mi, ma, frames.termini, frames.central_ring,
                                   tuple(int(i) + n for i in frames.alkene),
                                   tuple(int(i) + n for i in frames.ring_carbons),
                                   {int(i): int(i) for i in np.concatenate((ai, mi))})
    contact = float(np.linalg.norm(ma[:, None, :] - ant[None, :, :], axis=-1).min())
    if min_contact is not None and contact < min_contact:
        raise ValueError(f"Closest intermolecular atoms are {contact:.3f} Angstrom apart (min_contact {min_contact:g})")

    atoms = Atoms(list(frames.anthracene.get_chemical_symbols()) + list(frames.anhydride.get_chemical_symbols()),
                  positions=np.vstack((ant, ma)))
    atoms.new_array("global_index", np.arange(len(atoms)))
    atoms.new_array("pair_role", np.array([ROLE_LABELS[0]] * n + [ROLE_LABELS[1]] * len(ma)))
    metrics = {"max_forming_distance": max(geometry.forming_distances),
               "distance_imbalance": geometry.distance_imbalance, "axis_angle": geometry.axis_angle,
               "plane_angle": geometry.plane_angle, "approach_angle": geometry.approach_angle}
    assert set(metrics) == set(SHARED_METRICS)
    # Flat scalar keys, so the structure writes to Extended XYZ unchanged.
    atoms.info.update(requested_distance=float(distance), requested_distance_kind=distance_kind,
                      height=height, offset_x=offset[0], offset_y=offset[1], twist=float(twist),
                      roll=float(roll), fold=float(fold),
                      screen_distance=float(np.linalg.norm([offset[0], offset[1], height])),
                      min_intermolecular_distance=contact,
                      **{f"measured_{name}": float(value) for name, value in metrics.items()})
    return atoms, geometry


def _pair_sites(atoms, scale):
    """Rows and sites of the anthracene and the MA in one finite structure (any atom order).

    Membership comes from a pair_role array ("ANT"/"MA") when present, so close
    contacts cannot merge molecules, otherwise from the bond graph.
    """
    plain = Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)
    plain.new_array("global_index", np.arange(len(plain)))
    graph = mfg.build_bond_graph(plain, scale=scale)
    elements = dict(enumerate(plain.get_chemical_symbols()))
    if "pair_role" in atoms.arrays:
        roles = np.asarray(atoms.arrays["pair_role"]).astype(str)
        components = [list(np.flatnonzero(roles == label)) for label in ROLE_LABELS]
    else:
        components = mfg.connected_components(graph, atom_ids=mfg.atom_ids(plain))
    by_size = {len(c): c for c in components}
    if sorted(by_size) != sorted({ANTHRACENE_SIZE, ANHYDRIDE_SIZE}) or len(components) != 2:
        raise ValueError(f"Expected one anthracene ({ANTHRACENE_SIZE} atoms) and one maleic anhydride "
                         f"({ANHYDRIDE_SIZE} atoms); found molecules of sizes {sorted(len(c) for c in components)}")
    termini, ring = anthracene_sites(graph, elements, by_size[ANTHRACENE_SIZE])
    alkene, carbons, oxygen = anhydride_sites(graph, elements, by_size[ANHYDRIDE_SIZE])
    return {"xyz": plain.positions, "symbols": plain.get_chemical_symbols(),
            "anthracene": [int(i) for i in by_size[ANTHRACENE_SIZE]],
            "anhydride": [int(i) for i in by_size[ANHYDRIDE_SIZE]], "termini": termini, "central_ring": ring,
            "alkene": alkene, "ring_carbons": carbons, "bridging_oxygen": oxygen}


def structure_metrics(atoms, *, scale=1.3):
    """The shared Diels-Alder metrics, screen distance and closest contact of any anthracene + MA structure.

    The same equations as build_pair's info and a trajectory analysis
    (measure_diels_alder): max_forming_distance, distance_imbalance (Angstrom),
    axis_angle, plane_angle, approach_angle (acute degrees); screen_distance is
    the central-ring centroid to C=C midpoint distance and
    min_intermolecular_distance the closest anthracene-MA atom pair (Angstrom).
    """
    s = _pair_sites(atoms, scale)
    xyz = s["xyz"]
    ant, ma = np.asarray(s["anthracene"]), np.asarray(s["anhydride"])
    geometry = measure_diels_alder(ant, xyz[ant], ma, xyz[ma], s["termini"], s["central_ring"], s["alkene"],
                                   s["ring_carbons"], {int(i): int(i) for i in range(len(xyz))})
    return {"screen_distance": float(np.linalg.norm(xyz[list(s["alkene"])].mean(axis=0)
                                                    - xyz[list(s["central_ring"])].mean(axis=0))),
            "max_forming_distance": max(geometry.forming_distances),
            "distance_imbalance": geometry.distance_imbalance, "axis_angle": geometry.axis_angle,
            "plane_angle": geometry.plane_angle, "approach_angle": geometry.approach_angle,
            "min_intermolecular_distance": float(np.linalg.norm(xyz[ant][:, None] - xyz[ma][None], axis=-1).min())}


def measure_placement(atoms, *, scale=1.3):
    """Placement parameters (height, offset_x, offset_y, twist, roll, fold, screen_distance) of a structure.

    atoms holds one whole anthracene and one whole maleic anhydride (any atom
    order, finite coordinates, e.g. a reference complex or TS, or a builder
    pair). Membership comes from a pair_role array ("ANT"/"MA") when present,
    otherwise from the bond graph. The anthracene frame is fitted to that
    structure's own anthracene (best fit when it is distorted, as in a TS), with
    z from its bend; a flat anthracene's z is chosen so the MA lies on +z.

    The molecules' twofold symmetries make several parameter sets describe one
    geometry: anthracene's C2 about z (x, y -> -x, -y) and, for a flat MA, its
    C2 axis (through the C=C midpoint and the ring O); a bent MA's frame is fixed
    by its vinyl-H bend instead (anhydride_frame). Of these, the simplest is
    returned: smallest |fold|, then |twist|, then |roll|. For symmetric
    templates, build_pair with the returned parameters reproduces the structure;
    for distorted molecules (TS), up to their small deviation from symmetry.
    """
    s = _pair_sites(atoms, scale)
    xyz, symbols = s["xyz"], s["symbols"]
    origin, axes, bent = anthracene_frame(xyz, s["termini"], s["central_ring"],
                                          _outer_carbons(symbols, s["anthracene"], s["central_ring"]))
    midpoint = xyz[list(s["alkene"])].mean(axis=0) - origin
    if not bent and axes[2] @ midpoint < 0:
        # A flat anthracene's faces are equivalent (C2 about y: x -> -x, z -> -z).
        axes = axes * np.array([[-1.], [1.], [-1.]])
    _, body, ma_bent = anhydride_frame(xyz, s["alkene"], s["ring_carbons"], s["bridging_oxygen"],
                                       _hydrogens(symbols, s["anhydride"]))
    candidates = []
    for ant_c2z in (1., -1.):          # anthracene C2 about z: x, y -> -x, -y
        for ma_c2 in ((1.,) if ma_bent else (1., -1.)):    # flat MA: C2 about its ring axis b (a, c -> -a, -c)
            frame = axes * np.array([[ant_c2z], [ant_c2z], [1.]])
            rotation = (frame @ (body * np.array([[ma_c2], [1.], [ma_c2]])).T) @ REFERENCE_POSE.T
            position = frame @ midpoint
            roll = float(np.degrees(np.arcsin(np.clip(rotation[2, 1], -1., 1.))))
            twist = float(np.degrees(np.arctan2(-rotation[0, 1], rotation[1, 1])))
            fold = -float(np.degrees(np.arctan2(-rotation[2, 0], rotation[2, 2])))
            placement = {"height": float(position[2]), "offset_x": float(position[0]),
                         "offset_y": float(position[1]), "twist": twist, "roll": roll, "fold": fold,
                         "screen_distance": float(np.linalg.norm(position))}
            candidates.append(((round(abs(fold), 6), round(abs(twist), 6), round(abs(roll), 6)), placement))
    return min(candidates, key=lambda item: item[0])[1]
