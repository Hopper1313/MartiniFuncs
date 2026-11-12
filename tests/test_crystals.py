"""Organic-crystal tools: CIF reading, whole molecules, slabs, lattice matching, interfaces, workflow.

The crystal is a synthetic P2_1/c benzene (Z = 4) written as a CIF here, so the
tests need no CSD files. ToyMolecularCalculator (harmonic bonds within
molecules + Lennard-Jones between molecules, with stress) keeps molecules
intact so the MLIP code paths can be exercised in seconds; its numbers mean
nothing physically.
"""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.data import covalent_radii
from ase.geometry import cellpar_to_cell
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from MartiniFuncs.crystals import cif, interfaces, morphology, slabs, surfaces, bulk
from MartiniFuncs.structure.graph import periodic_pairs

CELLPAR = [7.4, 9.4, 6.9, 90, 97, 90]
QUAT = [0.9517853796263588, -0.2201987726392127, -0.15091832274011197, -0.15113157022609086]
COM = [0.44781996433148946, 0.15888707060765578, 0.4105887073806304]
OPS = ["x,y,z", "-x,y+1/2,-z+1/2", "-x,-y,-z", "x,-y+1/2,z+1/2"]


def benzene_cif(extra_sites=(), banner=False, ops=OPS):
    cell = cellpar_to_cell(CELLPAR)
    mol = molecule("C6H6")
    pos = Rotation.from_quat(QUAT).apply(mol.positions - mol.positions.mean(0)) + np.array(COM) @ cell
    frac = np.linalg.solve(cell.T, pos.T).T
    lines = (["# exported by some program", "#"] if banner else []) + [
        "data_benzene_test",
        *(f"_cell_{k} {v}" for k, v in zip(("length_a", "length_b", "length_c", "angle_alpha", "angle_beta",
                                             "angle_gamma"), CELLPAR)),
        "_symmetry_space_group_name_H-M 'P 21/c'",
        "loop_", "_symmetry_equiv_pos_as_xyz", *(f"'{op}'" for op in ops),
        "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
        "_atom_site_fract_z", "_atom_site_occupancy", "_atom_site_disorder_group"]
    for k, (s, f) in enumerate(zip(mol.get_chemical_symbols(), frac)):
        lines.append(f"{s}{k + 1} {s} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} 1.0 .")
    lines.extend(extra_sites)
    return "\n".join(lines) + "\n"


def write_cif_text(text, directory):
    path = Path(directory) / "benzene.cif"
    path.write_text(text)
    return path


class ToyMolecularCalculator(Calculator):
    """Harmonic bonds (to covalent-radius sums) within molecules, LJ between molecules; energy, forces, stress."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, k=10.0, epsilon=0.004, sigma=2.6, cutoff=7.0, **kwargs):
        super().__init__(**kwargs)
        self.k, self.epsilon, self.sigma, self.cutoff = k, epsilon, sigma, cutoff

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos, cell, pbc = self.atoms.positions, self.atoms.cell.array, self.atoms.pbc
        mol = self.atoms.arrays.get("mol_id", np.zeros(len(pos), int))
        radii = covalent_radii[self.atoms.numbers]
        i, j, S, d = periodic_pairs(pos, cell, pbc, self.cutoff)
        keep = i < j
        i, j, S, d = i[keep], j[keep], S[keep], d[keep]
        vec = pos[j] + S @ cell - pos[i]
        same = mol[i] == mol[j]
        r0 = radii[i] + radii[j]
        bonded = same & (d < 1.3 * r0)
        lj = ~same
        energy = np.sum(self.k * (d[bonded] - r0[bonded]) ** 2)
        dEdr = np.zeros(len(d))
        dEdr[bonded] = 2 * self.k * (d[bonded] - r0[bonded])
        sr6 = (self.sigma / d[lj]) ** 6
        shift = (self.sigma / self.cutoff) ** 6
        energy += np.sum(4 * self.epsilon * (sr6 ** 2 - sr6) - 4 * self.epsilon * (shift ** 2 - shift))
        dEdr[lj] = 4 * self.epsilon * (-12 * sr6 ** 2 + 6 * sr6) / d[lj]
        pair_force = (dEdr / np.where(d > 0, d, 1))[:, None] * vec        # force on i is +, on j is -
        forces = np.zeros_like(pos)
        np.add.at(forces, i, pair_force)
        np.add.at(forces, j, -pair_force)
        self.results["energy"] = self.results["free_energy"] = float(energy)
        self.results["forces"] = forces
        if pbc.all():
            virial = np.einsum("p,pa,pb->ab", dEdr / np.where(d > 0, d, 1), vec, vec)
            sigma = virial / self.atoms.get_volume()
            self.results["stress"] = sigma.ravel()[[0, 4, 8, 5, 2, 1]]
        else:
            self.results["stress"] = np.zeros(6)


def intramolecular_signatures(atoms):
    mol = atoms.arrays["mol_id"]
    return sorted(tuple(np.round(np.sort(pdist(atoms.positions[mol == m])), 3)) for m in np.unique(mol))


class CifTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.crystal = cif.read_cif(write_cif_text(benzene_cif(), self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_symmetry_expansion_and_molecules(self):
        c = self.crystal
        self.assertEqual(len(c.atoms), 48)
        self.assertEqual(c.n_molecules, 4)
        self.assertEqual(c.composition, {"C6H6": 4})
        self.assertAlmostEqual(c.z_prime, 1.0)
        self.assertTrue(c.symmetry.centrosymmetric)
        com = c.centers_of_mass @ np.linalg.inv(c.atoms.cell.array)
        self.assertTrue(np.all((com >= -1e-9) & (com < 1)))
        ref = tuple(np.round(np.sort(pdist(molecule("C6H6").positions)), 3))
        self.assertTrue(all(sig == ref for sig in intramolecular_signatures(c.atoms)))

    def test_banner_before_data_block(self):
        crystal = cif.read_cif(write_cif_text(benzene_cif(banner=True), self.tmp.name))
        self.assertEqual(crystal.n_molecules, 4)

    def test_disorder_policies(self):
        extra = ["X1 C 0.95 0.95 0.95 0.3 2"]
        path = write_cif_text(benzene_cif(extra_sites=extra), self.tmp.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(len(cif.read_cif(path).atoms), 48)
        with self.assertRaisesRegex(ValueError, "disordered"):
            cif.read_cif(path, disorder="error")

    def test_wrapped_molecules_are_made_whole(self):
        atoms = self.crystal.atoms.copy()
        atoms.positions += [3.1, 4.7, 2.2]
        atoms.wrap()
        whole = cif.make_molecules_whole(atoms)
        self.assertEqual(intramolecular_signatures(whole), intramolecular_signatures(self.crystal.atoms))

    def test_network_is_rejected(self):
        from ase import Atoms
        chain = Atoms("C", positions=[[0, 0, 0]], cell=[1.4, 10, 10], pbc=True).repeat((2, 1, 1))
        with self.assertRaisesRegex(ValueError, "framework or polymer"):
            cif.make_molecules_whole(chain)


class SlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.crystal = cif.read_cif(write_cif_text(benzene_cif(), tmp))

    def test_miller_basis_properties(self):
        cell = cellpar_to_cell([6.1, 7.3, 8.2, 81, 95, 103])
        for hkl in [(1, 0, 0), (0, 1, 1), (1, -1, 2), (2, 1, 3), (3, 2, -1)]:
            P = slabs.miller_basis(cell, hkl)
            self.assertEqual(round(np.linalg.det(P)), 1)
            self.assertEqual(P[0] @ hkl, 0)
            self.assertEqual(P[1] @ hkl, 0)
            self.assertEqual(P[2] @ hkl, 1)
            uv = P[:2] @ cell
            area = np.linalg.norm(np.cross(*uv))
            self.assertAlmostEqual(np.linalg.det(cell) / area, slabs.d_spacing(cell, hkl), places=8)

    def test_slab_keeps_molecules_and_stoichiometry(self):
        ref = intramolecular_signatures(self.crystal.atoms)[0]
        for hkl in [(0, 0, 1), (1, 1, 0), (1, 0, -1), (2, 1, 1)]:
            for term in slabs.terminations(self.crystal, hkl)[:2]:
                s = slabs.build_slab(self.crystal, hkl, layers=3, termination=term, min_lengths=12)
                self.assertEqual(s.n_molecules, 4 * 3 * round(np.linalg.det(s.inplane)))
                self.assertEqual(s.composition, {"C6H6": s.n_molecules})
                self.assertTrue(all(sig == ref for sig in intramolecular_signatures(s.atoms)))
                self.assertGreaterEqual(np.linalg.norm(s.atoms.cell[0]), 12 - 1e-6)
                self.assertGreaterEqual(np.linalg.norm(s.atoms.cell[1]), 12 - 1e-6)
                self.assertTrue(np.allclose(s.atoms.cell[2, :2], 0) and np.allclose(s.atoms.cell[:2, 2], 0))
                # layers: COM heights increase with the layer index and stay inside the slab
                z = s.atoms.positions[:, 2]
                self.assertGreater(z.min(), 0)
                self.assertLess(z.max(), s.atoms.cell[2, 2])
                self.assertEqual(sorted(set(s.atoms.arrays["layer"])), [0, 1, 2])
                d = periodic_pairs(s.atoms.positions, s.atoms.cell.array, s.atoms.pbc, 1.5)[3]
                self.assertEqual(len(d) // 2, s.n_molecules * 12)        # only the C-C and C-H bonds

    def test_layers_hold_their_molecules(self):
        # Molecule 0 moved onto the origin: its COM height is 0 up to round-off, the case
        # (inversion centres, e.g. anthracene) where np.mod(-1e-17, 1) = 1 put it a layer too high.
        atoms = self.crystal.atoms.copy()
        atoms.positions -= self.crystal.centers_of_mass[0]
        shifted = cif.crystal_from_atoms(atoms)
        for crystal in (self.crystal, shifted):
            for hkl in [(0, 0, 1), (1, 0, 0), (0, 1, 1), (1, 1, -1)]:
                bulk = slabs.orient(crystal, hkl)
                self.assertTrue(np.all((bulk.com_heights >= 0) & (bulk.com_heights < 1)))
                for term in slabs.terminations(bulk):
                    s = slabs.build_slab(crystal, hkl, layers=3, termination=term)
                    a = s.atoms
                    m, mol, layer = a.get_masses(), a.arrays["mol_id"], a.arrays["layer"]
                    z = np.array([m[mol == k] @ a.positions[mol == k, 2] / m[mol == k].sum() for k in range(s.n_molecules)])
                    lay = np.array([layer[mol == k][0] for k in range(s.n_molecules)])
                    # COMs of layer k lie within one d_hkl band above the band of layer k - 1
                    bottom = z[lay == 0].min()
                    band = np.floor((z - bottom) / s.d_spacing + 1e-6)
                    self.assertTrue(np.array_equal(band, lay), (hkl, term.index, band, lay))

    def test_thickness_and_size_limits(self):
        s = slabs.build_slab(self.crystal, (0, 1, 0), min_thickness=20)
        self.assertGreaterEqual(s.thickness, 20)
        self.assertLess(s.thickness - s.d_spacing, 20)
        with self.assertRaisesRegex(ValueError, "max_atoms"):
            slabs.build_slab(self.crystal, (0, 1, 0), layers=4, max_atoms=100)

    def test_flip(self):
        s = slabs.build_slab(self.crystal, (1, 1, 0), layers=2)
        f = slabs.flip_slab(s)
        self.assertEqual(f.hkl, (-1, -1, 0))
        self.assertGreater(np.linalg.det(f.atoms.cell.array), 0)
        self.assertEqual(intramolecular_signatures(f.atoms), intramolecular_signatures(s.atoms))

    def test_bfdh_extinctions(self):
        faces = {f["hkl"]: f for f in morphology.bfdh(self.crystal, 1)}
        self.assertEqual(faces[(0, 1, 0)]["multiple"], 2)     # 2_1 along b: 0k0, k odd absent
        self.assertEqual(faces[(0, 0, 1)]["multiple"], 2)     # c glide: h0l, l odd absent
        self.assertEqual(faces[(1, 0, 0)]["multiple"], 1)
        self.assertAlmostEqual(sum(f["area_fraction"] for f in faces.values()), 1.0, places=6)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.crystal = cif.read_cif(write_cif_text(benzene_cif(), tmp))
        cls.slab = slabs.build_slab(cls.crystal, (0, 0, 1), layers=2)

    def test_self_match_is_strain_free(self):
        best = interfaces.match_lattices(self.slab, self.slab, max_area=200)[0]
        self.assertAlmostEqual(best.area, self.slab.area, places=6)
        self.assertLess(best.max_strain, 1e-9)

    def test_strained_match_and_rigid_interface(self):
        other = slabs.build_slab(self.crystal, (0, 0, 1), layers=2)
        other.atoms.set_cell(other.atoms.cell.array * [[1.02], [1.02], [1]], scale_atoms=False)
        match = interfaces.match_lattices(self.slab, other, max_area=200, max_length_tol=0.05)[0]
        self.assertAlmostEqual(match.max_strain, abs(1 / 1.02 - 1), places=6)
        it = interfaces.build_interface(self.slab, other, match, gap=2.5, vacuum=10)
        a, b = it.arrays["slab"] == 0, it.arrays["slab"] == 1
        self.assertAlmostEqual(it.positions[b, 2].min() - it.positions[a, 2].max(), 2.5, places=6)
        self.assertEqual(intramolecular_signatures(it[b]), intramolecular_signatures(other.atoms))
        v = interfaces.set_collision_velocities(it, 1000.0, temperature_K=300, seed=3)
        m = it.get_masses()
        from ase import units
        vz = [(m[s] @ v[s, 2]) / m[s].sum() * units.fs * 1e5 for s in (a, b)]
        self.assertAlmostEqual(vz[0], 500.0, places=6)
        self.assertAlmostEqual(vz[1], -500.0, places=6)


class PropertyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            cls.crystal = cif.read_cif(write_cif_text(benzene_cif(), tmp))
        cls.calc = ToyMolecularCalculator()

    def test_toy_stress_matches_finite_differences(self):
        atoms = self.crystal.atoms.copy()
        atoms.calc = self.calc
        from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress
        np.testing.assert_allclose(atoms.get_stress(), calculate_numerical_stress(atoms, eps=1e-5), atol=1e-6)
        np.testing.assert_allclose(atoms.get_forces(), calculate_numerical_forces(atoms, eps=1e-5), atol=1e-6)

    def test_gas_reference_carries_no_charge_or_spin_by_default(self):
        # UMA reads atoms.info charge/spin; the crystal has none, so the gas molecule must not either
        mol, _ = bulk.relax_molecule(self.crystal.molecule(0), self.calc, fmax=0.1, steps=20)
        self.assertNotIn("spin", mol.info)
        self.assertNotIn("charge", mol.info)
        mol, _ = bulk.relax_molecule(self.crystal.molecule(0), self.calc, fmax=0.1, steps=20, charge=0, spin=1)
        self.assertEqual((mol.info["charge"], mol.info["spin"]), (0, 1))

    def test_lattice_energy_bookkeeping(self):
        result = bulk.lattice_energy(self.crystal, crystal_energy=-4.0, gas_energies={"C6H6": 0.5})
        self.assertAlmostEqual(result["lattice_energy_eV"], -1.5)
        self.assertAlmostEqual(result["lattice_energy_kJ_mol"], -1.5 * 96.4853, places=2)

    def test_relax_and_surface_energy(self):
        relaxed, info = bulk.relax_bulk(self.crystal, self.calc, fmax=0.02, steps=300, symmetrize=False)
        self.assertEqual(relaxed.n_molecules, 4)
        slab = slabs.build_slab(relaxed, (0, 0, 1), layers=2)
        r = surfaces.surface_energy(slab, self.calc, info["energy_per_molecule"], relax=False, crystal=relaxed)
        self.assertGreater(r["gamma_J_m2"], 0)

    def test_molecule_removal_energy(self):
        relaxed, info = bulk.relax_bulk(self.crystal, self.calc, fmax=0.05, steps=200, symmetrize=False)
        gas = bulk.relax_molecule(relaxed.molecule(0), self.calc)[1]
        latt = bulk.lattice_energy(relaxed, info["energy"], {"C6H6": gas})["lattice_energy_eV"]
        slab = slabs.build_slab(relaxed, (0, 0, 1), layers=3, min_lengths=12)
        r = surfaces.molecule_removal_energy(slab, self.calc, {"C6H6": gas}, relax=True, fmax=0.1, steps=100,
                                             lattice_energy=latt)
        # the removed molecule is the outermost one and leaves one molecule's worth of atoms behind
        self.assertEqual(r["molecule"], surfaces.surface_molecule(slab))
        self.assertEqual(len(r["vacancy_atoms"]), len(slab.atoms) - 12)
        # pulling a molecule out of a bound surface costs energy, less than from the bulk (2 |E_latt|)
        self.assertGreater(r["vertical_eV"], 0)
        self.assertGreater(r["adiabatic_eV"], 0)
        self.assertGreater(r["vertical_relaxed_eV"], 0)
        self.assertLess(r["vertical_over_lattice"], 2.0)
        self.assertEqual(len(r["slab_atoms"]), len(slab.atoms))
        # vertical = intermolecular binding: E(vacancy) + E(frozen) - E(slab), checked directly
        a = slab.atoms.copy(); a.calc = self.calc
        gone = a.arrays["mol_id"] == r["molecule"]
        v = a[~gone]; v.calc = self.calc
        m = a[gone]; m.set_cell(None); m.pbc = False; del m.arrays["mol_id"]; del m.arrays["layer"]; m.calc = self.calc
        self.assertAlmostEqual(r["vertical_eV"], v.get_potential_energy() + m.get_potential_energy()
                               - a.get_potential_energy(), places=8)
        # the same check for the vertical energy on the relaxed slab that the function returns
        rs = r["slab_atoms"].copy(); rs.calc = self.calc
        rv = rs[~gone]; rv.calc = self.calc
        rm = rs[gone]; rm.set_cell(None); rm.pbc = False; del rm.arrays["mol_id"]; del rm.arrays["layer"]; rm.calc = self.calc
        self.assertAlmostEqual(r["vertical_relaxed_eV"], rv.get_potential_energy() + rm.get_potential_energy()
                               - rs.get_potential_energy(), places=6)
        with self.assertWarnsRegex(UserWarning, "periodic images"):
            surfaces.molecule_removal_energy(slabs.build_slab(relaxed, (0, 0, 1), layers=2), self.calc, gas,
                                             relax=False)

    def test_workflow_end_to_end(self):
        from MartiniFuncs.crystals.workflow import run_crystal_workflow
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cif_text(benzene_cif(), tmp)
            results = run_crystal_workflow(path, self.calc, Path(tmp) / "out", faces=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                                           n_terminations=1,
                                           layers=2, min_lengths=None, bulk_fmax=0.05, slab_fmax=0.1,
                                           compute_eos=False, symmetrize=False, log=lambda *a: None)
            saved = json.loads((Path(tmp) / "out" / "results.json").read_text())
            self.assertEqual(len(saved["surfaces"]), 3)
            self.assertAlmostEqual(sum(results["wulff"]["area_fractions"].values()), 1.0, places=6)
            self.assertTrue((Path(tmp) / "out" / "bulk_relaxed.cif").exists())
            self.assertNotIn("removal", saved)
            # two face families cannot close a Wulff shape: reported, not raised
            partial = run_crystal_workflow(path, self.calc, Path(tmp) / "out2", faces=[(0, 1, 0), (0, 0, 1)],
                                           n_terminations=1, layers=2, min_lengths=None, bulk_fmax=0.05,
                                           slab_fmax=0.1, compute_eos=False, symmetrize=False, log=lambda *a: None)
            self.assertIn("error", partial["wulff"])


if __name__ == "__main__":
    unittest.main()
