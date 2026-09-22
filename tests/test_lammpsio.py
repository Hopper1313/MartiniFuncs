import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from ase.io import read

from MartiniFuncs.lammpsio import (
    iter_lammps_frames, read_lammps_frame, my_readlammps,
    split_lammpstrj, make_ase_obj_and_fix_atomtypes,
    read_lammps_input_settings, read_input_datafile,
)
from MartiniFuncs import geom_utils as geom


FIXTURES = Path(__file__).parent / "lammpsio_tests"


def dump(timestep=0, rows=None, header=None, box=None):
    rows = rows or ["9 1 H 1 2 3 0.1 0.2 0.3 1 2 3 7 1 0 -1",
                    "2 1 H 4 5 6 0.4 0.5 0.6 4 5 6 8 0 0 0"]
    header = header or "id type element x y z vx vy vz fx fy fz mol ix iy iz"
    box = box or "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10"
    return (f"ITEM: TIMESTEP\n{timestep}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n"
            f"{box}\nITEM: ATOMS {header}\n" + "\n".join(rows) + "\n")


class LammpsIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "dump.lammpstrj"

    def load(self, text, units="metal", **kwargs):
        self.path.write_text(text)
        return read_lammps_frame(self.path, units=units, **kwargs)

    def test_all_nine_frames_and_split_roundtrip(self):
        path = FIXTURES / "multiple_frames.lammpstrj"
        frames = list(iter_lammps_frames(path, units="metal"))
        self.assertEqual([f.timestep for f in frames], list(range(0, 9000, 1000)))
        out = Path(self.temp.name) / "frames"
        split_lammpstrj(path, out)
        self.assertEqual(len(list(out.glob("*.lammpstrj"))), 9)
        for i, original in enumerate(frames):
            split = read_lammps_frame(out / f"frame_{i}.lammpstrj", units="metal")
            self.assertEqual(original.timestep, split.timestep)
            assert_array_equal(original.columns["id"], split.columns["id"])
            assert_allclose(original.atoms.positions, split.atoms.positions)
            assert_allclose(original.atoms.arrays["lammps_forces"], split.atoms.arrays["lammps_forces"])

    def test_single_frame_split_and_no_final_newline(self):
        self.path.write_text(dump().rstrip())
        out = Path(self.temp.name) / "frames"
        split_lammpstrj(self.path, out)
        self.assertEqual(len(list(out.glob("*.lammpstrj"))), 1)
        self.assertEqual(my_readlammps(out / "frame_0.lammpstrj")[0], 0)

    def test_frame_index_and_legacy_alignment(self):
        self.path.write_text(dump(10) + dump(20).replace("1 2 3 7", "8 9 10 7"))
        first = read_lammps_frame(self.path, units="metal", index=0)
        last = read_lammps_frame(self.path, units="metal", index=-1)
        self.assertEqual(first.timestep, 10)
        self.assertEqual(last.timestep, 20)
        self.assertEqual(my_readlammps(self.path)[0], 20)
        self.assertEqual(my_readlammps(self.path, index=0)[0], 10)
        assert_allclose(last.atoms.arrays["lammps_forces"][1], [8, 9, 10])
        with self.assertRaisesRegex(ValueError, "different timesteps"):
            geom._source_vectors(first.atoms, my_readlammps(self.path), [1, 8])
        for index in (2, -3):
            with self.assertRaises(IndexError):
                read_lammps_frame(self.path, units="metal", index=index)
        with self.assertRaises(TypeError):
            read_lammps_frame(self.path, units="metal", index=":")

    def test_forward_stream_does_not_parse_later_bad_frame(self):
        self.path.write_text(dump() + dump(1).replace("4 5 6 8", "nan 5 6 8"))
        iterator = iter_lammps_frames(self.path, units="metal")
        self.addCleanup(iterator.close)
        self.assertEqual(next(iterator).timestep, 0)
        with self.assertRaisesRegex(ValueError, "frame 1"):
            next(iterator)

    def test_sparse_ids_and_integer_metadata(self):
        frame = self.load(dump(), id_offset=0)
        assert_array_equal(frame.atoms.arrays["global_index"], [2, 9])
        assert_array_equal(frame.atoms.arrays["molecule_id"], [8, 7])
        assert_array_equal(frame.atoms.arrays["image_flags"], [[0, 0, 0], [1, 0, -1]])
        assert_allclose(frame.atoms.arrays["lammps_forces"], [[4, 5, 6], [1, 2, 3]])
        self.assertEqual(frame.columns["mol"].dtype.kind, "i")
        self.assertEqual(frame.as_legacy_tuple()[-1][9]["fx"], 1)

    def test_real_units_normalized_and_raw_kept(self):
        frame = self.load(dump(), units="real")
        assert_allclose(frame.atoms.arrays["lammps_forces"][1], np.array([1, 2, 3]) / 23.060548, rtol=1e-7)
        self.assertEqual(frame.columns["fx"][1], 1)
        metal = self.load(dump(), units="metal")
        assert_allclose(frame.atoms.get_velocities(), 1000 * metal.atoms.get_velocities())

    def test_scaled_triclinic_mixed_pbc_origin(self):
        box = "ITEM: BOX BOUNDS xy xz yz pp ff pp\n-2 11 2\n3 12 -1\n5 13 1"
        f = self.load(dump(rows=["3 C 0.5 0.5 0.5"], header="id element xs ys zs", box=box))
        assert_allclose(f.atoms.cell, [[10, 0, 0], [2, 8, 0], [-1, 1, 8]])
        assert_array_equal(f.atoms.pbc, [True, False, True])
        assert_allclose(f.atoms.get_celldisp(), [-1, 3, 5])
        assert_allclose(f.atoms.positions, [[4.5, 7.5, 9]])

    def test_scaled_origin_forces_and_general_triclinic(self):
        box = "ITEM: BOX BOUNDS abc origin pp pp pp\n10 0 0 -1\n2 8 0 3\n-1 1 8 5"
        f = self.load(dump(rows=["3 C 0.5 0.5 0.5 1 2 3"],
                           header="id element xs ys zs fx fy fz", box=box), wrap=True)
        assert_allclose(f.atoms.positions, [[4.5, 7.5, 9]])
        assert_allclose(f.atoms.arrays["lammps_forces"], [[1, 2, 3]])

    def test_input_settings_are_literal_not_comment_inference(self):
        settings = read_lammps_input_settings(FIXTURES / "infile.in")
        self.assertEqual(settings["units"], "metal")
        self.assertEqual(settings["boundary"], ("p", "p", "p"))
        self.assertEqual(settings["dump_elements"]["1"], {1: "C", 2: "H", 3: "N", 4: "O"})
        self.assertNotIn("task", settings)
        self.assertNotIn("model", settings)
        self.path.write_text("echo screen\nvariable stride index 10\nclear\nunits metal\ntimestep 0.0005\n")
        initialized = read_lammps_input_settings(self.path)
        self.assertEqual(initialized["units"], "metal")
        self.assertEqual(initialized["timestep"], 0.0005)
        for text in ("units ${u}\n", "include other.in\n", "units real\nunits metal\n",
                     "units real\nclear\nunits metal\n", "read_data old.data\nclear\nunits metal\n"):
            self.path.write_text(text)
            with self.assertRaises(ValueError):
                read_lammps_input_settings(self.path)

    def test_sparse_atomic_data_file_without_blank_separator(self):
        self.path.write_text("test\n\n2 atoms\n1 atom types\n0 10 xlo xhi\n0 10 ylo yhi\n"
                             "0 10 zlo zhi\nMasses\n\n1 12.011 # C\n\nAtoms # atomic\n\n"
                             "9 1 1 2 3\n2 1 4 5 6\n")
        natoms, _, _, _, records, by_id = read_input_datafile(self.path)
        self.assertEqual(natoms, 2)
        self.assertEqual([r[0] for r in records], [2, 9])
        self.assertEqual(by_id[9]["symbol"], "C")

    def test_unwrapped_coordinates_and_source_forces_survive_wrap(self):
        text = dump(rows=["4 C 12 3 4 1 2 3"], header="id element xu yu zu fx fy fz")
        f = self.load(text, wrap=True)
        assert_allclose(f.atoms.positions, [[2, 3, 4]])
        assert_allclose(f.atoms.arrays["lammps_forces"], [[1, 2, 3]])
        self.assertIsNone(f.atoms.calc)

    def test_masses_and_type_mapping(self):
        f = self.load(dump(rows=["7 15 13.0 1 2 3"], header="id type mass x y z"), type_symbols={15: "C"})
        self.assertEqual(f.atoms.get_chemical_symbols(), ["C"])
        assert_allclose(f.atoms.get_masses(), [13])
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.load(dump(), type_symbols={1: "C"})

    def test_type_only_dump_uses_data_file_species_and_masses(self):
        datafile = Path(self.temp.name) / "before.data"
        datafile.write_text("test\n\n2 atoms\n1 atom types\n0 10 xlo xhi\n0 10 ylo yhi\n"
                            "0 10 zlo zhi\nMasses\n\n1 12.01115 # C\n\nAtoms # atomic\n\n"
                            "9 1 1 2 3\n2 1 4 5 6\n")
        frame = self.load(dump(rows=["9 1 1 2 3 1 2 3 4 5 6", "2 1 4 5 6 3 2 1 6 5 4"],
                               header="id type x y z vx vy vz fx fy fz"), input_datafile=datafile)
        self.assertEqual(frame.atoms.get_chemical_symbols(), ["C", "C"])
        from ase.calculators.lammps import convert
        expected_mass = convert(12.01115, "mass", "metal", "ASE")
        assert_allclose(frame.atoms.get_masses(), [expected_mass, expected_mass], rtol=0, atol=1e-12)
        assert_allclose(frame.atoms.arrays["lammps_forces"], [[6, 5, 4], [4, 5, 6]])

    def test_bad_rows_headers_and_units_fail_contextually(self):
        cases = [dump().replace("9 1 H", "2 1 H"),
                 dump().replace("9 1 H", "9 H"),
                 dump().replace("0.1", "nan"),
                 dump().replace(" fx fy fz ", " fx fy force_z "),
                 dump().replace(" mol ix iy iz", " id ix iy iz"),
                 dump().rsplit("\n", 2)[0],
                 dump().replace("ITEM: NUMBER", "ITEM: UNITS\nreal\nITEM: NUMBER")]
        for text in cases:
            with self.subTest(text=text[-60:]), self.assertRaisesRegex(ValueError, "frame 0"):
                self.load(text)
        with self.assertRaisesRegex(ValueError, "Unsupported physical"):
            self.load(dump(), units="lj")

    def test_fixture_wrapper_and_normalized_export(self):
        path = FIXTURES / "frame_0.lammpstrj"
        frame = read_lammps_frame(path, units="metal", input_datafile=FIXTURES / "frame_0.data")
        wrapped = make_ase_obj_and_fix_atomtypes(path, FIXTURES / "frame_0.data", units="metal")
        assert_allclose(frame.atoms.arrays["lammps_forces"], wrapped.arrays["lammps_forces"])
        pairs, shifts = geom.get_all_prereactive_pairs(frame.atoms, scale=1.3)
        self.assertEqual(len(pairs), 2)
        clusters = geom.get_pair_shell_clusters(frame.atoms, pairs, shifts, scale=1.3)
        records = geom.get_cluster_positions_and_forces(frame.atoms, frame, clusters)
        stream = io.StringIO()
        geom.pair_data_to_extxyz(records, stream)
        stream.seek(0)
        images = read(stream, format="extxyz", index=":")
        for i, image in enumerate(images):
            assert_allclose(image.arrays["lammps_forces"], records[i]["forces"], atol=5e-9)
            self.assertEqual(image.info["force_units"], "eV/Angstrom")
            self.assertEqual(image.info["timestep"], 0)
            assert_allclose(image.info["source_cell"], frame.atoms.cell)


if __name__ == "__main__":
    unittest.main()
