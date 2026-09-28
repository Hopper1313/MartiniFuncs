import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from ase import Atoms
from MartiniFuncs.reaction_detection import find_possible_products, scan_possible_products, PossibleReactionError
from MartiniFuncs.trajectory_analysis import analyze_run


def frame(index, merged):
    x = np.array([0., 1.4, 2.8, 4.2, 5.6])
    if not merged:
        x[2:] += 5
    atoms = Atoms('C5', positions=np.column_stack([x, np.zeros((5, 2))]), cell=[30]*3, pbc=True)
    atoms.new_array('global_index', np.array([10, 20, 40, 70, 90]))
    return SimpleNamespace(index=index, timestep=index*100, atoms=atoms)


class ReactionDetectionTests(unittest.TestCase):
    def test_size_ids_periodicity_and_reversibility(self):
        self.assertEqual(find_possible_products(frame(0, False).atoms, reactant_atom_counts=(2, 3)), [])
        records = scan_possible_products([frame(0, False), frame(1, True), frame(2, True), frame(3, False)],
                                         reactant_atom_counts=(2, 3))
        self.assertEqual([r['frame'] for r in records], [1, 2])
        self.assertEqual(records[0]['atom_ids'], [10, 20, 40, 70, 90])
        atoms = frame(0, True).atoms
        atoms.positions[:, 0] += 27
        atoms.wrap()
        atoms = atoms[[4, 2, 0, 3, 1]]
        hit = find_possible_products(atoms, reactant_atom_counts=(2, 3))[0]
        self.assertEqual(hit['atom_ids'], records[0]['atom_ids'])
        np.testing.assert_array_equal(atoms.arrays['global_index'][hit['local_indices']], hit['atom_ids'])

    def test_analysis_catches_product_before_stride_and_in_first_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'trajectory.lammpstrj').touch()
            for events, expected in (([frame(0, False), frame(1, True), frame(2, False)], 1),
                                     ([frame(0, True)], 0)):
                with patch('MartiniFuncs.trajectory_analysis.iter_lammps_frames', return_value=iter(events)):
                    with self.assertRaises(PossibleReactionError) as caught:
                        analyze_run(directory, units='metal', timestep_ps=.001, frame_stride=2,
                                    reaction_atom_counts=(2, 3))
                self.assertEqual(caught.exception.candidates[0]['frame'], expected)

    def test_current_cpd_mal_count(self):
        atoms = Atoms('C34', positions=[[i*1.4, 0, 0] for i in range(34)])
        self.assertEqual(find_possible_products(atoms, reactant_atom_counts=(11, 23))[0]['atom_count'], 34)
        self.assertEqual(find_possible_products(atoms, reactant_atom_counts=(11, 22)), [])


if __name__ == '__main__':
    unittest.main()
