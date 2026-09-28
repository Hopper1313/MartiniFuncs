"""Indexed parsing and ordered temporal/reaction behavior across chunk boundaries."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MartiniFuncs.frame_parallel import frame_offsets
from MartiniFuncs.lammpsio import iter_lammps_frames
from MartiniFuncs.reaction_detection import PossibleReactionError
from MartiniFuncs.trajectory_analysis import analyze_run


def dump_frame(step, symbols, xs, half=10):
    """Small text dump with nonconsecutive atom IDs and source force vectors."""
    return (f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(xs)}\n'
            'ITEM: BOX BOUNDS pp pp pp\n' + f'{-half} {half}\n'*3 +
            'ITEM: ATOMS id element x y z fx fy fz\n' +
            ''.join(f'{10+3*i} {symbol} {x} 0 0 0.000000000123 -2 3\n'
                    for i, (symbol, x) in enumerate(zip(symbols, xs))))


class ParallelFrameTests(unittest.TestCase):
    def test_seek_crlf_and_tor_across_variable_cell_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'test.lammpstrj')
            text = ''.join(dump_frame(i*100, ['He'], [x], 10+i*.1)
                           for i, x in enumerate([9., -9., -7., -5., -3.]))
            path.write_bytes(text.replace('\n', '\r\n').rstrip().encode())
            offsets = frame_offsets(path)
            self.assertEqual(len(offsets), 6)
            frames = list(iter_lammps_frames(path, units='metal',
                          byte_range=(int(offsets[2]), int(offsets[4])), frame_offset=2))
            self.assertEqual([f.index for f in frames], [2, 3])
            self.assertEqual([f.timestep for f in frames], [200, 300])
            options = dict(units='metal', timestep_ps=.0005, compute_pairs=False)
            serial = analyze_run(directory, **options)
            parallel = analyze_run(directory, pair_workers=2, pair_batch_size=1, **options)
            np.testing.assert_array_equal(serial['com_angstrom'], parallel['com_angstrom'])
            np.testing.assert_allclose(parallel['com_angstrom'][:, 0, 0], [9, 11.2, 13.2, 15.2, 17.2])
            self.assertEqual(serial['frames'], parallel['frames'])

    def test_worker_reaction_retains_evidence_and_precedes_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'test.lammpstrj').write_text(
                dump_frame(0, ['C']*5, [0, 1.4, 7.8, 9.2, 10.6], 20) +
                dump_frame(100, ['C']*5, [0, 1.4, 2.8, 4.2, 5.6], 20) +
                dump_frame(200, ['C']*5, [0, 1.4, 7.8, 9.2, 10.6], 20))
            with self.assertRaises(PossibleReactionError) as caught:
                analyze_run(directory, units='metal', timestep_ps=.0005,
                            reaction_atom_counts=(2, 3), frame_stride=2,
                            pair_workers=2, pair_batch_size=1)
            self.assertEqual(caught.exception.candidates[0]['frame'], 1)
            self.assertEqual(caught.exception.candidates[0]['atom_count'], 5)


if __name__ == '__main__':
    unittest.main()
