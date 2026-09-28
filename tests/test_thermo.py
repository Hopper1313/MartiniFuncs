from pathlib import Path
import tempfile
import unittest
import pandas as pd
from MartiniFuncs.thermo import read_lammps_thermo, attach_thermo, discover_thermo_file


class ThermoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'run.out'
        self.path.write_text('Driver startup\nStep Time Temp Press PotEng\n2000000 1000 400 25000 -1D3\nWARNING: something\n2002000 1001 401 24000 -1001\nLoop time of 1\n123 footer\n')

    def test_discovery_and_sparse_alignment(self):
        self.assertEqual(discover_thermo_file(self.root), self.path)
        thermo = read_lammps_thermo(self.path, units='metal')
        frames = pd.DataFrame(dict(frame=[0, 1, 2], timestep=[2000000, 2001000, 2002000], time_ps=[1000., 1000.5, 1001.]))
        joined, coverage = attach_thermo(frames, thermo)
        self.assertEqual(coverage['matched_frames'], 2)
        self.assertTrue(pd.isna(joined.loc[1, 'temperature_K']))
        self.assertEqual(joined.loc[0, 'potential_energy_eV'], -1000.)
        self.assertEqual(joined.loc[2, 'pressure_bar'], 24000.)

    def test_ambiguity_and_repeated_blocks(self):
        (self.root/'other.out').write_text(self.path.read_text())
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            discover_thermo_file(self.root)
        with self.path.open('a') as stream:
            stream.write('Step Time Temp\n2000000 1000 402\n')
        table = read_lammps_thermo(self.path, units='metal')
        frames = pd.DataFrame(dict(timestep=[2000000], time_ps=[1000.]))
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            attach_thermo(frames, table)
        self.assertEqual(attach_thermo(frames, table, block=1)[0].loc[0, 'temperature_K'], 402.)

    def test_malformed_rows_and_time_mismatch(self):
        table = read_lammps_thermo(self.path, units='metal')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            attach_thermo(pd.DataFrame(dict(timestep=[2000000], time_ps=[0.])), table)
        self.path.write_text('Step Temp Press\n0 300\n')
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            read_lammps_thermo(self.path, units='metal')
        with self.assertRaisesRegex(ValueError, 'metal'):
            read_lammps_thermo(self.path, units='real')

    def test_older_frames_recover_source_steps_preserving_order(self):
        dump = self.root/'source.lammpstrj'
        dump.write_text(''.join(f'ITEM: TIMESTEP\n{s}\nITEM: NUMBER OF ATOMS\n0\n'
                                for s in [2000000, 2001000, 2002000]))
        frames = pd.DataFrame(dict(frame=[2, 0], time_ps=[1001., 1000.]))
        thermo = read_lammps_thermo(self.path, units='metal')
        joined, coverage = attach_thermo(frames, thermo, dump_file=dump)
        self.assertEqual(joined.timestep.tolist(), [2002000, 2000000])
        self.assertEqual(joined.temperature_K.tolist(), [401., 400.])
        self.assertTrue(coverage['timesteps_recovered_from_dump'])
        self.assertNotIn('timestep', frames)
        with self.assertRaisesRegex(ValueError, 'Supply dump_file'):
            attach_thermo(frames, thermo)

    def test_integral_types_accepted_without_rounding(self):
        thermo = read_lammps_thermo(self.path, units='metal')
        for steps in ([2000000., 2002000.], ['2000000', '2002000']):
            joined, _ = attach_thermo(pd.DataFrame(dict(timestep=steps)), thermo)
            self.assertEqual(joined.temperature_K.tolist(), [400., 401.])
        for bad in (2000000.5, float('nan'), float(2**53), True):
            with self.assertRaises(ValueError):
                attach_thermo(pd.DataFrame(dict(timestep=[bad])), thermo)


if __name__ == '__main__':
    unittest.main()
