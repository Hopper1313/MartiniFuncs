"""Execute notebook thermo cells without trajectories or precomputed reports."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


NOTEBOOKS = ('examples/02_plot_md_statistics.ipynb',
             'examples/04_plot_md_statistics_from_directory.ipynb',
             'examples/workflows/04_molecular_statistics.ipynb')


class NotebookThermoTests(unittest.TestCase):
    def execute(self, notebook, directory, **options):
        cells = json.loads((Path(__file__).resolve().parents[1]/notebook).read_text(encoding='utf-8'))['cells']
        namespace = dict(THERMO_DIRECTORY=directory, SIMULATION_DIR=directory, RUN_DIR=directory,
                         LOAD_THERMO=True, THERMO_FILE=None, THERMO_BLOCK=None, UNITS='metal', **{})
        namespace.update(options)
        with patch('IPython.display.display'), patch('matplotlib.pyplot.show'), redirect_stdout(StringIO()):
            for c in cells:
                if c.get('id') in {'thermo-read', 'thermo-plot'}:
                    exec(compile(''.join(c['source']), notebook+':'+c['id'], 'exec'), namespace)
        self.addCleanup(plt.close, 'all')
        return namespace

    def test_read_plot_without_dump_or_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'run.out').write_text('Driver warning\nStep Time Temp Press\n0 0 400 25000\n10 .005 402 24900\nLoop time\n')
            for notebook in NOTEBOOKS:
                with self.subTest(notebook=notebook):
                    result = self.execute(notebook, directory)
                    self.assertEqual(len(result['thermo_selected']), 2)
                    self.assertEqual(result['thermo_x'], 'thermo_time_ps')
                    np.testing.assert_array_equal(result['thermo_axes'][0, 0].lines[0].get_ydata(), [400, 402])
                    np.testing.assert_array_equal(result['thermo_axes'][1, 0].lines[0].get_ydata(), [25000, 24900])

    def test_missing_disabled_multiple_blocks_and_no_time(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for notebook in NOTEBOOKS:
                self.assertIsNone(self.execute(notebook, directory)['thermo'])
            (directory/'run.out').write_text('Step Temp\n0 400\n10 401\nLoop time\nStep Temp\n0 500\n10 501\nLoop time\n')
            for notebook in NOTEBOOKS:
                with self.subTest(notebook=notebook):
                    self.assertIsNone(self.execute(notebook, directory, LOAD_THERMO=False)['thermo'])
                    self.assertIsNone(self.execute(notebook, directory)['thermo_selected'])
                    result = self.execute(notebook, directory, THERMO_BLOCK=1)
                    self.assertEqual(result['thermo_x'], 'timestep')
                    self.assertEqual(result['thermo_selected'].temperature_K.tolist(), [500, 501])
                    with self.assertRaisesRegex(ValueError, 'absent'):
                        self.execute(notebook, directory, THERMO_BLOCK=7)
