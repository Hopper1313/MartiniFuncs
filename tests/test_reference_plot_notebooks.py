"""Both plotting notebooks can draw reference MSD from cache without source files."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from MartiniFuncs.postprocessing import cache_analysis


class ReferencePlotNotebookTests(unittest.TestCase):
    def test_reference_cells_refresh_module_missing_new_function(self):
        import MartiniFuncs.md_statistics as statistics
        root = Path(__file__).resolve().parents[1]
        for name in ('examples/02_plot_md_statistics.ipynb',
                     'examples/04_plot_md_statistics_from_directory.ipynb',
                     'examples/workflows/04_molecular_statistics.ipynb',
                     'examples/workflows/08_diffusion_and_equilibration.ipynb'):
            cells = json.loads((root/name).read_text(encoding='utf-8'))['cells']
            source = ''.join(next(c for c in cells if c.get('id') == 'reference-msd-plot')['source'])
            original = statistics.get_molecular_reference_msd
            del statistics.get_molecular_reference_msd
            try:
                with redirect_stdout(StringIO()):
                    exec(source, {'result': {'com_angstrom': None}})
                self.assertTrue(callable(statistics.get_molecular_reference_msd))
            finally:
                if not hasattr(statistics, 'get_molecular_reference_msd'):
                    statistics.get_molecular_reference_msd = original

    def test_04_reference_plots_follow_lag_plots(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ('examples/04_plot_md_statistics_from_directory.ipynb',
                         'examples/workflows/04_molecular_statistics.ipynb'):
            cells = json.loads((root/relative).read_text(encoding='utf-8'))['cells']
            ids = [cell.get('id') for cell in cells]
            after = ids.index('directory-md-08')+1
            self.assertEqual(ids[after:after+2], ['reference-msd-guide', 'reference-msd-plot'])

    def test_both_notebooks_use_cache_without_source_dump(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory/'source'
            source.mkdir()
            dump = source/'trajectory.lammpstrj'
            shutil.copyfile(root/'tests/lammpsio_tests/multiple_frames.lammpstrj', dump)
            cache = directory/'cache'
            cache_analysis(source, cache, units='metal', timestep_ps=.0005, progress=None)
            dump.unlink()  # Cache-only plotting must not attempt to open this.
            for filename in ('02_plot_md_statistics.ipynb', '04_plot_md_statistics_from_directory.ipynb'):
                with self.subTest(notebook=filename):
                    notebook = json.loads((root/'examples'/filename).read_text(encoding='utf-8'))
                    namespace = {}
                    with patch('matplotlib.pyplot.show'), patch('IPython.display.display'), redirect_stdout(StringIO()):
                        for cell in notebook['cells']:
                            identity = cell.get('id')
                            if identity in {'reference-cache-config', 'directory-md-01', 'reference-cache-load',
                                            'directory-md-03', 'reference-msd-plot'}:
                                exec(''.join(cell['source']), namespace)
                                if identity in {'reference-cache-config', 'directory-md-01'}:
                                    namespace['CACHE_DIR'] = cache
                    self.assertEqual(len(namespace['reference_fig'].axes), 4)
                    for name, measured in namespace['reference_msd'].items():
                        self.assertEqual(measured['squared_displacement_angstrom2'].shape, (9, 40))
                        np.testing.assert_array_equal(measured['msd_angstrom2'][0], 0)
                        self.assertEqual(len(namespace['reference_molecule_ids'][name]), 40)
                    plt.close('all')
                    # Windows cannot remove files while notebook memmaps remain open.
                    for value in namespace['result'].values():
                        if isinstance(value, np.memmap):
                            value._mmap.close()
                    namespace.clear()
