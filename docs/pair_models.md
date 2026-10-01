# Pair models

The pair analysis has two parts:

- **generic machinery** (`MartiniFuncs.pairs`): the periodic center-distance
  screen, candidate caching, selection after the analysis, lifetimes and
  structure export;
- **a pair model**: the chemistry, meaning which molecules can pair, where
  their screen points are, their reactive sites, and the measured metrics.

The model is chosen per analysis. In the CLI it is `--pair-model`; in Python it
is `analyze_run(pair_model=...)`. The default is `cpd_n4t`. The model name is
recorded in the cache (`pair_analysis.model`, plus the candidate metric names),
so `report` and `extract` always select with the model that made the cache.
Caches written before pair models existed are cpd_n4t caches and load as such.

## The cpd_n4t model

`MartiniFuncs.chemistry.cpd_n4t`: cyclopentadiene (C5H6, 11 atoms) is the
first member and the N4T maleimide dienophile (C11H9NO2, 23 atoms) the second.
34-atom adducts take no part. Dimerization is not modelled, because it is not
observed experimentally.

| | |
|---|---|
| screen | whole-CPD COM to dienophile reactive C=C COM, minimum image (`--reactive-radius`, default 3.25 Angstrom) |
| sites | CPD diene termini and four diene carbons; dienophile C=C and four ring carbons |
| metrics | `max_forming_distance`, `distance_imbalance` (Angstrom); `axis_angle`, `plane_angle`, `approach_angle` (acute degrees) |
| criteria | `CpdN4tCriteria(max_forming_distance, max_distance_imbalance, max_axis_angle, max_plane_angle, max_approach_angle)` |
| export labels | `pair_role` = `CPD` / `MAL` / `shell`; columns `cpd_molecule`, `dienophile_molecule`; index field `cpd_shift` |

The equations are in `chemistry/cpd_n4t/selection.py` (`measure_pair_geometry`).

## Criteria

A model's criteria class subclasses `pairs.model.PairCriteria`. Each field is
an optional upper threshold on one metric, declared with `threshold(metric, ...)`.

A candidate passes when both of these hold:

- its geometry was measured;
- every non-None threshold is met (`metric <= threshold`; equality passes).

The CLI builds one `--max-...` flag per field, with help text from the field
metadata. A flag that belongs to a different model is refused.

## Adding a model

Write a `PairModel` subclass and register an instance:

```python
from dataclasses import dataclass
from MartiniFuncs.pairs.model import PairCriteria, PairModel, register_pair_model, threshold

@dataclass(frozen=True)
class MyCriteria(PairCriteria):
    max_site_distance: float | None = threshold("site_distance", help="site-site distance")

class MyModel(PairModel):
    name = "my_pair"
    roles = ("donor", "acceptor")          # column/field names: donor_molecule, donor_shift
    role_labels = ("DON", "ACC")           # per-atom pair_role in exports
    criteria_class = MyCriteria
    metric_names = ("site_distance",)
    metric_ranges = {"site_distance": (0, 8)}   # report histogram ranges
    species = frozenset({"C2H6O", "C3H6O"})      # or None to skip the check

    def classify(self, components, elements):     # 0, 1 or None per molecule
        ...
    def screen_atoms(self, graph, elements, component, role):   # None = whole molecule
        ...
    def sites(self, graph, elements, component, role):           # raise ValueError if absent
        ...
    def measure(self, first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows):
        ...                                       # return a dataclass geometry
    def metrics(self, geometry):                  # {metric name: value}
        ...

register_pair_model(MyModel())
```

**What the engine guarantees:**
- pairs are `[first IDs, second IDs]`;
- the periodic image shift applies to the first member;
- `measure` receives whole coordinates in Angstrom, already jointly placed.

**Registration:**
- Register the model in an importable module, not a notebook cell or `__main__`.
  Spawned workers rebuild the model from its class.
- To make the model a built-in, add it to `_BUILTIN` in `pairs/model.py`, and
  `--pair-model` will list it.

`tests/test_pair_models.py` runs a toy model through the whole pipeline: both
backends, workers, selection, and the CLI.
