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
| collection | `CpdN4tCollection(reactive_radius=3.25)`: whole-CPD COM to dienophile reactive C=C COM, minimum image (`--reactive-radius`) |
| sites | CPD diene termini, four diene carbons and the CH2 bridge carbon; dienophile C=C, four ring carbons and the imide N |
| metrics | `max_forming_distance`, `distance_imbalance` (Angstrom); `axis_angle`, `plane_angle`, `approach_angle` (acute degrees); `ring_dihedral` (signed CH2 - termini midpoint - C=C midpoint - N dihedral, degrees: exo near 0, endo near 180) and `ring_dihedral_deviation` (degrees from the nearest of 0 and 180) |
| criteria | `CpdN4tCriteria(max_screen_distance, require_geometry, max_forming_distance, max_distance_imbalance, max_axis_angle, max_plane_angle, max_approach_angle, max_ring_dihedral_deviation)` |
| export labels | `pair_role` = `CPD` / `MAL` / `shell`; columns `cpd_molecule`, `dienophile_molecule`; index field `cpd_shift` |

The equations are in `chemistry/cpd_n4t/selection.py` (`measure_pair_geometry`).

## Collection and selection

A pair analysis has two stages, each with its own settings class:

1. **Collection** (`pairs.model.PairCollection`; the model's `collection_class`):
   which pairs an analysis measures and caches. Its `reactive_radius` is the
   screen between the model's screen centers; the model's subclass gives it a
   default, because what the radius means depends on the chemistry. It is the
   only pair setting recorded in a cache.
2. **Selection** (`pairs.model.PairCriteria`; the model's `criteria_class`):
   which collected pairs a report, export or `select_pairs` uses. Each field is
   an optional upper threshold on one metric, declared with
   `threshold(metric, ...)`. Every model inherits `max_screen_distance` (at most
   the collected radius) and `require_geometry`.

A collected pair is selected when every non-None threshold is met (`metric <=
threshold`; equality passes) and, if any geometric threshold is set or
`require_geometry=True`, its geometry was measured. With no criteria every
collected pair is used.

The CLI builds one `--max-...` flag per threshold field, with help text from the
field metadata, plus `--geometry-required`. A flag that belongs to a different
model is refused.

## Adding a model

Write a `PairModel` subclass and register an instance:

```python
from dataclasses import dataclass
from MartiniFuncs.pairs.model import PairCollection, PairCriteria, PairModel, register_pair_model, threshold

@dataclass(frozen=True)
class MyCollection(PairCollection):
    reactive_radius: float = 4.0                 # Angstrom between screen centers

@dataclass(frozen=True)
class MyCriteria(PairCriteria):
    max_site_distance: float | None = threshold("site_distance", help="site-site distance")

class MyModel(PairModel):
    name = "my_pair"
    roles = ("donor", "acceptor")          # column/field names: donor_molecule, donor_shift
    role_labels = ("DON", "ACC")           # per-atom pair_role in exports
    collection_class = MyCollection
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

The collection's default radius belongs to the model because its meaning
depends on the model's screen centers (`screen_atoms`): a radius chosen for CPD
COM to C=C COM says nothing about another reaction. Without a subclass the
model keeps the base `PairCollection`, which has no default, so every analysis
must pass `pair_collection=PairCollection(reactive_radius=...)`.

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

## Building CPD + N4T pairs at a chosen geometry

`MartiniFuncs.chemistry.cpd_n4t.build` makes finite pair structures, for example
as starting points for reaction or barrier calculations. The N4T template is
fixed; the CPD is placed as a rigid body:

```python
from MartiniFuncs.chemistry.cpd_n4t.build import build_pair, rdkit_templates

cpd, n4t = rdkit_templates()        # or templates_from_frame(atoms), or your own relaxed Atoms
atoms, geometry = build_pair(cpd, n4t, distance=3.0, ring_dihedral=180, plane_angle=15)
```

- `distance` is the CPD termini midpoint to C=C midpoint distance by default;
  `distance_kind="max_forming"` or `"screen"` targets the longer forming C...C
  distance or the analysis screen distance (CPD COM to C=C COM) instead.
- `ring_dihedral` is the signed CH2 - termini midpoint - C=C midpoint - N dihedral
  (exo near 0, endo near 180), met exactly.
- `plane_angle` tilts the CPD ring about its termini axis; positive opens the CH2
  side away from the N4T. The measured `plane_angle` and `approach_angle` match it
  to within the templates' non-planarity.
- `face` picks the N4T face; `min_contact` rejects overlapping structures.

The structure is measured with the same equations as the analysis; the returned
`geometry` and `atoms.info` (`measured_*`, `min_intermolecular_distance`) report
what was built. Rows are CPD then N4T, with `global_index` and `pair_role`.
