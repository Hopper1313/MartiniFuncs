"""Pair models: the chemistry-specific part of a pair analysis behind one interface.

A pair model says which molecules can form a pair (two roles, e.g. diene and
dienophile), where each molecule's screen center is, which reactive sites it
has, and how a placed pair is measured. Everything else is generic and lives
in MartiniFuncs.pairs: the periodic center-distance screen, candidate caching,
selection after the fact, lifetimes and structure export.

A pair analysis has two stages, each with its own settings class:

1. Collection (PairCollection; cpd_n4t: CpdN4tCollection) decides what an
   analysis measures and caches: every pair whose screen centers are closer
   than reactive_radius, with all of the model's metrics. It is the only pair
   setting recorded in a cache.
2. Selection (PairCriteria; cpd_n4t: CpdN4tCriteria) picks pairs from what was
   collected: optional upper thresholds on the screen distance and the model's
   metrics. It never changes a cache; reports and exports apply it.

Models are registered by name. The name is recorded with an analysis, so
report/extract use the model that produced a cache. Built-in models are
imported on first use; register_pair_model adds others.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from importlib import import_module
from types import MappingProxyType

import numpy as np

DEFAULT_PAIR_MODEL = "cpd_n4t"

_REGISTRY: dict = {}
# Built-in models, imported lazily: name -> "module:attribute".
_BUILTIN = {"cpd_n4t": "MartiniFuncs.chemistry.cpd_n4t.model:CpdN4tModel"}


SCREEN_METRIC = "screen_distance"


@dataclass(frozen=True)
class PairCollection:
    """Stage 1: what a pair analysis collects and caches.

    Every candidate whose screen centers (the model's screen_atoms, nearest
    periodic image) are closer than reactive_radius (Angstrom, strict) is
    measured and stored with all of the model's metrics; later selections can
    only narrow this set. A model's subclass gives reactive_radius its default,
    because what the radius means depends on the model's screen centers.
    """
    reactive_radius: float

    def __post_init__(self):
        if (isinstance(self.reactive_radius, (bool, np.bool_)) or self.reactive_radius is None
                or not np.isfinite(self.reactive_radius) or self.reactive_radius <= 0):
            raise ValueError("reactive_radius must be positive and finite (Angstrom)")


def threshold(metric, *, upper=None, unit="Angstrom", help=None):
    """A PairCriteria field: an optional upper threshold on one metric.

    upper bounds the threshold itself (e.g. 90 for acute angles, in degrees).
    """
    return field(default=None, metadata=dict(metric=metric, upper=upper, unit=unit, help=help))


@dataclass(frozen=True)
class PairCriteria:
    """Stage 2: selection of collected pairs (base class; all None selects everything).

    Each threshold field is an optional upper bound on one metric; a metric
    strictly above its bound rejects the pair, equality passes.
    max_screen_distance narrows the collected reactive_radius (it cannot exceed
    it). A pair whose geometry could not be measured is rejected when
    require_geometry is True or any geometric (non-screen) threshold is set.
    """
    max_screen_distance: float | None = threshold(
        SCREEN_METRIC, help="screen-center distance; at most the collected reactive_radius")
    require_geometry: bool = False

    def __post_init__(self):
        if not isinstance(self.require_geometry, (bool, np.bool_)):
            raise ValueError("require_geometry must be True or False")
        for f in fields(self):
            if "metric" not in f.metadata:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
            upper = f.metadata.get("upper")
            if upper is not None and value > upper:
                raise ValueError(f"{f.name} must be at most {upper:g} {f.metadata.get('unit', '')}".rstrip())

    @classmethod
    def thresholds(cls):
        """{field name: metric name} for every threshold field, in field order."""
        return {f.name: f.metadata["metric"] for f in fields(cls) if "metric" in f.metadata}

    def active(self):
        """{metric name: threshold} for the non-None thresholds."""
        return {metric: getattr(self, name) for name, metric in self.thresholds().items()
                if getattr(self, name) is not None}

    def requires_geometry(self):
        """True if unmeasured geometry rejects a pair: require_geometry or any geometric threshold."""
        return bool(self.require_geometry) or any(metric != SCREEN_METRIC for metric in self.active())

    def rejection_reasons(self, metrics, *, geometry_ok=True, error=None):
        """Reasons, in field order, why a pair is rejected; () when it is selected.

        metrics maps metric names to values: the model's metrics when the
        geometry was measured (geometry_ok), plus screen_distance when known.
        An unmeasured geometry is a reason only if requires_geometry(); error
        is its message. A threshold whose metric is missing although it should
        be available raises ValueError rather than passing silently.
        """
        reasons = []
        if not geometry_ok and self.requires_geometry():
            reasons.append(error or "geometry could not be measured")
        for name, metric in self.thresholds().items():
            limit = getattr(self, name)
            if limit is None:
                continue
            if metric not in metrics:
                if geometry_ok or metric == SCREEN_METRIC:
                    raise ValueError(f"{name} needs the {metric!r} value, which was not supplied")
                continue
            if metrics[metric] > limit:
                reasons.append(f"{name}: {metrics[metric]:.6g} > {limit:.6g}")
        return tuple(reasons)


class PairModel(ABC):
    """Chemistry of one kind of molecular pair.

    Class attributes
    ----------------
    name            registry key, recorded with every analysis
    roles           identifiers of the two pair members, used in column and
                    field names (e.g. "cpd_molecule", "cpd_shift")
    role_labels     per-atom pair_role labels written in structure exports
    collection_class
                    the PairCollection subclass for stage 1 (its reactive_radius
                    default is this model's screen radius); the base
                    PairCollection has no default, so a radius is then required
    criteria_class  the PairCriteria subclass with this model's thresholds (stage 2)
    metric_names    the geometry metrics measure() reports, in column order
    metric_ranges   {metric: (low, high)} histogram ranges for reports
    species         molecular formulas the model expects in a trajectory, or None

    Pairs are [first-member atom IDs, second-member atom IDs]; the periodic
    image shift always applies to the first member.
    """

    name: str
    roles: tuple
    role_labels: tuple
    criteria_class: type
    metric_names: tuple
    metric_ranges: dict = MappingProxyType({})   # read-only default; subclasses assign a dict
    species: frozenset | None = None
    collection_class: type = PairCollection

    @abstractmethod
    def classify(self, components, elements):
        """Role of each component: 0, 1, or None for molecules that take no part.

        components are sorted persistent atom-ID lists; elements maps atom ID to
        symbol. Raise ValueError for molecules the model does not recognize.
        """

    def screen_atoms(self, graph, elements, component, role):
        """Atom IDs whose mass-weighted center is the screen point; None means the whole molecule."""
        return None

    @abstractmethod
    def sites(self, graph, elements, component, role):
        """Reactive sites of one molecule (any picklable value); raise ValueError if absent."""

    @abstractmethod
    def measure(self, first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows):
        """Measure a jointly placed pair; return a dataclass geometry or raise ValueError.

        *_ids are persistent atom IDs in placement order, *_xyz whole Angstrom
        coordinates with the image shift already applied to the first member,
        frame_rows maps persistent ID to the row in the source frame.
        """

    @abstractmethod
    def metrics(self, geometry):
        """{metric name: value} for every name in metric_names."""

    def check_species(self, species):
        """Raise ValueError if a trajectory's molecular formulas do not fit this model."""
        if self.species is not None and set(species) != set(self.species):
            raise ValueError(f"Pair model {self.name!r} expects molecules {sorted(self.species)}, "
                             f"found {sorted(set(species))}")

    def metric_row(self, distance, geometry):
        """Candidate row: screen distance, metric_names values, geometry_ok (NaN metrics if None)."""
        if geometry is None:
            return [float(distance)] + [np.nan] * len(self.metric_names) + [0.]
        values = self.metrics(geometry)
        return [float(distance)] + [values[name] for name in self.metric_names] + [1.]

    def __reduce__(self):
        # Sent to worker processes by name and class: the worker uses its registered
        # model of that name, or registers a fresh instance of the (importable) class.
        return _restore_pair_model, (self.name, type(self))


def get_collection(model=None, collection=None):
    """The stage-1 PairCollection for a model: collection, or the model's default when None.

    model is a PairModel, a registered name or None (the default model).
    Raises ValueError when collection is not the model's collection_class, or
    when None is given and the model's collection has no default radius.
    """
    model = get_pair_model(model)
    cls = model.collection_class
    if collection is None:
        try:
            return cls()
        except TypeError:
            raise ValueError(f"Pair model {model.name!r} has no default collection; pass "
                             f"pair_collection={cls.__name__}(reactive_radius=...)") from None
    if not isinstance(collection, cls):
        raise ValueError(f"pair_collection must be {cls.__name__} for pair model {model.name!r}")
    return collection


def default_reactive_radius(model=None):
    """The model's default collection radius in Angstrom, or None if it has none (for help text)."""
    try:
        return get_collection(model).reactive_radius
    except ValueError:
        return None


def _restore_pair_model(name, cls):
    if name in _REGISTRY or name in _BUILTIN:
        return get_pair_model(name)
    return register_pair_model(cls())


def register_pair_model(model):
    """Register a PairModel instance under model.name (replacing any previous one)."""
    if not isinstance(model, PairModel):
        raise TypeError("register_pair_model expects a PairModel instance")
    _REGISTRY[model.name] = model
    return model


def get_pair_model(model=None):
    """A PairModel from an instance, a registered name, or None (the default model)."""
    if isinstance(model, PairModel):
        return model
    name = DEFAULT_PAIR_MODEL if model is None else model
    if name not in _REGISTRY and name in _BUILTIN:
        module, attribute = _BUILTIN[name].split(":")
        register_pair_model(getattr(import_module(module), attribute)())
    if name not in _REGISTRY:
        raise ValueError(f"Unknown pair model {name!r}; available: {', '.join(pair_model_names())}")
    return _REGISTRY[name]


def pair_model_names():
    """Names of the built-in and registered pair models."""
    return sorted(set(_BUILTIN) | set(_REGISTRY))


# Active only inside site_cache(); see model_sites.
_SITE_MEMO: dict | None = None


@contextmanager
def site_cache():
    """Reuse reactive-site identification across frames with unchanged molecules.

    A molecule's sites depend only on its atom IDs, elements and intramolecular
    bonds, so results are keyed by exactly those; any bond change misses the
    cache. Failures (ValueError) are cached and re-raised the same way.
    """
    global _SITE_MEMO
    previous, _SITE_MEMO = _SITE_MEMO, {}
    try:
        yield
    finally:
        _SITE_MEMO = previous


def model_sites(model, graph, elements, ids, role):
    """model.sites for one molecule, memoized inside site_cache()."""
    if _SITE_MEMO is None:
        return model.sites(graph, elements, ids, role)
    members = set(ids)
    bonds = tuple(sorted((i, j, tuple(int(x) for x in s)) for i in members for j, s in graph.get(i, [])
                         if j in members and i < j))
    key = (model.name, role, tuple(ids), tuple(elements[i] for i in ids), bonds)
    if key not in _SITE_MEMO:
        try:
            _SITE_MEMO[key] = model.sites(graph, elements, ids, role)
        except ValueError as exc:
            _SITE_MEMO[key] = exc
    value = _SITE_MEMO[key]
    if isinstance(value, ValueError):
        raise ValueError(*value.args)
    return value
