"""Anthracene + maleic anhydride Diels-Alder pairs: the pair model and its thresholds (model),
reactive sites (sites), a pair builder with controlled distance and orientation (build), and
an interactive HTML builder page (viewer), and Diels-Alder alignment across a crystal
interface (contact: contact_sites, alignment, best_registry, contact_alignment, best_shift).

The UMA-s-1p2p1 reference structures of the hand-off (isolated molecules,
reactant complex, TS, adduct; charge 0, spin 1) are package data:
build.reference_structure(name).
"""
from MartiniFuncs.chemistry.anthracene_ma.model import AnthraceneMaCollection, AnthraceneMaCriteria, AnthraceneMaModel

__all__ = ["AnthraceneMaCollection", "AnthraceneMaCriteria", "AnthraceneMaModel"]
