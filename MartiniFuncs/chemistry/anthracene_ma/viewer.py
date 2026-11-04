"""Interactive HTML page for the anthracene + maleic anhydride pair builder, with optional optimizer.

builder_page renders both molecules with Psience/McUtils (Molecule.plot, backend
"x3d"), each in the builder's frame (build.PairFrames): the anthracene fixed, the
maleic anhydride (MA) in its reference pose inside one X3D Transform.

Builder mode: the sliders set build_pair's parameters and the page applies the
same rigid placement (R = Rz(twist) Rx(roll) Ry(-fold), then the C=C midpoint at
(offset_x, offset_y, height)) to that Transform, so the rendered atoms and bonds
are McUtils' own.

Optimizer (when served by MartiniFuncs.optimize.server): the shown pair is sent
to a worker (UMA, AIMNet2, MACE-OMOL, GFN2-xTB) for a constrained optimization
(forming distances fixed or restrained, selected atoms fixed, or free), and the
steps stream back. Optimization frames move atoms individually: every McUtils
sphere is tied to its atom and every bond cylinder to its two atoms (position
along the bond, length fraction, and the sideways offset of double-bond
cylinders in a frame built from a neighbouring atom), recorded when the page is
generated. Click atoms to choose the ones to fix.

Every finished optimization is kept as a saved result (model, charge/spin,
optimizer, settings, constraint, convergence, final energy and force, final
geometry and metrics, optionally the whole path). The results list shows any
of them (optimizing a shown result continues from it), copies its settings,
and compares each against a chosen reference: energy differences only for the
same model, charge and spin, metric differences for any. The UMA reactant
complex, TS and product (and the separated reactants' energy) are built-in
entries. Results live on the server (in its --results folder, or its memory)
and export to / import from JSON; opened as a file the page keeps them in memory.

The forming C...C pairs (minimax matching) are drawn as a line or, below a
cutoff distance, as bonds (two extra cylinders styled like McUtils' bonds).

Forces and vibrations (server): the shown geometry's forces (a single point) or
Hessian (analytic, finite differences or auto) with any model; force arrows,
and normal modes listed by frequency, animated, with displacement arrows.
Optimization frames carry their own forces. Each calculation attaches to the
shown saved result (its final geometry, same model, charge and spin) or is
saved as a new single-point result, so it reloads and exports with the results.

The page recomputes the shared Diels-Alder metrics (as structure_metrics) for
every shown geometry, exports it (or the whole optimization path) as XYZ, and
prints the build_pair call for builder geometries. Opened as a file it works as
the builder alone. x3dom comes from the server's assets/x3dom/ when it has them
(server --assets; the MLIP container ships x3dom 1.8.3), otherwise from
x3dom.org, which needs an internet connection.
"""
import html
import json
import re
from pathlib import Path

import numpy as np

from MartiniFuncs.chemistry.anthracene_ma.build import (PairFrames, measure_placement, reference_energy,
                                                        reference_structure, structure_metrics, to_builder_frame)

BONDED = 1.9      # Angstrom: atom pairs considered when tying a cylinder to its bond
# The 33-atom UMA references shown on the page: (reference_structure name, label).
REFERENCE_LABELS = (("reactant_complex", "UMA reactant complex"), ("transition_state", "UMA TS"),
                    ("product", "UMA product"))


def _psience_nodes(atoms):
    """The shape nodes of a Psience/McUtils X3D figure of atoms (Angstrom), without viewpoint or background."""
    from MartiniFuncs.io.psience import ase_atoms_to_molecule
    markup = ase_atoms_to_molecule(atoms, energy_evaluator=None).plot(
        backend="x3d", background="white", units="Angstroms")._repr_html_()
    start = markup.index(">", markup.index("<Scene")) + 1
    nodes = markup[start:markup.index("</Scene>")]
    return re.sub(r"<(Viewpoint|Background|Environment)\b[^>]*>(</\1>)?", "", nodes)


def _neighbour(xyz, i, j):
    """An atom bonded to i (else to j), other than i and j, to orient double-bond offsets; None if absent."""
    for a, b in ((i, j), (j, i)):
        distances = np.linalg.norm(xyz - xyz[a], axis=1)
        for k in np.argsort(distances):
            if k not in (a, b) and distances[k] < BONDED:
                return int(k)
    return None


def _bond_frame(xyz, i, j, k):
    """Unit vectors (along i->j, toward k perpendicular to the bond, their cross product)."""
    e1 = (xyz[j] - xyz[i]) / np.linalg.norm(xyz[j] - xyz[i])
    if k is None:
        helper = np.array([1., 0., 0.]) if abs(e1[0]) < 0.9 else np.array([0., 1., 0.])
    else:
        helper = xyz[k] - xyz[i]
    e2 = helper - np.dot(helper, e1) * e1
    e2 /= np.linalg.norm(e2)
    return e1, e2, np.cross(e1, e2)


def scene(atoms, first_index=0):
    """(markup, spheres, cylinders) of one molecule's McUtils scene, tied to its atoms.

    Sphere shapes get onclick="pickAtom(index)" (index = first_index + row).
    spheres: [{"id", "atom"}]; cylinders: [{"id", "i", "j", "k", "t", "s", "o2", "o3"}]
    with i, j the bonded atoms, t the centre's fraction along i->j, s the height
    as a fraction of the bond length, and (o2, o3) the centre's sideways offset in
    the bond frame oriented by neighbour k (k -1: none). Indices include first_index.
    """
    xyz = atoms.positions
    markup = _psience_nodes(atoms)
    spheres, cylinders = [], []
    for match in re.finditer(r"<Transform\b([^>]*)>(.*?)</Transform>", markup, flags=re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        node = attrs["id"][:-len("-transform")]
        centre = np.array(attrs["translation"].split(), dtype=float)
        if "<Sphere" in match.group(2):
            atom = int(np.argmin(np.linalg.norm(xyz - centre, axis=1)))
            spheres.append({"id": node, "atom": first_index + atom})
            continue
        height = float(re.search(r'<Cylinder\b[^>]*\bheight="([^"]+)"', match.group(2)).group(1))
        best = None
        for i in range(len(xyz)):
            for j in range(len(xyz)):
                bond = xyz[j] - xyz[i]
                length = np.linalg.norm(bond)
                if i == j or length > BONDED:
                    continue
                t = float(np.dot(centre - xyz[i], bond) / length ** 2)
                perpendicular = np.linalg.norm(centre - xyz[i] - t * bond)
                if 0 <= t <= 1 and (best is None or perpendicular < best[0]):
                    best = (perpendicular, i, j, t, height / length)
        if best is None:
            continue           # not on any bond; left static
        _, i, j, t, s = best
        k = _neighbour(xyz, i, j)
        _, e2, e3 = _bond_frame(xyz, i, j, k)
        offset = centre - (xyz[i] + t * (xyz[j] - xyz[i]))
        cylinders.append({"id": node, "i": first_index + i, "j": first_index + j,
                          "k": -1 if k is None else first_index + k, "t": round(t, 6), "s": round(s, 6),
                          "o2": round(float(np.dot(offset, e2)), 6), "o3": round(float(np.dot(offset, e3)), 6)})
    pick = {sphere["id"]: sphere["atom"] for sphere in spheres}
    markup = re.sub(r'(<Transform id="([^"]+)-transform"[^>]*>)<Shape>',
                    lambda m: m.group(1) + (f'<Shape onclick="pickAtom({pick[m.group(2)]})">'
                                            if m.group(2) in pick else "<Shape>"), markup)
    return markup, spheres, cylinders


def forming_bond_markup(markup, cylinders, symbols):
    """Two hidden X3D cylinders (each in a Switch) for the forming C-C bonds, styled like McUtils' C-C bonds.

    markup, cylinders and symbols are one molecule's scene() markup and
    cylinders and its atom symbols; the radius and colour of its first C-C
    bond cylinder are copied. The page places and shows them (forming-bond-K,
    forming-bond-switch-K, K = 0, 1) from the forming pairs of the shown geometry.
    """
    radius, colour = 0.1, "0.565 0.565 0.565"
    for c in cylinders:
        if symbols[c["i"]] == symbols[c["j"]] == "C":
            size = re.search(r'<Transform id="%s-transform"[^>]*\bscale="([\d.]+) ' % re.escape(c["id"]), markup)
            tint = re.search(r'diffuseColor="([^"]+)" id="%s-appearance-material"' % re.escape(c["id"]), markup)
            if size and tint:
                radius, colour = float(size.group(1)), tint.group(1)
                break
    return "".join(f'<switch id="forming-bond-switch-{k}" whichChoice="-1"><transform id="forming-bond-{k}" '
                   f'translation="0 0 0" rotation="1 0 0 0" scale="{radius:g} 1 {radius:g}"><shape><appearance>'
                   f'<material diffuseColor="{colour}"></material></appearance><cylinder radius="1" height="1">'
                   f'</cylinder></shape></transform></switch>' for k in range(2))


ARROW_COLOURS = {"force": "0.10 0.45 0.80", "mode": "0.88 0.00 0.48"}


def arrow_markup(kind, count, colour):
    """count hidden X3D arrows (shaft cylinder + cone head, each in a Switch) for the page to place.

    Ids: KIND-arrow-switch-I, KIND-arrow-I (origin, rotation), KIND-arrow-I-shaft,
    KIND-arrow-I-head; the page sets them from a vector per atom (force or mode).
    """
    material = f'<appearance><material diffuseColor="{colour}" specularColor="0.3 0.3 0.3"></material></appearance>'
    return "".join(
        f'<switch id="{kind}-arrow-switch-{i}" whichChoice="-1"><transform id="{kind}-arrow-{i}" translation="0 0 0" '
        f'rotation="1 0 0 0"><transform id="{kind}-arrow-{i}-shaft" translation="0 0.5 0" scale="0.045 1 0.045">'
        f'<shape>{material}<cylinder radius="1" height="1"></cylinder></shape></transform>'
        f'<transform id="{kind}-arrow-{i}-head" translation="0 1.15 0" scale="1 1 1"><shape>{material}'
        f'<cone bottomRadius="0.11" height="0.3"></cone></shape></transform></transform></switch>'
        for i in range(count))


def reference_entries(frames):
    """The page's UMA reference rows: metrics, energy and builder-frame positions of each REFERENCE_LABELS
    structure (to_builder_frame), then the separated reactants (energy only: anthracene + MA)."""
    entries = []
    for name, label in REFERENCE_LABELS:
        structure = reference_structure(name)
        entries.append({"key": name, "name": label, "model": "uma", "charge": 0, "spin": 1,
                        "energy": reference_energy(name),
                        "positions": np.round(to_builder_frame(structure, frames).positions, 6).tolist(),
                        **structure_metrics(structure)})
    entries.append({"key": "separated", "name": "UMA separated reactants", "model": "uma", "charge": 0, "spin": 1,
                    "energy": reference_energy("anthracene") + reference_energy("maleic_anhydride"),
                    "positions": None})
    return entries


def default_presets(frames):
    """Starting placements: the stacked pose, two contrasts, and the UMA reference complex, TS and product (rigid)."""
    presets = [{"name": "Stacked (reactive)", "height": 3.2},
               {"name": "C=C crossed (twist 90)", "height": 3.3, "twist": 90.0},
               {"name": "Ring edge-on (fold 90)", "height": 3.6, "fold": 90.0}]
    for name, label in REFERENCE_LABELS:
        placement = measure_placement(reference_structure(name))
        presets.append({"name": label + " (rigid)", **{k: round(v, 3) for k, v in placement.items()
                                                       if k != "screen_distance"}})
    return presets


def builder_page(frames=None, *, presets=None, title="Anthracene + MA Pair Builder"):
    """The interactive builder page as an HTML string.

    frames is a build.PairFrames (default: the UMA-relaxed templates); presets
    a list of {"name", "height", "offset_x", "offset_y", "twist", "roll", "fold"}
    placements (default: default_presets). Reference rows (reference_entries)
    hold the metrics, UMA energies and geometries of the flexible UMA reactant
    complex, TS and product, which the page can show and compare against.
    """
    from ase import Atoms
    frames = frames or PairFrames()
    presets = default_presets(frames) if presets is None else presets
    anthracene = Atoms(frames.anthracene.get_chemical_symbols(), positions=frames.anthracene_xyz)
    anhydride = Atoms(frames.anhydride.get_chemical_symbols(), positions=frames.anhydride_xyz)
    n = len(anthracene)
    ant_markup, ant_spheres, ant_cylinders = scene(anthracene, 0)
    ma_markup, ma_spheres, ma_cylinders = scene(anhydride, n)
    references = reference_entries(frames)
    data = {
        "anthracene": {"symbols": anthracene.get_chemical_symbols(), "xyz": np.round(anthracene.positions, 6).tolist()},
        "anhydride": {"symbols": anhydride.get_chemical_symbols(), "xyz": np.round(anhydride.positions, 6).tolist()},
        # Rows of the combined pair (anthracene 0..23, then MA): sites for the metrics.
        "termini": [int(i) for i in frames.termini], "alkene": [n + int(i) for i in frames.alkene],
        "central_ring": [int(i) for i in frames.central_ring], "ring_carbons": [n + int(i) for i in frames.ring_carbons],
        "spheres": ant_spheres + ma_spheres, "cylinders": ant_cylinders + ma_cylinders,
        "presets": presets, "references": references,
    }
    return (_PAGE.replace("@@TITLE@@", html.escape(title))
            .replace("@@ANTHRACENE@@", ant_markup).replace("@@ANHYDRIDE@@", ma_markup)
            .replace("@@FORMING_BONDS@@", forming_bond_markup(ant_markup, ant_cylinders,
                                                              anthracene.get_chemical_symbols()))
            .replace("@@ARROWS@@", "".join(arrow_markup(kind, n + len(anhydride), colour)
                                           for kind, colour in ARROW_COLOURS.items()))
            .replace("@@DATA@@", json.dumps(data).replace("</", "<\\/")))


def write_builder_html(path, frames=None, *, presets=None, title="Anthracene + MA Pair Builder"):
    """Write builder_page to path; return the path."""
    path = Path(path)
    path.write_text(builder_page(frames, presets=presets, title=title), encoding="utf-8")
    return path


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<link rel="stylesheet" href="assets/x3dom/x3dom.css">
<script src="assets/x3dom/x3dom.js"></script>
<script>
  // x3dom from the server's assets (the MLIP container ships it), else from x3dom.org (needs internet).
  if (!window.x3dom) document.write('<link rel="stylesheet" href="https://www.x3dom.org/download/x3dom.css">' +
    '<script src="https://www.x3dom.org/download/1.8.3/x3dom-full.js"><\/script>');
</script>
<style>
  :root { --bg: #f6f7f9; --panel: #ffffff; --ink: #1d2330; --muted: #5d6677; --line: #dde1e8;
          --accent: #c4561b; --accent-soft: #fbeee6; --good: #1f7a3f; --bad: #b3261e; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 14px 24px 4px; }
  header h1 { margin: 0; font-size: 20px; }
  header p { margin: 4px 0 0; color: var(--muted); max-width: 1100px; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 400px; gap: 16px; padding: 10px 24px 24px;
         align-items: start; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } .view { position: static; } }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
  .view { padding: 0; overflow: hidden; position: sticky; top: 10px; display: flex; flex-direction: column; }
  #builder-x3d { display: block; width: 100%; height: calc(100vh - 150px); min-height: 420px; border: none; }
  #builder-x3d canvas { width: 100% !important; height: 100% !important; }
  .view-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px 10px; border-top: 1px solid var(--line); }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 10px; }
  .control { display: grid; grid-template-columns: 96px 1fr 70px; align-items: center; gap: 8px; margin: 6px 0; }
  .control label { font-weight: 600; }
  .control input[type=number] { width: 70px; padding: 2px 4px; }
  .control input[type=range] { width: 100%; }
  .field { display: grid; grid-template-columns: 110px 1fr; align-items: center; gap: 8px; margin: 6px 0; }
  .field label { font-weight: 600; }
  .field input, .field select { width: 100%; padding: 2px 4px; font: inherit; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .hint { color: var(--muted); font-size: 12px; margin: 2px 0 10px; }
  button { font: inherit; padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px;
           background: #fff; cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: .5; cursor: default; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  td, th { padding: 3px 6px; border-bottom: 1px solid var(--line); text-align: right; }
  td:first-child, th:first-child { text-align: left; }
  .refs th { font-weight: 600; color: var(--muted); font-size: 12px; }
  pre { background: var(--accent-soft); padding: 8px; border-radius: 6px; white-space: pre-wrap;
        font-size: 12px; margin: 8px 0; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; align-items: center; }
  .checks label { margin-right: 14px; }
  .legend span { display: inline-block; margin-right: 12px; font-size: 12px; color: var(--muted); }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  .stack { display: flex; flex-direction: column; gap: 16px; }
  .mode { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: var(--accent-soft); color: var(--accent); }
  .status { font-size: 13px; margin: 8px 0 4px; min-height: 1.4em; }
  .status.good { color: var(--good); } .status.bad { color: var(--bad); }
  .offline { background: #f1f3f6; border-radius: 6px; padding: 8px; color: var(--muted); font-size: 13px; }
  svg.plot { width: 100%; height: 110px; display: block; margin-top: 6px; }
  .frames { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; margin-top: 8px; }
  .results td, .results th { vertical-align: top; padding: 4px 4px; }
  .results th { font-size: 12px; color: var(--muted); font-weight: 600; }
  .results small, #metrics small { color: var(--muted); font-size: 11px; }
  .results tr.shown td { background: var(--accent-soft); }
  .results .name { font-weight: 600; outline: none; border-bottom: 1px dashed transparent; }
  .results .name[contenteditable=true]:hover, .results .name:focus { border-bottom-color: var(--muted); }
  .results td.acts { white-space: nowrap; width: 1%; }
  .results td.acts button { display: block; width: 100%; margin: 0 0 3px; }
  .results td:nth-child(2), .results th:nth-child(2) { text-align: left; }
  .results td:nth-child(2) small { display: block; line-height: 1.3; }
  button.mini { padding: 1px 6px; font-size: 12px; margin: 0 0 2px 2px; }
  #metrics th { font-size: 12px; color: var(--muted); font-weight: 600; }
  .vib-info { font-size: 13px; margin: 6px 0; }
  .vib-info .warn { color: var(--bad); }
  #vib-box[hidden] { display: none; }
  .checks select, .checks input { font: inherit; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>@@TITLE@@</h1>
  <p>Anthracene stays fixed in its frame: origin at the central-ring centroid, <b>x</b> the long axis,
     <b>y</b> from C9 to C10, <b>z</b> the ring normal. Maleic anhydride (MA) is placed rigidly: its C=C midpoint at
     (offset x, offset y, height); with all angles 0 its C=C is parallel to C9&ndash;C10 and the two rings are stacked.
     Molecules are rendered with Psience/McUtils (X3D). Drag to rotate, scroll to zoom; click atoms to select them.</p>
</header>
<main>
  <section class="panel view">
    <x3d id="builder-x3d" showStat="false" showLog="false">
      <scene>
        <viewpoint id="vp-oblique" description="Oblique" position="0 -15.6 9" orientation="1 0 0 1.047" centerOfRotation="0 0 1.5"></viewpoint>
        <viewpoint id="vp-top" description="Top" position="0 0 18" orientation="0 0 1 0" centerOfRotation="0 0 1.5"></viewpoint>
        <viewpoint id="vp-long" description="Along the long axis" position="18 0 1.5" orientation="0.577 0.577 0.577 2.094" centerOfRotation="0 0 1.5"></viewpoint>
        <viewpoint id="vp-short" description="Along C9-C10" position="0 -18 1.5" orientation="1 0 0 1.5708" centerOfRotation="0 0 1.5"></viewpoint>
        <background skyColor="1 1 1"></background>
        <group id="anthracene-scene">@@ANTHRACENE@@</group>
        <transform id="anhydride-placement" translation="0 0 3.2" rotation="1 0 0 0">@@ANHYDRIDE@@</transform>
        <switch id="forming-switch" whichChoice="0">
          <shape>
            <appearance><material emissiveColor="0.85 0.35 0.05" diffuseColor="0 0 0"></material></appearance>
            <lineset vertexCount="2 2"><coordinate id="forming-coords" point="0 0 0 0 0 0 0 0 0 0 0 0"></coordinate></lineset>
          </shape>
        </switch>
        @@FORMING_BONDS@@
        @@ARROWS@@
        <switch id="axes-switch" whichChoice="-1">
          <shape>
            <appearance><material emissiveColor="1 1 1" diffuseColor="0 0 0"></material></appearance>
            <lineset vertexCount="2 2 2">
              <coordinate point="0 0 0 4 0 0 0 0 0 0 4 0 0 0 0 0 0 4"></coordinate>
              <color color="0.85 0.1 0.1 0.85 0.1 0.1 0.1 0.6 0.1 0.1 0.6 0.1 0.1 0.25 0.9 0.1 0.25 0.9"></color>
            </lineset>
          </shape>
        </switch>
      </scene>
    </x3d>
    <div class="view-bar">
      <button data-view="vp-oblique">Oblique</button>
      <button data-view="vp-top">Top (down z)</button>
      <button data-view="vp-long">Along long axis (x)</button>
      <button data-view="vp-short">Along C9&ndash;C10 (y)</button>
      <span class="checks" style="margin-left:auto">
        <label title="forming C&middot;&middot;&middot;C pairs: a line, bonds, or bonds below the cutoff (auto)">forming C&middot;&middot;&middot;C
          <select id="forming-style"><option value="auto">auto</option><option value="line">line</option>
            <option value="bond">bond</option><option value="off">off</option></select></label>
        <label title="auto: drawn as a bond below this C&middot;&middot;&middot;C distance">bond &lt;
          <input type="number" id="forming-cutoff" value="1.8" step="0.05" min="1" max="3" style="width:58px"> &Aring;</label>
        <label><input type="checkbox" id="show-axes"> frame axes</label>
      </span>
    </div>
    <div class="view-bar legend">
      <span class="mode" id="mode-label">builder</span>
      <span><i class="swatch" style="background:#d9590d"></i>forming C&middot;&middot;&middot;C pairs (minimax matching; bonds when formed)</span>
      <span><i class="swatch" style="background:#e0a000"></i>selected atoms</span>
      <span><i class="swatch" style="background:#d91a1a"></i>x</span>
      <span><i class="swatch" style="background:#1a991a"></i>y</span>
      <span><i class="swatch" style="background:#1a40e6"></i>z</span>
    </div>
  </section>

  <div class="stack">
    <section class="panel">
      <h2>Placement</h2>
      <div class="presets" id="presets"></div>
      <div class="control">
        <label for="kind">Distance</label>
        <select id="kind">
          <option value="height">height (z of C=C midpoint)</option>
          <option value="centers">centers (ring centroid to C=C midpoint)</option>
          <option value="max_forming">max forming C&middot;&middot;&middot;C</option>
        </select>
        <span></span>
      </div>
      <div id="controls"></div>
      <p class="hint">Distances in &Aring;, angles in degrees. fold lifts the anhydride ring away from the anthracene
        (rotation about the C=C); roll tilts the C=C out of the plane; twist turns it in the plane.</p>
    </section>

    <section class="panel">
      <h2>Optimize</h2>
      <div class="offline" id="opt-offline">Optimizer offline. Serve this page with
        <code>python -m MartiniFuncs.optimize.server</code> (or the container's <code>server</code> app) to
        optimize the shown pair.</div>
      <div id="opt-controls">
        <div class="field"><label for="opt-model">Model</label><select id="opt-model"></select></div>
        <div class="field"><label for="opt-constraint">Constraint</label>
          <select id="opt-constraint">
            <option value="fix_distances">fix forming C&middot;&middot;&middot;C distances</option>
            <option value="restrain_distances">restrain forming distances (harmonic)</option>
            <option value="fix_atoms">fix selected atoms</option>
            <option value="none">none (free)</option>
          </select></div>
        <div class="field" id="opt-k-row"><label for="opt-k">k (eV/&Aring;&sup2;)</label><input id="opt-k" type="number" value="10" step="1" min="0"></div>
        <div class="field"><label for="opt-atoms">Selected atoms</label><input id="opt-atoms" type="text"></div>
        <p class="hint">Rows 1&ndash;24 anthracene, 25&ndash;33 MA (as in the XYZ export). Default: the four forming
          carbons. Click atoms in the view to add or remove them.</p>
        <div class="field"><label for="opt-optimizer">Optimizer</label>
          <select id="opt-optimizer">
            <option value="BFGS">BFGS</option><option value="LBFGS">LBFGS</option><option value="FIRE">FIRE</option>
            <option value="sella_min">Sella (minimum)</option><option value="sella_ts">Sella (transition state)</option>
          </select></div>
        <div class="field" id="opt-sella-row"><label for="opt-sella-hessian">Sella Hessian</label>
          <select id="opt-sella-hessian" title="Sella's own iterative estimate, or the model's Hessian (method as under Forces &amp; vibrations)">
            <option value="sella">Sella's own (iterative)</option>
            <option value="model">model Hessian when Sella needs one</option>
            <option value="every">model Hessian every N steps</option></select></div>
        <div class="field" id="opt-sella-every-row"><label for="opt-sella-every">N (steps)</label>
          <input id="opt-sella-every" type="number" value="5" step="1" min="1"></div>
        <div class="pair">
          <div class="field"><label for="opt-fmax">fmax (eV/&Aring;)</label><input id="opt-fmax" type="number" value="0.05" step="0.01" min="0.001"></div>
          <div class="field"><label for="opt-steps">max steps</label><input id="opt-steps" type="number" value="300" step="10" min="1"></div>
          <div class="field"><label for="opt-charge">charge</label><input id="opt-charge" type="number" value="0" step="1"></div>
          <div class="field"><label for="opt-spin">spin mult.</label><input id="opt-spin" type="number" value="1" step="1" min="1"></div>
        </div>
        <div class="row">
          <button class="primary" id="opt-run">Optimize shown geometry</button>
          <button id="opt-cancel" disabled>Cancel</button>
          <button id="opt-builder" disabled>Back to builder</button>
        </div>
        <div class="status" id="opt-status"></div>
        <svg class="plot" id="opt-plot" viewBox="0 0 360 110" preserveAspectRatio="none"></svg>
        <div class="frames">
          <input type="range" id="opt-frame" min="0" max="0" value="0" disabled>
          <span id="opt-frame-label">no frames</span>
        </div>
      </div>
    </section>

    <section class="panel" id="analysis-panel">
      <h2>Forces &amp; vibrations</h2>
      <div class="row">
        <button id="calc-forces">Calculate forces</button>
        <button id="calc-hessian">Calculate Hessian</button>
        <select id="hessian-method" title="Hessian: the model's own second derivatives, central finite differences of the forces (6N force calls), or analytic when the model has it">
          <option value="auto">auto</option><option value="analytic">analytic</option>
          <option value="finite_diff">finite differences</option></select>
      </div>
      <p class="hint">On the shown geometry, with the model, charge and spin of the Optimize panel. The result attaches
        to the shown saved result (same model), otherwise it is saved as a new single-point result.</p>
      <div class="status" id="analysis-status"></div>
      <div class="row">
        <label><input type="checkbox" id="show-forces" checked> force arrows</label>
        <label title="longest arrow 1.5 Å"><input type="checkbox" id="force-auto" checked> auto scale</label>
        <label>or <input type="number" id="force-scale" value="1" step="0.1" min="0.01" style="width:60px"> Å per eV/Å</label>
      </div>
      <div class="hint" id="force-info"></div>
      <div id="vib-box" hidden>
        <div class="field"><label for="vib-mode">Mode</label><select id="vib-mode"></select></div>
        <div class="control"><label for="vib-amp">amplitude</label>
          <input type="range" id="vib-amp" min="0.05" max="1" step="0.05" value="0.4"><span id="vib-amp-label">0.40 Å</span></div>
        <div class="row">
          <button id="vib-play">Play</button>
          <label><input type="checkbox" id="vib-arrows" checked> mode arrows</label>
        </div>
        <div class="vib-info" id="vib-info"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Saved results</h2>
      <div class="hint" id="results-store"></div>
      <table class="results" id="results"></table>
      <div class="row">
        <button id="results-export">Export JSON</button>
        <button id="results-import-button">Import JSON</button>
        <input type="file" id="results-import" accept=".json,application/json" hidden>
        <label title="store every optimization step with the result (larger files)"><input type="checkbox" id="results-path" checked> keep paths</label>
      </div>
      <div class="status" id="results-msg"></div>
      <p class="hint">Finished optimizations are saved here. The ref button picks the reference for &Delta;E and the
        metric differences; energies compare only for the same model, charge and spin. Show loads a result (its path,
        if kept) into the view, and optimizing it continues from it; Settings copies its model and settings to the
        Optimize panel. Click a name to rename it.</p>
    </section>

    <section class="panel">
      <h2>Measured metrics</h2>
      <table id="metrics"></table>
      <h2 style="margin-top:14px">Reference structures (flexible UMA; Show them from Saved results)</h2>
      <table class="refs" id="references"></table>
    </section>

    <section class="panel">
      <h2>Export</h2>
      <div class="row">
        <button class="primary" id="download-xyz">Download shown XYZ</button>
        <button id="download-path" disabled>Download optimization path</button>
        <button id="copy-python">Copy Python call</button>
      </div>
      <pre id="python-call"></pre>
    </section>
  </div>
</main>

<script type="application/json" id="builder-data">@@DATA@@</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("builder-data").textContent);
  var ANT = D.anthracene.xyz, MA = D.anhydride.xyz, NA = ANT.length;
  var SYMBOLS = D.anthracene.symbols.concat(D.anhydride.symbols);
  var CONTROLS = [
    {key: "distance", label: "distance", min: 0.5, max: 8, step: 0.01, value: 3.2},
    {key: "offset_x", label: "offset x", min: -5, max: 5, step: 0.05, value: 0},
    {key: "offset_y", label: "offset y", min: -5, max: 5, step: 0.05, value: 0},
    {key: "twist", label: "twist", min: -180, max: 180, step: 1, value: 0},
    {key: "roll", label: "roll", min: -90, max: 90, step: 1, value: 0},
    {key: "fold", label: "fold", min: -180, max: 180, step: 1, value: 0}
  ];
  var state = {kind: "height"};
  CONTROLS.forEach(function (c) { state[c.key] = c.value; });
  // source: what the shown frames are, {kind: "job" | "result", id, name, model, charge, spin}
  var view = {mode: "builder", positions: null, built: null, frames: [], frame: 0, model: null, source: null};
  var job = null;     // the running or last optimization: {id, frames, source, request, startLabel, parentId}
  var KCAL = 23.0605;  // kcal/mol per eV
  var selected = new Set([D.termini[0], D.termini[1], D.alkene[0], D.alkene[1]]);

  // --- linear algebra (row-major 3x3) ---
  function mul(A, B) {
    var C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++)
      for (var k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  }
  function apply(R, v) {
    return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function mid(a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]; }
  function norm(a) { return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function unit(a) { var n = norm(a); return n > 1e-12 ? scale(a, 1 / n) : [0, 0, 0]; }
  function acute(a, b) {   // as diels_alder.acute_angle: unsigned, 0-90 degrees
    var c = Math.abs(dot(a, b)) / (norm(a) * norm(b));
    return Math.acos(Math.min(1, Math.max(0, c))) * 180 / Math.PI;
  }
  function planeNormal(points) {   // as diels_alder.plane_normal: smallest-variance direction (Jacobi)
    var c = [0, 0, 0];
    points.forEach(function (p) { c = add(c, p); });
    c = scale(c, 1 / points.length);
    var A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    points.forEach(function (p) { var d = sub(p, c);
      for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) A[i][j] += d[i] * d[j]; });
    var V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (var sweep = 0; sweep < 50; sweep++) {
      var off = Math.abs(A[0][1]) + Math.abs(A[0][2]) + Math.abs(A[1][2]);
      if (off < 1e-14) break;
      [[0, 1], [0, 2], [1, 2]].forEach(function (pq) {
        var p = pq[0], q = pq[1];
        if (Math.abs(A[p][q]) < 1e-18) return;
        var theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        var cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
        for (var k = 0; k < 3; k++) {   // A <- J^T A J, V <- V J
          var akp = A[k][p], akq = A[k][q];
          A[k][p] = cs * akp - sn * akq; A[k][q] = sn * akp + cs * akq;
        }
        for (var k2 = 0; k2 < 3; k2++) {
          var apk = A[p][k2], aqk = A[q][k2];
          A[p][k2] = cs * apk - sn * aqk; A[q][k2] = sn * apk + cs * aqk;
        }
        for (var k3 = 0; k3 < 3; k3++) {
          var vkp = V[k3][p], vkq = V[k3][q];
          V[k3][p] = cs * vkp - sn * vkq; V[k3][q] = sn * vkp + cs * vkq;
        }
      });
    }
    var m = 0;
    for (var i = 1; i < 3; i++) if (A[i][i] < A[m][m]) m = i;
    return [V[0][m], V[1][m], V[2][m]];
  }
  var RAD = Math.PI / 180;
  // build.placement_rotation: Rz(twist) Rx(roll) Ry(-fold)
  function rotation(twist, roll, fold) {
    var t = twist * RAD, r = roll * RAD, f = -fold * RAD;
    var Rz = [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];
    var Rx = [[1, 0, 0], [0, Math.cos(r), -Math.sin(r)], [0, Math.sin(r), Math.cos(r)]];
    var Ry = [[Math.cos(f), 0, Math.sin(f)], [0, 1, 0], [-Math.sin(f), 0, Math.cos(f)]];
    return mul(Rz, mul(Rx, Ry));
  }
  function axisAngle(R) {  // rotation matrix -> x3dom "x y z angle"
    var tr = R[0][0] + R[1][1] + R[2][2], w, x, y, z, s;
    if (tr > 0) { s = Math.sqrt(tr + 1) * 2; w = s / 4; x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s; }
    else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) { s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2; w = (R[2][1] - R[1][2]) / s; x = s / 4; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s; }
    else if (R[1][1] > R[2][2]) { s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2; w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s; y = s / 4; z = (R[1][2] + R[2][1]) / s; }
    else { s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2; w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = s / 4; }
    var angle = 2 * Math.acos(Math.min(1, Math.max(-1, w))), k = Math.sqrt(Math.max(0, 1 - w * w));
    return k < 1e-9 ? [1, 0, 0, 0] : [x / k, y / k, z / k, angle];
  }
  function alignY(d) {     // x3dom rotation taking the cylinder axis (0,1,0) onto unit vector d
    var y = [0, 1, 0], axis = cross(y, d), s = norm(axis), c = dot(y, d);
    if (s < 1e-9) return c > 0 ? [1, 0, 0, 0] : [1, 0, 0, Math.PI];
    return [axis[0] / s, axis[1] / s, axis[2] / s, Math.atan2(s, c)];
  }
  function fmt(v, d) { return v.toFixed(d === undefined ? 3 : d); }
  function vec(v) { return v.map(function (x) { return x.toFixed(5); }).join(" "); }

  // --- the builder (mirrors build.build_pair) ---
  function placeAt(R, position) { return MA.map(function (v) { return add(apply(R, v), position); }); }
  function maxForming(P) {
    var t = D.termini, a = D.alkene;
    return Math.min(Math.max(norm(sub(P[t[0]], P[a[0]])), norm(sub(P[t[1]], P[a[1]]))),
                    Math.max(norm(sub(P[t[0]], P[a[1]])), norm(sub(P[t[1]], P[a[0]]))));
  }
  function solveHeight(s, R) {
    var ox = s.offset_x, oy = s.offset_y;
    if (s.kind === "height") return s.distance;
    if (s.kind === "centers") {
      var sq = s.distance * s.distance - ox * ox - oy * oy;
      if (sq <= 0) throw new Error("centers distance must exceed the in-plane offset");
      return Math.sqrt(sq);
    }
    // max_forming: lowest height (grid 0-30 A, then bisection) where the longer forming distance reaches the request
    var f = function (h) { return maxForming(ANT.concat(placeAt(R, [ox, oy, h]))) - s.distance; };
    var prev = f(0), step = 0.05;
    for (var h = step; h <= 30 + 1e-9; h += step) {
      var cur = f(h);
      if (prev < 0 && cur >= 0) {
        var lo = h - step, hi = h;
        for (var it = 0; it < 80; it++) { var m = (lo + hi) / 2; if (f(m) < 0) lo = m; else hi = m; }
        return (lo + hi) / 2;
      }
      prev = cur;
    }
    throw new Error("no height at these offsets and angles gives that max forming distance");
  }
  function build(s) {
    var R = rotation(s.twist, s.roll, s.fold), height = solveHeight(s, R);
    var position = [s.offset_x, s.offset_y, height];
    return {R: R, position: position, height: height, positions: ANT.concat(placeAt(R, position))};
  }

  // --- metrics of any shown geometry (as build.structure_metrics) ---
  function metrics(P) {
    var t = D.termini, a = D.alkene;
    var options = [[a[0], a[1]], [a[1], a[0]]].map(function (partner) {
      var d = [norm(sub(P[t[0]], P[partner[0]])), norm(sub(P[t[1]], P[partner[1]]))];
      return {partner: partner, d: d, key: [Math.max(d[0], d[1]), d[0] + d[1]]};
    });
    options.sort(function (x, y) { return x.key[0] - y.key[0] || x.key[1] - y.key[1]; });
    var best = options[0];
    var ring = D.central_ring.map(function (i) { return P[i]; });
    var centroid = scale(ring.reduce(add, [0, 0, 0]), 1 / ring.length);
    var dieneNormal = planeNormal(ring), dienophileNormal = planeNormal(D.ring_carbons.map(function (i) { return P[i]; }));
    var termMid = mid(P[t[0]], P[t[1]]), alkMid = mid(P[a[0]], P[a[1]]);
    var contact = Infinity;
    for (var i = 0; i < NA; i++) for (var j = NA; j < P.length; j++) contact = Math.min(contact, norm(sub(P[i], P[j])));
    return {
      pairs: [[t[0], best.partner[0]], [t[1], best.partner[1]]],
      screen_distance: norm(sub(alkMid, centroid)),
      max_forming_distance: Math.max(best.d[0], best.d[1]), forming_1: best.d[0], forming_2: best.d[1],
      distance_imbalance: Math.abs(best.d[0] - best.d[1]),
      axis_angle: acute(sub(P[t[1]], P[t[0]]), sub(P[a[1]], P[a[0]])),
      plane_angle: acute(dieneNormal, dienophileNormal),
      approach_angle: acute(dieneNormal, sub(alkMid, termMid)),
      min_intermolecular_distance: contact
    };
  }

  // --- the McUtils scene: rigid MA placement (builder) or per-atom geometry (optimization frames) ---
  var placement = document.getElementById("anhydride-placement");
  var original = {};      // node id -> attributes as rendered, restored when returning to the builder
  function node(id) { return document.getElementById(id); }
  function remember(id, attrs) {
    if (original[id]) return;
    var e = node(id + "-transform"), c = node(id);
    if (!e) return;
    original[id] = {translation: e.getAttribute("translation"), rotation: e.getAttribute("rotation"),
                    height: c && attrs ? c.getAttribute("height") : null};
  }
  D.spheres.forEach(function (s) { remember(s.id, false); });
  D.cylinders.forEach(function (c) { remember(c.id, true); });
  var perAtom = false;
  function showPerAtom(P) {
    placement.setAttribute("translation", "0 0 0");
    placement.setAttribute("rotation", "1 0 0 0");
    D.spheres.forEach(function (s) { var e = node(s.id + "-transform"); if (e) e.setAttribute("translation", vec(P[s.atom])); });
    D.cylinders.forEach(function (c) {
      var pi = P[c.i], pj = P[c.j], bond = sub(pj, pi), length = norm(bond), e1 = unit(bond);
      var helper = c.k >= 0 ? sub(P[c.k], pi) : (Math.abs(e1[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]);
      var e2 = unit(sub(helper, scale(e1, dot(helper, e1)))), e3 = cross(e1, e2);
      var centre = add(add(pi, scale(bond, c.t)), add(scale(e2, c.o2), scale(e3, c.o3)));
      var e = node(c.id + "-transform"), cyl = node(c.id);
      if (!e) return;
      e.setAttribute("translation", vec(centre));
      e.setAttribute("rotation", vec(alignY(e1)));
      if (cyl) cyl.setAttribute("height", (c.s * length).toFixed(5));
    });
    perAtom = true;
  }
  function showRigid(built) {
    if (perAtom) {
      Object.keys(original).forEach(function (id) {
        var o = original[id], e = node(id + "-transform"), c = node(id);
        e.setAttribute("translation", o.translation);
        if (o.rotation !== null) e.setAttribute("rotation", o.rotation);
        if (o.height !== null && c) c.setAttribute("height", o.height);
      });
      perAtom = false;
    }
    placement.setAttribute("rotation", vec(axisAngle(built.R)));
    placement.setAttribute("translation", vec(built.position));
  }
  function highlight() {
    D.spheres.forEach(function (s) {
      var m = node(s.id + "-appearance-material");
      if (m) m.setAttribute("emissiveColor", selected.has(s.atom) ? "0.55 0.38 0" : "0 0 0");
    });
    document.getElementById("opt-atoms").value =
      Array.from(selected).sort(function (a, b) { return a - b; }).map(function (i) { return i + 1; }).join(", ");
  }
  window.pickAtom = function (atom) {
    if (selected.has(atom)) selected.delete(atom); else selected.add(atom);
    highlight();
  };

  // --- page state ---
  var LABELS = {
    screen_distance: ["screen distance", "Å", "central-ring centroid to C=C midpoint"],
    max_forming_distance: ["max forming", "Å", "longer forming C···C"],
    forming_1: ["forming d₁", "Å", "C9-side C···C"],
    forming_2: ["forming d₂", "Å", "C10-side C···C"],
    distance_imbalance: ["imbalance |d₁−d₂|", "Å", ""],
    axis_angle: ["axis angle", "°", "C9-C10 vs C=C"],
    plane_angle: ["plane angle", "°", "central ring vs MA ring carbons"],
    approach_angle: ["approach angle", "°", "ring normal vs termini-midpoint to C=C"],
    min_intermolecular_distance: ["closest contact", "Å", "any anthracene-MA atom pair"]
  };
  var METRIC_KEYS = ["screen_distance", "max_forming_distance", "forming_1", "forming_2", "distance_imbalance",
                     "axis_angle", "plane_angle", "approach_angle", "min_intermolecular_distance"];
  function esc(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]; });
  }
  function signed(v, d) { return (v < 0 ? "−" : "+") + Math.abs(v).toFixed(d); }
  function comparable(a, b) { return !!(a && b) && a.model === b.model && a.charge === b.charge && a.spin === b.spin; }
  function shownEnergy() {   // the shown frame's model energy (restraint removed) with its model, charge and spin
    if (view.mode !== "frames" || !view.source) return null;
    var f = view.frames[view.frame];
    if (f.energy === null || f.energy === undefined) return null;
    return {energy: f.energy - (f.restraint_energy || 0), model: view.source.model, charge: view.source.charge,
            spin: view.source.spin};
  }
  // Forming C...C pairs: a line, or a bond (cylinder as McUtils draws them) per pair, by the chosen style.
  function updateForming(P, m) {
    var style = node("forming-style").value, cutoff = parseFloat(node("forming-cutoff").value), line = [];
    m.pairs.forEach(function (pair, k) {
      var a = P[pair[0]], b = P[pair[1]], d = norm(sub(b, a));
      var bond = style === "bond" || (style === "auto" && d < cutoff);
      var asLine = style === "line" || (style === "auto" && !bond);
      line.push(a, asLine ? b : a);          // a zero-length segment hides the line
      var sw = node("forming-bond-switch-" + k), t = node("forming-bond-" + k);
      if (!sw || !t) return;
      sw.setAttribute("whichChoice", bond ? "0" : "-1");
      if (bond) {
        var r = t.getAttribute("scale").split(/\s+/)[0];
        t.setAttribute("translation", vec(mid(a, b)));
        t.setAttribute("rotation", vec(alignY(unit(sub(b, a)))));
        t.setAttribute("scale", r + " " + d.toFixed(5) + " " + r);
      }
    });
    node("forming-coords").setAttribute("point", line.map(vec).join(" "));
    view.formingBonds = m.pairs.map(function (pair) {
      var d = norm(sub(P[pair[1]], P[pair[0]]));
      return style === "bond" || (style === "auto" && d < cutoff);
    });
  }
  function render() {
    var P, rows = [], ref = referenceResult(), rm = ref && ref.metrics;
    if (view.mode === "builder") {
      try { view.built = build(state); } catch (err) {
        document.getElementById("metrics").innerHTML = '<tr><td colspan="3">' + esc(err.message) + "</td></tr>"; return;
      }
      P = view.built.positions;
      showRigid(view.built);
    } else {
      P = view.frames[view.frame].positions;
      showPerAtom(P);
    }
    view.positions = P;
    var m = metrics(P);
    updateForming(P, m);
    rows.push('<tr><th></th><th>shown</th><th title="shown minus the reference chosen under Saved results">' +
              (ref ? "Δ vs " + esc(ref.name) : "") + "</th></tr>");
    if (view.mode === "builder") rows.push('<tr><td>height</td><td>' + fmt(view.built.height) + ' Å</td><td></td></tr>');
    METRIC_KEYS.forEach(function (k) {
      var L = LABELS[k], d = L[1] === "°" ? 1 : 3;
      var delta = rm && rm[k] !== undefined ? signed(m[k] - rm[k], d) + " " + L[1] : "";
      rows.push('<tr title="' + L[2] + '"><td>' + L[0] + '</td><td>' + fmt(m[k], d) + " " + L[1] + "</td><td>" + delta + "</td></tr>");
    });
    if (view.mode === "frames") {
      var f = view.frames[view.frame], e = shownEnergy(), dE = "";
      if (e && ref && ref.energy !== null && ref.energy !== undefined) {
        dE = comparable(e, ref) ? signed(e.energy - ref.energy, 4) + " eV<br><small>" + signed((e.energy - ref.energy) * KCAL, 2) + " kcal/mol</small>"
          : '<span title="different model, charge or spin than the reference: energies are not comparable">n/a</span>';
      }
      if (e) rows.push('<tr title="model energy (any restraint energy removed)"><td>energy</td><td>' + fmt(e.energy, 4) + ' eV</td><td>' + dE + "</td></tr>");
      if (view.frames.length > 1 && e) {
        var e0 = view.frames[0].energy - (view.frames[0].restraint_energy || 0);
        rows.push('<tr><td>ΔE from first frame</td><td>' + signed(e.energy - e0, 4) + ' eV<br><small>' + signed((e.energy - e0) * KCAL, 2) + ' kcal/mol</small></td><td></td></tr>');
      }
      if (f.restraint_energy) rows.push('<tr><td>restraint energy</td><td>' + fmt(f.restraint_energy, 4) + ' eV</td><td></td></tr>');
      if (f.fmax !== null && f.fmax !== undefined) rows.push('<tr><td>max force</td><td>' + fmt(f.fmax, 4) + ' eV/Å</td><td></td></tr>');
      var shownF = shownAnalysis().forces;
      if (shownF) {
        var norms = shownF.map(norm), rms = Math.sqrt(norms.reduce(function (a, v) { return a + v * v; }, 0) / norms.length);
        rows.push('<tr title="largest atomic force, any calculation on this geometry"><td>max |F|</td><td>' +
                  fmt(Math.max.apply(null, norms), 4) + ' eV/Å<br><small>RMS ' + fmt(rms, 4) + '</small></td><td></td></tr>');
      }
    }
    document.getElementById("metrics").innerHTML = rows.join("");
    updateAnalysisDisplay();
    var what = view.source && view.source.kind === "result" ? view.source.name : "optimization";
    document.getElementById("mode-label").textContent = view.mode === "builder" ? "builder" :
      what + " · frame " + view.frame + " / " + (view.frames.length - 1) + " (" + view.model + ")";
    document.getElementById("python-call").textContent = view.mode === "builder" ? pythonCall(state) :
      "# " + what + " (" + view.model + ", step " + view.frames[view.frame].step + "): use Download shown XYZ";
    markShown();
  }
  function toBuilder() {
    view.mode = "builder";
    document.getElementById("opt-builder").disabled = true;
    document.getElementById("download-path").disabled = true;
    render();
  }

  // --- builder controls ---
  var controls = document.getElementById("controls"), inputs = {};
  CONTROLS.forEach(function (c) {
    var row = document.createElement("div"); row.className = "control";
    row.innerHTML = '<label for="r-' + c.key + '">' + c.label + '</label>' +
      '<input type="range" id="r-' + c.key + '" min="' + c.min + '" max="' + c.max + '" step="' + c.step + '" value="' + c.value + '">' +
      '<input type="number" id="n-' + c.key + '" min="' + c.min + '" max="' + c.max + '" step="' + c.step + '" value="' + c.value + '">';
    controls.appendChild(row);
    var range = row.querySelector('input[type=range]'), number = row.querySelector('input[type=number]');
    inputs[c.key] = {range: range, number: number};
    range.addEventListener("input", function () {
      number.value = range.value; state[c.key] = parseFloat(range.value); view.mode = "builder"; render(); });
    number.addEventListener("change", function () {
      var v = parseFloat(number.value); if (!isFinite(v)) return;
      range.value = v; state[c.key] = v; view.mode = "builder"; render();
    });
  });
  document.getElementById("kind").addEventListener("change", function (e) {
    // switching the distance kind keeps the shown builder structure: the new distance is its measured value
    if (view.built) {
      var m = metrics(view.built.positions), kind = e.target.value;
      state.distance = kind === "height" ? view.built.height : kind === "centers" ? m.screen_distance : m.max_forming_distance;
    }
    state.kind = e.target.value; syncInputs(); view.mode = "builder"; render();
  });
  function syncInputs() {
    document.getElementById("kind").value = state.kind;
    CONTROLS.forEach(function (c) {
      var v = Math.round(state[c.key] / c.step) * c.step;
      inputs[c.key].range.value = v; inputs[c.key].number.value = +v.toFixed(4);
    });
  }
  D.presets.forEach(function (p) {
    var b = document.createElement("button"); b.textContent = p.name;
    b.addEventListener("click", function () {
      state.kind = "height"; state.distance = p.height; state.offset_x = p.offset_x || 0; state.offset_y = p.offset_y || 0;
      state.twist = p.twist || 0; state.roll = p.roll || 0; state.fold = p.fold || 0;
      syncInputs(); view.mode = "builder"; render();
    });
    document.getElementById("presets").appendChild(b);
  });
  document.querySelectorAll("[data-view]").forEach(function (b) {
    b.addEventListener("click", function () { document.getElementById(b.dataset.view).setAttribute("set_bind", "true"); });
  });
  document.getElementById("forming-style").addEventListener("change", function (e) {
    document.getElementById("forming-switch").setAttribute("whichChoice", e.target.value === "off" ? "-1" : "0");
    render();
  });
  document.getElementById("forming-cutoff").addEventListener("change", function () { render(); });
  document.getElementById("show-axes").addEventListener("change", function (e) {
    document.getElementById("axes-switch").setAttribute("whichChoice", e.target.checked ? "0" : "-1"); });
  function pythonCall(s) {
    return "from MartiniFuncs.chemistry.anthracene_ma.build import build_pair\n" +
      "atoms, geometry = build_pair(distance=" + fmt(s.distance, 4) + ', distance_kind="' + s.kind + '",\n' +
      "                             offset=(" + fmt(s.offset_x, 4) + ", " + fmt(s.offset_y, 4) + "), twist=" + fmt(s.twist, 2) +
      ", roll=" + fmt(s.roll, 2) + ", fold=" + fmt(s.fold, 2) + ")";
  }
  (function () {   // reference table
    var keys = ["screen_distance", "max_forming_distance", "distance_imbalance", "axis_angle", "plane_angle", "approach_angle"];
    document.getElementById("references").innerHTML =
      "<tr><th>structure</th>" + keys.map(function (k) { return "<th>" + LABELS[k][0] + "</th>"; }).join("") + "</tr>" +
      D.references.filter(function (r) { return r.positions; }).map(function (r) {
        return "<tr><td>" + r.name + "</td>" + keys.map(function (k) {
          return "<td>" + fmt(r[k], LABELS[k][1] === "°" ? 1 : 2) + "</td>"; }).join("") + "</tr>";
      }).join("");
  })();

  // --- export ---
  function xyzBlock(P, comment) {
    return [P.length, comment].concat(P.map(function (p, i) {
      return SYMBOLS[i] + " " + p.map(function (v) { return v.toFixed(8); }).join(" "); })).join("\n") + "\n";
  }
  function shownComment() {
    var m = metrics(view.positions), base = " screen_distance=" + fmt(m.screen_distance, 4) +
      " max_forming_distance=" + fmt(m.max_forming_distance, 4) + " rows: anthracene 1-" + NA + ", MA " + (NA + 1) + "-" + SYMBOLS.length;
    if (view.mode === "builder") {
      var s = state;
      return "anthracene+MA build_pair distance=" + fmt(s.distance, 4) + " distance_kind=" + s.kind +
        " height=" + fmt(view.built.height, 4) + " offset_x=" + fmt(s.offset_x, 4) + " offset_y=" + fmt(s.offset_y, 4) +
        " twist=" + fmt(s.twist, 2) + " roll=" + fmt(s.roll, 2) + " fold=" + fmt(s.fold, 2) + base;
    }
    var f = view.frames[view.frame];
    var name = view.source && view.source.kind === "result" ? ' result="' + view.source.name.replace(/"/g, "'") + '"' : "";
    return "anthracene+MA optimized model=" + view.model + name + " step=" + f.step +
      (f.energy === null || f.energy === undefined ? "" : " energy=" + fmt(f.energy, 6)) + base;
  }
  function download(name, text, type) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], {type: type || "chemical/x-xyz"}));
    a.download = name; document.body.appendChild(a); a.click(); a.remove();
  }
  window.builderState = function () {
    return {state: JSON.parse(JSON.stringify(state)), mode: view.mode, positions: view.positions,
            metrics: metrics(view.positions), xyz: xyzBlock(view.positions, shownComment()), frames: view.frames.length,
            source: view.source ? view.source.id || view.source.kind : null, formingBonds: view.formingBonds,
            metricsTable: document.getElementById("metrics").innerText};
  };
  function fileName(text) { return String(text).replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, ""); }
  window.setBuilder = function (params) { Object.assign(state, params); syncInputs(); view.mode = "builder"; render(); };
  document.getElementById("download-xyz").addEventListener("click", function () {
    var name = view.mode === "builder" ? "anthracene_ma_" + state.kind + "_" + fmt(state.distance, 2) + "_tw" + Math.round(state.twist) +
      "_ro" + Math.round(state.roll) + "_fo" + Math.round(state.fold) + ".xyz" :
      "anthracene_ma_" + fileName(view.source && view.source.kind === "result" ? view.source.name : view.model) +
      "_step" + view.frames[view.frame].step + ".xyz";
    download(name, xyzBlock(view.positions, shownComment()));
  });
  document.getElementById("download-path").addEventListener("click", function () {
    var label = view.source && view.source.kind === "result" ? view.source.name : view.model;
    download("anthracene_ma_" + fileName(label) + "_path.xyz", view.frames.map(function (f) {
      return xyzBlock(f.positions, "step=" + f.step + " energy=" + fmt(f.energy, 6) + " fmax=" + fmt(f.fmax, 5) + " model=" + view.model);
    }).join(""));
  });
  document.getElementById("copy-python").addEventListener("click", function () {
    navigator.clipboard.writeText(document.getElementById("python-call").textContent); });

  // --- optimizer (needs the server) ---
  var el = function (id) { return document.getElementById(id); };
  var poll = null;
  function setStatus(text, kind) { var s = el("opt-status"); s.textContent = text; s.className = "status " + (kind || ""); }
  function plot() {
    var svg = el("opt-plot"), F = view.frames;
    if (F.length < 2) { svg.innerHTML = ""; return; }
    var E = F.map(function (f) { return f.energy; }), lo = Math.min.apply(null, E), hi = Math.max.apply(null, E);
    var span = hi - lo || 1e-6, W = 360, H = 110, pad = 6;
    var pts = E.map(function (e, i) {
      return (pad + (W - 2 * pad) * i / (E.length - 1)).toFixed(1) + "," + (pad + (H - 2 * pad) * (hi - e) / span).toFixed(1); });
    var x = pad + (W - 2 * pad) * view.frame / (E.length - 1);
    svg.innerHTML = '<polyline fill="none" stroke="#c4561b" stroke-width="1.6" points="' + pts.join(" ") + '"/>' +
      '<line x1="' + x + '" x2="' + x + '" y1="0" y2="' + H + '" stroke="#5d6677" stroke-dasharray="3 3"/>' +
      '<text x="' + (W - 4) + '" y="12" text-anchor="end" font-size="10" fill="#5d6677">ΔE range ' +
      fmt((hi - lo) * 23.0605, 2) + ' kcal/mol</text>';
  }
  function showFrame(i) {
    view.frame = i; el("opt-frame").value = i;
    el("opt-frame-label").textContent = "frame " + i + " / " + (view.frames.length - 1);
    view.mode = "frames"; el("opt-builder").disabled = false; render(); plot();
  }
  function showFrames(frames, source, index) {   // frames of a job or a saved result, with what they are
    view.frames = frames; view.source = source; view.model = source.model;
    el("opt-frame").max = frames.length - 1; el("opt-frame").disabled = frames.length < 2;
    el("download-path").disabled = frames.length < 2;
    showFrame(index === undefined ? frames.length - 1 : index);
  }
  el("opt-frame").addEventListener("input", function (e) { showFrame(parseInt(e.target.value, 10)); });
  el("opt-builder").addEventListener("click", toBuilder);
  el("opt-constraint").addEventListener("change", function () {
    el("opt-k-row").style.display = el("opt-constraint").value === "restrain_distances" ? "" : "none"; });
  el("opt-k-row").style.display = "none";
  el("opt-atoms").addEventListener("change", function () {
    selected = new Set(el("opt-atoms").value.split(/[\s,;]+/).map(function (x) { return parseInt(x, 10) - 1; })
      .filter(function (i) { return i >= 0 && i < SYMBOLS.length; }));
    highlight();
  });
  function request() {
    var m = metrics(view.positions), mode = el("opt-constraint").value;
    var constraint = {mode: mode};
    if (mode === "fix_distances" || mode === "restrain_distances") {
      constraint.pairs = m.pairs;     // targets: the start distances (fix_distances records them; the engine keeps them)
      constraint.targets = m.pairs.map(function (p) { return norm(sub(view.positions[p[0]], view.positions[p[1]])); });
    }
    if (mode === "restrain_distances") constraint.k = parseFloat(el("opt-k").value);
    if (mode === "fix_atoms") constraint.atoms = Array.from(selected);
    return {model: el("opt-model").value, symbols: SYMBOLS, positions: view.positions, constraint: constraint,
            optimizer: el("opt-optimizer").value, fmax: parseFloat(el("opt-fmax").value),
            steps: parseInt(el("opt-steps").value, 10), charge: parseInt(el("opt-charge").value, 10),
            spin: parseInt(el("opt-spin").value, 10), sella_hessian: el("opt-sella-hessian").value,
            sella_hessian_every: parseInt(el("opt-sella-every").value, 10), hessian_method: el("hessian-method").value};
  }
  function sellaRows() {
    var sella = el("opt-optimizer").value.indexOf("sella") === 0;
    el("opt-sella-row").style.display = sella ? "" : "none";
    el("opt-sella-every-row").style.display = sella && el("opt-sella-hessian").value === "every" ? "" : "none";
  }
  el("opt-optimizer").addEventListener("change", sellaRows);
  el("opt-sella-hessian").addEventListener("change", sellaRows);
  sellaRows();
  function running(on) {
    el("opt-run").disabled = on; el("opt-cancel").disabled = !on;
  }
  function startLabel() {
    if (view.mode === "builder") {
      return "builder: " + state.kind + " " + fmt(state.distance, 3) + " Å, offset (" + fmt(state.offset_x, 2) + ", " +
        fmt(state.offset_y, 2) + "), twist " + fmt(state.twist, 1) + ", roll " + fmt(state.roll, 1) + ", fold " + fmt(state.fold, 1);
    }
    return (view.source && view.source.kind === "result" ? view.source.name : "unsaved optimization") + ", frame " + view.frame;
  }
  el("opt-run").addEventListener("click", function () {
    var body = request();
    var pending = {frames: [], request: body, startLabel: startLabel(),
                   parentId: view.mode === "frames" && view.source && view.source.kind === "result" ? view.source.id : null,
                   source: {kind: "job", model: body.model, charge: body.charge, spin: body.spin}};
    running(true); setStatus("submitting…");
    fetch("optimize", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.statusText); return j; }); })
      .then(function (j) {
        pending.id = j.job; pending.source.id = "job:" + j.job; job = pending;
        poll = setInterval(pollJob, 400);
      })
      .catch(function (err) { running(false); setStatus(err.message, "bad"); });
  });
  el("opt-cancel").addEventListener("click", function () {
    if (job) fetch("jobs/" + job.id + "/cancel", {method: "POST", body: "{}"});
  });
  function pollJob() {
    var current = job;
    fetch("jobs/" + current.id + "?since=" + current.frames.length).then(function (r) { return r.json(); }).then(function (j) {
      var watching = view.frames === current.frames;
      var follow = (current.frames.length === 0 && j.frames.length > 0) ||
        (watching && view.mode === "frames" && view.frame === current.frames.length - 1);
      Array.prototype.push.apply(current.frames, j.frames);
      if (follow) showFrames(current.frames, current.source);
      else if (watching && current.frames.length) { el("opt-frame").max = current.frames.length - 1; el("opt-frame").disabled = false; plot(); }
      var last = current.frames[current.frames.length - 1];
      var info = last ? " · step " + last.step + ", max force " + fmt(last.fmax, 3) + " eV/Å" : "";
      var done = ["converged", "not_converged", "cancelled", "error"].indexOf(j.status) >= 0;
      setStatus(j.message + info, done ? (j.status === "converged" ? "good" : j.status === "error" ? "bad" : "") : "");
      if (done) {
        clearInterval(poll); poll = null; running(false);
        if (j.status !== "error" && current.frames.length) {
          var result = resultFromJob(current, j.status);
          saveResult(result);
          if (view.frames === current.frames) { view.source = sourceOf(result); render(); }
          setStatus(j.message + info + " · saved as “" + result.name + "”", j.status === "converged" ? "good" : "");
        }
      }
    }).catch(function (err) { setStatus("lost contact with the server: " + err.message, "bad"); });
  }
  fetch("models").then(function (r) { if (!r.ok) throw new Error(); return r.json(); }).then(function (models) {
    var select = el("opt-model"), any = false;
    Object.keys(models).forEach(function (name) {
      var m = models[name], o = document.createElement("option");
      o.value = name; o.textContent = m.label + (m.available ? "" : " (unavailable)");
      o.disabled = !m.available; o.title = m.reason || ""; select.appendChild(o);
      if (m.available && !any) { select.value = name; any = true; }
    });
    el("opt-offline").style.display = "none";
    if (!any) setStatus("no model available on the server", "bad");
  }).catch(function () {
    el("opt-controls").querySelectorAll("input, select, button").forEach(function (x) { x.disabled = true; });
    ["calc-forces", "calc-hessian", "hessian-method"].forEach(function (id) { el(id).disabled = true; });
    el("analysis-status").textContent = "Calculations need the server; saved forces and modes still display.";
  });

  // --- forces and vibrations of the shown geometry ---
  var anim = {on: false, raf: 0, start: 0, last: 0}, analysisJob = null;
  function hideArrows(kind) {
    for (var i = 0; i < SYMBOLS.length; i++) { var sw = node(kind + "-arrow-switch-" + i); if (sw) sw.setAttribute("whichChoice", "-1"); }
  }
  function setArrow(kind, i, origin, v) {
    var sw = node(kind + "-arrow-switch-" + i), length = norm(v);
    if (!sw) return;
    if (length < 0.03) { sw.setAttribute("whichChoice", "-1"); return; }
    var head = Math.min(0.3, 0.45 * length), shaft = length - head;
    node(kind + "-arrow-" + i).setAttribute("translation", vec(origin));
    node(kind + "-arrow-" + i).setAttribute("rotation", vec(alignY(unit(v))));
    node(kind + "-arrow-" + i + "-shaft").setAttribute("translation", "0 " + (shaft / 2).toFixed(4) + " 0");
    node(kind + "-arrow-" + i + "-shaft").setAttribute("scale", "0.045 " + Math.max(shaft, 1e-4).toFixed(4) + " 0.045");
    node(kind + "-arrow-" + i + "-head").setAttribute("translation", "0 " + (shaft + head / 2).toFixed(4) + " 0");
    node(kind + "-arrow-" + i + "-head").setAttribute("scale", "1 " + (head / 0.3).toFixed(4) + " 1");
    sw.setAttribute("whichChoice", "0");
  }
  // What is known for the shown geometry: frame forces, or the shown saved result's own calculations.
  function shownAnalysis() {
    if (view.mode !== "frames" || !view.source || !view.frames.length) return {};
    var f = view.frames[view.frame], out = {};
    var r = view.source.kind === "result" ? findResult(view.source.id) : null;
    if (r && r.analysis && view.frame === view.frames.length - 1) {
      out.vib = r.analysis.vib || null;
      if (r.analysis.forces) out.forces = r.analysis.forces.forces;
    }
    if (!out.forces && f.forces) out.forces = f.forces;
    out.result = r;
    return out;
  }
  function vibLabel(f) { return f < 0 ? fmt(-f, 1) + "i cm⁻¹" : fmt(f, 1) + " cm⁻¹"; }
  var vibShown = null;     // the vib object the mode list was built for
  function updateAnalysisDisplay() {
    var a = shownAnalysis(), P = view.positions;
    // force arrows
    if (a.forces && el("show-forces").checked) {
      var big = Math.max.apply(null, a.forces.map(norm)) || 1;
      var k = el("force-auto").checked ? 1.5 / big : parseFloat(el("force-scale").value) || 1;
      a.forces.forEach(function (F, i) { setArrow("force", i, P[i], scale(F, k)); });
      el("force-info").textContent = "Force arrows: " + fmt(k, 2) + " Å per eV/Å; largest " + fmt(big, 4) + " eV/Å.";
    } else {
      hideArrows("force");
      el("force-info").textContent = a.forces ? "" : "No forces for the shown geometry: Calculate forces, or show an optimization frame.";
    }
    // modes
    if (a.vib !== vibShown) {
      vibShown = a.vib || null;
      stopAnimation(true);
      el("vib-box").hidden = !vibShown;
      if (vibShown) {
        var select = el("vib-mode"), first = -1;
        select.innerHTML = "";
        vibShown.frequencies.forEach(function (f, j) {
          var o = document.createElement("option");
          o.value = j; o.textContent = (j + 1) + ": " + vibLabel(f) + (f < 0 ? " (imaginary)" : "");
          select.appendChild(o);
          if (f < 0 && first < 0) first = j;
        });
        select.value = first >= 0 ? first : 0;
      }
    }
    if (vibShown) drawMode(); else hideArrows("mode");
  }
  function currentMode() { return vibShown ? vibShown.modes[parseInt(el("vib-mode").value, 10) || 0] : null; }
  function drawMode() {
    var v = vibShown, j = parseInt(el("vib-mode").value, 10) || 0, amp = parseFloat(el("vib-amp").value), mode = v.modes[j];
    el("vib-amp-label").textContent = amp.toFixed(2) + " Å";
    if (el("vib-arrows").checked) {
      mode.forEach(function (d, i) { setArrow("mode", i, view.positions[i], scale(d, 3 * amp)); });
    } else hideArrows("mode");
    var imaginary = v.frequencies.filter(function (f) { return f < 0; }).length;
    var lines = ["<b>Mode " + (j + 1) + " of " + v.frequencies.length + "</b>: " + vibLabel(v.frequencies[j]) +
                 " · μ " + fmt(v.reduced_masses[j], 2) + " amu · k " + fmt(v.force_constants[j], 3) + " mdyn/Å",
                 imaginary + " imaginary · Hessian " + (v.method === "analytic" ? "analytic" : "finite differences") +
                 (v.requested === "auto" ? " (auto)" : "") + " · " + esc(v.model) + (v.check !== null && v.check !== undefined ?
                 " · FD check " + fmt(100 * v.check, 2) + "%" : "")];
    if (v.note) lines.push('<span class="warn">' + esc(v.note) + "</span>");
    if (v.fmax !== undefined && v.fmax > 0.05) lines.push('<span class="warn">max |F| ' + fmt(v.fmax, 3) +
      " eV/Å: not a stationary point, so the frequencies are approximate (low modes most).</span>");
    el("vib-info").innerHTML = lines.join("<br>");
  }
  function frameLoop(ts) {
    if (!anim.on) return;
    anim.raf = requestAnimationFrame(frameLoop);
    if (ts - anim.last < 33) return;
    anim.last = ts;
    var mode = currentMode(), amp = parseFloat(el("vib-amp").value), P0 = view.positions;
    if (!mode) return stopAnimation();
    var w = amp * Math.sin(2 * Math.PI * (ts - anim.start) / 1400);
    var P = P0.map(function (p, i) { return add(p, scale(mode[i], w)); });
    showPerAtom(P);
    updateForming(P, metrics(P));
  }
  function startAnimation() {
    anim.on = true; anim.start = performance.now(); anim.last = 0;
    el("vib-play").textContent = "Pause";
    anim.raf = requestAnimationFrame(frameLoop);
  }
  function stopAnimation(quiet) {
    var was = anim.on;
    anim.on = false; cancelAnimationFrame(anim.raf);
    el("vib-play").textContent = "Play";
    if (was && !quiet) {      // back to the resting geometry
      if (view.mode === "builder") showRigid(view.built); else showPerAtom(view.positions);
      updateForming(view.positions, metrics(view.positions));
    }
  }
  el("vib-play").addEventListener("click", function () { if (anim.on) stopAnimation(); else startAnimation(); });
  el("vib-mode").addEventListener("change", function () { if (vibShown) drawMode(); });
  el("vib-amp").addEventListener("input", function () { if (vibShown) drawMode(); });
  el("vib-arrows").addEventListener("change", function () { if (vibShown) drawMode(); });
  ["show-forces", "force-auto"].forEach(function (id) { el(id).addEventListener("change", updateAnalysisDisplay); });
  el("force-scale").addEventListener("change", function () { el("force-auto").checked = false; updateAnalysisDisplay(); });

  function analysisStatus(text, kind) { var s = el("analysis-status"); s.textContent = text; s.className = "status " + (kind || ""); }
  function runAnalysis(task) {
    if (analysisJob) return;
    var r = view.mode === "frames" && view.source && view.source.kind === "result" ? findResult(view.source.id) : null;
    var body = {model: el("opt-model").value, symbols: SYMBOLS, positions: view.positions,
                charge: parseInt(el("opt-charge").value, 10), spin: parseInt(el("opt-spin").value, 10)};
    if (task === "hessian") body.method = el("hessian-method").value;
    var target = {task: task, body: body, label: startLabel(), parentId: r ? r.id : null,
                  attach: r && !r.builtin && view.frame === view.frames.length - 1 && comparable(r, body) ? r : null,
                  base: r ? r.name : view.mode === "builder" ? "builder pose" : "optimization frame " + view.frame};
    analysisJob = target;
    el("calc-forces").disabled = el("calc-hessian").disabled = true;
    analysisStatus("submitting…");
    send("POST", task, body).then(function (j) {
      target.id = j.job;
      var timer = setInterval(function () {
        fetch("jobs/" + j.job + "?since=0").then(function (x) { return x.json(); }).then(function (state) {
          var done = ["done", "cancelled", "error"].indexOf(state.status) >= 0;
          analysisStatus(state.message, state.status === "error" ? "bad" : state.status === "done" ? "good" : "");
          if (!done) return;
          clearInterval(timer); finishAnalysis();
          if (state.status === "done") storeAnalysis(target, state.result);
        }).catch(function (err) { clearInterval(timer); finishAnalysis(); analysisStatus("lost contact with the server: " + err.message, "bad"); });
      }, 500);
    }).catch(function (err) { finishAnalysis(); analysisStatus(err.message, "bad"); });
  }
  function finishAnalysis() { analysisJob = null; el("calc-forces").disabled = el("calc-hessian").disabled = false; }
  function storeAnalysis(t, res) {
    var b = t.body, stamp = {model: b.model, charge: b.charge, spin: b.spin, computed: new Date().toISOString()};
    var forces = Object.assign({energy: res.energy, forces: res.forces, fmax: res.fmax}, stamp);
    var vib = t.task === "hessian" ? Object.assign({method: res.method, requested: res.requested, step: res.step,
      check: res.check, note: res.note, energy: res.energy, fmax: res.fmax, frequencies: res.frequencies,
      reduced_masses: res.reduced_masses, force_constants: res.force_constants, modes: res.modes, masses: res.masses,
      hessian: res.hessian}, stamp) : null;
    var r = t.attach;
    if (r) {
      r.analysis = r.analysis || {};
      r.analysis.forces = forces;
      if (vib) r.analysis.vib = vib;
      saveResult(r); renderResults(); render();
      analysisStatus(el("analysis-status").textContent + " · stored with “" + r.name + "”", "good");
      return;
    }
    r = {id: "r" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
         name: t.base + " · " + b.model + (vib ? " Hessian" : " forces"), created: stamp.computed,
         model: b.model, charge: b.charge, spin: b.spin, status: "single point", steps: 0,
         energy: res.energy, restraint_energy: 0, fmax: res.fmax, fmax_kind: "raw",
         positions: b.positions, metrics: metrics(b.positions), start: t.label, parent: t.parentId, path: null,
         analysis: {forces: forces, vib: vib}};
    saveResult(r); showResult(r);
    analysisStatus(el("analysis-status").textContent + " · saved as “" + r.name + "”", "good");
  }
  el("calc-forces").addEventListener("click", function () { runAnalysis("forces"); });
  el("calc-hessian").addEventListener("click", function () { runAnalysis("hessian"); });

  // --- saved results: the built-in UMA references plus every finished optimization ---
  var RESULTS = [], refId = "ref:reactant_complex", store = {mode: "page", location: ""};
  D.references.forEach(function (r) {
    RESULTS.push({id: "ref:" + r.key, name: r.name, builtin: true, model: r.model, charge: r.charge, spin: r.spin,
                  energy: r.energy, restraint_energy: 0, fmax: null, status: "reference", positions: r.positions,
                  metrics: r.positions ? metrics(r.positions) : null,
                  start: "UMA-s-1p2p1 hand-off structure" + (r.positions ? "" : " (energy of anthracene + MA, apart)")});
  });
  function findResult(id) {
    for (var i = 0; i < RESULTS.length; i++) if (RESULTS[i].id === id) return RESULTS[i];
    return null;
  }
  function referenceResult() { return findResult(refId); }
  function sourceOf(r) { return {kind: "result", id: r.id, name: r.name, model: r.model, charge: r.charge, spin: r.spin}; }
  var OPTIMIZER_LABELS = {BFGS: "BFGS", LBFGS: "LBFGS", FIRE: "FIRE", sella_min: "Sella min", sella_ts: "Sella TS"};
  function constraintText(c) {
    if (!c) return "";
    var d = (c.targets || []).map(function (x) { return fmt(x, 2); }).join("/");
    if (c.mode === "fix_distances") return "fixed C···C " + d + " Å";
    if (c.mode === "restrain_distances") return "restrained C···C " + d + " Å (k " + c.k + ")";
    if (c.mode === "fix_atoms") return "fixed atoms " + c.atoms.map(function (i) { return i + 1; }).join(",");
    return "free";
  }
  function resultFromJob(run, status) {
    var req = run.request, last = run.frames[run.frames.length - 1];
    var mine = RESULTS.filter(function (r) { return !r.builtin; }).length;
    var keep = el("results-path").checked;
    return {
      id: "r" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      name: req.model + " " + OPTIMIZER_LABELS[req.optimizer] + ", " + constraintText(req.constraint) + " #" + (mine + 1),
      created: new Date().toISOString(), job: run.id, model: req.model, charge: req.charge, spin: req.spin,
      optimizer: req.optimizer, fmax_target: req.fmax, max_steps: req.steps, constraint: req.constraint,
      sella_hessian: req.optimizer.indexOf("sella") === 0 ? req.sella_hessian : undefined,
      sella_hessian_every: req.sella_hessian === "every" ? req.sella_hessian_every : undefined,
      hessian_method: req.optimizer.indexOf("sella") === 0 && req.sella_hessian !== "sella" ? req.hessian_method : undefined,
      status: status, converged: status === "converged", steps: last.step,
      energy: last.energy - (last.restraint_energy || 0), restraint_energy: last.restraint_energy || 0,
      fmax: last.fmax, fmax_kind: req.optimizer.indexOf("sella") === 0 ? "raw" : "constraint-projected",
      positions: last.positions, metrics: metrics(last.positions), start: run.startLabel, parent: run.parentId,
      path: keep ? run.frames.map(function (f) {
        return {step: f.step, energy: f.energy, restraint_energy: f.restraint_energy || 0, fmax: f.fmax, positions: f.positions}; }) : null
    };
  }
  function resultsMessage(text, kind) { var s = el("results-msg"); s.textContent = text; s.className = "status " + (kind || ""); }
  function send(method, path, body) {
    return fetch(path, {method: method, headers: {"Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined})
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.statusText); return j; }); });
  }
  function saveResult(r) {
    if (!findResult(r.id)) RESULTS.push(r);
    renderResults();
    if (store.mode !== "page") send("POST", "results", r).catch(function (err) {
      resultsMessage("could not save “" + r.name + "” on the server: " + err.message + " (export to keep it)", "bad"); });
  }
  function deleteResult(r) {
    RESULTS.splice(RESULTS.indexOf(r), 1);
    if (refId === r.id) refId = "ref:reactant_complex";
    renderResults(); render();
    if (store.mode !== "page") send("DELETE", "results/" + encodeURIComponent(r.id)).catch(function (err) {
      resultsMessage("could not delete on the server: " + err.message, "bad"); });
  }
  function showResult(r) {
    if (!r || !r.positions) return;
    var frames = r.path && r.path.length ? r.path : [{step: r.steps || 0, energy: r.energy === null ? null : r.energy + (r.restraint_energy || 0),
                                                       restraint_energy: r.restraint_energy || 0, fmax: r.fmax, positions: r.positions}];
    showFrames(frames, sourceOf(r));
  }
  function setValue(id, value) { if (value !== undefined && value !== null) el(id).value = value; }
  function useSettings(r) {
    var option = el("opt-model").querySelector('option[value="' + r.model + '"]');
    if (option && !option.disabled) el("opt-model").value = r.model;
    else resultsMessage("model " + r.model + " is not available on this server; the other settings were copied", "bad");
    setValue("opt-optimizer", r.optimizer); setValue("opt-fmax", r.fmax_target); setValue("opt-steps", r.max_steps);
    setValue("opt-sella-hessian", r.sella_hessian); setValue("opt-sella-every", r.sella_hessian_every);
    setValue("hessian-method", r.hessian_method); sellaRows();
    setValue("opt-charge", r.charge); setValue("opt-spin", r.spin);
    if (r.constraint) {
      setValue("opt-constraint", r.constraint.mode); setValue("opt-k", r.constraint.k);
      el("opt-k-row").style.display = r.constraint.mode === "restrain_distances" ? "" : "none";
      if (r.constraint.mode === "fix_atoms") { selected = new Set(r.constraint.atoms); highlight(); }
    }
  }
  function energyCell(r, ref) {
    if (r.energy === null || r.energy === undefined) return "—";
    var absolute = ' title="E = ' + fmt(r.energy, 5) + ' eV"';
    if (!ref || ref.energy === null || ref.energy === undefined) return "<span" + absolute + ">—</span>";
    if (!comparable(r, ref)) return '<span title="different model, charge or spin than the reference: energies are not comparable">n/a</span>';
    var d = r.energy - ref.energy;
    return "<span" + absolute + ">" + signed(d * KCAL, 2) + "<br><small>" + signed(d, 4) + " eV</small></span>";
  }
  function renderResults() {
    var ref = referenceResult();
    var rows = ['<tr><th title="reference for ΔE and metric differences">ref</th><th>result</th>' +
                '<th title="vs the reference">ΔE kcal/mol</th><th title="max forming C···C">d<sub>max</sub> Å</th><th></th></tr>'];
    RESULTS.forEach(function (r) {
      var detail = r.builtin ? r.start : [r.status ? r.status + (r.steps !== undefined ? ", " + r.steps + " steps" : "") : "",
        r.optimizer && OPTIMIZER_LABELS[r.optimizer],
        r.sella_hessian === "model" ? "model Hessian" : r.sella_hessian === "every" ? "model Hessian every " + r.sella_hessian_every : "",
        constraintText(r.constraint), r.fmax !== null && r.fmax !== undefined ? "fmax " + fmt(r.fmax, 3) + " (" + (r.fmax_kind === "raw" ? "raw" : "projected") + ")" : "",
        r.analysis && r.analysis.vib ? r.analysis.vib.frequencies.length + " modes, " +
          r.analysis.vib.frequencies.filter(function (f) { return f < 0; }).length + " imaginary" :
          r.analysis && r.analysis.forces ? "forces" : "",
        r.parent ? "from " + ((findResult(r.parent) || {}).name || r.parent) : "", r.created ? localTime(r.created) : ""]
        .filter(Boolean).join(" · ");
      var model = r.model + " · q " + r.charge + ", 2S+1 " + r.spin;
      rows.push('<tr data-id="' + esc(r.id) + '"><td><input type="radio" name="result-ref" value="' + esc(r.id) + '"' +
        (r.id === refId ? " checked" : "") + '></td><td><span class="name"' + (r.builtin ? "" : ' contenteditable="true" spellcheck="false"') +
        ">" + esc(r.name) + "</span><small>" + esc(model) + "</small><small>" + esc(detail) + "</small></td><td>" +
        (r.id === refId ? "ref" : energyCell(r, ref)) + "</td><td>" +
        (r.metrics ? fmt(r.metrics.max_forming_distance, 3) : "—") + '</td><td class="acts">' +
        (r.positions ? '<button class="mini" data-act="show">Show</button>' : "") +
        (r.builtin ? "" : '<button class="mini" data-act="settings" title="copy the model and settings to the Optimize panel">Settings</button>' +
          '<button class="mini" data-act="delete" title="delete this result">✕</button>') + "</td></tr>");
    });
    el("results").innerHTML = rows.join("");
    el("results-store").textContent = store.mode === "folder" ? "Results are saved on the server in " + store.location + "." :
      store.mode === "server" ? "Results are kept in the server's memory until it stops: export them to keep them, or start the server with --results DIR." :
      "Results are kept in this page only: export them to keep them.";
    markShown();
  }
  function localTime(iso) {
    var d = new Date(iso);
    if (isNaN(d)) return "";
    var two = function (n) { return (n < 10 ? "0" : "") + n; };
    return d.getFullYear() + "-" + two(d.getMonth() + 1) + "-" + two(d.getDate()) + " " + two(d.getHours()) + ":" + two(d.getMinutes());
  }
  function markShown() {
    var id = view.mode === "frames" && view.source ? view.source.id : null;
    document.querySelectorAll("#results tr[data-id]").forEach(function (tr) { tr.classList.toggle("shown", tr.dataset.id === id); });
  }
  el("results").addEventListener("change", function (e) {
    if (e.target.name === "result-ref") { refId = e.target.value; renderResults(); render(); }
  });
  el("results").addEventListener("click", function (e) {
    var button = e.target.closest("button[data-act]");
    if (!button) return;
    var r = findResult(button.closest("tr").dataset.id);
    if (!r) return;
    if (button.dataset.act === "show") showResult(r);
    else if (button.dataset.act === "settings") useSettings(r);
    else if (button.dataset.act === "delete") {
      if (button.dataset.armed) deleteResult(r);
      else { button.dataset.armed = "1"; button.textContent = "delete?"; setTimeout(function () { button.textContent = "✕"; delete button.dataset.armed; }, 3000); }
    }
  });
  el("results").addEventListener("keydown", function (e) {
    if (e.target.classList.contains("name") && e.key === "Enter") { e.preventDefault(); e.target.blur(); }
  });
  el("results").addEventListener("focusout", function (e) {
    if (!e.target.classList.contains("name")) return;
    var r = findResult(e.target.closest("tr").dataset.id), name = e.target.textContent.trim();
    if (!r || r.builtin) return;
    if (name && name !== r.name) {
      r.name = name;
      if (view.source && view.source.id === r.id) { view.source.name = name; render(); }
      saveResult(r);
    } else e.target.textContent = r.name;
  });
  el("results-export").addEventListener("click", function () {
    var mine = RESULTS.filter(function (r) { return !r.builtin; });
    download("anthracene_ma_results.json", JSON.stringify({format: "MartiniFuncs anthracene_ma results", version: 1,
      exported: new Date().toISOString(), symbols: SYMBOLS, results: mine}), "application/json");
    resultsMessage("exported " + mine.length + " result" + (mine.length === 1 ? "" : "s"));
  });
  el("results-import-button").addEventListener("click", function () { el("results-import").click(); });
  function importResults(text) {
    var data = JSON.parse(text), list = Array.isArray(data) ? data : data.results, added = 0, skipped = 0;
    if (!Array.isArray(list)) throw new Error("no results in this file");
    list.forEach(function (r) {
      var ok = r && typeof r.id === "string" && !/^ref:/.test(r.id) && Array.isArray(r.positions) && r.positions.length === SYMBOLS.length;
      if (!ok || findResult(r.id)) { skipped++; return; }
      r.metrics = metrics(r.positions); r.builtin = false;
      saveResult(r); added++;
    });
    return {added: added, skipped: skipped};
  }
  window.importResults = importResults;
  el("results-import").addEventListener("change", function (e) {
    var file = e.target.files[0];
    if (!file) return;
    file.text().then(function (text) {
      var n = importResults(text);
      resultsMessage("imported " + n.added + " result" + (n.added === 1 ? "" : "s") + (n.skipped ? ", skipped " + n.skipped + " (already present or not this system)" : ""), "good");
    }).catch(function (err) { resultsMessage("import failed: " + err.message, "bad"); });
    e.target.value = "";
  });
  window.pageResults = function () {
    return {ref: refId, store: store.mode, results: RESULTS.map(function (r) {
      return {id: r.id, name: r.name, builtin: !!r.builtin, model: r.model, energy: r.energy, status: r.status,
              max_forming_distance: r.metrics ? r.metrics.max_forming_distance : null, path: r.path ? r.path.length : 0}; })};
  };
  window.showResult = function (id) { showResult(findResult(id)); };
  window.analysisState = function () {
    var a = shownAnalysis();
    var shown = function (kind) { var n = 0; for (var i = 0; i < SYMBOLS.length; i++) {
      var sw = node(kind + "-arrow-switch-" + i); if (sw && sw.getAttribute("whichChoice") === "0") n++; } return n; };
    return {forces: !!a.forces, modes: a.vib ? a.vib.frequencies.length : 0, animating: anim.on,
            forceArrows: shown("force"), modeArrows: shown("mode"), busy: !!analysisJob,
            status: el("analysis-status").textContent, vibInfo: el("vib-info").innerText};
  };
  window.setReference = function (id) { refId = id; renderResults(); render(); };
  fetch("results").then(function (r) { if (!r.ok) throw new Error(); return r.json(); }).then(function (j) {
    store = {mode: j.persistent ? "folder" : "server", location: j.location || ""};
    (j.results || []).forEach(function (r) { if (!findResult(r.id)) { r.metrics = metrics(r.positions); RESULTS.push(r); } });
    renderResults();
  }).catch(function () { renderResults(); });

  highlight();
  renderResults();
  render();
})();
</script>
</body>
</html>
"""


def main(argv=None):
    """python -m MartiniFuncs.chemistry.anthracene_ma.viewer OUTPUT.html [--rdkit]"""
    import argparse
    from MartiniFuncs.chemistry.anthracene_ma.build import rdkit_templates
    parser = argparse.ArgumentParser(description="Write the interactive anthracene + MA pair-builder page.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rdkit", action="store_true", help="RDKit/MMFF templates instead of the UMA-relaxed ones")
    args = parser.parse_args(argv)
    frames = PairFrames(*rdkit_templates()) if args.rdkit else PairFrames()
    print(write_builder_html(args.output, frames).resolve())


if __name__ == "__main__":
    main()
