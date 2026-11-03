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

The page recomputes the shared Diels-Alder metrics (as structure_metrics) for
every shown geometry, exports it (or the whole optimization path) as XYZ, and
prints the build_pair call for builder geometries. Opened as a file it works as
the builder alone. It loads x3dom from x3dom.org, so it needs an internet
connection.
"""
import html
import json
import re
from pathlib import Path

import numpy as np

from MartiniFuncs.chemistry.anthracene_ma.build import (PairFrames, measure_placement, reference_structure,
                                                        structure_metrics)

BONDED = 1.9      # Angstrom: atom pairs considered when tying a cylinder to its bond


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


def default_presets(frames):
    """Starting placements: the stacked pose, two contrasts, and the UMA reference complex and TS."""
    presets = [{"name": "Stacked (reactive)", "height": 3.2},
               {"name": "C=C crossed (twist 90)", "height": 3.3, "twist": 90.0},
               {"name": "Ring edge-on (fold 90)", "height": 3.6, "fold": 90.0}]
    for name, label in (("reactant_complex", "UMA reactant complex"), ("transition_state", "UMA TS")):
        placement = measure_placement(reference_structure(name))
        presets.append({"name": label + " (rigid)", **{k: round(v, 3) for k, v in placement.items()
                                                       if k != "screen_distance"}})
    return presets


def builder_page(frames=None, *, presets=None, title="Anthracene + MA Pair Builder"):
    """The interactive builder page as an HTML string.

    frames is a build.PairFrames (default: the UMA-relaxed templates); presets
    a list of {"name", "height", "offset_x", "offset_y", "twist", "roll", "fold"}
    placements (default: default_presets). Reference rows show the metrics of
    the flexible UMA reactant complex and TS (structure_metrics).
    """
    from ase import Atoms
    frames = frames or PairFrames()
    presets = default_presets(frames) if presets is None else presets
    anthracene = Atoms(frames.anthracene.get_chemical_symbols(), positions=frames.anthracene_xyz)
    anhydride = Atoms(frames.anhydride.get_chemical_symbols(), positions=frames.anhydride_xyz)
    n = len(anthracene)
    ant_markup, ant_spheres, ant_cylinders = scene(anthracene, 0)
    ma_markup, ma_spheres, ma_cylinders = scene(anhydride, n)
    references = [{"name": label, **structure_metrics(reference_structure(name))}
                  for name, label in (("reactant_complex", "UMA reactant complex"),
                                      ("transition_state", "UMA TS"))]
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
<link rel="stylesheet" href="https://www.x3dom.org/download/x3dom.css">
<script src="https://www.x3dom.org/download/1.8.3/x3dom-full.js"></script>
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
        <label><input type="checkbox" id="show-forming" checked> forming C&middot;&middot;&middot;C</label>
        <label><input type="checkbox" id="show-axes"> frame axes</label>
      </span>
    </div>
    <div class="view-bar legend">
      <span class="mode" id="mode-label">builder</span>
      <span><i class="swatch" style="background:#d9590d"></i>forming C&middot;&middot;&middot;C pairs (minimax matching)</span>
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

    <section class="panel">
      <h2>Measured metrics</h2>
      <table id="metrics"></table>
      <h2 style="margin-top:14px">Reference structures (flexible UMA)</h2>
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
  var view = {mode: "builder", positions: null, built: null, frames: [], frame: 0, job: null, model: null};
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
  function render() {
    var P, heightRow = "";
    if (view.mode === "builder") {
      try { view.built = build(state); } catch (err) {
        document.getElementById("metrics").innerHTML = '<tr><td colspan="2">' + err.message + "</td></tr>"; return;
      }
      P = view.built.positions;
      showRigid(view.built);
      heightRow = '<tr><td>height</td><td>' + fmt(view.built.height) + ' Å</td></tr>';
    } else {
      P = view.frames[view.frame].positions;
      showPerAtom(P);
    }
    view.positions = P;
    var m = metrics(P);
    var lines = [P[m.pairs[0][0]], P[m.pairs[0][1]], P[m.pairs[1][0]], P[m.pairs[1][1]]];
    document.getElementById("forming-coords").setAttribute("point", lines.map(vec).join(" "));
    var rows = [heightRow];
    ["screen_distance", "max_forming_distance", "forming_1", "forming_2", "distance_imbalance", "axis_angle",
     "plane_angle", "approach_angle", "min_intermolecular_distance"].forEach(function (k) {
      var L = LABELS[k];
      rows.push('<tr title="' + L[2] + '"><td>' + L[0] + '</td><td>' + fmt(m[k], L[1] === "°" ? 1 : 3) + " " + L[1] + "</td></tr>");
    });
    if (view.mode === "frames") {
      var f = view.frames[view.frame], e0 = view.frames[0].energy;
      rows.push('<tr><td>energy</td><td>' + fmt(f.energy, 4) + ' eV</td></tr>');
      rows.push('<tr><td>ΔE from start</td><td>' + fmt(f.energy - e0, 4) + ' eV (' + fmt((f.energy - e0) * 23.0605, 2) + ' kcal/mol)</td></tr>');
      if (f.restraint_energy) rows.push('<tr><td>restraint energy</td><td>' + fmt(f.restraint_energy, 4) + ' eV</td></tr>');
      rows.push('<tr><td>max force</td><td>' + fmt(f.fmax, 4) + ' eV/Å</td></tr>');
    }
    document.getElementById("metrics").innerHTML = rows.join("");
    document.getElementById("mode-label").textContent = view.mode === "builder" ? "builder" :
      "optimization frame " + view.frame + " / " + (view.frames.length - 1) + " (" + view.model + ")";
    document.getElementById("python-call").textContent = view.mode === "builder" ? pythonCall(state) :
      "# optimized geometry (" + view.model + ", step " + view.frames[view.frame].step + "): use Download shown XYZ";
  }
  function toBuilder() {
    view.mode = "builder";
    document.getElementById("opt-builder").disabled = true;
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
  document.getElementById("show-forming").addEventListener("change", function (e) {
    document.getElementById("forming-switch").setAttribute("whichChoice", e.target.checked ? "0" : "-1"); });
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
      D.references.map(function (r) {
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
    return "anthracene+MA optimized model=" + view.model + " step=" + f.step + " energy=" + fmt(f.energy, 6) + base;
  }
  function download(name, text) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], {type: "chemical/x-xyz"}));
    a.download = name; document.body.appendChild(a); a.click(); a.remove();
  }
  window.builderState = function () {
    return {state: JSON.parse(JSON.stringify(state)), mode: view.mode, positions: view.positions,
            metrics: metrics(view.positions), xyz: xyzBlock(view.positions, shownComment()), frames: view.frames.length};
  };
  window.setBuilder = function (params) { Object.assign(state, params); syncInputs(); view.mode = "builder"; render(); };
  document.getElementById("download-xyz").addEventListener("click", function () {
    var name = view.mode === "builder" ? "anthracene_ma_" + state.kind + "_" + fmt(state.distance, 2) + "_tw" + Math.round(state.twist) +
      "_ro" + Math.round(state.roll) + "_fo" + Math.round(state.fold) + ".xyz" : "anthracene_ma_" + view.model + "_step" + view.frames[view.frame].step + ".xyz";
    download(name, xyzBlock(view.positions, shownComment()));
  });
  document.getElementById("download-path").addEventListener("click", function () {
    download("anthracene_ma_" + view.model + "_path.xyz", view.frames.map(function (f) {
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
    if (mode === "fix_distances" || mode === "restrain_distances") constraint.pairs = m.pairs;
    if (mode === "restrain_distances") {
      constraint.k = parseFloat(el("opt-k").value);
      constraint.targets = m.pairs.map(function (p) { return norm(sub(view.positions[p[0]], view.positions[p[1]])); });
    }
    if (mode === "fix_atoms") constraint.atoms = Array.from(selected);
    return {model: el("opt-model").value, symbols: SYMBOLS, positions: view.positions, constraint: constraint,
            optimizer: el("opt-optimizer").value, fmax: parseFloat(el("opt-fmax").value),
            steps: parseInt(el("opt-steps").value, 10), charge: parseInt(el("opt-charge").value, 10),
            spin: parseInt(el("opt-spin").value, 10)};
  }
  function running(on) {
    el("opt-run").disabled = on; el("opt-cancel").disabled = !on;
  }
  el("opt-run").addEventListener("click", function () {
    var body = request();
    running(true); setStatus("submitting…");
    fetch("optimize", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.statusText); return j; }); })
      .then(function (j) {
        view.job = j.job; view.model = body.model; view.frames = []; el("download-path").disabled = true;
        el("opt-frame").disabled = false;
        poll = setInterval(pollJob, 400);
      })
      .catch(function (err) { running(false); setStatus(err.message, "bad"); });
  });
  el("opt-cancel").addEventListener("click", function () {
    if (view.job) fetch("jobs/" + view.job + "/cancel", {method: "POST", body: "{}"});
  });
  function pollJob() {
    fetch("jobs/" + view.job + "?since=" + view.frames.length).then(function (r) { return r.json(); }).then(function (j) {
      var follow = view.mode !== "frames" || view.frame === view.frames.length - 1 || view.frames.length === 0;
      view.frames = view.frames.concat(j.frames);
      if (view.frames.length) {
        el("opt-frame").max = view.frames.length - 1;
        if (follow) showFrame(view.frames.length - 1);
        el("download-path").disabled = false;
      }
      var last = view.frames[view.frames.length - 1];
      var info = last ? " · step " + last.step + ", max force " + fmt(last.fmax, 3) + " eV/Å" : "";
      var done = ["converged", "not_converged", "cancelled", "error"].indexOf(j.status) >= 0;
      setStatus(j.message + info, done ? (j.status === "converged" ? "good" : j.status === "error" ? "bad" : "") : "");
      if (done) { clearInterval(poll); poll = null; running(false); }
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
  });

  highlight();
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
