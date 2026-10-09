"""Recompute exported pair/cluster structures with a calculator, outside the periodic MD.

Reads the pairs.extxyz / clusters.extxyz written by extract/collect and runs one
task per structure:

- ForceDecomposition (single points at the MD geometry): F_isolated of the pair
  (and optionally of each molecule) and F_env = F_MD - F_isolated, with energies;
- FragmentRelaxation: relax the isolated pair (and optionally each molecule) and
  report deformation energies;
- FrozenShellOptimization: optimize the pair inside its fixed shell (clusters).

Modules: fragments (which rows form the pair, its members and each shell
molecule), evaluation (isolated single points and geometry optimization with an
ASE calculator), tasks (the three tasks), driver (file in, file out). The CLI
command is `python -m MartiniFuncs recompute`.
"""
