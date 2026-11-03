# MLIP container (Apptainer / Singularity)

UMA, MACE-OMOL and AIMNet2 on CPU, with ASE, Sella, geomeTRIC, tblite (GFN2-xTB), RDKit,
McUtils, Psience and MartiniFuncs, in one image (~1.0 GB `.sif`).

| App | Python environment | Models |
|---|---|---|
| `uma` | `/opt/envs/uma`: fairchem-core 2.22, aimnet 0.2.5 (aimnetcentral), e3nn 0.6 | UMA, AIMNet2 |
| `mace` | `/opt/envs/mace`: mace-torch 0.3.17, e3nn 0.4.4 | MACE-OMOL |
| `base` | system Python 3.12 (shared by both) | tblite only |

UMA and MACE cannot share an environment: MACE pins e3nn to exactly 0.4.4, UMA needs e3nn >= 0.5.
Both environments are built on one base layer (CPU torch 2.13, numpy, ASE, Sella, geomeTRIC,
tblite, RDKit, McUtils, Psience, MartiniFuncs), so torch and numpy are stored once.
fairchem imports ray and wandb at load time, so they are installed although unused.

## Build (WSL2 Ubuntu, from the repository root)

```bash
apptainer build --fakeroot --ignore-subuid mlip.sif containers/mlip/mlip.def
```

`--ignore-subuid` is needed under WSL2, where the default fakeroot mode fails to mount
`binfmt_misc`; on a regular Linux machine plain `--fakeroot` works. The build takes about 3
minutes and ends with an import test of both environments. Versions are pinned in the
`requirements-*.txt` files.

## Run

Model weights are not in the image; bind them to `/models`:

```bash
apptainer run --app selftest \
  --bind "/mnt/c/Users/nicho/Documents/Martini collab/uma_data:/models/uma" \
  --bind /mnt/c/Users/nicho/.cache/mace:/models/mace \
  --bind "/mnt/c/Users/nicho/Documents/Martini collab/aimnet2_2025_b973c_d3_0.jpt:/models/aimnet/aimnet2_2025_b973c_d3_0.jpt" \
  mlip.sif
```

`selftest` evaluates the anthracene + maleic anhydride reference structures with each model and
compares against `reference_energies.json` (tested values; agreement ~1e-7 eV). Other model
locations: `--env UMA_CHECKPOINT=...,MACE_MODEL=...,AIMNET_MODEL=...`.

Run your own scripts with `apptainer run --app uma mlip.sif script.py` (or `--app mace`). Your home
folder is mounted automatically; add `--bind /mnt/c` to reach Windows files from WSL. UMA uses
charge 0 / spin 1 from `atoms.info`; AIMNet2 takes `charge`/`mult` in `AIMNet2ASE`.
