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

Everything runs on CPU. The model weights are **not** in the image: each user downloads them
under the providers' terms with `fetch_models.sh`. GFN2-xTB needs no weights, so the image is
usable right away.

## Quick start (use the published image)

You need Linux, or Windows with WSL2 (Ubuntu), and [Apptainer](https://apptainer.org/docs/admin/main/installation.html)
(on Ubuntu/WSL2: `sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer`).
Run everything below in a Linux / WSL terminal, from a clone of this repository.

1. **Get the image** (release [mlip-bb6e1dc](https://github.com/Hopper1313/MartiniFuncs/releases/tag/mlip-bb6e1dc),
   about 1 GB) into `~/containers/` and check the download:
   ```bash
   mkdir -p ~/containers && cd ~/containers
   gh release download mlip-bb6e1dc --repo Hopper1313/MartiniFuncs --pattern 'mlip.sif*'
   sha256sum -c mlip.sif.sha256
   ```
   `gh` (GitHub CLI) works while the repository is private, for anyone with access. For a public
   repository, plain downloads work too:
   `curl -LO https://github.com/Hopper1313/MartiniFuncs/releases/download/mlip-bb6e1dc/mlip.sif`
   (and the same for `mlip.sif.sha256`). Or build the image yourself (see Build below); use the
   release whose name matches the commit you have checked out.
2. **Get the models** into `~/mlip_models` (choose another folder with `--models DIR` or `MLIP_MODELS`):
   ```bash
   export HF_TOKEN=hf_...                     # for UMA, see the table below
   containers/mlip/fetch_models.sh --accept-asl   # --accept-asl: only if the ASL covers your use (MACE-OMOL)
   ```
   Anything you skip (or can't get yet) is reported; the rest works. `fetch_models.sh uma` (or
   `mace`, `aimnet`) fetches one model later. At the end it runs the self-test (below).
3. **Start the builder page with the optimizer:**
   ```bash
   containers/mlip/run_server.sh              # options: --port 8001, --results DIR, --models DIR, --sif FILE
   ```
   and open http://127.0.0.1:8000/ (from Windows too). Optimization results are saved as JSON
   files in `./pair_builder_results`. x3dom is served from the image, so the page needs no internet.

| Model | File in the models folder | Where it comes from | Terms |
|---|---|---|---|
| UMA (uma-s-1p2p1, task omol) | `uma/uma-s-1p2p1.pt` | Hugging Face `facebook/UMA`, `checkpoints/` | Gated: request access on https://huggingface.co/facebook/UMA (approval can take a day or two), then make a read token at https://huggingface.co/settings/tokens and `export HF_TOKEN=...` |
| MACE-OMOL (extra large) | `mace/MACE-omol-0-extra-large-1024.model` | https://github.com/ACEsuit/mace-foundations/releases (release `mace_omol_0`) | Academic Software License (https://github.com/gabor1/ASL); `fetch_models.sh` downloads it only with `--accept-asl` |
| AIMNet2 2025 (B97-3c + D3, member 0) | `aimnet/aimnet2_2025_b973c_d3_0.pt` | aimnetcentral model registry (https://isayevlab.github.io/aimnetcentral/), SHA-256 checked | see the aimnetcentral repository |
| GFN2-xTB | none | tblite, in the image | LGPL-3.0 |

Read each provider's terms yourself; the table is a pointer, not legal advice. Don't upload the
weights to this repository or attach them to releases.

`run_server.sh` binds whichever files are present to `/models/...` in the container and sets
`UMA_CHECKPOINT`, `MACE_MODEL` and `AIMNET_MODEL` accordingly; missing models show as
"unavailable" in the page. `run_server.sh --source .` uses this checkout's MartiniFuncs instead of
the copy inside the image (for development, without rebuilding).

### Self-test

```bash
containers/mlip/run_server.sh --selftest
```

evaluates the anthracene + maleic anhydride reference structures with each model found and compares
against `reference_energies.json` (agreement ~1e-7 eV on the maintainer's machine). The AIMNet2
reference was computed with the earlier `.jpt` export of the same model; `fetch_models.sh` gets the
registry's `.pt` file, which `run_server.sh` prefers when both are present.

## Build (WSL2 Ubuntu, from the repository root)

```bash
apptainer build --fakeroot --ignore-subuid mlip.sif containers/mlip/mlip.def
```

`--ignore-subuid` is needed under WSL2, where the default fakeroot mode fails to mount
`binfmt_misc`; on a regular Linux machine plain `--fakeroot` works. The build takes about 4
minutes, needs internet (PyPI, the PyTorch CPU index, the npm registry for x3dom) and ends with an
import test of both environments. Versions are pinned in the `requirements-*.txt` files.
MartiniFuncs is copied into the image at build time: rebuild after code changes (or use
`run_server.sh --source .`).

### Publishing a release (maintainers)

```bash
apptainer build --fakeroot --ignore-subuid ~/containers/mlip.sif containers/mlip/mlip.def
containers/mlip/run_server.sh --selftest                  # with your own model files
cd ~/containers && sha256sum mlip.sif > mlip.sif.sha256
gh release create mlip-$(git -C ~/path/to/MartiniFuncs rev-parse --short HEAD) \
    mlip.sif mlip.sif.sha256 --title "MLIP container" \
    --notes "Built from commit ...; CPU only; models not included (containers/mlip/README.md)."
```

GitHub release assets may be up to 2 GB each (the image is about 1 GB). Name the release after the
commit it was built from, so the image and the scripts in the repository match.

## Advanced: running apps directly

Without the scripts, bind the model files to `/models` yourself, e.g.:

```bash
apptainer run --app selftest \
  --bind ~/mlip_models/uma:/models/uma \
  --bind ~/mlip_models/mace:/models/mace \
  --bind ~/mlip_models/aimnet:/models/aimnet \
  --env AIMNET_MODEL=/models/aimnet/aimnet2_2025_b973c_d3_0.pt \
  ~/containers/mlip.sif
```

Other model locations: `--env UMA_CHECKPOINT=...,MACE_MODEL=...,AIMNET_MODEL=...`. The `server` app
takes the server's options (`--port`, `--host`, `--results DIR`, `--assets DIR`).

Run your own scripts with `apptainer run --app uma mlip.sif script.py` (or `--app mace`). Your home
folder is mounted automatically; add `--bind /mnt/c` to reach Windows files from WSL. UMA uses
charge 0 / spin 1 from `atoms.info`; AIMNet2 takes `charge`/`mult` in `AIMNet2ASE`.
