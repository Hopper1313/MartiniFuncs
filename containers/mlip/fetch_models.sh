#!/usr/bin/env bash
# Download the model weights the MLIP container uses into one folder (default ~/mlip_models):
#
#   uma/uma-s-1p2p1.pt                       UMA (Meta FAIR Chemistry). Gated on Hugging Face: request
#                                            access at https://huggingface.co/facebook/UMA, then create
#                                            a read token and export HF_TOKEN=hf_... before running.
#   mace/MACE-omol-0-extra-large-1024.model  MACE-OMOL (ACEsuit, mace-foundations release mace_omol_0).
#                                            Academic Software License: read https://github.com/gabor1/ASL
#                                            and pass --accept-asl to download it.
#   aimnet/aimnet2_2025_b973c_d3_0.pt        AIMNet2 2025, B97-3c + D3 (Isayev lab, aimnetcentral model
#                                            registry; checked against the registry's SHA-256).
#
# GFN2-xTB needs no download. The weights are not redistributed with the image: each user
# downloads them under the providers' terms. Downloads run inside the image (no other tools needed).
#
#   ./fetch_models.sh [--accept-asl] [--no-check] [--sif FILE] [--models DIR] [uma] [mace] [aimnet]
#
# With no model names, all three are fetched. Files already present are kept. Afterwards the
# models are checked against the reference energies (run_server.sh --selftest) unless --no-check.
# MLIP_SIF and MLIP_MODELS set the image and folder as in run_server.sh.
set -euo pipefail

SIF="${MLIP_SIF:-$HOME/containers/mlip.sif}"
MODELS="${MLIP_MODELS:-$HOME/mlip_models}"
ACCEPT_ASL=0
CHECK=1
WANT=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "fetch_models.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --sif) SIF="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --accept-asl) ACCEPT_ASL=1; shift ;;
        --no-check) CHECK=0; shift ;;
        -h|--help) usage; exit 0 ;;
        uma|mace|aimnet) WANT+=("$1"); shift ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
[ ${#WANT[@]} -gt 0 ] || WANT=(uma mace aimnet)

command -v apptainer >/dev/null || die "apptainer not found (see containers/mlip/README.md)"
[ -f "$SIF" ] || die "no image at $SIF (download mlip.sif, or pass --sif / set MLIP_SIF)"
case "$MODELS" in *:*) die "the models path must not contain ':'" ;; esac
mkdir -p "$MODELS/uma" "$MODELS/mace" "$MODELS/aimnet"
MODELS="$(cd "$MODELS" && pwd)"
echo "Models folder: $MODELS"

# Python inside the image's uma environment, with the models folder at /mlip_models.
image_python() { apptainer exec --bind "$MODELS:/mlip_models" "$@"; }
FAILED=()

fetch_uma() {
    if [ -f "$MODELS/uma/uma-s-1p2p1.pt" ]; then echo "UMA: present"; return; fi
    if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
        echo "UMA: skipped. It is gated: request access at https://huggingface.co/facebook/UMA (approval can"
        echo "     take a while), create a read token at https://huggingface.co/settings/tokens, then"
        echo "     export HF_TOKEN=hf_... and run ./fetch_models.sh uma"
        FAILED+=(UMA); return
    fi
    echo "UMA: downloading facebook/UMA checkpoints/uma-s-1p2p1.pt ..."
    if image_python "$SIF" /opt/envs/uma/bin/python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
staging = "/mlip_models/uma/.download"
path = hf_hub_download("facebook/UMA", "uma-s-1p2p1.pt", subfolder="checkpoints", local_dir=staging)
os.replace(path, "/mlip_models/uma/uma-s-1p2p1.pt")
shutil.rmtree(staging, ignore_errors=True)
PY
    then echo "UMA: done"; else echo "UMA: download failed (access not granted yet, or a bad token?)"; FAILED+=(UMA); fi
}

fetch_mace() {
    local name=MACE-omol-0-extra-large-1024.model
    if [ -f "$MODELS/mace/$name" ]; then echo "MACE-OMOL: present"; return; fi
    if [ "$ACCEPT_ASL" != 1 ]; then
        echo "MACE-OMOL: skipped. Its weights are under the Academic Software License"
        echo "     (https://github.com/gabor1/ASL). If your use is covered, run ./fetch_models.sh --accept-asl mace"
        FAILED+=(MACE-OMOL); return
    fi
    echo "MACE-OMOL: downloading $name ..."
    if image_python "$SIF" python - "$name" <<'PY'
import os, sys, urllib.request
name = sys.argv[1]
url = f"https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/{name}"
partial = f"/mlip_models/mace/{name}.part"
with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
    while chunk := response.read(1 << 20):
        out.write(chunk)
os.replace(partial, f"/mlip_models/mace/{name}")
PY
    then echo "MACE-OMOL: done"; else echo "MACE-OMOL: download failed"; FAILED+=(MACE-OMOL); fi
}

fetch_aimnet() {
    if [ -f "$MODELS/aimnet/aimnet2_2025_b973c_d3_0.pt" ] || [ -f "$MODELS/aimnet/aimnet2_2025_b973c_d3_0.jpt" ]; then
        echo "AIMNet2: present"; return
    fi
    echo "AIMNet2: downloading aimnet2-b973c-2025-d3_0 from the aimnetcentral registry ..."
    if image_python --env AIMNET_CACHE_DIR=/mlip_models/aimnet "$SIF" /opt/envs/uma/bin/python -c \
        'from aimnet.calculators.model_registry import get_model_path; print(get_model_path("aimnet2-b973c-2025-d3_0"))'
    then echo "AIMNet2: done"; else echo "AIMNet2: download failed"; FAILED+=(AIMNet2); fi
}

for model in "${WANT[@]}"; do "fetch_$model"; done

if [ ${#FAILED[@]} -gt 0 ]; then echo "Not fetched: ${FAILED[*]} (see above). The others are usable."; fi
if [ "$CHECK" = 1 ]; then
    echo "Checking the models against the reference energies ..."
    "$(dirname "$0")/run_server.sh" --sif "$SIF" --models "$MODELS" --selftest
fi
