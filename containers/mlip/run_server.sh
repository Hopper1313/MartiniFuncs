#!/usr/bin/env bash
# Start the anthracene + MA builder page with the optimizer from the MLIP container, binding
# whichever model weights are present in the models folder (see fetch_models.sh).
#
#   ./run_server.sh [options] [server options, e.g. --port 8001]
#   then open http://127.0.0.1:8000/
#
# Options (or the environment variables in brackets):
#   --sif FILE       the container image                     [MLIP_SIF, default ~/containers/mlip.sif]
#   --models DIR     model weights, as fetch_models.sh makes  [MLIP_MODELS, default ~/mlip_models]
#   --results DIR    saved optimization results               [MLIP_RESULTS, default ./pair_builder_results]
#   --source DIR     use this MartiniFuncs checkout instead of the copy in the image (development)
#   --selftest       check the models against the reference energies instead of serving
#   -h, --help
# Models missing from the folder show as "unavailable" in the page; GFN2-xTB needs no weights.
set -euo pipefail

SIF="${MLIP_SIF:-$HOME/containers/mlip.sif}"
MODELS="${MLIP_MODELS:-$HOME/mlip_models}"
RESULTS="${MLIP_RESULTS:-$PWD/pair_builder_results}"
SOURCE=""
APP=server
PASS=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "run_server.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --sif) SIF="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --selftest) APP=selftest; shift ;;
        -h|--help) usage; exit 0 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

command -v apptainer >/dev/null || die "apptainer not found (see containers/mlip/README.md)"
[ -f "$SIF" ] || die "no image at $SIF (download mlip.sif, or pass --sif / set MLIP_SIF)"
case "$MODELS$RESULTS$SOURCE" in *:*) die "paths must not contain ':' (apptainer bind syntax)" ;; esac

[ -d "$MODELS" ] && MODELS="$(cd "$MODELS" && pwd)"
BINDS=()
ENVS=("AIMNET_CACHE_DIR=/models/aimnet")
FOUND=()
MISSING=()
if [ -f "$MODELS/uma/uma-s-1p2p1.pt" ]; then
    BINDS+=(--bind "$MODELS/uma:/models/uma"); FOUND+=(UMA)
else
    MISSING+=("UMA ($MODELS/uma/uma-s-1p2p1.pt)")
fi
if [ -f "$MODELS/mace/MACE-omol-0-extra-large-1024.model" ]; then
    BINDS+=(--bind "$MODELS/mace:/models/mace"); FOUND+=(MACE-OMOL)
else
    MISSING+=("MACE-OMOL ($MODELS/mace/MACE-omol-0-extra-large-1024.model)")
fi
AIMNET=""
for name in aimnet2_2025_b973c_d3_0.pt aimnet2_2025_b973c_d3_0.jpt; do
    if [ -f "$MODELS/aimnet/$name" ]; then AIMNET="$name"; break; fi
done
if [ -n "$AIMNET" ]; then
    BINDS+=(--bind "$MODELS/aimnet:/models/aimnet"); ENVS+=("AIMNET_MODEL=/models/aimnet/$AIMNET"); FOUND+=(AIMNet2)
else
    MISSING+=("AIMNet2 ($MODELS/aimnet/aimnet2_2025_b973c_d3_0.pt)")
fi

if [ -n "$SOURCE" ]; then
    [ -d "$SOURCE/MartiniFuncs" ] || die "--source $SOURCE has no MartiniFuncs package folder"
    BINDS+=(--bind "$(cd "$SOURCE" && pwd):/opt/live/MartiniFuncs")
    ENVS+=("PYTHONPATH=/opt/live/MartiniFuncs")
fi

echo "Image:  $SIF"
echo "Models: ${FOUND[*]:-none} + GFN2-xTB"
for m in "${MISSING[@]}"; do echo "  missing: $m  (fetch_models.sh)"; done

if [ "$APP" = selftest ]; then
    exec apptainer run --app selftest "${BINDS[@]}" --env "$(IFS=,; echo "${ENVS[*]}")" "$SIF" "${PASS[@]}"
fi

mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"
BINDS+=(--bind "$RESULTS:$RESULTS")
echo "Results: $RESULTS"
echo "Open http://127.0.0.1:8000/ (or your --port) when it says Serving; Ctrl+C stops it."
exec apptainer run --app server "${BINDS[@]}" --env "$(IFS=,; echo "${ENVS[*]}")" "$SIF" --results "$RESULTS" "${PASS[@]}"
