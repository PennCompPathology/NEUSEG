#!/usr/bin/env bash
set -euo pipefail

# Get project root (parent directory of this script)
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
project_root="$(dirname "$script_dir")"

# Input and output directories relative to project root
WSIS_DIR="$project_root/Data" # .svs files (WSI)
OUTPUT_ROOT="$project_root/GM_WM_Seg_Results"

echo "WSIS_DIR: $WSIS_DIR"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"

# --- Log everything to a timestamped file as well ---
LOG_FILE="$OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Collect slides
shopt -s nullglob
slides=("$WSIS_DIR"/*.svs)
shopt -u nullglob

num_slides=${#slides[@]}
if (( num_slides == 0 )); then
  echo "No .svs files found in: $WSIS_DIR"
  exit 1
fi

echo "Found $num_slides slide(s)."
echo

# Track failures
FAIL_PRE=()
FAIL_SEG=()

# ---------- Step 1: Nuclei Segmentation ----------
nuclei_py="$script_dir/Nuclei_Segmentation.py"

for i in "${!slides[@]}"; do
  slide="${slides[$i]}"
  base="$(basename "$slide")"
  slide_name="${base%.svs}"

  idx=$((i + 1))
  echo "🛠  [$idx/$num_slides] Nuclei Segmentation: $base"

  if ! python "$nuclei_py" "$slide" "$OUTPUT_ROOT"; then
    echo "❌ Nuclei Segmentation failed for: $base"
    FAIL_PRE+=("$base")
  else
    echo "✅ Nuclei Segmentation completed: $base"
  fi
  echo
done

# ---------- Step 2: GMM Segmentation ----------
gmm_py="$script_dir/GMM_Segmentation.py"

# GMM debug outputs
GMM_DEBUG_ROOT="$project_root/GMM_Debug"
mkdir -p "$GMM_DEBUG_ROOT"

for i in "${!slides[@]}"; do
  slide="${slides[$i]}"
  base="$(basename "$slide")"
  slide_name="${base%.svs}"

  # per-slide debug dir
  debug_path="$GMM_DEBUG_ROOT/$slide_name"
  mkdir -p "$debug_path"

  idx=$((i + 1))
  echo "🧠  [$idx/$num_slides] GMM Segmentation: $base"

  if ! python "$gmm_py" "$slide" "$OUTPUT_ROOT" "$debug_path"; then
    echo "❌ GMM Segmentation failed for: $base"
    FAIL_SEG+=("$base")
  else
    echo "✅ GMM Segmentation completed: $base"
  fi
  echo
done

# ---------- Write failure lists (if any) ----------
FAILED_PRE_FILE="$OUTPUT_ROOT/failed_nucleiSeg.txt"
FAILED_SEG_FILE="$OUTPUT_ROOT/failed_GMMSeg.txt"

: > "$FAILED_PRE_FILE"
: > "$FAILED_SEG_FILE"

if (( ${#FAIL_PRE[@]} > 0 )); then
  printf "%s\n" "${FAIL_PRE[@]}" > "$FAILED_PRE_FILE"
  echo "⚠️  Nuclei segmentation failures: ${#FAIL_PRE[@]} (see $FAILED_PRE_FILE)"
fi

if (( ${#FAIL_SEG[@]} > 0 )); then
  printf "%s\n" "${FAIL_SEG[@]}" > "$FAILED_SEG_FILE"
  echo "⚠️  GMM segmentation failures: ${#FAIL_SEG[@]} (see $FAILED_SEG_FILE)"
fi

echo "📝 Full log saved to: $LOG_FILE"

# Return non-zero if anything failed
if (( ${#FAIL_PRE[@]} > 0 || ${#FAIL_SEG[@]} > 0 )); then
  exit 2
fi

