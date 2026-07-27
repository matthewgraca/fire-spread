#!/bin/bash
# ==============================================================================
# GOFER Pipeline Runner
#
# Runs the full GOFER pipeline: ingest followed by processing.
# All default arguments are written out explicitly for clarity.
#
# Usage:
#   ./scripts/gofer/run.sh
# ==============================================================================

set -e  # Exit on any error

# --- Configuration (edit these) ---
MANIFEST="manifests/example.csv"
GOES_DIR="data/goes"
TEMP_DIR="temp"
OUT_DIR="out"
DEM="data/dem/SRTMGL3_NC.003_SRTMGL3_DEM_doy2000042000000_aid0001.tif"
START_STEP="aggregate"
CLEAN=true

# --- Phase 1: Ingest ---
echo "============================================================"
echo "Phase 1: Ingesting GOES data"
echo "============================================================"

python scripts/gofer/ingest.py \
    --manifest "$MANIFEST" \
    --goes-dir "$GOES_DIR" \
    --temp-dir "$TEMP_DIR"

# --- Phase 2: Process ---
echo ""
echo "============================================================"
echo "Phase 2: Processing pipeline"
echo "============================================================"

CLEAN_FLAG=""
if [ "$CLEAN" = true ]; then
    CLEAN_FLAG="--clean"
fi

python scripts/gofer/run_pipeline.py \
    --manifest "$MANIFEST" \
    --step "$START_STEP" \
    --goes-dir "$GOES_DIR" \
    --temp-dir "$TEMP_DIR" \
    --out-dir "$OUT_DIR" \
    --dem "$DEM" \
    $CLEAN_FLAG

echo ""
echo "============================================================"
echo "Done. Outputs in: $OUT_DIR/"
echo "============================================================"
