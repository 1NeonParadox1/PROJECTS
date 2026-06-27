#!/bin/bash
# run_all_cases.sh
# ================
# Runs all CFD cases in sequence for the serpentine cooling channel project.
# Source OpenFOAM environment before running:
#   source /opt/openfoam11/etc/bashrc
#   bash scripts/run_all_cases.sh

set -e  # exit on error

CASES=(case_Re500 case_Re1000 case_Re2000 case_rib)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================================"
echo "  Serpentine Cooling Channel - Running All CFD Cases"
echo "========================================================"
echo ""

# Check OpenFOAM is loaded
if ! command -v buoyantSimpleFoam &> /dev/null; then
    echo "[ERROR] OpenFOAM not found. Source environment first:"
    echo "        source /opt/openfoam11/etc/bashrc"
    exit 1
fi

echo "OpenFOAM found: $(foamVersion 2>/dev/null || echo 'version unknown')"
echo ""

for CASE in "${CASES[@]}"; do
    CASE_DIR="$ROOT_DIR/$CASE"
    echo "----------------------------------------------------"
    echo "  Running: $CASE"
    echo "----------------------------------------------------"

    cd "$CASE_DIR"

    # Step 1: Generate mesh
    echo "[1/3] Generating mesh with blockMesh..."
    # Use the simple (working) blockMesh for straight channel cases
    # For rib case, use the rib blockMeshDict
    if [ "$CASE" == "case_rib" ]; then
        blockMesh > log.blockMesh 2>&1
    else
        # Use the simple validated channel dict
        cp system/blockMeshDict.simple system/blockMeshDict.backup 2>/dev/null || true
        blockMesh > log.blockMesh 2>&1
    fi

    # Check mesh quality
    echo "[2/3] Checking mesh quality..."
    checkMesh > log.checkMesh 2>&1
    MESH_OK=$(grep -c "Mesh OK." log.checkMesh 2>/dev/null || echo 0)
    if [ "$MESH_OK" -gt 0 ]; then
        echo "      Mesh OK ✓"
    else
        echo "      [WARN] Mesh check issues - see log.checkMesh"
    fi

    # Step 2: Run solver
    echo "[3/3] Running buoyantSimpleFoam (this takes 10-30 min)..."
    buoyantSimpleFoam > log.buoyantSimpleFoam 2>&1 &
    PID=$!

    # Monitor convergence in background
    echo "      PID=$PID | Monitoring residuals..."
    while kill -0 $PID 2>/dev/null; do
        if [ -f log.buoyantSimpleFoam ]; then
            ITER=$(grep -c "^Time" log.buoyantSimpleFoam 2>/dev/null || echo "?")
            echo -ne "      Iteration: ~$ITER / 2000\r"
        fi
        sleep 10
    done
    echo ""
    echo "      Solver finished."

    cd "$ROOT_DIR"
    echo ""
done

echo "========================================================"
echo "  ALL CASES COMPLETE"
echo "========================================================"
echo ""
echo "Post-processing:"
echo "  python3 scripts/calc_nusselt.py"
echo ""
echo "Visualisation (open each case in ParaView):"
echo "  paraFoam -case case_Re500"
echo "  paraFoam -case case_rib"
