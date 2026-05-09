#!/bin/bash
# AutoComp optimization runner -- runs beam search on internal kernels
# Usage: bash run_optimize.sh [prob_id]
#   If prob_id is provided, runs only that kernel. Otherwise runs all listed.
#
# Environment: PyTorch Native (TorchNeuron) Beta 2 + NKI 0.3.0 (GA)
# Setup: Follow PyTorch Native setup guide in AGENTS.md (extract DLC,
#         install host runtime, create venv from workspace wheels)

set -e

# SDK 2.29.1 NxDI venv (standard DLAMI) or PyTorch Native venv (DLC)
if [ -d "$HOME/workspace/native_venv" ]; then
    source $HOME/workspace/native_venv/bin/activate
else
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
fi
export AWS_REGION=us-east-1
export WANDB_MODE=disabled

cd /home/ubuntu/autocomp

# Ensure output directory exists (fix for set -e + tee interaction)
mkdir -p output

# Install package if needed
pip install -e . -q 2>/dev/null

# Kernels to optimize:
#   21 = Cross-Entropy Loss (316 LOC, online LSE, float32, nki-library)

if [ -n "$1" ]; then
    KERNELS="$1"
else
    KERNELS="21"
fi

# Warm-up: run each kernel's test harness once to trigger library rehydration
# and first-time NKI compilation. This ensures beam search iterations hit warm caches.
echo "=========================================="
echo "Warm-up phase: pre-compiling kernels"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

for PROB_ID in $KERNELS; do
    TEST_FILE=$(ls tests/trn-internal/${PROB_ID}_*_test.py 2>/dev/null | head -1)
    if [ -n "$TEST_FILE" ]; then
        echo "Warming up: $TEST_FILE"
        python "$TEST_FILE" 2>&1 || echo "Warm-up for prob_id=$PROB_ID returned non-zero (may be OK)"
    else
        echo "WARNING: No test harness found for prob_id=$PROB_ID, skipping warm-up"
    fi
done

echo "Warm-up complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

for PROB_ID in $KERNELS; do
    case $PROB_ID in
        8) NAME="deltanet_recurrent" ;;
        9) NAME="flash_attn_d256" ;;
        10) NAME="openfold3_trimul" ;;
        11) NAME="whisper_megakernel" ;;
        12) NAME="flashvsr_fp32attn" ;;
        14) NAME="ltx2_crossattn" ;;
        15) NAME="gemm" ;;
        21) NAME="cross_entropy" ;;
        22) NAME="hunyuan_dit_attn" ;;
        5) NAME="fft256" ;;
        6) NAME="mamba_scan" ;;
        4) NAME="trimul" ;;
        *) NAME="kernel_${PROB_ID}" ;;
    esac

    echo "=========================================="
    echo "Optimizing kernel: $NAME (prob_id=$PROB_ID)"
    echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "=========================================="

    # Patch run_search.py with the target prob_id
    sed -i "s/prob_id = [0-9]*/prob_id = $PROB_ID/" autocomp/search/run_search.py

    # Run the beam search
    python -m autocomp.search.run_search 2>&1 | tee "output/optimize_${NAME}_$(date -u '+%Y%m%d_%H%M%S').log"

    echo ""
    echo "Completed: $NAME at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
done

echo "All optimizations complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
