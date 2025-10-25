#!/bin/bash
# Automated profiling sweep across CP sizes and sequence lengths
# Customize the arrays below to test different configurations

set -e

echo "=========================================="
echo "CP Configuration Sweep"
echo "=========================================="
echo ""

# ============================================
# CONFIGURATION - Modify these as needed
# ============================================

# CP sizes to test (must have enough GPUs)
CP_SIZES=(1 2 4 8)

# Sequence lengths for each CP size
# You can customize per CP size if needed
declare -A SEQ_LENS_BY_CP
SEQ_LENS_BY_CP[1]="2048,4096,8192,16384"           # CP=1: shorter sequences due to memory
SEQ_LENS_BY_CP[2]="2048,4096,8192,16384,32768"     # CP=2: can handle more
SEQ_LENS_BY_CP[4]="2048,4096,8192,16384,32768,65536"  # CP=4: even longer
SEQ_LENS_BY_CP[8]="2048,4096,8192,16384,32768,65536,131072"  # CP=8: longest sequences

# Other configuration
BATCH_SIZES="1,2"
HIDDEN_SIZE=2048
NUM_LAYERS=16
NUM_ITERATIONS=5
NUM_WARMUP=2

# ============================================

# Verify GPU availability
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected GPUs: $NUM_GPUS"
echo ""

# Create output directory
OUTPUT_DIR="profile_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  CP sizes: ${CP_SIZES[@]}"
echo "  Batch sizes: $BATCH_SIZES"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Warmup: $NUM_WARMUP"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run profiling for a specific CP size
run_profile() {
    local CP_SIZE=$1
    local SEQ_LENS=$2
    local OUTPUT_FILE="$OUTPUT_DIR/cp${CP_SIZE}_profile.json"
    local LOG_FILE="$OUTPUT_DIR/cp${CP_SIZE}_profile.log"
    
    echo ""
    echo "=========================================="
    echo "Profiling CP=$CP_SIZE"
    echo "  Sequence lengths: $SEQ_LENS"
    echo "=========================================="
    
    START_TIME=$(date +%s)
    
    if [ $CP_SIZE -eq 1 ]; then
        # Single GPU - use python directly
        python tests/cp/profile_context_parallel.py \
            --cp_size $CP_SIZE \
            --seq_lens "$SEQ_LENS" \
            --batch_sizes "$BATCH_SIZES" \
            --hidden_size $HIDDEN_SIZE \
            --num_layers $NUM_LAYERS \
            --num_iterations $NUM_ITERATIONS \
            --num_warmup $NUM_WARMUP \
            --output "$OUTPUT_FILE" \
            2>&1 | tee "$LOG_FILE"
    else
        # Multi-GPU - use torchrun
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((CP_SIZE-1))) \
        torchrun --nproc_per_node=$CP_SIZE tests/cp/profile_context_parallel.py \
            --cp_size $CP_SIZE \
            --seq_lens "$SEQ_LENS" \
            --batch_sizes "$BATCH_SIZES" \
            --hidden_size $HIDDEN_SIZE \
            --num_layers $NUM_LAYERS \
            --num_iterations $NUM_ITERATIONS \
            --num_warmup $NUM_WARMUP \
            --output "$OUTPUT_FILE" \
            2>&1 | tee "$LOG_FILE"
    fi
    
    PROFILE_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $PROFILE_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ CP=$CP_SIZE completed successfully in ${DURATION}s"
        echo "CP=$CP_SIZE: ${DURATION}s [SUCCESS]" >> "$OUTPUT_DIR/timing.txt"
    else
        echo ""
        echo "✗ CP=$CP_SIZE failed after ${DURATION}s"
        echo "CP=$CP_SIZE: ${DURATION}s [FAILED]" >> "$OUTPUT_DIR/timing.txt"
    fi
    
    echo ""
}

# Main execution
echo "Starting profiling sweep..."
echo "" > "$OUTPUT_DIR/timing.txt"

TOTAL_START=$(date +%s)

for CP_SIZE in "${CP_SIZES[@]}"; do
    # Check if we have enough GPUs
    if [ $CP_SIZE -gt $NUM_GPUS ]; then
        echo "⚠ Skipping CP=$CP_SIZE (requires $CP_SIZE GPUs, only $NUM_GPUS available)"
        echo "CP=$CP_SIZE: SKIPPED (insufficient GPUs)" >> "$OUTPUT_DIR/timing.txt"
        continue
    fi
    
    # Get sequence lengths for this CP size
    SEQ_LENS="${SEQ_LENS_BY_CP[$CP_SIZE]}"
    if [ -z "$SEQ_LENS" ]; then
        # Default if not specified
        SEQ_LENS="2048,4096,8192,16384"
    fi
    
    run_profile $CP_SIZE "$SEQ_LENS"
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "=========================================="
echo "Profiling Sweep Complete!"
echo "=========================================="
echo ""
echo "Total time: ${TOTAL_DURATION}s ($((TOTAL_DURATION/60)) minutes)"
echo ""
echo "Timing Summary:"
cat "$OUTPUT_DIR/timing.txt"
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "=========================================="

