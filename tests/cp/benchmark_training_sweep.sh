#!/bin/bash
# Automated training benchmark sweep: CP vs Non-CP
# Compares performance across different sequence lengths and CP sizes

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Training Benchmark: CP vs Non-CP"
echo "=========================================="
echo ""
echo "Script dir: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected GPUs: $NUM_GPUS"

# Benchmark parameters
BATCH_SIZE=2
NUM_STEPS=50
HIDDEN_SIZE=2048
NUM_LAYERS=16
NUM_HEADS=8
HEAD_DIM=64
VOCAB_SIZE=32000
LEARNING_RATE=1e-4

# Create output directory (use absolute path)
OUTPUT_DIR="$SCRIPT_DIR/training_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Steps: $NUM_STEPS"
echo "  Hidden size: $HIDDEN_SIZE"
echo "  Layers: $NUM_LAYERS"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Test configurations
# Format: "seq_len:cp_sizes"
# cp_sizes is comma-separated list (1 means single GPU)
CONFIGS=(
    "2048:1"           # Short seq, single GPU only
    "4096:1,2"         # Medium seq, single + CP=2
    "8192:1,2,4"       # Long seq, single + CP=2,4
    "16384:2,4"        # Very long, CP only (too big for single GPU)
    "32768:4,8"        # Ultra long, large CP only
)

run_benchmark() {
    local SEQ_LEN=$1
    local CP_SIZE=$2
    local MODE=$3
    local OUTPUT_FILE="$OUTPUT_DIR/${MODE}_seq${SEQ_LEN}_cp${CP_SIZE}.json"
    
    echo ""
    echo "=========================================="
    echo "Benchmark: seq_len=$SEQ_LEN, CP=$CP_SIZE, mode=$MODE"
    echo "=========================================="
    
    START_TIME=$(date +%s)
    
    if [ "$MODE" == "single" ]; then
        # Single GPU
        cd "$PROJECT_ROOT"
        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/benchmark_training.py" \
            --mode single \
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --hidden_size $HIDDEN_SIZE \
            --num_layers $NUM_LAYERS \
            --num_heads $NUM_HEADS \
            --head_dim $HEAD_DIM \
            --vocab_size $VOCAB_SIZE \
            --learning_rate $LEARNING_RATE \
            --output "$OUTPUT_FILE" \
            2>&1 | tee "$OUTPUT_DIR/${MODE}_seq${SEQ_LEN}_cp${CP_SIZE}.log"
    else
        # Multi-GPU with CP
        cd "$PROJECT_ROOT"
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((CP_SIZE-1))) \
        torchrun --nproc_per_node=$CP_SIZE "$SCRIPT_DIR/benchmark_training.py" \
            --mode cp \
            --cp_size $CP_SIZE \
            --seq_len $SEQ_LEN \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --hidden_size $HIDDEN_SIZE \
            --num_layers $NUM_LAYERS \
            --num_heads $NUM_HEADS \
            --head_dim $HEAD_DIM \
            --vocab_size $VOCAB_SIZE \
            --learning_rate $LEARNING_RATE \
            --output "$OUTPUT_FILE" \
            2>&1 | tee "$OUTPUT_DIR/${MODE}_seq${SEQ_LEN}_cp${CP_SIZE}.log"
    fi
    
    BENCH_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $BENCH_EXIT_CODE -eq 0 ]; then
        echo "✓ Completed in ${DURATION}s"
        echo "seq_len=$SEQ_LEN, cp=$CP_SIZE, mode=$MODE: ${DURATION}s [SUCCESS]" >> "$OUTPUT_DIR/timing.txt"
    else
        echo "✗ Failed after ${DURATION}s (likely OOM)"
        echo "seq_len=$SEQ_LEN, cp=$CP_SIZE, mode=$MODE: ${DURATION}s [FAILED - OOM]" >> "$OUTPUT_DIR/timing.txt"
    fi
}

# Run benchmarks
echo "Starting benchmark sweep..."
echo "" > "$OUTPUT_DIR/timing.txt"

TOTAL_START=$(date +%s)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r SEQ_LEN CP_SIZES <<< "$config"
    
    IFS=',' read -ra CP_ARRAY <<< "$CP_SIZES"
    
    for CP_SIZE in "${CP_ARRAY[@]}"; do
        # Skip if not enough GPUs
        if [ $CP_SIZE -gt $NUM_GPUS ]; then
            echo "⚠ Skipping seq_len=$SEQ_LEN, CP=$CP_SIZE (need $CP_SIZE GPUs, have $NUM_GPUS)"
            continue
        fi
        
        # Determine mode
        if [ $CP_SIZE -eq 1 ]; then
            MODE="single"
        else
            MODE="cp"
        fi
        
        # Run benchmark (continue on failure - OOM is expected for some configs)
        run_benchmark $SEQ_LEN $CP_SIZE $MODE || true
    done
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "=========================================="
echo "Benchmark Sweep Complete!"
echo "=========================================="
echo ""
echo "Total time: ${TOTAL_DURATION}s ($((TOTAL_DURATION/60)) minutes)"
echo ""
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "Timing Summary:"
cat "$OUTPUT_DIR/timing.txt"
echo ""

# Generate comparison report
echo "=========================================="
echo "Generating Comparison Report"
echo "=========================================="

python "$SCRIPT_DIR/analyze_training_benchmarks.py" --input_dir "$OUTPUT_DIR"

echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "View comparison report:"
echo "  cat $OUTPUT_DIR/comparison_report.md"
echo ""
echo "=========================================="

