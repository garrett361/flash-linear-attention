# Training Performance Comparison - Context Parallelism vs Single GPU

## 📊 Complete Results Table

| Seq Length | CP Size | GPUs | Tokens/sec | Peak Mem/GPU | Total Memory | Avg Loss | Status |
|------------|---------|------|------------|--------------|--------------|----------|--------|
| 2K         | 1       | 1    | 49,804     | 8.49 GB      | 8.49 GB      | 0.3212   | ✓      |
| 4K         | 1       | 1    | 98,666     | 12.23 GB     | 12.23 GB     | 0.3310   | ✓      |
| 4K         | 2       | 2    | 72,899     | 8.50 GB      | 17.00 GB     | 0.3213   | ✓      |
| 8K         | 1       | 1    | 144,741    | 19.81 GB     | 19.81 GB     | 0.3131   | ✓      |
| 8K         | 2       | 2    | **147,858**| 12.23 GB     | 24.46 GB     | 0.3219   | ✓      |
| 8K         | 4       | 4    | 107,319    | 8.50 GB      | 34.00 GB     | 0.3155   | ✓      |
| 16K        | 1       | 1    | **OOM**    | -            | -            | -        | ❌     |
| 16K        | 2       | 2    | **228,057**| 19.81 GB     | 39.62 GB     | 0.3271   | ✓      |
| 16K        | 4       | 4    | 205,719    | 12.23 GB     | 48.92 GB     | 0.3261   | ✓      |
| 32K        | 1       | 1    | **OOM**    | -            | -            | -        | ❌     |
| 32K        | 4       | 4    | **352,734**| 19.81 GB     | 79.24 GB     | 0.3248   | ✓      |
| 32K        | 8       | 8    | 252,612    | 12.23 GB     | 97.84 GB     | 0.3213   | ✓      |

---

## 🎯 Quick Summary

### Best Results

| Metric | Value | Configuration |
|--------|-------|---------------|
| **Highest Throughput** | 352,734 tokens/s | 32K + CP=4 |
| **Lowest Memory/GPU** | 8.50 GB | 8K + CP=4 |
| **Longest Sequence** | 32,768 tokens | CP=4 or CP=8 (impossible on single GPU) |

### Key Insights

✅ **8K + CP=2**: Similar throughput (147K vs 144K) with lower memory per GPU  
✅ **16K+**: Context Parallelism is **required** (single GPU OOMs)  
✅ **32K + CP=4**: Peak performance at 353K tokens/s  
✅ **Short sequences (≤4K)**: Single GPU achieves higher throughput

---

## 📈 Performance by Sequence Length

### 2K Tokens
- **Single GPU is best**: 49,804 tokens/s
- No benefit from CP at this length

### 4K Tokens  
- **Single GPU is best**: 98,666 tokens/s
- CP=2 is slower: 72,899 tokens/s (overhead > benefit)

### 8K Tokens
- **Single GPU**: 144,741 tokens/s, 19.81 GB
- **CP=2**: 147,858 tokens/s, 12.23 GB ✓
- **CP=4**: 107,319 tokens/s, 8.50 GB

### 16K Tokens (CP Required)
- **Single GPU**: ❌ Out of Memory
- **CP=2**: 228,057 tokens/s ✓ (enables training)
- **CP=4**: 205,719 tokens/s (better memory efficiency)

### 32K Tokens (CP Essential)
- **Single GPU**: ❌ Out of Memory  
- **CP=4**: **352,734 tokens/s** 🏆 (best overall performance)
- **CP=8**: 252,612 tokens/s (lower throughput but less memory per GPU)

---

## 💾 Memory per GPU Analysis

| Sequence | Single GPU | CP=2 | CP=4 | CP=8 |
|----------|------------|------|------|------|
| 2K       | 8.49 GB    | -    | -    | -    |
| 4K       | 12.23 GB   | 8.50 GB | - | - |
| 8K       | 19.81 GB   | 12.23 GB | 8.50 GB | - |
| 16K      | **OOM ❌** | 19.81 GB | 12.23 GB | - |
| 32K      | **OOM ❌** | - | 19.81 GB | 12.23 GB |

**Key Finding**: Memory per GPU scales roughly as `total_memory / cp_size`, enabling much longer contexts.

---

## 🎯 Recommendations

### Use Single GPU for:
- ❌ Sequences ≤ 4K tokens
- ❌ Quick prototyping / debugging
- ❌ When you only have 1 GPU available

### Use CP=2 for:
- ✅ 8K-16K token sequences
- ✅ Similar throughput with memory savings
- ✅ Good balance of performance and resources

### Use CP=4 for:
- ✅ 16K-32K token sequences  
- ✅ Maximum throughput (353K tokens/s at 32K)
- ✅ When single GPU OOMs

### Use CP=8 for:
- ✅ 32K+ ultra-long sequences
- ✅ Maximum memory efficiency per GPU
- ✅ When lower-memory GPUs are available

---

## ⚠️ Important Notes

1. **Training quality is preserved**: Loss values are consistent across all configurations
2. **Total GPU memory increases**: While per-GPU memory decreases, total memory used across all GPUs is higher
3. **Real benefit is capability extension**: CP enables 4x longer sequences
4. **Per-GPU memory scales with CP size**: More GPUs = less memory needed per GPU

---

## 📊 Visual Summary

### Throughput Comparison
```
2K:   Single GPU ████████████ 49,804 tokens/s
4K:   Single GPU ████████████████████ 98,666 tokens/s  
8K:   Single GPU ████████████████████████████████ 144,741 tokens/s
8K:   CP=2       ████████████████████████████████ 147,858 tokens/s
16K:  CP=2       ████████████████████████████████████████████ 228,057 tokens/s
32K:  CP=4       ████████████████████████████████████████████████████████████████ 352,734 tokens/s 🏆
```

### Memory per GPU
```
8K:   Single GPU ████████████████████ 19.81 GB
8K:   CP=2       ████████████ 12.23 GB
8K:   CP=4       ████████ 8.50 GB
```

---

**Generated**: October 24, 2025  
**Total Runs**: 10 successful benchmarks  
**Model**: GatedDeltaNet (2048 hidden, 16 layers, 50 steps each)

