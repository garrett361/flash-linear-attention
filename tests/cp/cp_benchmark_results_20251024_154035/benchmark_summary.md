# Context Parallelism Benchmark Results
**Total Configurations Tested:** 16
**Successful Runs:** 16

## CP Size = 1
| Seq Length | Chunk/GPU | Peak Memory | Throughput | Fwd Time | Bwd Time |
|------------|-----------|-------------|------------|----------|----------|
| 4,096 | 4,096 | 2.92 GB | 116233 tok/s | 31.6 ms | 38.9 ms |
| 8,192 | 8,192 | 5.68 GB | 309705 tok/s | 31.0 ms | 21.9 ms |
| 16,384 | 16,384 | 11.19 GB | 277298 tok/s | 84.9 ms | 33.3 ms |
| 32,768 | 32,768 | 22.23 GB | 476349 tok/s | 83.5 ms | 54.1 ms |

## CP Size = 2
| Seq Length | Chunk/GPU | Peak Memory | Throughput | Fwd Time | Bwd Time |
|------------|-----------|-------------|------------|----------|----------|
| 4,096 | 2,048 | 1.54 GB | 124170 tok/s | 40.6 ms | 25.4 ms |
| 8,192 | 4,096 | 2.92 GB | 224242 tok/s | 44.0 ms | 29.1 ms |
| 16,384 | 8,192 | 5.68 GB | 459707 tok/s | 48.8 ms | 22.5 ms |
| 32,768 | 16,384 | 11.19 GB | 797414 tok/s | 52.2 ms | 30.0 ms |

## CP Size = 4
| Seq Length | Chunk/GPU | Peak Memory | Throughput | Fwd Time | Bwd Time |
|------------|-----------|-------------|------------|----------|----------|
| 4,096 | 1,024 | 0.86 GB | 144458 tok/s | 18.4 ms | 38.3 ms |
| 8,192 | 2,048 | 1.54 GB | 261398 tok/s | 27.0 ms | 35.7 ms |
| 16,384 | 4,096 | 2.92 GB | 357955 tok/s | 67.2 ms | 24.3 ms |
| 32,768 | 8,192 | 5.68 GB | 695405 tok/s | 67.5 ms | 26.7 ms |

## CP Size = 8
| Seq Length | Chunk/GPU | Peak Memory | Throughput | Fwd Time | Bwd Time |
|------------|-----------|-------------|------------|----------|----------|
| 4,096 | 512 | 0.51 GB | 77382 tok/s | 64.5 ms | 41.4 ms |
| 8,192 | 1,024 | 0.85 GB | 226180 tok/s | 29.9 ms | 42.6 ms |
| 16,384 | 2,048 | 1.54 GB | 289312 tok/s | 48.2 ms | 65.1 ms |
| 32,768 | 4,096 | 2.92 GB | 401434 tok/s | 120.5 ms | 42.8 ms |


