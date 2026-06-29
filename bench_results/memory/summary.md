# VortexDB Memory Profiling — HNSW Index

**Setup:** dim=128, M=16, ef_construction=200, L2, Apple Silicon (ARM NEON)  
**Tool:** `/usr/bin/time -l` (macOS) + `resource.getrusage()` per checkpoint  
**Leak check:** `leaks --atExit` → 0 leaks

## Per-checkpoint breakdown

| Vectors | Peak RSS | Raw vectors | HNSW overhead | Overhead B/vec | Total B/vec |
|--------:|--------:|------------:|--------------:|---------------:|------------:|
|  50,000 | 108 MB  |    24 MB    |    46 MB      |    964 B/vec   | 1,476 B/vec |
| 100,000 | 158 MB  |    49 MB    |    72 MB      |    750 B/vec   | 1,262 B/vec |
| 150,000 | 207 MB  |    73 MB    |    95 MB      |    667 B/vec   | 1,179 B/vec |
| 200,000 | 262 MB  |    98 MB    |   126 MB      |    661 B/vec   | 1,173 B/vec |

## Summary at 200K vectors

| Metric                  | Value              |
|-------------------------|--------------------|
| Raw vector storage      | 512 B/vec          |
| HNSW graph overhead     | 661 B/vec          |
| **Total**               | **1,173 B/vec**    |
| Amplification           | 2.3× raw           |
| Peak RSS                | 262 MB             |

## Notes

- Overhead decreases from 964 → 661 B/vec as N grows (fixed engine state amortized over more vectors)
- HNSW overhead breaks down roughly as: layer-0 neighbor list (128 B) + upper layers (96 B) + node metadata (16 B) + allocator fragmentation (~420 B)
- faiss HNSW uses ~200–300 B/vec overhead (flat contiguous arrays vs per-node std::vector)
- macOS `leaks` confirms **0 memory leaks** at 10K vectors
