# VectorFile Design

## What is VectorFile?

VectorFile is the persistent storage layer for raw float32 vector data. Every vector inserted into a collection is written to a VectorFile, where it is stored in a fixed-size slot on disk and accessed through a memory-mapped region.

The slot index assigned by `append()` is used directly as the node ID in the HNSW index. There is no translation table — slot 42 in VectorFile holds the vector for node 42 in the graph.

**Why a dedicated file for vectors?**

The HNSW graph structure (neighbor lists, layer assignments, entry point) is serialized to a separate file at checkpoint time. Vectors are not stored inside the graph file because they are large and access patterns differ: during a search, the graph structure is traversed to find candidate nodes, then the vectors at those nodes are fetched for distance computation. Keeping them separate avoids reading the entire graph file to get vector data, and allows the graph to be compact.

**Why mmap instead of read()?**

A normal `read()` call copies data from the OS page cache into your buffer — two copies of the data are in memory at the same time. mmap maps the OS page cache directly into the process's address space, so there is only one copy. `read(slot)` returns a pointer directly into the mapped region — no copy, no buffer.

## How VectorFile Fits in the System

```
Insert request: (node_id=42, vec=[...])
    │
    ▼
WAL.append(Insert, node_id=42, vec)   ← durability first
    │
    ▼
HnswIndex.insert(node_id=42, vec)     ← update graph in memory
    │
    ▼
VectorFile.append(vec)                ← persist vector data
(returns slot == node_id)
```

During a search, `HnswIndex` traverses the graph to find candidate node IDs, then calls `VectorFile.read(node_id)` to fetch the vector for distance computation.

During crash recovery, `WAL.replay()` re-calls both `HnswIndex.insert()` and `VectorFile.append()` for each Insert record after the checkpoint. The vector data in the WAL payload is the source of truth — VectorFile is rebuilt by replaying it.

## On-Disk Format

The file is split into two regions: a fixed-size header followed by a flat array of vector slots.

```
[FileHeader: 64B][slot 0: dim×4B][slot 1: dim×4B]...[slot capacity-1: dim×4B]
```

**Header (64 bytes):**

| Field | Size | Description |
|-------|------|-------------|
| `magic` | 8B | Identifies this as a VectorFile (`0x564442564543540A`) |
| `version` | 4B | Format version, currently `1` |
| `dim` | 4B | Vector dimension — fixed for the file's lifetime |
| `slot_count` | 8B | Number of slots written so far |
| `capacity` | 8B | Total number of slots the file currently has space for |
| `_pad` | 32B | Reserved, zeroed |

The header is padded to 64 bytes so the first slot starts at a 64-byte aligned offset. `sizeof(float) = 4`, and most SIMD distance kernels assume 16-byte or 32-byte alignment — 64-byte alignment satisfies both.

**Slot array:**

Slot `i` starts at byte offset `64 + i × dim × 4`. The layout is a flat array of `float32` values, no per-slot metadata. Given a slot index, the vector is at a fixed, calculable offset — no scanning, no indirection.

`slot_count` is the number of slots that have been written. `capacity` is the total number of slots allocated (the file has physical space for `capacity` slots but only `slot_count` contain valid data).

## mmap Mechanics

The entire file — header and all slots — is mapped with a single `mmap(MAP_SHARED)` call. Every `append()` and `read()` goes through this mapping.

**The mapped region**

`mmap` returns a starting address — a pointer into the process's virtual address space. From that address, `map_size` bytes are reserved and backed by the file:

```
map_  →  0x7f8a00000000
         │
         ├── [0..63]                    FileHeader
         ├── [64 .. 64+dim×4-1]         slot 0
         ├── [64+dim×4 .. 64+2×dim×4-1] slot 1
         │   ...
         └── [64+(capacity-1)×dim×4 ..]  slot capacity-1
```

There is no physical memory behind this address range until it is accessed — the OS loads pages from disk on demand (page fault). The `Impl` struct holds three things to describe the mapping:

```cpp
void*  map_;      // mmap return value — base address of the region
size_t map_size;  // 64 + capacity × dim × 4
int    fd;        // file descriptor, kept open for ftruncate and remap
```

All pointer arithmetic is derived from `map_`:

```cpp
FileHeader* header()   { return (FileHeader*)map_; }
float*      data_ptr() { return (float*)((uint8_t*)map_ + 64); }
```

`header()` casts `map_` directly to `FileHeader*` — reading or writing any header field is a read or write into the first 64 bytes of the file. `data_ptr()` skips the header and points at the start of the slot array. `read(slot)` returns `data_ptr() + slot × dim`, which is a pointer directly into the mapped file at the right offset — no copy, no seek.

**The reserved address range is as large as the file**

`mmap` reserves a region of virtual address space exactly as large as the file:

```cpp
map_size = st.st_size;   // file is 2MB → reserve 2MB of virtual address space
map_ = mmap(nullptr, map_size, ...);
```

This is how the entire file can be read — every byte in the file has a corresponding virtual address. Reserving virtual address space is nearly free: on a 64-bit system the virtual address space is 128TB, and "reserving" a range only adds one VMA record in the kernel. No physical memory (RAM) is allocated just from the reservation.

The three layers each work independently:

```
virtual address space reserved:  entire file size  (nearly free — just a VMA record)
page cache actually in RAM:       only pages that have been accessed  (loaded on demand)
disk space actually allocated:    only slots that have been written   (sparse file)
```

A 512MB VectorFile with 1000 vectors written occupies 512MB of virtual address space, a few MB of RAM (only the pages for those 1000 vectors), and a few MB of disk (only the physical blocks for those 1000 vectors).

**One mapping covers the whole file**

The design uses a single `mmap` call covering header and all slots together. The alternative — mapping header and slots separately, or mapping individual slots on demand — would require managing multiple mappings and their lifetimes. A single contiguous mapping is simpler: one pointer, one size, one `munmap` on close.

**MAP_SHARED**

With `MAP_SHARED`, the mmap region and the underlying file are two views of the same data. Writes through the mapping modify the OS page cache, and the OS will flush those dirty pages to disk asynchronously. There is no separate `write()` call — `memcpy` into the mapping is sufficient.

```
memcpy(mmap_ptr + offset, vec, dim * 4)
    ↓
OS marks the page as dirty
    ↓
OS flushes dirty page to disk (asynchronously)
```

This is not an extra copy: the mmap region *is* the page cache. The vector data goes directly from the caller's buffer into the page that backs the file on disk.

**Page faults on read**

When `read(slot)` returns a pointer and the caller dereferences it, one of two things happens:

- The page is already in the OS page cache (loaded by a prior access) → pure memory read, no I/O.
- The page is not in cache → the CPU raises a page fault, the OS loads the page from disk into the page cache, and the access completes transparently.

HNSW search follows graph edges, visiting nodes in a non-sequential order. Each edge traversal may touch a different page. Accessing nodes that happen to share a page is "free" (the page is already in cache); accessing nodes on separate pages triggers individual page faults.

**MADV_RANDOM**

By default, the OS prefetches pages sequentially: accessing page N hints that page N+1 will be needed soon. This is optimal for workloads that scan data in order (like iterating a vector array), but counterproductive for HNSW search, which jumps between arbitrary node IDs.

`madvise(MADV_RANDOM)` disables sequential prefetch. The OS loads only the page actually accessed, not its neighbors. This avoids wasting I/O bandwidth and page cache space on pages that will not be used.

## Growth: ftruncate + remap

The file is pre-allocated with space for `kInitialCapacity = 1024` slots. When `append()` is called and `slot_count == capacity`, the file must grow:

```
1. new_cap  = capacity × 2
2. new_size = 64 + new_cap × dim × 4
3. ftruncate(fd, new_size)    — extend file; new bytes are zero-filled
4. munmap(old_map, old_size)  — release old mapping (old pointers now invalid)
5. mmap(new_size)             — map the larger file
6. madvise(MADV_RANDOM)       — re-apply hint on new mapping
7. header()->capacity = new_cap
```

**Growth is application-controlled, not OS-controlled**

`ftruncate` is a syscall we call explicitly — the OS does not know about our "double on full" strategy and does not trigger growth automatically. The OS only executes what we ask: change the file size to `new_size`. The decision of when to grow and by how much is entirely in our `grow()` function, analogous to how `std::vector` manages its own capacity with `malloc`/`free` — except the underlying resource is a file instead of heap memory.

**Why ftruncate instead of write()?**

`ftruncate` extends the file without writing any bytes. The new region becomes a sparse hole — the file system records the new size in the inode but allocates no physical disk blocks for the extended area.

Sparse files are a file system feature (supported on APFS, ext4, xfs — our target platforms). The file system's inode stores a block map: "logical block N → physical block M". For a sparse region, those entries are empty. When that region is read, the file system sees the empty entry and returns zeros without doing any I/O. When that region is first written, the file system allocates a new physical block and fills in the map entry.

The practical consequence: `ftruncate` doubling capacity is nearly instant regardless of how large the new size is. Only `append()` calls that actually write data allocate physical disk space. The difference is visible with standard tools:

```bash
ls -l vectors.vdb    # logical size: e.g. 2MB (capacity × dim × 4 + 64)
du -sh vectors.vdb   # actual disk usage: e.g. 512KB (only written slots)
```

Writing actual zeros with `write()` would allocate all blocks immediately and take time proportional to the extended size — the same amount of work as filling the file with real data.

**Why munmap + mmap instead of mremap?**

`mremap` is Linux-specific. `munmap` + `mmap` works on both Linux and macOS. The performance difference is negligible here — growth is infrequent (O(log N) times for N inserts).

**The pointer invalidation problem**

After `munmap`, any pointer previously returned by `read()` points into unmapped memory. Dereferencing it is undefined behavior. The API comment documents this constraint:

> Callers must not cache raw pointers returned by `read()` across an `append()` call that may trigger a remap.

In practice the engine fetches a pointer, uses it for distance computation immediately, and does not hold onto it. The HNSW graph stores node IDs, not pointers — it calls `read(node_id)` each time it needs a vector.

## Crash Safety

VectorFile does not provide crash safety by itself. It relies on the WAL:

- The WAL record is written and fsynced **before** `VectorFile.append()` is called.
- If the process crashes after the WAL write but before `VectorFile.append()`, the vector data is missing from VectorFile.
- On restart, `WAL.replay()` replays the Insert record and calls `VectorFile.append()` again, reconstructing the missing data.

**Can VectorFile lose data that the WAL thinks exists?**

Yes. The WAL is fsynced on every operation; VectorFile relies on the OS to flush dirty mmap pages asynchronously. After a crash, the WAL may contain Insert records whose vectors never made it to disk in VectorFile. This is expected and handled by replay.

**Can VectorFile contain data the WAL doesn't know about?**

No. VectorFile is only written after the WAL record is durable. If the process crashes before the WAL write, VectorFile is not written either.

## API

### `VectorFile(path, dim)`

**Purpose:** open or create the vector storage file for a collection.

If the file does not exist, creates it with a fresh header and pre-allocates space for `kInitialCapacity` slots using `ftruncate`. Maps the file with `mmap(MAP_SHARED)`.

If the file exists, opens it and maps it. Validates `magic`, `version`, and `dim` — throws `std::runtime_error` if any field does not match. The `dim` mismatch check prevents silently reading garbage data when a file from a different collection is accidentally opened.

**Parameters:**
- `path` — file path (e.g. `data/vectors.vdb`)
- `dim` — vector dimension; must match the collection schema

**Throws:** `std::runtime_error` on I/O failure, bad magic, version mismatch, or dim mismatch.

---

### `append(vec) → uint32_t`

**Purpose:** persist a vector and return its stable slot index.

Writes `vec` (a `float*` array of length `dim`) into the next free slot via `memcpy` into the mmap region. If the file is full, triggers a grow before writing.

**Returns:** the slot index assigned to this vector. Slot indices are assigned sequentially starting from 0 and are never reused. The caller uses this slot index as the node ID in `HnswIndex`.

**Note:** if this call triggers a grow, any pointer previously returned by `read()` is invalidated.

---

### `read(slot) → const float*`

**Purpose:** fetch a vector by slot index for distance computation.

Returns a zero-copy pointer directly into the mmap region. The caller uses this pointer to compute distance between the query vector and the stored vector. No data is copied.

**Returns:** pointer to `dim` floats at slot `slot`. Valid until the next `append()` that triggers a remap.

**Precondition:** `slot < slot_count()`. Passing an out-of-range slot is undefined behavior (asserted in debug builds).

---

### `slot_count() → size_t`

**Purpose:** return how many vectors have been written.

Reads `header()->slot_count` from the mmap region. Used by the engine to validate slot indices and to know how many vectors exist in the collection.

## Design Decisions

**Why 64-byte header?**

64 bytes is a common cache line size on both ARM and x86. Padding the header to 64 bytes ensures the first slot starts at a cache-line boundary — avoids a false sharing situation where the last bytes of the header and the first bytes of slot 0 share a cache line, causing header writes (slot_count updates) to invalidate the cache line holding slot 0's vector data.

**Why double capacity on grow?**

Doubling gives amortized O(1) appends — each vector is moved at most once across all grows (like `std::vector`). A fixed-increment growth strategy would give O(N) total remap cost for N inserts. The downside of doubling is potentially wasting up to 50% of allocated space, but the physical space is only allocated on write (sparse allocation), so unwritten slots don't consume disk space.

**Why not store slot index in each slot?**

The slot index is implicit from position, the same way array indices are implicit. Storing it would duplicate information already derivable from the slot's offset in the file, waste 4 bytes per slot, and add nothing.

**Why not use MAP_PRIVATE?**

`MAP_PRIVATE` creates a copy-on-write mapping — writes go to anonymous memory and are never reflected in the file. That would make VectorFile a read-only view of the file, which is not useful for a write-capable storage layer.

**Why not msync explicitly?**

`msync` flushes dirty pages to disk synchronously. We don't call it because VectorFile's durability comes from the WAL, not from VectorFile itself. The OS will flush dirty pages on its own schedule. If the OS hasn't flushed VectorFile when a crash happens, the WAL replay will reconstruct the missing data. Calling `msync` after every `append` would add I/O cost equivalent to the WAL sync, doubling the write amplification for no benefit.
