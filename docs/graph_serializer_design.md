# Graph Serializer Design

## What is GraphSerializer?

GraphSerializer writes the complete in-memory HNSW graph state to a binary file on disk (checkpoint / snapshot), and reads it back to reconstruct the index after a restart.

It is not involved in normal read/write operations. It is called once at checkpoint time to persist the current state, and once at startup to restore it.

## Where it Fits in the System

During normal operation, the HNSW index lives entirely in memory:

```
insert/search/remove → HnswIndex (in memory)
                        ↓
                       WAL (on disk, for crash safety)
```

GraphSerializer is only invoked at checkpoint time:

```
checkpoint trigger
    ↓
GraphSerializer::serialize(index, "graph.bin")  ← snapshot memory → disk
    ↓
wal.truncate_before(checkpoint_lsn)             ← drop WAL before snapshot
```

At restart, the recovery sequence is:

```
GraphSerializer::deserialize("graph.bin", cfg)  ← load snapshot
    ↓
wal.replay(checkpoint_lsn, on_insert, on_delete)← replay WAL after snapshot
    ↓
index fully restored
```

Without checkpoints, every restart would replay the entire WAL from record zero — time grows unboundedly with the number of writes. Checkpoints bound recovery time to the number of writes since the last checkpoint.

## Why Not Just Dump Memory to Disk?

The HNSW graph is stored in memory as C++ objects:

```
unordered_map<NodeId, HnswNode>
    └── HnswNode {
            id: 42,
            layer: 2,
            neighbors: vector<vector<NodeId>>  ← pointer into heap
                           [0] → [13, 77, 5, ...]  ← another heap allocation
                           [1] → [13, 5]
                           [2] → [13]
        }

unordered_map<NodeId, vector<float>>
    └── 42 → [0.1, 0.3, ...]  ← another heap allocation
```

These objects are full of **pointers** — virtual memory addresses valid only for the current process run. Writing raw memory to disk and reading it back would produce dangling pointers: the next process launch assigns different virtual addresses, so every pointer in the dump would point at garbage.

There is also a layout problem: `unordered_map` and `vector` are not contiguous blobs. Their data is scattered across multiple separate heap allocations. There is no single region of memory to dump.

`serialize()` solves both problems: it traverses the C++ objects, extracts the meaningful values (node IDs, layer numbers, neighbor IDs, float coordinates), and writes them to disk in a flat, pointer-free binary format. `deserialize()` does the reverse — reads those values and `new`s fresh `unordered_map` and `vector` objects, filling them with the values from the file.

## What Gets Serialized

The HNSW index has two kinds of state:

**Graph topology** — the structure of the graph:
- `entry_point`: the node where all searches start
- `max_layer`: the highest layer in the graph
- `live_count`: number of non-tombstoned nodes
- Per node: `id`, `layer`, `tombstone`, and `neighbors[level][i]` for every layer the node participates in

**Vector data** — the raw float32 arrays:
- Per node: `vec` (dim floats), needed to compute distances during search

Vectors are serialized alongside the graph even though they are also stored in VectorFile. This keeps the snapshot self-contained: `deserialize()` reconstructs a fully operational index from the single graph file, without needing access to VectorFile or the WAL.

The alternative would be to skip vectors in the graph file and reload them from VectorFile during deserialization. That would reduce snapshot file size but couple two components together at restore time. The current design keeps the checkpoint simple and independent.

## On-Disk Format

All values are host byte order (no endianness conversion).

```
[GraphFileHeader: 32B]
per node (node_count times):
    [id: 4B][layer: 4B][tombstone: 1B]
    [vec: dim × 4B]
    per layer (layer+1 times):
        [nbr_count: 4B][nbr_ids: nbr_count × 4B]
```

**File header (32 bytes):**

| Field | Size | Description |
|-------|------|-------------|
| `magic` | 8B | `0x47524150480A0000` — identifies this as a graph snapshot |
| `version` | 4B | Format version, currently `1` |
| `node_count` | 4B | Total number of nodes (live + tombstoned) |
| `entry_point` | 4B | NodeId of the graph entry point (`kInvalidNode` if empty) |
| `max_layer` | 4B | Highest layer in the graph (-1 if empty) |
| `dim` | 4B | Vector dimension — validated against `cfg.dim` on open |
| `live_count` | 4B | Number of non-tombstoned nodes |

**Per-node record:**

Each node is written as a contiguous block. After `id`, `layer`, and `tombstone` come `dim` floats for the vector. Then for each layer from 0 to `layer`, the neighbor count followed by that many NodeIds. The neighbor list for layer 0 typically has the most entries (up to M0 = 2M); upper layers have fewer.

There is no padding between records. The file is read sequentially during deserialization.

## Internal Access: snapshot() and restore()

`HnswIndex` uses the Pimpl pattern — all internal state (`nodes`, `vecs`, `entry_point`, etc.) is defined in `hnsw_index.cpp` and is not visible from the header. GraphSerializer needs to read and write that state.

The naive solution would be to add `friend class GraphSerializer` to `HnswIndex`. This fails because the `Impl` struct is defined inside `hnsw_index.cpp`, not in the header — even with a `friend` declaration, `graph_serializer.cpp` cannot see `Impl`'s fields.

Instead, `HnswIndex` exposes two methods specifically for serialization:

```cpp
struct NodeData {
    NodeId id;
    int    layer;
    bool   tombstone;
    std::vector<float>               vec;
    std::vector<std::vector<NodeId>> neighbors;
};

std::vector<NodeData> snapshot() const;   // pack all nodes into a portable struct
void restore(NodeId entry_point, int max_layer,
             size_t live_count, std::vector<NodeData> nodes); // rebuild Impl from NodeData
```

`snapshot()` iterates the internal `nodes` map and packs each node's fields into a `NodeData`. `restore()` unpacks `NodeData` back into `nodes` and `vecs`, bypassing the normal `insert()` algorithm. `insert()` runs HNSW's random layer assignment and greedy graph construction — calling it during deserialization would produce a different graph than what was serialized.

These methods are part of `HnswIndex`'s public API but are not part of the regular insert/search/remove interface. They are documented as "for GraphSerializer use only."

## Crash Safety

GraphSerializer writes to a permanent path directly (`O_CREAT | O_TRUNC`) rather than writing to a temporary file and renaming. The tradeoff:

**With direct write:** if a crash happens mid-write, the snapshot file is partially written and corrupt. `deserialize()` will detect this either via a bad magic number (if the header was not completed) or a truncated read error (if the file ends mid-node). In either case, the snapshot is unusable and the system must fall back to a full WAL replay from the beginning.

**With tmp + rename:** if a crash happens mid-write, the old snapshot file is still intact. The rename is atomic — the old snapshot survives until the new one is complete. Recovery can use the previous checkpoint plus the WAL from that point.

The current implementation uses direct write for simplicity. A production system would use tmp + rename (the same pattern as WAL `truncate_before`) to guarantee that a valid snapshot always exists on disk.

Because `truncate_before()` is called only after `serialize()` returns successfully, a crash during serialization leaves the WAL intact — the system recovers by replaying the full WAL without the new checkpoint. Correctness is preserved; only recovery time is affected.

## Node Order in the File

Nodes are written in `std::unordered_map` iteration order, which is not deterministic across runs. This is fine: deserialization reads them all into the map regardless of order. The graph's behavior (search results, neighbor lists) depends only on the map contents, not on which order they were inserted into the map.

## Tombstoned Nodes

Tombstoned nodes are included in the snapshot. They are serialized with `tombstone = 1`. After deserialization, they remain in the `nodes` map but not in `live_count`. Search traversal may still follow edges to tombstoned nodes as graph intermediaries, but tombstoned nodes are filtered from search results.

Excluding tombstoned nodes from the snapshot would produce a smaller file but would break the graph: if node A is tombstoned but is still in B's neighbor list, excluding A leaves a dangling reference. Cleaning up those references requires a compaction step (not implemented yet).

## API

### `GraphSerializer::serialize(index, path)`

**Purpose:** write a complete snapshot of `index` to `path`.

Calls `index.snapshot()` to get a copy of all node data, then writes the file header followed by per-node records. Closes the file on completion.

**Parameters:**
- `index` — the `HnswIndex` to snapshot
- `path` — destination file path; created or overwritten

**Throws:** `std::runtime_error` on I/O failure.

---

### `GraphSerializer::deserialize(path, cfg) → unique_ptr<HnswIndex>`

**Purpose:** reconstruct an `HnswIndex` from a snapshot file.

Reads the file header, validates `magic` and `version`, then reads each node record into a `vector<NodeData>`. Constructs a new `HnswIndex(cfg)` and calls `restore()` to populate it.

**Parameters:**
- `path` — snapshot file path
- `cfg` — `HnswConfig` for the new index; `cfg.dim` is used to read the correct number of floats per vector

**Returns:** a fully operational `HnswIndex` ready for search and insert.

**Throws:** `std::runtime_error` if the file cannot be opened, magic does not match, version is unsupported, or the file is truncated.

## Design Decisions

**Why serialize vectors inside the graph file?**

The graph file is meant to be a complete, self-contained checkpoint. If vectors were omitted, `deserialize()` would need an additional argument (a VectorFile or a callback to fetch vectors by node ID). This couples the two components at restore time and complicates the recovery sequence. The snapshot file is larger, but checkpoints are infrequent and the file is only read once per restart.

**Why not fsync the snapshot file?**

`serialize()` does not call `fsync` before returning. On a crash, the snapshot file may not have reached persistent storage. However, `truncate_before()` is not called until after `serialize()` returns — if the snapshot did not make it to disk, the WAL is still intact and a full replay is possible. An fsync would guarantee the snapshot is durable before the WAL is truncated, at the cost of a synchronous disk write. This is a correctness vs. performance tradeoff left for a later checkpoint policy layer to decide.

**Why `int32_t` for `layer` and `max_layer`?**

`HnswNode::layer` is `int` in the in-memory representation (consistent with the HNSW paper's use of signed integer levels). Using `int32_t` in the file format makes the field size explicit and portable — `int` width varies by platform.

**Why magic `0x47524150480A0000`?**

The first 5 bytes are `GRAPH` in ASCII. The `0x0A` is a newline — following the same convention as VectorFile's magic, where a newline byte makes the file identifiable as a VortexDB file to tools like `file(1)`. The trailing two zero bytes pad the magic to 8 bytes.
