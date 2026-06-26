# Metadata Serializer Design

## What is MetadataSerializer?

MetadataSerializer writes the complete in-memory MetadataIndex to a binary file (`metadata.bin`) at checkpoint time, and reads it back on startup. It plays the same role for MetadataIndex as GraphSerializer plays for HnswIndex — a one-shot utility called once per checkpoint, not a runtime component.

## Where it Fits in the System

During normal operation, MetadataIndex lives entirely in memory:

```
insert(id, vec, meta) → MetadataIndex (in memory)
                         ↓
                        WAL (metadata bytes appended to each Insert record)
```

MetadataSerializer is only invoked at checkpoint time:

```
checkpoint():
  1. GraphSerializer::serialize(index, "graph.bin")
  2. MetadataSerializer::serialize(meta, "metadata.bin")   ← here
  3. wal.truncate_before(current_lsn)
```

At restart, recovery loads both snapshots then replays the WAL:

```
startup:
  1. GraphSerializer::deserialize("graph.bin")     → HnswIndex
  2. MetadataSerializer::deserialize("metadata.bin") → MetadataIndex
  3. wal.replay(0, dim, on_insert, on_delete)      → applies post-checkpoint records
                                                      to both HnswIndex and MetadataIndex
```

## Why a Separate Snapshot File?

MetadataIndex is rebuilt from two sources:

- **Before the checkpoint LSN**: captured in `metadata.bin`
- **After the checkpoint LSN**: replayed from WAL (each Insert record carries metadata bytes)

Without `metadata.bin`, every recovery would need to replay the entire WAL history from the beginning — the same problem that motivates `graph.bin`. With the snapshot, recovery only replays the WAL records written after the last checkpoint.

## Why Not Dump Memory Directly?

MetadataIndex uses `std::unordered_map` and `std::vector` internally. These contain heap pointers that are only valid within the current process — writing raw struct bytes to disk would produce garbage on the next run. Instead, MetadataSerializer iterates over all entries and serializes each (field, value, id) triple as length-prefixed strings and fixed-width numbers.

## Accessing the Data: for_each_string / for_each_numeric

MetadataIndex uses the Pimpl pattern — its internal maps are defined in the `.cpp` file and are invisible to any other class. MetadataSerializer cannot access them directly.

MetadataIndex exposes two iteration callbacks for this purpose:

```cpp
idx.for_each_string([&](const std::string& field, const std::string& value, NodeId id) {
    // called once per (field, value, id) string entry
});
idx.for_each_numeric([&](const std::string& field, double value, NodeId id) {
    // called once per (field, value, id) numeric entry
});
```

This is the same approach HnswIndex uses with `snapshot()` — the index owns its data and decides how to expose it; the serializer just consumes what it is given.

## On-Disk Format

```
[magic:         8B = 0x4D455441440A0000]   "METAD\n\0\0" — identifies file type
[version:       4B = 1]
[num_strings:   4B]                         total string entries
[num_numerics:  4B]                         total numeric entries

string entries (num_strings ×):
  [field_len: 4B][field bytes]
  [val_len:   4B][val bytes]
  [id:        4B]

numeric entries (num_numerics ×):
  [field_len: 4B][field bytes]
  [value:     8B double]
  [id:        4B]
```

All integers are little-endian. The magic number serves the same purpose as in `graph.bin` — it prevents accidentally loading a wrong file and allows the `file(1)` command to identify the format.

String lengths are stored as `uint32_t` immediately before the string bytes. This allows arbitrary-length field and value names without a fixed-size header.

## Crash Safety

`metadata.bin` is written before `wal.truncate_before()`, following the same snapshot-before-truncate ordering as `graph.bin`:

```
crash after serialize, before truncate:
  metadata.bin is complete ✓
  WAL still has all post-checkpoint records ✓
  recovery: load snapshot + replay WAL → correct state ✓

crash during serialize (partial write):
  metadata.bin is incomplete — magic check or read will fail
  WAL is still intact ✓
  recovery: no snapshot → empty MetadataIndex + full WAL replay → correct state ✓
```

A partial `metadata.bin` is safe to ignore. If `deserialize()` throws (bad magic, truncated file), the Engine falls back to an empty MetadataIndex and relies entirely on WAL replay to rebuild it. This is the same fallback used when no `metadata.bin` exists yet (before the first checkpoint).
