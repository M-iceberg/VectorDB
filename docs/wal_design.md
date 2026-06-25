# WAL Design

## What is the WAL?

The WAL (Write-Ahead Log) is a crash-safe durability layer for the vector database. Every insert and delete operation is written to the WAL *before* it is applied to the in-memory HNSW index. On crash and restart, the engine replays the WAL to reconstruct any operations that happened after the last checkpoint.

**Why is this necessary?** The HNSW graph structure lives in memory and is periodically checkpointed to disk. Vector data is backed by memory-mapped files (VectorFile, Day 13), but the graph topology — neighbor lists, layer assignments, entry point — is reconstructed in memory on startup from the checkpoint. Memory is volatile: on crash or power loss, any in-memory state that hasn't been checkpointed is gone. The WAL provides a persistent record of every operation since the last checkpoint so they can be replayed on restart to bring the index back to its pre-crash state.

**Why write to WAL before applying to the index?** If the process crashes between the two steps, we need to know what state to recover to. Writing to the WAL first guarantees that any acknowledged operation is on disk. On restart:
- If a record is in the WAL → the operation was durable, replay it into the index.
- If a record is not in the WAL → the client never got a success response, so it knows to retry. Nothing to recover.

If we applied to the index first and then wrote the WAL, a crash in between would leave the index updated but the WAL missing that operation — on restart, we'd have no record of it and the data would be silently lost.

**What keeps the WAL data from being lost on crash?** `sync()` forces the OS to flush the WAL data from its in-memory page cache to disk before returning success to the client.

A normal `write()` call only puts data into the OS page cache — an in-memory buffer managed by the kernel. The OS flushes it to disk asynchronously, at its own pace (potentially seconds later). A crash in that window loses the data.

```
write()  →  OS page cache (memory)  →  (OS, async)  →  disk
```

`sync()` blocks until the disk confirms the data has reached non-volatile storage:

```
write()  →  OS page cache
sync()   →  force flush  →  disk controller  →  physical media
            blocks until disk confirms write complete
```

On macOS, `F_FULLFSYNC` is used instead of `fdatasync`. macOS's `fdatasync` only flushes data to the disk controller's write cache — it does not guarantee the data has reached flash storage. `F_FULLFSYNC` sends an explicit flush-cache command to the drive, ensuring data is on the physical medium before returning. On Linux, `fdatasync` provides the equivalent guarantee.

The client receives a success response only after `sync()` returns. This means: once the client sees success, the WAL record is on physical storage and will survive any crash.

Without the WAL, any insert or delete that happened since the last checkpoint would be lost on crash. The WAL bounds that loss to at most one client batch.

## How WAL, Checkpoint, and VectorFile Work Together

**During normal operation:**
- Vector data is stored in a memory-mapped file (VectorFile) — on disk, accessed through memory addresses.
- The graph structure (neighbor lists, layer assignments, entry point) lives in memory.

**Checkpoint (done periodically):**
- Serialize the current graph structure to disk (snapshot).
- Record the LSN of the last operation included in the snapshot.
- Truncate the WAL before that LSN — those records are no longer needed.

**After a crash, on restart:**
1. Load the most recent checkpoint → rebuild the graph to its checkpointed state.
2. Replay WAL records from the checkpoint LSN onward → re-apply any insert/delete that happened after the checkpoint.
3. The graph is fully recovered to its pre-crash state.

The WAL does not rebuild the entire graph from scratch. It only covers the gap between the last checkpoint and the crash. The more frequently checkpoints are taken, the fewer WAL records need to be replayed.

**Why loading the checkpoint doesn't require replaying all LSNs:**

The WAL is a sequence of operations, each assigned a monotonically increasing LSN:

```
LSN 0  →  Insert(node_id=5, vec=[...])
LSN 1  →  Insert(node_id=6, vec=[...])
LSN 2  →  Delete(node_id=3)
LSN 3  →  Insert(node_id=7, vec=[...])
...
```

A checkpoint is not just a record of which LSN was reached — it is a **complete snapshot of the graph at that moment**: all nodes, all neighbor lists at every layer, the entry point, and layer assignments. Loading the checkpoint directly restores the graph to that exact state without re-executing any prior operation. The effect of every LSN before the checkpoint is already baked into the snapshot. Only the operations after the checkpoint LSN need to be replayed from the WAL to bring the graph fully up to date.

## Role in the System

```
Client request (insert / delete)
    │
    ▼
WAL.append()          ← write to disk first
    │
    ▼
WAL.sync()            ← fdatasync: guarantee durability before responding
    │
    ▼
HnswIndex.insert()    ← apply to in-memory graph
    │
    ▼
Return success to client
```

On crash between `sync()` and `HnswIndex.insert()`, the operation is in the WAL but not in the graph. On restart, the engine replays the WAL to re-apply it.

On crash before `sync()`, the operation is not yet durable. The client gets no success response, so it knows to retry.

## On-Disk Format

The WAL file is a flat sequence of records written one after another, never overwritten. Each record has two parts: a **header** and a **payload**.

**Header** is a fixed-size prefix (17 bytes) that appears at the start of every record. The reader always reads the header first — since its size is fixed and known, no prior information is needed to find it. The header tells the reader how to interpret the rest of the record:

```
[crc32: 4B][payload_length: 4B][type: 1B][timestamp_us: 8B]
```

| Field | Size | Description |
|-------|------|-------------|
| `crc32` | 4B | CRC32 checksum over everything after this field — used to detect corruption |
| `payload_length` | 4B | How many bytes of payload follow — tells the reader exactly how much to read next |
| `type` | 1B | What kind of operation this is: Insert (1), Delete (2), Checkpoint (3) |
| `timestamp_us` | 8B | Unix timestamp in microseconds when the record was written |

**Payload** is the variable-length data that follows the header. Its length is given by `payload_length` in the header. The reader first reads the fixed-size header, extracts `payload_length`, then reads exactly that many bytes for the payload.

```
[crc32: 4B][payload_length: 4B][type: 1B][timestamp_us: 8B][payload: N B]
 ←————————————————— header (17B, fixed) ——————————————————→←— payload —→
```

**Payload formats:**
- `Insert`: `[node_id: 4B][vec: dim × 4B]` — node id followed by the raw float vector. The full vector is stored in the WAL so recovery is self-contained: the engine can replay an insert purely from the WAL record without reading from VectorFile.
- `Delete`: `[node_id: 4B]`

Storing the full vector in the WAL (rather than just a VectorFile slot reference) preserves strict WAL-first ordering. The alternative — writing to VectorFile first, then recording the slot in the WAL — breaks the invariant: if the process crashes between the VectorFile write and the WAL append, the vector exists on disk but there is no WAL record of it, so recovery has no way to know about it. With the vector in the WAL payload, the WAL is always written first and is the single source of truth for recovery.

The checkpoint LSN is stored in the checkpoint file itself, not in the WAL. Recovery reads the checkpoint file to get the LSN, then calls `iterate(checkpoint_lsn, cb)` to replay only the WAL records after that point.

**Why separate header and payload?** The header is fixed-size so the reader always knows where to start. Without it, the reader would have no way to know where one record ends and the next begins. `payload_length` in the header is what allows variable-length payloads — different operations (Insert vs Delete) have different payload sizes, and the header tells the reader how much to read each time.

## LSN (Log Sequence Number)

LSN is the sequential index of a record in the file — first record is LSN 0, second is LSN 1, and so on. It is **not stored inside WAL records**. To find a record's LSN, you count from the beginning of the file. This is like an array element knowing its index by its position — the element doesn't need to store the index itself.

**LSN is stored in two places:**
- **In memory** — `next_lsn` variable in `Impl`, incremented after every `append()`. Tracks what LSN to assign to the next record.
- **In the checkpoint file** — when a checkpoint is written, the current LSN is stored in the checkpoint file. On restart, recovery reads this value to know where to start replaying from.

**Consequences of not storing LSN in records:**

`iterate(start_lsn, cb)` cannot seek directly to `start_lsn` — it must scan from the beginning of the file, counting records until it reaches `start_lsn`, then start calling `cb`. WAL records are variable-length with no index, so there is no way to jump to a specific LSN without reading everything before it.

`truncate_before(lsn)` similarly cannot truncate in-place. It must rewrite the file: read all records with LSN >= `lsn` into a temporary file, then rename the temporary file over the original. `rename` is atomic on POSIX — the old file is replaced in one step with no intermediate state.

```
wal.log:   [LSN 0][LSN 1][LSN 2][LSN 3][LSN 4]
                                  ↑ checkpoint_lsn = 3

truncate_before(3):
  temp.log: [LSN 3][LSN 4]
  rename temp.log → wal.log

wal.log:   [LSN 3][LSN 4]   (LSN 0/1/2 removed)
```

Both operations are sequential reads — WAL is designed for sequential access only, not random access. For recovery this is fine: replay always starts at `checkpoint_lsn` and reads forward to the end.

**`base_lsn` and the sidecar file:**

After `truncate_before(3)`, the WAL file contains only `[LSN 3][LSN 4]`. But `iterate()` reads records by position — it has no way to know that the first record in the file is LSN 3 rather than LSN 0. Without extra information, it would assign LSN 0 to the first record, making `replay(checkpoint_lsn=3)` replay nothing (all records would appear to have LSN < 3).

To fix this, `truncate_before` writes the new base LSN to a companion sidecar file (`wal.log.base`) — a single `uint64_t`. On open, `Wal` reads this sidecar to find `base_lsn` and starts counting from there instead of from 0. The sidecar is updated atomically after the WAL file is renamed, so there is no window where the WAL file and the sidecar are inconsistent after a crash.

## Crash Recovery

On `Wal` construction, the WAL file on disk (the file at the `path` passed to the constructor) is scanned from the beginning:
1. Read each record header and payload
2. Verify CRC — if mismatch, stop (corrupt tail from partial write on crash)
3. Truncate the file to the last valid record
4. Set `next_lsn` to the count of valid records

This ensures the WAL is always in a consistent state before new appends begin. A partial tail record left by a crash is silently discarded — the operation was never acknowledged to the client, so discarding it is safe.

## API

### `Wal(path)`

**Purpose:** initialize the WAL, recover from any previous crash, and prepare for new appends.

On first use, creates a new empty WAL file at `path`. On subsequent opens (after a clean shutdown or crash), scans the existing file to find the last valid record, truncates any corrupt tail left by a partial write, and sets `next_lsn` to the count of valid records so new appends continue from the right LSN.

**Parameters:**
- `path` — file path for the WAL (e.g. `data/wal.log`)

**Throws:** `std::runtime_error` if the file cannot be opened for writing.

---

### `append(type, payload, length) → Lsn`

**Purpose:** record one operation in the WAL before applying it to the index.

This is always called first — before the operation is applied to the HNSW index. Writing the WAL record first ensures that even if the process crashes before the index is updated, the operation can be replayed on restart.

**Parameters:**
- `type` — `WalRecordType::Insert`, `Delete`, or `Checkpoint`
- `payload` — pointer to the record payload bytes
- `length` — byte length of the payload

**Returns:** the LSN assigned to this record (monotonically increasing `uint64_t`). The checkpoint uses this LSN to mark the boundary between "already snapshotted" and "needs replay".

Uses `writev` to write header and payload in a single syscall, minimizing the crash window where a partial record could be written.

**Note:** the write is buffered by the OS page cache — the record is not durable until `sync()` is called.

---

### `sync()`

**Purpose:** make all previously appended records durable on disk before telling the client the operation succeeded.

After `sync()` returns, all WAL records written since the last `sync()` are guaranteed to be on physical storage. The client success response is sent only after this call returns — this is what makes the WAL's crash-safety guarantee meaningful.

Multiple `append()` calls can be batched before a single `sync()` to amortize the cost of the flush across several operations.

On macOS: `fcntl(fd, F_FULLFSYNC)` — forces the drive to flush its write cache to physical media.
On Linux: `fdatasync(fd)` — flushes data to the disk controller.

**Throws:** `std::runtime_error` on failure.

---

### `iterate(start_lsn, cb)`

**Purpose:** replay WAL records after a checkpoint LSN during crash recovery.

Reads records sequentially from the beginning of the file, skipping any with LSN < `start_lsn`, and calls `cb` for each record at or after `start_lsn`. Stops silently on a corrupt or truncated record (corrupt tail is expected after a crash and is not an error).

Used at startup: load the checkpoint first (which gives a `checkpoint_lsn`), then call `iterate(checkpoint_lsn, cb)` to re-apply all operations that happened after the checkpoint. This brings the index back to its pre-crash state without replaying the full history.

**Parameters:**
- `start_lsn` — first LSN to deliver to `cb`; records before this LSN are skipped. Pass `0` to iterate all records.
- `cb` — callback invoked for each record: `void(Lsn, WalRecordType, const void* payload, uint32_t length)`

---

### `truncate_before(lsn)`

**Purpose:** bound WAL file growth by removing records that are no longer needed for recovery.

Called after a successful checkpoint: once the graph state up to `lsn` is safely on disk in the checkpoint file, the WAL records before `lsn` are redundant — recovery can start from the checkpoint instead. Removing them keeps the WAL file from growing indefinitely.

**Parameters:**
- `lsn` — truncation point; all records with LSN < `lsn` are removed

**Implementation:**
1. Read all records with logical LSN >= `lsn` directly from the WAL file.
2. Write them to a temp file (`wal.log.tmp`).
3. Write the new `base_lsn` to the sidecar file (`wal.log.base`).
4. `rename(tmp, wal.log)` — atomic on POSIX, no intermediate state visible on crash.
5. Update `base_lsn` in memory and reopen the write fd on the new file.

The sidecar must be written **before** the rename (step 3 before step 4). If the order were reversed and the process crashed between rename and sidecar write, the WAL file would contain only records >= `lsn` but the sidecar would still have the old `base_lsn`. On restart, `iterate()` would assign LSNs starting from the old base — the first record in the file (actually LSN `lsn`) would be treated as LSN 0, so `replay(checkpoint_lsn=lsn)` would see no records >= `lsn` and replay nothing. Records written after the checkpoint but before the crash would be lost.

Writing the sidecar first means a crash between steps 3 and 4 leaves the old WAL file intact with the new `base_lsn` recorded. On restart, `iterate()` starts from the new base — the old WAL contains all records >= `lsn`, so some already-checkpointed records may be replayed again. This is safe because replay is idempotent (inserting a node that already exists is a no-op).

---

### `replay(start_lsn, on_insert, on_delete)`

**Purpose:** reconstruct in-memory state from WAL records after a checkpoint.

Wraps `iterate()` and dispatches each record to the appropriate callback based on record type. Called during crash recovery after loading the checkpoint snapshot.

**Parameters:**
- `start_lsn` — first LSN to replay; pass `checkpoint_lsn` from the checkpoint file
- `on_insert(id, vec, dim)` — called for each Insert record; should call `hnsw.insert(id, vec)` and `vector_file.append(vec)`
- `on_delete(id)` — called for each Delete record; should call `hnsw.remove(id)`

**Implementation:** calls `iterate(start_lsn, cb)` and inside the callback, parses the payload by type:
- Insert: reads `node_id` (first 4 bytes), then `vec` (remaining bytes), computes `dim = (len - 4) / 4`
- Delete: reads `node_id` (4 bytes)
- Checkpoint: silently skipped — checkpoint records mark truncation points, not operations to replay

---

### `current_lsn() → Lsn`

**Purpose:** inspect the next LSN without appending anything.

Useful for recording the current WAL position before taking a checkpoint — the checkpoint stores this value so recovery knows where to start replaying from.

**Returns:** the LSN that will be assigned to the next `append` call. Equal to `base_lsn + number of valid records in the file`.

## Design Decisions

**What is CRC32?**
CRC32 (Cyclic Redundancy Check) produces a 32-bit integer from an arbitrary block of data. The individual bits have no independent meaning — the value is the remainder of a polynomial division over the input data, treated as a binary polynomial, divided by a fixed generator polynomial (`0xEDB88320`, the IEEE 802.3 standard). The practical implementation uses a lookup table: for each input byte, index into the table and XOR with the running value, which avoids doing actual polynomial arithmetic.

CRC32 guarantees detection of all single-bit errors and all burst errors of length ≤ 32 bits — sufficient for catching partial writes and bit flips. It cannot correct errors, only detect them. It is also not cryptographically secure (two different inputs can produce the same CRC32), but that is not a requirement here.

In our WAL, CRC32 covers everything in the record after the `crc32` field itself: `payload_length + type + timestamp_us + payload`. The `crc32` field cannot cover itself — its value is not known until the computation is complete.

**Why CRC32 and not a stronger hash?**
CRC32 is sufficient for detecting accidental corruption (partial writes, bit flips). The WAL is not a security boundary — it doesn't need to be collision-resistant. CRC32 is fast and simple to implement without dependencies.

**Why `writev` for append?**
Writing header and payload in a single `writev` syscall reduces the crash window. If the process crashes between two separate `write` calls, we'd have a header with no payload. With `writev`, the kernel writes both atomically at the syscall boundary (though not at the disk level — the CRC catches any partial write on the next open).

**Why implicit LSN?**
LSN is the sequential index of a record in the file — it is inherently derivable by counting records from the beginning. Storing it explicitly in each record would be redundant: the record would be telling the reader information the reader already knows from counting. Like an array element storing its own index, it adds no information.

**Why `F_FULLFSYNC` on macOS?**
`fdatasync` on macOS does not guarantee that data reaches non-volatile storage — it only flushes to the disk's write cache. `F_FULLFSYNC` forces the drive to flush its cache, providing the same durability guarantee as `fdatasync` on Linux with a write-through or write-back drive.
