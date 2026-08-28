# Crash-Atomic Checkpoints

VectorDB publishes checkpoints as immutable generations. The collection
directory contains a small `checkpoint.current` manifest and exactly one live
`checkpoint-N/` directory after successful cleanup:

```text
collection/
  schema.bin
  vectors.vdb
  wal.log
  wal.log.base
  checkpoint.current
  checkpoint-7/
    graph.bin
    metadata.bin
    id_map.bin
```

`checkpoint.current` records the generation plus the byte size and CRC32 of
all three snapshot files. Startup validates the manifest CRC and every file
digest before deserializing anything. Root-level snapshot files from the old
format remain readable for backward compatibility.

## Publication protocol

1. Write `graph.bin`, `metadata.bin`, and `id_map.bin` into a new hidden
   generation directory.
2. `fsync` every file and the temporary directory.
3. Atomically rename the directory to `checkpoint-N/`, then `fsync` the
   collection directory.
4. Write and `fsync` `checkpoint.current.tmp`; atomically rename it to
   `checkpoint.current`, then `fsync` the collection directory again.
5. Truncate the WAL only after the new manifest is durable.
6. Remove older and orphaned generations after WAL truncation succeeds.

The manifest rename is the commit point. A process restart can therefore see
only the previous complete generation or the new complete generation, never a
partially overwritten snapshot.

## WAL ordering

Insert IDs are reserved before graph construction. Vector and metadata records
are appended to the WAL and synchronously committed before the HNSW graph,
mmap vector file, ID maps, or metadata index are updated. Batch inserts use one
group commit for the entire batch and then construct the graph in parallel
using the reserved IDs.

If the process dies immediately after WAL sync but before graph construction,
startup replays the committed records and reconstructs the missing in-memory
state.

## Fault-injection coverage

`EngineRecoveryTest.AtomicCheckpointSurvivesEveryPublishPhase` sends SIGKILL
at four deterministic boundaries:

- after snapshot files are synced;
- after the generation directory is published;
- after the manifest is published;
- after WAL truncation.

`EngineRecoveryTest.WalCommitPrecedesBatchGraphMutation` kills the process
after batch WAL sync but before HNSW construction. Every case must recover all
committed vectors. `CorruptPublishedSnapshotIsRejected` flips a byte in a
published graph and verifies that checksum validation rejects it during open.
