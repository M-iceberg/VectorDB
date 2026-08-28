# VectorDB API Reference

---

## Python SDK

### Open a database

```python
import vectordb

db = vectordb.open(data_dir: str) -> VectorDB
```

Opens or creates a database rooted at `data_dir`. On open, runs crash recovery for every collection found on disk (load checkpoint + replay WAL).

---

### Collection management

```python
db.create_collection(name: str, dimension: int, metric: str = "l2") -> None
```

Creates a new collection. `metric`: `"l2"` | `"cosine"` | `"ip"` (inner product).  
Raises `RuntimeError` if a collection with that name already exists.

```python
db.drop_collection(name: str) -> None
```

Destroys the collection and deletes all on-disk files (`wal.log`, `vectors.vdb`, `graph.bin`, `metadata.bin`, `schema.bin`). Irreversible.

```python
db.list_collections() -> list[dict]
```

Returns a list of `{"name": str, "dimension": int, "metric": str}` dicts for every open collection.

---

### Insert

```python
db.insert(
    collection: str,
    *,
    ids:      str | int | list[str | int],
    vectors:  np.ndarray,               # shape (dim,) or (N, dim), dtype float32
    metadata: dict | list[dict] = None, # optional, one dict per vector
) -> None
```

Inserts one or many vectors. `ids` and `vectors` must have the same length. IDs are stored and returned as strings; integer IDs are coerced to strings automatically.

**Metadata fields:**
- String values → stored as string, queryable with `$eq` (default equality filter)
- Numeric values (`int`, `float`) → stored as `double`, queryable with `$gte` / `$lte`
- Boolean values → stored as `"1"` / `"0"` strings

Each insert is WAL-synced (`fdatasync`) before returning.

---

### Search

```python
db.search(
    collection: str,
    *,
    query:     np.ndarray,          # shape (dim,), dtype float32
    top_k:     int = 10,
    ef_search: int = 64,            # beam width; higher = better recall, lower QPS
    filters:   dict = None,         # optional metadata filter
) -> list[dict]
```

Returns a list of `{"id": str, "distance": float}` dicts sorted by ascending distance.

`ef_search` must be ≥ `top_k`. Higher values increase recall at the cost of QPS. Recommended: `ef_search = 2 × top_k` to `5 × top_k` for most workloads. Set `ef_search = 200` for >99% Recall@10 on SIFT-1M.

**Filter syntax:**

```python
# String equality
filters = {"category": "ml"}

# Numeric range (inclusive bounds)
filters = {"year": {"$gte": 2023, "$lte": 2025}}

# Combined (AND — all keys must match)
filters = {"category": "ml", "year": {"$gte": 2023}}
```

Supported operators: `$gte` (≥), `$lte` (≤). Filters are applied post-HNSW (post-filtering). Recall degrades at selectivity < 1% — see [benchmarks](benchmarks.md#filtered-search--selectivity-vs-qps-and-recall-day-28).

---

### Delete

```python
db.remove(collection: str, id: str | int) -> None
```

Soft-deletes the vector. The ID never appears in subsequent search results. The delete is WAL-synced before returning. Storage is reclaimed on the next checkpoint.

---

### Checkpoint

```python
db.checkpoint(collection: str) -> None
```

Serializes the HNSW graph and metadata index to `graph.bin` and `metadata.bin`, writes the checkpoint LSN, then truncates the WAL up to that LSN.

Call periodically to bound recovery time. A checkpoint every 50K inserts keeps recovery under ~60s on typical hardware. Without checkpoints, recovery replays the full WAL at O(N log N) cost.

---

## gRPC API

Defined in [`proto/vectordb.proto`](../proto/vectordb.proto). The gRPC server (`server/grpc_server.h`) is a thin wrapper around `Engine` — the same object used by the Python SDK.

> **Status:** gRPC interface is designed and the proto is defined. The C++ server implementation (`GrpcServer`) is stubbed but not wired up. Intended as Phase 1.5 work to support non-Python clients.

### Service definition

```protobuf
service VectorDB {
  rpc CreateCollection (CreateCollectionRequest) returns (CreateCollectionResponse);
  rpc DropCollection   (DropCollectionRequest)   returns (DropCollectionResponse);
  rpc Insert           (InsertRequest)           returns (InsertResponse);
  rpc Delete           (DeleteRequest)           returns (DeleteResponse);
  rpc Search           (SearchRequest)           returns (SearchResponse);
}
```

### Messages

```protobuf
message CreateCollectionRequest { string name = 1; uint32 dim = 2; string metric = 3; }
message CreateCollectionResponse { bool ok = 1; }

message InsertRequest {
  string         collection = 1;
  uint32         id         = 2;
  repeated float vector     = 3 [packed = true];
}
message InsertResponse { bool ok = 1; }

message DeleteRequest  { string collection = 1; uint32 id = 2; }
message DeleteResponse { bool ok = 1; }

message SearchRequest {
  string         collection = 1;
  repeated float query      = 2 [packed = true];
  uint32         top_k      = 3;
  uint32         ef_search  = 4;
}
message SearchResult   { uint32 id = 1; float distance = 2; }
message SearchResponse { repeated SearchResult results = 1; }
```

**Note:** the proto currently uses `uint32 id` (integer). The Python SDK evolved to support string IDs; the proto would need to be updated to `string id` to match.

---

## Phase 2 — Distributed design (not implemented)

This section describes how VortexDB would be extended for production-scale distribution. It is intentionally not implemented — the goal of this project is to understand single-node internals, not build a cloud service.

### Motivation

Single-node limits:
- **Capacity:** one machine's RAM limits how many vectors can be searched at HNSW speed. At 1B vectors × 512 B/vec = 512 GB — exceeds a single node.
- **Write throughput:** single `shared_mutex` serializes all writes. High write concurrency requires partitioning.
- **Availability:** single node = single point of failure. Replicas needed for production SLA.

### Proposed architecture

```
                    ┌─────────────────┐
   Client ─────────►│  Query Router    │
                    │  (stateless)     │
                    └────────┬────────┘
                             │ hash(vector_id) % num_shards
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌─────────┐   ┌─────────┐   ┌─────────┐
         │ Shard 0 │   │ Shard 1 │   │ Shard 2 │
         │ Primary │   │ Primary │   │ Primary │
         │ VortexDB│   │ VortexDB│   │ VortexDB│
         └────┬────┘   └────┬────┘   └────┬────┘
              │              │              │
         ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
         │ Replica │   │ Replica │   │ Replica │
         └─────────┘   └─────────┘   └─────────┘
```

**Sharding:** vectors partitioned by consistent hashing on vector ID. Each shard owns 1/N of the vector space. A shard is one VortexDB node (the existing single-node code, unchanged).

**Replication:** each shard has one primary and one or more read replicas. Primary streams WAL records to replicas asynchronously. Replicas serve read queries; primary handles all writes.

**Query fan-out:** a k-NN query is broadcast to all shards. Each shard returns its local top-k. The query router merges and re-ranks across shards, returning the global top-k. This requires `O(shards × top_k)` merging work at the router — cheap.

**Metadata coordination:** `create_collection` / `drop_collection` must be consistent across all shards. Options:
- External consensus (etcd / ZooKeeper) for collection-level metadata
- Two-phase commit (simpler, lower availability during coordinator failure)

### What changes in the single-node code

Almost nothing. Each shard still runs the existing VortexDB binary. Changes are confined to:
1. A new `router` service that parses queries and fans them out via gRPC
2. WAL streaming from primary to replicas (add a gRPC stream endpoint to the existing server)
3. A collection registry in etcd to track which shards own which collections

The single-node Engine, HnswIndex, WAL, and MetadataIndex are unchanged. This is the value of having clean layer boundaries from the start.

### Trade-offs not addressed

- **Cross-shard nearest-neighbor accuracy:** if vectors cluster non-uniformly, some shards hold denser neighborhoods. A query near a cluster boundary may have true nearest neighbors split across two shards. With hash-based sharding, fan-out to all shards always finds the global top-k — correctness is guaranteed. However, if you switch to vector-space-region sharding (assign vectors to shards by spatial location, e.g. k-means centroids) to avoid querying all shards, boundary queries still need to fan out to multiple adjacent regions to be correct. Fanning out to all shards would fix it, but that defeats the entire purpose of region sharding, which was to query fewer shards. In practice, production systems using region-based partitioning either accept slightly lower recall near shard boundaries, or use an overlap zone where boundary vectors are replicated to neighboring shards.
- **Hot shard:** popular ID ranges concentrate writes on one shard. Mitigated by consistent hashing with virtual nodes.
- **Rebalancing:** adding a shard requires migrating vectors. Non-trivial with a live index.
