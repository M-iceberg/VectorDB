#pragma once
// gRPC service implementation — added Phase 1.5 / Optional.
// Requires protobuf + gRPC dependencies; not compiled in Phase 1 MVP.
// See proto/vectordb.proto for the service definition.

namespace vectordb {
class Engine;
}

namespace vectordb {

// TODO Optional: GrpcServer wraps Engine and exposes it over gRPC.
//               Implement after Python SDK is working (Day 21+).

}  // namespace vectordb
