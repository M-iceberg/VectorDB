// -----------------------------------------------------------------------------
// grpc_server.h — gRPC service implementation  (Optional / Phase 1.5)
//
// Wraps Engine and exposes it as a gRPC service defined in proto/vectordb.proto.
// Not compiled in Phase 1 MVP — requires protobuf + gRPC dependencies.
//
// When implemented, GrpcServer will:
//   - Accept connections on a configurable port
//   - Translate each RPC into an Engine method call
//   - Stream SearchResponse results back to the client
//   - Map C++ exceptions to gRPC status codes
// -----------------------------------------------------------------------------
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
