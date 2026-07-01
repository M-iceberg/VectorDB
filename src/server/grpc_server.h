// -----------------------------------------------------------------------------
// grpc_server.h — gRPC service implementation (Phase 1.5, not yet wired up)
//
// Wraps Engine and exposes it as a gRPC service defined in proto/vectordb.proto.
// Requires protobuf + gRPC dependencies; not compiled in the current build.
//
// When implemented, GrpcServer will:
//   - Accept connections on a configurable port
//   - Translate each RPC into an Engine method call
//   - Map C++ exceptions to gRPC status codes
// -----------------------------------------------------------------------------
#pragma once

namespace vectordb { class Engine; }

namespace vectordb {

// GrpcServer wraps Engine and exposes it over the gRPC service defined in
// proto/vectordb.proto. See docs/api.md for the full interface specification.
class GrpcServer;

}  // namespace vectordb
