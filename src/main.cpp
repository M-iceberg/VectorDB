// -----------------------------------------------------------------------------
// main.cpp — standalone binary entry point
//
// Prints the version string and exits. The database is accessed via the
// Python SDK (vectordb.open) or the gRPC server (grpc_server.h, Phase 1.5).
//
// Usage:
//   vectordb --data-dir /var/lib/vectordb [--port 50051]
// -----------------------------------------------------------------------------
#include <iostream>
#include "core/version.h"

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "vectordb v" << VECTORDB_VERSION << "\n";
    return 0;
}
