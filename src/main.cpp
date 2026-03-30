// -----------------------------------------------------------------------------
// main.cpp — Server entry point
//
// Day 1: prints the version string and exits (skeleton only).
// Day 18+: will initialize the Engine with a data directory, optionally
//           start the gRPC server (Phase 1.5), and block until shutdown.
//
// Usage (future):
//   vectordb --data-dir /var/lib/vectordb [--port 50051]
// -----------------------------------------------------------------------------
#include <iostream>
#include "core/version.h"
// TODO Day 18: #include "server/engine.h" — wire up Engine + gRPC server

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "vectordb v" << VECTORDB_VERSION << "\n";
    return 0;
}
