#include <iostream>
#include "core/version.h"
// TODO Day 18: #include "server/engine.h" — wire up Engine + gRPC server

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "vectordb v" << VECTORDB_VERSION << "\n";
    return 0;
}
