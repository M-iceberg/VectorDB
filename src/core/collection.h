// -----------------------------------------------------------------------------
// collection.h — Named vector collection metadata
//
// A Collection is a lightweight descriptor that binds together a name,
// dimensionality, and distance metric. The Engine owns one Collection per
// named namespace and uses it to validate insert/search requests.
// Actual storage (HnswIndex, VectorFile, Wal, MetadataIndex) is owned
// separately by the Engine::Impl per-collection state.
//
// API:
//   CollectionSchema         — POD: { name, dim, metric }
//
//   Collection(schema)
//       Constructs from a schema. Validates that dim > 0.
//
//   schema() → const CollectionSchema&
//       Read-only access to name, dim, and metric.
// -----------------------------------------------------------------------------
#pragma once
#include "distance.h"
#include <cstddef>
#include <string>

namespace vectordb {

struct CollectionSchema {
    std::string name;
    size_t      dim = 0;
    Metric      metric = Metric::L2;
};

// Lightweight metadata container for a named vector collection.
// Actual storage and index objects are owned by the Engine.
class Collection {
public:
    explicit Collection(CollectionSchema schema);
    ~Collection();

    const CollectionSchema& schema() const;

    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;

private:
    CollectionSchema schema_;
};

}  // namespace vectordb
