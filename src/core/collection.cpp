// -----------------------------------------------------------------------------
// collection.cpp — CollectionSchema + Collection implementation
//
// Minimal implementation: stores the schema and provides read-only access.
// Future: may validate schema on construction (e.g. dim > 0, name non-empty).
// -----------------------------------------------------------------------------
#include "collection.h"

namespace vectordb {

Collection::Collection(CollectionSchema schema)
    : schema_(std::move(schema)) {}

Collection::~Collection() = default;

const CollectionSchema& Collection::schema() const {
    return schema_;
}

}  // namespace vectordb
