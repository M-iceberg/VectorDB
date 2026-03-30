#include "collection.h"

namespace vectordb {

Collection::Collection(CollectionSchema schema)
    : schema_(std::move(schema)) {}

Collection::~Collection() = default;

const CollectionSchema& Collection::schema() const {
    return schema_;
}

}  // namespace vectordb
