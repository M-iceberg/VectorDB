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
