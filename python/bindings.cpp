// Python bindings for VectorDB
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "server/engine.h"
#include <limits>
#include <stdexcept>

namespace py = pybind11;
using namespace vectordb;

namespace {

Metric parse_metric(const std::string& s) {
    if (s == "cosine") return Metric::Cosine;
    if (s == "ip")     return Metric::InnerProduct;
    if (s == "l2")     return Metric::L2;
    throw std::invalid_argument("unknown metric '" + s + "': use 'l2', 'cosine', or 'ip'");
}

std::string metric_to_str(Metric m) {
    switch (m) {
        case Metric::Cosine:        return "cosine";
        case Metric::InnerProduct:  return "ip";
        default:                    return "l2";
    }
}

// Convert Python dict {"color": "red", "price": 9.9} -> MetadataEntry.
// bool is a subclass of int in Python, so check py::bool_ first.
MetadataEntry parse_metadata(const py::dict& d) {
    MetadataEntry entry;
    for (auto item : d) {
        std::string key = item.first.cast<std::string>();
        auto val = item.second;
        if (py::isinstance<py::bool_>(val)) {
            entry.strings[key] = val.cast<bool>() ? "1" : "0";
        } else if (py::isinstance<py::str>(val)) {
            entry.strings[key] = val.cast<std::string>();
        } else if (py::isinstance<py::float_>(val) || py::isinstance<py::int_>(val)) {
            entry.numerics[key] = val.cast<double>();
        } else {
            throw std::invalid_argument("metadata value for '" + key +
                                        "' must be str, float, int, or bool");
        }
    }
    return entry;
}

// Convert Python filter dict -> std::vector<FieldFilter>.
// Supported syntax:
//   {"color": "red"}                      -> Eq("color", "red")
//   {"price": {"$gte": 10.0}}             -> Range("price", 10.0, +inf)
//   {"price": {"$lte": 100.0}}            -> Range("price", -inf, 100.0)
//   {"price": {"$gte": 10, "$lte": 100}}  -> Range("price", 10, 100)
// Multiple keys are ANDed by the Engine.
std::vector<FieldFilter> parse_filters(const py::dict& d) {
    std::vector<FieldFilter> filters;
    const double INF = std::numeric_limits<double>::infinity();
    for (auto item : d) {
        std::string field = item.first.cast<std::string>();
        auto val = item.second;
        FieldFilter f;
        f.field = field;
        if (py::isinstance<py::str>(val)) {
            f.op      = FieldFilter::Op::Eq;
            f.str_val = val.cast<std::string>();
        } else if (py::isinstance<py::dict>(val)) {
            f.op = FieldFilter::Op::Range;
            f.lo = -INF;
            f.hi =  INF;
            for (auto op_item : val.cast<py::dict>()) {
                std::string op  = op_item.first.cast<std::string>();
                double      num = op_item.second.cast<double>();
                if      (op == "$gte") f.lo = num;
                else if (op == "$lte") f.hi = num;
                else throw std::invalid_argument(
                    "unknown filter op '" + op + "': use '$gte' or '$lte'");
            }
        } else {
            throw std::invalid_argument("filter value for '" + field +
                                        "' must be a string or a dict");
        }
        filters.push_back(std::move(f));
    }
    return filters;
}

}  // namespace

PYBIND11_MODULE(_vectordb, m) {
    m.doc() = "VectorDB -- persistent C++ vector search engine";

    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&>(), py::arg("data_dir"))

        .def("create_collection",
            [](Engine& self, const std::string& name, size_t dim,
               const std::string& metric) {
                CollectionSchema s;
                s.name   = name;
                s.dim    = dim;
                s.metric = parse_metric(metric);
                py::gil_scoped_release release;
                self.create_collection(s);
            },
            py::arg("name"), py::arg("dim"), py::arg("metric") = "l2")

        .def("drop_collection",
            [](Engine& self, const std::string& name) {
                py::gil_scoped_release release;
                self.drop_collection(name);
            }, py::arg("name"))

        .def("list_collections",
            [](Engine& self) {
                py::list out;
                for (auto& c : self.list_collections()) {
                    py::dict d;
                    d["name"]   = c.name;
                    d["dim"]    = c.dim;
                    d["metric"] = metric_to_str(c.metric);
                    out.append(d);
                }
                return out;
            })

        // Batch insert: one fdatasync per batch instead of one per vector.
        // vecs: 2-D float32 numpy array (N × dim).
        // user_ids: list of N strings, or empty list for auto-assign.
        // metadatas: list of N dicts, or None for no metadata.
        .def("insert_batch",
            [](Engine& self, const std::string& collection,
               const std::vector<std::string>& user_ids,
               py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
               py::object metadatas, int num_threads) -> py::list {
                auto buf = vecs.request();
                if (buf.ndim != 2)
                    throw std::invalid_argument("vectors must be 2-D");
                size_t count = static_cast<size_t>(buf.shape[0]);
                size_t dim = self.collection_dimension(collection);
                if (static_cast<size_t>(buf.shape[1]) != dim)
                    throw std::invalid_argument(
                        "vector dimension does not match collection dimension");
                if (!user_ids.empty() && user_ids.size() != count)
                    throw std::invalid_argument(
                        "user_ids length must match vector count");
                if (num_threads < 0)
                    throw std::invalid_argument("num_threads must be non-negative");
                const float* ptr = static_cast<const float*>(buf.ptr);

                std::vector<MetadataEntry> metas;
                if (!metadatas.is_none()) {
                    auto lst = metadatas.cast<py::list>();
                    if (static_cast<size_t>(py::len(lst)) != count)
                        throw std::invalid_argument(
                            "metadatas length must match vector count");
                    metas.reserve(count);
                    for (size_t i = 0; i < count; ++i) {
                        auto item = lst[i];
                        if (item.is_none())
                            metas.emplace_back();
                        else
                            metas.push_back(parse_metadata(item.cast<py::dict>()));
                    }
                }

                std::vector<std::string> assigned;
                {
                    py::gil_scoped_release release;
                    assigned = self.insert_batch(
                        collection, user_ids, ptr, count, metas, num_threads);
                }

                py::list out;
                for (auto& s : assigned) out.append(s);
                return out;
            },
            py::arg("collection"), py::arg("user_ids"), py::arg("vectors"),
            py::arg("metadatas") = py::none(), py::arg("num_threads") = 0)

        // insert with optional metadata dict; returns the effective user_id string
        .def("insert",
            [](Engine& self, const std::string& collection,
               const std::string& user_id,
               py::array_t<float, py::array::c_style | py::array::forcecast> vec,
               py::object metadata) -> std::string {
                auto buf = vec.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("vector must be 1-D");
                if (static_cast<size_t>(buf.shape[0]) !=
                    self.collection_dimension(collection))
                    throw std::invalid_argument(
                        "vector dimension does not match collection dimension");
                MetadataEntry meta;
                if (!metadata.is_none())
                    meta = parse_metadata(metadata.cast<py::dict>());
                std::string assigned;
                {
                    py::gil_scoped_release release;
                    assigned = self.insert(collection, user_id,
                                           static_cast<const float*>(buf.ptr), meta);
                }
                return assigned;
            },
            py::arg("collection"), py::arg("user_id"), py::arg("vector"),
            py::arg("metadata") = py::none())

        .def("remove",
            [](Engine& self, const std::string& collection,
               const std::string& user_id) {
                py::gil_scoped_release release;
                self.remove(collection, user_id);
            },
            py::arg("collection"), py::arg("user_id"))

        // search with optional filters dict
        .def("search",
            [](Engine& self, const std::string& collection,
               py::array_t<float, py::array::c_style | py::array::forcecast> query,
               int top_k, int ef_search, py::object filters) {
                auto buf = query.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("query must be 1-D");
                if (static_cast<size_t>(buf.shape[0]) !=
                    self.collection_dimension(collection))
                    throw std::invalid_argument(
                        "query dimension does not match collection dimension");
                if (top_k <= 0)
                    throw std::invalid_argument("top_k must be positive");
                if (ef_search <= 0)
                    throw std::invalid_argument("ef_search must be positive");
                SearchRequest req;
                req.collection = collection;
                req.query      = static_cast<const float*>(buf.ptr);
                req.top_k      = top_k;
                req.ef_search  = ef_search;
                if (!filters.is_none())
                    req.filters = parse_filters(filters.cast<py::dict>());
                std::vector<SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = self.search(req);
                }
                py::list out;
                for (auto& r : results) {
                    py::dict d;
                    d["user_id"]  = r.user_id;
                    d["distance"] = r.distance;
                    out.append(d);
                }
                return out;
            },
            py::arg("collection"), py::arg("query"),
            py::arg("top_k") = 10, py::arg("ef_search") = 64,
            py::arg("filters") = py::none())

        .def("checkpoint",
            [](Engine& self, const std::string& collection) {
                py::gil_scoped_release release;
                self.checkpoint(collection);
            }, py::arg("collection"))

        // Batch search: run n_queries in parallel.
        // queries: 2-D float32 numpy array (n_queries × dim).
        // Returns a list of n_queries result lists, each sorted by distance.
        .def("search_batch",
            [](Engine& self, const std::string& collection,
               py::array_t<float, py::array::c_style | py::array::forcecast> queries,
               int top_k, int ef_search, int num_threads, py::object filters) {
                auto buf = queries.request();
                if (buf.ndim != 2)
                    throw std::invalid_argument("queries must be 2-D");
                int n_queries = static_cast<int>(buf.shape[0]);
                if (static_cast<size_t>(buf.shape[1]) !=
                    self.collection_dimension(collection))
                    throw std::invalid_argument(
                        "query dimension does not match collection dimension");
                if (top_k <= 0)
                    throw std::invalid_argument("top_k must be positive");
                if (ef_search <= 0)
                    throw std::invalid_argument("ef_search must be positive");
                if (num_threads < 0)
                    throw std::invalid_argument("num_threads must be non-negative");
                const float* ptr = static_cast<const float*>(buf.ptr);

                Engine::BatchSearchRequest req;
                req.collection  = collection;
                req.queries     = ptr;
                req.n_queries   = n_queries;
                req.top_k       = top_k;
                req.ef_search   = ef_search;
                req.num_threads = num_threads;
                if (!filters.is_none())
                    req.filters = parse_filters(filters.cast<py::dict>());

                std::vector<std::vector<SearchResult>> batch;
                {
                    py::gil_scoped_release release;
                    batch = self.search_batch(req);
                }

                py::list out;
                for (auto& results : batch) {
                    py::list row;
                    for (auto& r : results) {
                        py::dict d;
                        d["user_id"]  = r.user_id;
                        d["distance"] = r.distance;
                        row.append(d);
                    }
                    out.append(row);
                }
                return out;
            },
            py::arg("collection"), py::arg("queries"),
            py::arg("top_k") = 10, py::arg("ef_search") = 64,
            py::arg("num_threads") = 0,
            py::arg("filters") = py::none());

    m.def("open", [](const std::string& data_dir) {
        return std::make_unique<Engine>(data_dir);
    }, py::arg("data_dir"));
}
