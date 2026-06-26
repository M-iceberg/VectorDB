// Python bindings for VortexDB — Day 21
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "server/engine.h"
#include <stdexcept>

namespace py = pybind11;
using namespace vectordb;

namespace {

// Convert "l2" / "cosine" / "ip" string to Metric enum.
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

// Validate that arr is a 1-D contiguous float32 array of length dim.
const float* validate_vec(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
    size_t dim)
{
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("vector must be 1-D, got " +
                                    std::to_string(buf.ndim) + "-D");
    if (static_cast<size_t>(buf.shape[0]) != dim)
        throw std::invalid_argument("vector length " +
                                    std::to_string(buf.shape[0]) +
                                    " does not match collection dim " +
                                    std::to_string(dim));
    return static_cast<const float*>(buf.ptr);
}

}  // namespace

PYBIND11_MODULE(_vectordb, m) {
    m.doc() = "VortexDB — high-performance vector search engine";

    // ------------------------------------------------------------------
    // Engine class
    // ------------------------------------------------------------------
    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&>(), py::arg("data_dir"),
             "Open (or create) a VortexDB database at data_dir.")

        // Collection management
        .def("create_collection",
            [](Engine& self, const std::string& name, size_t dim,
               const std::string& metric) {
                CollectionSchema s;
                s.name   = name;
                s.dim    = dim;
                s.metric = parse_metric(metric);
                self.create_collection(s);
            },
            py::arg("name"), py::arg("dim"), py::arg("metric") = "l2",
            "Create a new collection. metric: 'l2' | 'cosine' | 'ip'.")

        .def("drop_collection", &Engine::drop_collection, py::arg("name"),
             "Drop a collection and delete its files.")

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
            },
            "Return a list of dicts describing every open collection.")

        // Insert: zero-copy numpy → float*
        .def("insert",
            [](Engine& self, const std::string& collection, uint32_t id,
               py::array_t<float, py::array::c_style | py::array::forcecast> vec) {
                auto buf = vec.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("vector must be 1-D");
                self.insert(collection, id,
                            static_cast<const float*>(buf.ptr));
            },
            py::arg("collection"), py::arg("id"), py::arg("vector"),
            "Insert a vector (1-D numpy float32 array) with the given id.")

        // Remove
        .def("remove", &Engine::remove,
             py::arg("collection"), py::arg("id"),
             "Soft-delete a vector by id.")

        // Search: returns list of {"id": ..., "distance": ...}
        .def("search",
            [](Engine& self, const std::string& collection,
               py::array_t<float, py::array::c_style | py::array::forcecast> query,
               int top_k, int ef_search) {
                auto buf = query.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("query must be 1-D");
                SearchRequest req;
                req.collection = collection;
                req.query      = static_cast<const float*>(buf.ptr);
                req.top_k      = top_k;
                req.ef_search  = ef_search;
                auto results = self.search(req);
                py::list out;
                for (auto& r : results) {
                    py::dict d;
                    d["id"]       = r.id;
                    d["distance"] = r.distance;
                    out.append(d);
                }
                return out;
            },
            py::arg("collection"), py::arg("query"),
            py::arg("top_k") = 10, py::arg("ef_search") = 64,
            "Search for top_k nearest neighbors. Returns list of {id, distance}.")

        // Checkpoint
        .def("checkpoint", &Engine::checkpoint, py::arg("collection"),
             "Snapshot the collection to disk and truncate the WAL.");

    // ------------------------------------------------------------------
    // Module-level open() convenience function
    // ------------------------------------------------------------------
    m.def("open", [](const std::string& data_dir) {
        return std::make_unique<Engine>(data_dir);
    }, py::arg("data_dir"),
    "Open (or create) a VortexDB database. Returns an Engine instance.");
}
