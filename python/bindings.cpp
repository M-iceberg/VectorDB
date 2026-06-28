// Python bindings for VortexDB
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
    m.doc() = "VortexDB -- high-performance vector search engine";

    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&>(), py::arg("data_dir"))

        .def("create_collection",
            [](Engine& self, const std::string& name, size_t dim,
               const std::string& metric) {
                CollectionSchema s;
                s.name   = name;
                s.dim    = dim;
                s.metric = parse_metric(metric);
                self.create_collection(s);
            },
            py::arg("name"), py::arg("dim"), py::arg("metric") = "l2")

        .def("drop_collection", &Engine::drop_collection, py::arg("name"))

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

        // insert with optional metadata dict
        .def("insert",
            [](Engine& self, const std::string& collection, uint32_t id,
               py::array_t<float, py::array::c_style | py::array::forcecast> vec,
               py::object metadata) {
                auto buf = vec.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("vector must be 1-D");
                MetadataEntry meta;
                if (!metadata.is_none())
                    meta = parse_metadata(metadata.cast<py::dict>());
                self.insert(collection, id,
                            static_cast<const float*>(buf.ptr), meta);
            },
            py::arg("collection"), py::arg("id"), py::arg("vector"),
            py::arg("metadata") = py::none())

        .def("remove", &Engine::remove,
             py::arg("collection"), py::arg("id"))

        // search with optional filters dict
        .def("search",
            [](Engine& self, const std::string& collection,
               py::array_t<float, py::array::c_style | py::array::forcecast> query,
               int top_k, int ef_search, py::object filters) {
                auto buf = query.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("query must be 1-D");
                SearchRequest req;
                req.collection = collection;
                req.query      = static_cast<const float*>(buf.ptr);
                req.top_k      = top_k;
                req.ef_search  = ef_search;
                if (!filters.is_none())
                    req.filters = parse_filters(filters.cast<py::dict>());
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
            py::arg("filters") = py::none())

        .def("checkpoint", &Engine::checkpoint, py::arg("collection"));

    m.def("open", [](const std::string& data_dir) {
        return std::make_unique<Engine>(data_dir);
    }, py::arg("data_dir"));
}
