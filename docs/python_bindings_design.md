# Python Bindings Design

## What are the Python bindings?

The Python bindings expose the C++ Engine to Python using pybind11. The result is a native Python extension module (`_vectordb.so`) that Python imports like any other package. There is no server, no network, no serialization — Python calls directly into the same C++ code that the unit tests use.

## Build

The bindings are opt-in. They are not compiled as part of the default build:

```bash
cmake -B build -DVECTORDB_PYTHON=ON -DPython3_EXECUTABLE=$(which python3)
cmake --build build --target _vectordb
```

pybind11 is fetched automatically via FetchContent (same pattern as googletest). NumPy must be installed in the target Python environment.

The compiled `.so` is placed directly into `python/vectordb/` so the package can be imported from there without any install step:

```bash
cd python
python3 -c "import vectordb; db = vectordb.open('/tmp/test')"
```

## How numpy array → float* works

A numpy `float32` array is a contiguous block of 32-bit floats in memory — the same layout as a C++ `float[]`. pybind11's `array_t<float>` gives direct access to that memory via a buffer protocol:

```cpp
py::array_t<float, py::array::c_style | py::array::forcecast> vec

auto buf = vec.request();          // get buffer descriptor
const float* ptr = static_cast<const float*>(buf.ptr);  // raw pointer into numpy's memory
self.insert(collection, id, ptr); // C++ reads directly — no copy
```

`buf.ptr` points into the numpy array's own allocation. No data is copied. The C++ function runs while Python still owns the memory — this is safe because the call is synchronous (C++ returns before Python can garbage-collect the array).

**The two flags on `array_t`:**

- `c_style` — requires the array to be row-major contiguous. If the user passes a non-contiguous array (e.g. `arr[::2]`), pybind11 makes a contiguous copy before calling into C++. For a normal `np.float32` array this is always zero-copy.
- `forcecast` — if the user passes `float64` (numpy's default dtype), pybind11 silently converts it to `float32` rather than raising a type error.

So "zero-copy" holds when the input is already a contiguous `float32` array. A contiguous copy happens only for non-contiguous arrays or wrong dtypes. In practice, embeddings from sentence-transformers and OpenAI are returned as `float32`, so the common path is always zero-copy.

## API

```python
import vectordb
import numpy as np

db = vectordb.open("/path/to/data")               # opens or creates the DB

db.create_collection("items", dim=128, metric="l2")   # metric: "l2" | "cosine" | "ip"
db.drop_collection("items")
cols = db.list_collections()                       # list of {name, dim, metric}

vec = np.random.rand(128).astype(np.float32)
db.insert("items", id=42, vector=vec)              # zero-copy numpy → C++
db.remove("items", id=42)

results = db.search("items", query=vec, top_k=10, ef_search=64)
# results: [{"id": ..., "distance": ...}, ...]

db.checkpoint("items")                             # flush snapshot to disk
```

## Error handling

C++ exceptions propagate to Python automatically via pybind11. A `std::runtime_error` (e.g. collection not found) becomes a Python `RuntimeError`. A `std::invalid_argument` (e.g. wrong vector dimension) becomes a `ValueError`. No explicit mapping is needed — pybind11 registers these translations by default.

## File structure

```
python/
  CMakeLists.txt        — pybind11_add_module, links vortex_core/storage/server
  bindings.cpp          — all pybind11 binding code
  vectordb/
    __init__.py         — re-exports Engine and open() from _vectordb
    _vectordb.*.so      — compiled extension (generated, not in git)
```

## What is NOT in Day 21

- **Metadata / filters** (Day 22): `insert` does not accept a metadata dict yet. `search` does not accept filters.
- **Batch insert**: insert takes one vector at a time. Batch support (2-D numpy array) is a future addition.
- **`pip install .` packaging** (Day 23): the package is not yet installable via pip. It must be used from the `python/` directory with the `.so` in place.
