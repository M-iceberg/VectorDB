// x86 runtime dispatch: selects AVX2 or naive fallback via cpuid — added Day 4.
// Compiled only on x86_64 (see src/core/CMakeLists.txt).
#include "distance.h"

#if defined(VECTORDB_ARCH_X86)
// TODO Day 4: Override DistanceCompute::create() here.
//             Use __builtin_cpu_supports("avx2") to pick AVX2 vs naive at runtime.
//             Remove create() from distance_naive.cpp when this is wired up.
#endif
