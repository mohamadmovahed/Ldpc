#include "simd_avx2.hpp"

namespace ldpc {
namespace detail {

namespace {
const LayeredMinSum<Avx2F32> kFloat;
const LayeredMinSum<Avx2I16> kInt16;
const LayeredMinSum<Avx2I8> kInt8;
}  // namespace

const LayeredKernel* avx2_kernel(Precision precision) noexcept {
    switch (precision) {
        case Precision::Float32: return &kFloat;
        case Precision::Int16: return &kInt16;
        case Precision::Int8: return &kInt8;
    }
    return nullptr;
}

}  // namespace detail
}  // namespace ldpc
