#include "simd_avx512.hpp"

namespace ldpc {
namespace detail {

namespace {
const LayeredMinSum<Avx512F32> kFloat;
const LayeredMinSum<Avx512I16> kInt16;
const LayeredMinSum<Avx512I8> kInt8;
}  // namespace

const LayeredKernel* avx512_kernel(Precision precision) noexcept {
    switch (precision) {
        case Precision::Float32: return &kFloat;
        case Precision::Int16: return &kInt16;
        case Precision::Int8: return &kInt8;
    }
    return nullptr;
}

}  // namespace detail
}  // namespace ldpc
