#include "simd_neon.hpp"

namespace ldpc {
namespace detail {

namespace {
const LayeredMinSum<NeonF32> kFloat;
const LayeredMinSum<NeonI16> kInt16;
const LayeredMinSum<NeonI8> kInt8;
}  // namespace

const LayeredKernel* neon_kernel(Precision precision) noexcept {
    switch (precision) {
        case Precision::Float32: return &kFloat;
        case Precision::Int16: return &kInt16;
        case Precision::Int8: return &kInt8;
    }
    return nullptr;
}

}  // namespace detail
}  // namespace ldpc
