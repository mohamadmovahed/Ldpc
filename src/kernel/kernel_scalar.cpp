#include "simd_scalar.hpp"

namespace ldpc {
namespace detail {

namespace {
const LayeredMinSum<ScalarF32> kFloat;
const LayeredMinSum<ScalarI16> kInt16;
const LayeredMinSum<ScalarI8> kInt8;
}  // namespace

const LayeredKernel* scalar_kernel(Precision precision) noexcept {
    switch (precision) {
        case Precision::Float32: return &kFloat;
        case Precision::Int16: return &kInt16;
        case Precision::Int8: return &kInt8;
    }
    return nullptr;
}

}  // namespace detail
}  // namespace ldpc
