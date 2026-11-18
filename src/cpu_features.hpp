#pragma once

namespace ldpc {
namespace detail {

/// Runtime CPU (and OS register-state) support for the SIMD backends.
struct CpuFeatures {
    bool avx2 = false;
    bool avx512bw = false;  ///< AVX-512F and AVX-512BW
};

const CpuFeatures& cpu_features() noexcept;

}  // namespace detail
}  // namespace ldpc
