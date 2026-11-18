#include "../cpu_features.hpp"
#include "kernel.hpp"

namespace ldpc {
namespace detail {

namespace {

const LayeredKernel* kernel_for(SimdBackend backend, Precision precision) noexcept {
    switch (backend) {
        case SimdBackend::Scalar:
            return scalar_kernel(precision);
        case SimdBackend::AVX2:
#ifdef LDPC_HAVE_AVX2
            if (cpu_features().avx2) {
                return avx2_kernel(precision);
            }
#endif
            return nullptr;
        case SimdBackend::AVX512:
#ifdef LDPC_HAVE_AVX512
            if (cpu_features().avx512bw) {
                return avx512_kernel(precision);
            }
#endif
            return nullptr;
        case SimdBackend::NEON:
#ifdef LDPC_HAVE_NEON
            return neon_kernel(precision);
#else
            return nullptr;
#endif
        case SimdBackend::Auto:
            return nullptr;
    }
    return nullptr;
}

constexpr SimdBackend kPreferenceOrder[] = {SimdBackend::AVX512, SimdBackend::AVX2, SimdBackend::NEON,
                                            SimdBackend::Scalar};

}  // namespace

const LayeredKernel* select_kernel(SimdBackend backend, Precision precision, SimdBackend* resolved) noexcept {
    if (backend != SimdBackend::Auto) {
        *resolved = backend;
        return kernel_for(backend, precision);
    }
    for (SimdBackend candidate : kPreferenceOrder) {
        if (const LayeredKernel* kernel = kernel_for(candidate, precision)) {
            *resolved = candidate;
            return kernel;
        }
    }
    return nullptr;
}

}  // namespace detail

bool is_backend_available(SimdBackend backend) noexcept {
    SimdBackend resolved;
    return detail::select_kernel(backend, Precision::Float32, &resolved) != nullptr;
}

SimdBackend best_available_backend() noexcept {
    SimdBackend resolved = SimdBackend::Scalar;
    detail::select_kernel(SimdBackend::Auto, Precision::Float32, &resolved);
    return resolved;
}

const char* to_string(Algorithm algorithm) noexcept {
    switch (algorithm) {
        case Algorithm::NormalizedMinSum: return "normalized-min-sum";
        case Algorithm::OffsetMinSum: return "offset-min-sum";
    }
    return "unknown";
}

const char* to_string(Precision precision) noexcept {
    switch (precision) {
        case Precision::Float32: return "float32";
        case Precision::Int16: return "int16";
        case Precision::Int8: return "int8";
    }
    return "unknown";
}

const char* to_string(SimdBackend backend) noexcept {
    switch (backend) {
        case SimdBackend::Auto: return "auto";
        case SimdBackend::Scalar: return "scalar";
        case SimdBackend::AVX2: return "avx2";
        case SimdBackend::AVX512: return "avx512";
        case SimdBackend::NEON: return "neon";
    }
    return "unknown";
}

}  // namespace ldpc
