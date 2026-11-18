#pragma once

namespace ldpc {

/// Check-node update rule. Both use a layered (row-serial) schedule.
///
/// Offset min-sum is the default because it is robust on codes with long
/// degree-2 chains (DVB, 802.11n, 5G NR, ...), where normalized min-sum with a
/// small factor (e.g. 0.75) can stall with a few residual bit errors.
enum class Algorithm {
    NormalizedMinSum,  ///< min-sum with messages scaled by `normalization_factor`
    OffsetMinSum,      ///< min-sum with messages reduced by `offset`
};

/// Arithmetic used inside the decoder. Input/output LLRs are always float.
enum class Precision {
    Float32,  ///< best error-rate performance
    Int16,    ///< saturating 16-bit fixed point, 2x the SIMD lanes of Float32
    Int8,     ///< saturating 8-bit fixed point, 4x the SIMD lanes of Float32 (fastest)
};

/// Instruction set used by the decoding kernel.
enum class SimdBackend {
    Auto,    ///< best backend supported by the build and the running CPU
    Scalar,  ///< portable C++ (relies on compiler auto-vectorisation)
    AVX2,    ///< x86-64 AVX2
    AVX512,  ///< x86-64 AVX-512F + AVX-512BW
    NEON,    ///< ARMv8 / AArch64 Advanced SIMD
};

struct DecoderConfig {
    Algorithm algorithm = Algorithm::OffsetMinSum;
    Precision precision = Precision::Float32;
    SimdBackend backend = SimdBackend::Auto;

    /// Maximum number of layered iterations (one iteration = one pass over all checks).
    int max_iterations = 50;
    /// Stop a frame as soon as all parity checks are satisfied.
    bool early_termination = true;

    /// Scaling applied to check-node messages (NormalizedMinSum), in (0, 1].
    float normalization_factor = 0.8f;
    /// Offset subtracted from check-node magnitudes (OffsetMinSum), in LLR units, >= 0.
    /// Low-rate codes may prefer ~0.25, high-rate codes ~0.5.
    float offset = 0.5f;

    /// Fixed-point quantisation: an input LLR x is mapped to round(x * scale)
    /// and saturated. 0 selects a default (Int16: 64, Int8: 2). Ignored for Float32.
    float fixed_point_scale = 0.0f;
};

/// Per-frame decoding outcome.
struct DecodeResult {
    int iterations = 0;      ///< iterations performed for this frame
    bool converged = false;  ///< true if the output satisfies every parity check
};

const char* to_string(Algorithm algorithm) noexcept;
const char* to_string(Precision precision) noexcept;
const char* to_string(SimdBackend backend) noexcept;

/// True if `backend` was compiled in and is supported by the running CPU.
bool is_backend_available(SimdBackend backend) noexcept;
/// The backend `SimdBackend::Auto` resolves to on this machine.
SimdBackend best_available_backend() noexcept;

}  // namespace ldpc
