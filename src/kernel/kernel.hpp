#pragma once

// Internal interface between the public decoder classes and the SIMD kernels.
//
// Every SIMD backend is compiled in its own translation unit with its own
// instruction-set flags. To keep code compiled for one ISA from leaking into
// another through shared inline/template symbols (an ODR hazard that crashes
// on CPUs lacking the ISA), kernels only exchange plain data through this
// header and use no standard-library templates.

#include <cstddef>
#include <cstdint>

#include "ldpc/decoder_config.hpp"

namespace ldpc {
namespace detail {

/// Tanner graph in CSR form (edges ordered by check node).
struct Graph {
    const std::uint32_t* row_ptr;  ///< num_rows + 1 offsets into col_idx
    const std::uint32_t* col_idx;  ///< variable node of every edge
    std::uint32_t num_rows;
    std::uint32_t num_cols;
    std::uint32_t num_edges;
    std::uint32_t max_row_degree;
};

struct KernelParams {
    bool normalized;  ///< true: normalized min-sum, false: offset min-sum
    float factor;     ///< normalization factor
    float offset;     ///< offset in LLR units
    float scale;      ///< LLR -> internal representation scale (1 for float)
    int max_iterations;
    bool early_termination;
};

/// One batch of at most `lanes()` frames.
struct KernelIO {
    const float* llr_in;    ///< num_frames * num_cols
    std::uint32_t num_frames;
    std::uint8_t* bits_out;  ///< may be null
    float* llr_out;          ///< may be null
    DecodeResult* results;   ///< may be null
};

/// Stateless decoding kernel; all mutable state lives in the caller's workspace.
class LayeredKernel {
public:
    virtual unsigned lanes() const noexcept = 0;
    /// Bytes of 64-byte aligned scratch memory required by decode().
    virtual std::size_t workspace_bytes(const Graph& graph) const noexcept = 0;
    virtual void decode(const Graph& graph, const KernelParams& params, void* workspace,
                        const KernelIO& io) const noexcept = 0;

protected:
    ~LayeredKernel() = default;
};

// Factories, one per backend translation unit. They return a pointer to a
// static instance, or null when the precision is unsupported.
const LayeredKernel* scalar_kernel(Precision precision) noexcept;
const LayeredKernel* avx2_kernel(Precision precision) noexcept;
const LayeredKernel* avx512_kernel(Precision precision) noexcept;
const LayeredKernel* neon_kernel(Precision precision) noexcept;

/// Resolves `backend` (Auto allowed) and returns the kernel, or null if the
/// backend is not available on this build/CPU.
const LayeredKernel* select_kernel(SimdBackend backend, Precision precision, SimdBackend* resolved) noexcept;

}  // namespace detail
}  // namespace ldpc
