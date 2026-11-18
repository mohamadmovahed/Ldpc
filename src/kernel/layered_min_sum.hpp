#pragma once

// Layered min-sum decoder, generic over a SIMD "traits" type S. Each vector
// lane decodes a different frame (inter-frame SIMD), so the algorithm needs
// no gathers or shuffles and works for any parity-check matrix.
//
// Memory layout (T = S::T, W = S::W lanes):
//   L[v * W + lane]   a-posteriori value of variable v
//   R[e * W + lane]   check-to-variable message of edge e (row-major edges)
//
// Everything is in an anonymous namespace on purpose: see kernel.hpp.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernel.hpp"

namespace ldpc {
namespace detail {
namespace {

constexpr std::size_t kAlignment = 64;

/// Input LLR given to unused lanes of a partially filled batch.
constexpr float kIdleLaneLlr = 1.0e4f;

/// Variables per block when converting between frame-major and lane-major layouts.
constexpr std::uint32_t kTransposeBlock = 64;

/// How many rows ahead the posterior values are prefetched.
constexpr std::uint32_t kPrefetchRows = 16;

inline void prefetch(const void* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address);
#else
    (void)address;
#endif
}

constexpr std::size_t align_up(std::size_t bytes) noexcept {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

template <class S>
class LayeredMinSum final : public LayeredKernel {
    using T = typename S::T;
    using V = typename S::V;
    static constexpr unsigned W = S::W;

public:
    unsigned lanes() const noexcept override { return W; }

    std::size_t workspace_bytes(const Graph& g) const noexcept override {
        return align_up(bytes_for(g.num_cols)) + align_up(bytes_for(g.num_edges)) +
               align_up(bytes_for(g.max_row_degree)) + kAlignment;
    }

    void decode(const Graph& g, const KernelParams& p, void* workspace, const KernelIO& io) const noexcept override {
        auto* base = static_cast<unsigned char*>(workspace);
        T* posterior = reinterpret_cast<T*>(base);
        base += align_up(bytes_for(g.num_cols));
        T* messages = reinterpret_cast<T*>(base);
        base += align_up(bytes_for(g.num_edges));
        T* row_buffer = reinterpret_cast<T*>(base);
        base += align_up(bytes_for(g.max_row_degree));
        T* lane_scratch = reinterpret_cast<T*>(base);  // W values, used for sign extraction

        load_input(g, p, io, posterior);

        const std::uint64_t active = io.num_frames >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << io.num_frames) - 1;
        std::uint64_t done = 0;
        const typename S::Scale scale = S::make_scale(p.factor);
        const V offset = S::splat(S::quantize(p.offset, p.scale));

        int iteration = 0;
        while (iteration < p.max_iterations) {
            ++iteration;
            // The first iteration treats all old messages as zero, which saves
            // clearing the message memory.
            if (iteration == 1) {
                update_all_checks<true>(g, p.normalized, scale, offset, posterior, messages, row_buffer);
            } else {
                update_all_checks<false>(g, p.normalized, scale, offset, posterior, messages, row_buffer);
            }
            if (p.early_termination && iteration < p.max_iterations) {
                const std::uint64_t unsatisfied = unsatisfied_lanes(g, posterior, lane_scratch);
                const std::uint64_t converged = active & ~done & ~unsatisfied;
                write_lanes(g, p, io, posterior, converged, unsatisfied, iteration);
                done |= converged;
                if (done == active) {
                    return;
                }
            }
        }
        const std::uint64_t unsatisfied = unsatisfied_lanes(g, posterior, lane_scratch);
        write_lanes(g, p, io, posterior, active & ~done, unsatisfied, iteration);
    }

private:
    static constexpr std::size_t bytes_for(std::size_t vectors) noexcept { return vectors * W * sizeof(T); }

    /// Transposes frame-major input into the lane-interleaved layout, in
    /// cache-sized blocks of variables.
    static void load_input(const Graph& g, const KernelParams& p, const KernelIO& io, T* posterior) noexcept {
        const T confident_zero = S::quantize(kIdleLaneLlr, p.scale);  // idle lanes converge at once
        for (std::uint32_t v0 = 0; v0 < g.num_cols; v0 += kTransposeBlock) {
            const std::uint32_t v1 = v0 + kTransposeBlock < g.num_cols ? v0 + kTransposeBlock : g.num_cols;
            for (std::uint32_t lane = 0; lane < W; ++lane) {
                T* dst = posterior + lane;
                if (lane < io.num_frames) {
                    const float* llr = io.llr_in + std::size_t{lane} * g.num_cols;
                    for (std::uint32_t v = v0; v < v1; ++v) {
                        dst[std::size_t{v} * W] = S::quantize(llr[v], p.scale);
                    }
                } else {
                    for (std::uint32_t v = v0; v < v1; ++v) {
                        dst[std::size_t{v} * W] = confident_zero;
                    }
                }
            }
        }
    }

    template <bool kFirstIteration>
    static void update_all_checks(const Graph& g, bool normalized, const typename S::Scale& scale, const V& offset,
                                  T* posterior, T* messages, T* row_buffer) noexcept {
        for (std::uint32_t r = 0; r < g.num_rows; ++r) {
            if (r + kPrefetchRows < g.num_rows) {
                prefetch_check(g, r + kPrefetchRows, posterior);
            }
            update_check<kFirstIteration>(g, r, normalized, scale, offset, posterior, messages, row_buffer);
        }
    }

    /// Hides the latency of the random posterior accesses of an upcoming row.
    static void prefetch_check(const Graph& g, std::uint32_t r, const T* posterior) noexcept {
        for (std::uint32_t e = g.row_ptr[r]; e < g.row_ptr[r + 1]; ++e) {
            prefetch(posterior + std::size_t{g.col_idx[e]} * W);
        }
    }

    /// Processes one check node (one layer row) for all lanes.
    template <bool kFirstIteration>
    static void update_check(const Graph& g, std::uint32_t r, bool normalized, const typename S::Scale& scale,
                             const V& offset, T* posterior, T* messages, T* row_buffer) noexcept {
        const std::uint32_t first = g.row_ptr[r];
        const std::uint32_t degree = g.row_ptr[r + 1] - first;
        const std::uint32_t* vars = g.col_idx + first;
        T* msg = messages + std::size_t{first} * W;

        // Pass 1: extrinsic inputs, two smallest magnitudes and the sign parity.
        V min1 = S::splat(S::kMax);
        V min2 = min1;
        V parity = S::zero();
        for (std::uint32_t k = 0; k < degree; ++k) {
            const V belief = S::load(posterior + std::size_t{vars[k]} * W);
            const V extrinsic = kFirstIteration ? belief : S::sub_sat(belief, S::load(msg + k * W));
            S::store(row_buffer + k * W, extrinsic);
            const V magnitude = S::abs(extrinsic);
            parity = S::bit_xor(parity, extrinsic);
            min2 = S::min(min2, S::max(min1, magnitude));
            min1 = S::min(min1, magnitude);
        }

        V out1, out2;  // corrected magnitudes for "not the minimum" / "the minimum" edges
        if (normalized) {
            out1 = S::scale(min1, scale);
            out2 = S::scale(min2, scale);
        } else {
            out1 = S::sub_floor0(min1, offset);
            out2 = S::sub_floor0(min2, offset);
        }

        // Pass 2: new messages and posterior update. An edge whose magnitude
        // equals min1 receives min2; with ties min2 == min1, so this is exact.
        for (std::uint32_t k = 0; k < degree; ++k) {
            const V extrinsic = S::load(row_buffer + k * W);
            const V magnitude = S::select_eq(S::abs(extrinsic), min1, out2, out1);
            const V message = S::copy_sign(magnitude, S::bit_xor(parity, extrinsic));
            S::store(msg + k * W, message);
            S::store(posterior + std::size_t{vars[k]} * W, S::add_sat(extrinsic, message));
        }
    }

    /// Bit `lane` is set if that lane's hard decision violates any parity check.
    static std::uint64_t unsatisfied_lanes(const Graph& g, const T* posterior, T* lane_scratch) noexcept {
        V any = S::zero();
        for (std::uint32_t r = 0; r < g.num_rows; ++r) {
            if (r + kPrefetchRows < g.num_rows) {
                prefetch_check(g, r + kPrefetchRows, posterior);
            }
            V parity = S::zero();
            for (std::uint32_t e = g.row_ptr[r]; e < g.row_ptr[r + 1]; ++e) {
                parity = S::bit_xor(parity, S::load(posterior + std::size_t{g.col_idx[e]} * W));
            }
            any = S::bit_or(any, parity);
        }
        S::store(lane_scratch, any);
        std::uint64_t mask = 0;
        for (std::uint32_t lane = 0; lane < W; ++lane) {
            mask |= std::uint64_t{S::is_negative(lane_scratch[lane])} << lane;
        }
        return mask;
    }

    static void write_lanes(const Graph& g, const KernelParams& p, const KernelIO& io, const T* posterior,
                            std::uint64_t lanes, std::uint64_t unsatisfied, int iteration) noexcept {
        lanes &= io.num_frames >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << io.num_frames) - 1;
        if (!lanes) {
            return;
        }
        const float inverse_scale = 1.0f / p.scale;
        // Transpose back in cache-sized blocks of variables.
        for (std::uint32_t v0 = 0; v0 < g.num_cols; v0 += kTransposeBlock) {
            const std::uint32_t v1 = v0 + kTransposeBlock < g.num_cols ? v0 + kTransposeBlock : g.num_cols;
            for (std::uint32_t lane = 0; lane < io.num_frames; ++lane) {
                if (!((lanes >> lane) & 1u)) {
                    continue;
                }
                const T* src = posterior + lane;
                const std::size_t frame_offset = std::size_t{lane} * g.num_cols;
                if (io.bits_out) {
                    std::uint8_t* bits = io.bits_out + frame_offset;
                    for (std::uint32_t v = v0; v < v1; ++v) {
                        bits[v] = S::is_negative(src[std::size_t{v} * W]) ? 1 : 0;
                    }
                }
                if (io.llr_out) {
                    float* llr = io.llr_out + frame_offset;
                    for (std::uint32_t v = v0; v < v1; ++v) {
                        llr[v] = static_cast<float>(src[std::size_t{v} * W]) * inverse_scale;
                    }
                }
            }
        }
        if (io.results) {
            for (std::uint32_t lane = 0; lane < io.num_frames; ++lane) {
                if ((lanes >> lane) & 1u) {
                    io.results[lane].iterations = iteration;
                    io.results[lane].converged = !((unsatisfied >> lane) & 1u);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Helpers shared by the fixed-point traits of every backend.

/// round(x * scale) (halves round up) saturated to [-max, max]; NaN maps to 0.
///
/// Rounds by adding max + 0.5 and truncating a value that is always positive,
/// which avoids a sign-dependent select: compilers tend to turn that into a
/// branch, and LLR signs are random, so it would mispredict constantly.
template <class T>
inline T quantize_fixed(float x, float scale, T max) noexcept {
    const float limit = static_cast<float>(max);
    float v = x * scale;
    v = v == v ? v : 0.0f;
    v = v < limit ? v : limit;
    v = v > -limit ? v : -limit;
    return static_cast<T>(static_cast<int>(v + (limit + 0.5f)) - static_cast<int>(max));
}

inline bool float_is_negative(float x) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return (bits >> 31) != 0;
}

/// Normalization factor in Q15 for mulhrs-style fixed-point multiplies.
inline std::int16_t factor_q15(float factor) noexcept {
    const float q = factor * 32768.0f + 0.5f;
    if (q >= 32767.0f) {
        return 32767;
    }
    return q <= 0.0f ? std::int16_t{0} : static_cast<std::int16_t>(q);
}

}  // namespace
}  // namespace detail
}  // namespace ldpc
