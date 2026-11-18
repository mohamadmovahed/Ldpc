#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ldpc/decoder_config.hpp"
#include "ldpc/sparse_matrix.hpp"

namespace ldpc {

/// Single-threaded, SIMD-accelerated layered min-sum LDPC decoder.
///
/// SIMD is applied across frames: `batch_size()` frames are decoded
/// simultaneously, one per vector lane. Decoding fewer frames costs the same
/// time as a full batch, so pass frames in multiples of `batch_size()` for
/// maximum throughput.
///
/// LLR convention: llr = log(P(bit = 0) / P(bit = 1)); positive means "0".
/// Frames are stored contiguously: frame f occupies llr[f * n() .. (f+1) * n()).
///
/// A Decoder owns scratch memory and is not thread-safe; use one instance per
/// thread or `ParallelDecoder`.
class Decoder {
public:
    /// Throws std::invalid_argument for an invalid configuration or matrix,
    /// or if the requested SIMD backend is unavailable.
    explicit Decoder(SparseMatrix h, const DecoderConfig& config = {});
    explicit Decoder(std::shared_ptr<const SparseMatrix> h, const DecoderConfig& config = {});
    ~Decoder();
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    /// Decodes one frame: `llr` holds n() values, `bits` receives n() hard decisions (0/1).
    DecodeResult decode(const float* llr, std::uint8_t* bits);

    /// Decodes `num_frames` frames.
    ///  - llr:     num_frames * n() input LLRs
    ///  - bits:    num_frames * n() hard decisions (may be null)
    ///  - results: num_frames results (may be null)
    ///  - llr_out: num_frames * n() a-posteriori LLRs (may be null)
    void decode_batch(const float* llr, std::size_t num_frames, std::uint8_t* bits,
                      DecodeResult* results = nullptr, float* llr_out = nullptr);

    /// Convenience overload: decodes llr.size() / n() frames; resizes `bits`.
    std::vector<DecodeResult> decode_batch(const std::vector<float>& llr, std::vector<std::uint8_t>& bits);

    std::size_t n() const noexcept;           ///< codeword length (columns of H)
    std::size_t num_checks() const noexcept;  ///< parity checks (rows of H)
    std::size_t batch_size() const noexcept;  ///< frames decoded in parallel (SIMD lanes)
    SimdBackend backend() const noexcept;     ///< resolved backend (never Auto)
    const DecoderConfig& config() const noexcept;
    const SparseMatrix& matrix() const noexcept;
    const std::shared_ptr<const SparseMatrix>& shared_matrix() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ldpc
