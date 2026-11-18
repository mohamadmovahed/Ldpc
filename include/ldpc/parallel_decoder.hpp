#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ldpc/decoder_config.hpp"
#include "ldpc/sparse_matrix.hpp"

namespace ldpc {

/// Multi-threaded decoder: splits a batch of frames into SIMD-sized chunks
/// and decodes them on a persistent pool of worker threads, each with its own
/// SIMD `Decoder`. Results are identical to `Decoder`.
///
/// Calls to `decode_batch` on one instance must not overlap.
class ParallelDecoder {
public:
    /// `num_threads == 0` uses std::thread::hardware_concurrency().
    explicit ParallelDecoder(SparseMatrix h, const DecoderConfig& config = {}, unsigned num_threads = 0);
    explicit ParallelDecoder(std::shared_ptr<const SparseMatrix> h, const DecoderConfig& config = {},
                             unsigned num_threads = 0);
    ~ParallelDecoder();
    ParallelDecoder(ParallelDecoder&&) noexcept;
    ParallelDecoder& operator=(ParallelDecoder&&) noexcept;
    ParallelDecoder(const ParallelDecoder&) = delete;
    ParallelDecoder& operator=(const ParallelDecoder&) = delete;

    /// Same contract as Decoder::decode_batch.
    void decode_batch(const float* llr, std::size_t num_frames, std::uint8_t* bits,
                      DecodeResult* results = nullptr, float* llr_out = nullptr);
    std::vector<DecodeResult> decode_batch(const std::vector<float>& llr, std::vector<std::uint8_t>& bits);

    std::size_t n() const noexcept;
    std::size_t batch_size() const noexcept;  ///< SIMD lanes per thread
    unsigned num_threads() const noexcept;
    SimdBackend backend() const noexcept;
    const DecoderConfig& config() const noexcept;
    const SparseMatrix& matrix() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ldpc
