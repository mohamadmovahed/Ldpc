#include "ldpc/parallel_decoder.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

#include "ldpc/decoder.hpp"
#include "thread_pool.hpp"

namespace ldpc {

namespace {

unsigned resolve_thread_count(unsigned requested) {
    if (requested > 0) {
        return requested;
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

}  // namespace

struct ParallelDecoder::Impl {
    std::vector<Decoder> decoders;  // one per worker
    detail::ThreadPool pool;

    Impl(std::shared_ptr<const SparseMatrix> h, const DecoderConfig& config, unsigned threads)
        : decoders(make_decoders(std::move(h), config, threads)), pool(threads) {}

    static std::vector<Decoder> make_decoders(std::shared_ptr<const SparseMatrix> h, const DecoderConfig& config,
                                              unsigned threads) {
        std::vector<Decoder> result;
        result.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            result.emplace_back(h, config);
        }
        return result;
    }

    const Decoder& front() const noexcept { return decoders.front(); }
};

ParallelDecoder::ParallelDecoder(SparseMatrix h, const DecoderConfig& config, unsigned num_threads)
    : ParallelDecoder(std::make_shared<const SparseMatrix>(std::move(h)), config, num_threads) {}

ParallelDecoder::ParallelDecoder(std::shared_ptr<const SparseMatrix> h, const DecoderConfig& config,
                                 unsigned num_threads)
    : impl_(std::make_unique<Impl>(std::move(h), config, resolve_thread_count(num_threads))) {}

ParallelDecoder::~ParallelDecoder() = default;
ParallelDecoder::ParallelDecoder(ParallelDecoder&&) noexcept = default;
ParallelDecoder& ParallelDecoder::operator=(ParallelDecoder&&) noexcept = default;

void ParallelDecoder::decode_batch(const float* llr, std::size_t num_frames, std::uint8_t* bits,
                                   DecodeResult* results, float* llr_out) {
    const std::size_t lanes = batch_size();
    const std::size_t n = this->n();
    const std::size_t num_chunks = (num_frames + lanes - 1) / lanes;
    impl_->pool.parallel_for(num_chunks, [&](std::size_t chunk, unsigned worker) {
        const std::size_t first = chunk * lanes;
        const std::size_t count = std::min(lanes, num_frames - first);
        impl_->decoders[worker].decode_batch(llr + first * n, count, bits ? bits + first * n : nullptr,
                                             results ? results + first : nullptr,
                                             llr_out ? llr_out + first * n : nullptr);
    });
}

std::vector<DecodeResult> ParallelDecoder::decode_batch(const std::vector<float>& llr,
                                                        std::vector<std::uint8_t>& bits) {
    const std::size_t frames = llr.size() / n();
    if (frames * n() != llr.size()) {
        throw std::invalid_argument("ParallelDecoder::decode_batch: input size is not a multiple of n()");
    }
    bits.resize(llr.size());
    std::vector<DecodeResult> results(frames);
    decode_batch(llr.data(), frames, bits.data(), results.data(), nullptr);
    return results;
}

std::size_t ParallelDecoder::n() const noexcept { return impl_->front().n(); }
std::size_t ParallelDecoder::batch_size() const noexcept { return impl_->front().batch_size(); }
unsigned ParallelDecoder::num_threads() const noexcept { return impl_->pool.size(); }
SimdBackend ParallelDecoder::backend() const noexcept { return impl_->front().backend(); }
const DecoderConfig& ParallelDecoder::config() const noexcept { return impl_->front().config(); }
const SparseMatrix& ParallelDecoder::matrix() const noexcept { return impl_->front().matrix(); }

}  // namespace ldpc
