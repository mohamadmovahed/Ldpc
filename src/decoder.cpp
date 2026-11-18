#include "ldpc/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "aligned_buffer.hpp"
#include "kernel/kernel.hpp"
#include "structure.hpp"

namespace ldpc {

namespace {

float default_fixed_point_scale(Precision precision) {
    switch (precision) {
        case Precision::Float32: return 1.0f;
        case Precision::Int16: return 64.0f;
        case Precision::Int8: return 2.0f;
    }
    return 1.0f;
}

void validate(const SparseMatrix& h, const DecoderConfig& c) {
    if (h.rows() == 0 || h.cols() == 0) {
        throw std::invalid_argument("Decoder: empty parity-check matrix");
    }
    if (c.max_iterations < 0) {
        throw std::invalid_argument("Decoder: max_iterations must be >= 0");
    }
    if (!(c.normalization_factor > 0.0f && c.normalization_factor <= 1.0f)) {
        throw std::invalid_argument("Decoder: normalization_factor must be in (0, 1]");
    }
    if (!(c.offset >= 0.0f) || !std::isfinite(c.offset)) {
        throw std::invalid_argument("Decoder: offset must be a finite value >= 0");
    }
    if (!(c.fixed_point_scale >= 0.0f) || !std::isfinite(c.fixed_point_scale)) {
        throw std::invalid_argument("Decoder: fixed_point_scale must be a finite value >= 0");
    }
}

}  // namespace

struct Decoder::Impl {
    std::shared_ptr<const SparseMatrix> h;
    DecoderConfig config;
    SimdBackend backend = SimdBackend::Scalar;
    const detail::LayeredKernel* kernel = nullptr;
    std::vector<std::uint32_t> check_offsets;    // CSR of H with rows in schedule order
    std::vector<std::uint32_t> check_variables;
    detail::Graph graph{};
    detail::KernelParams params{};
    detail::AlignedBuffer workspace;

    Impl(std::shared_ptr<const SparseMatrix> matrix, const DecoderConfig& cfg) : h(std::move(matrix)), config(cfg) {
        if (!h) {
            throw std::invalid_argument("Decoder: null parity-check matrix");
        }
        validate(*h, config);

        kernel = detail::select_kernel(config.backend, config.precision, &backend);
        if (!kernel) {
            throw std::invalid_argument(std::string("Decoder: SIMD backend '") + to_string(config.backend) +
                                        "' is not available on this build or CPU");
        }

        build_schedule();
        graph.row_ptr = check_offsets.data();
        graph.col_idx = check_variables.data();
        graph.num_rows = static_cast<std::uint32_t>(h->rows());
        graph.num_cols = static_cast<std::uint32_t>(h->cols());
        graph.num_edges = static_cast<std::uint32_t>(h->nnz());
        graph.max_row_degree = static_cast<std::uint32_t>(h->max_row_degree());

        params.normalized = config.algorithm == Algorithm::NormalizedMinSum;
        params.factor = config.normalization_factor;
        params.offset = config.offset;
        params.scale = config.precision == Precision::Float32
                           ? 1.0f
                           : (config.fixed_point_scale > 0.0f ? config.fixed_point_scale
                                                              : default_fixed_point_scale(config.precision));
        params.max_iterations = config.max_iterations;
        params.early_termination = config.early_termination;

        workspace = detail::AlignedBuffer(kernel->workspace_bytes(graph));
    }

    void build_schedule() {
        check_offsets.reserve(h->rows() + 1);
        check_variables.reserve(h->nnz());
        check_offsets.push_back(0);
        for (std::uint32_t r : detail::layered_check_order(*h)) {
            const auto vars = h->row(r);
            check_variables.insert(check_variables.end(), vars.begin(), vars.end());
            check_offsets.push_back(static_cast<std::uint32_t>(check_variables.size()));
        }
    }

    void decode_batch(const float* llr, std::size_t num_frames, std::uint8_t* bits, DecodeResult* results,
                      float* llr_out) {
        if (num_frames > 0 && !llr) {
            throw std::invalid_argument("Decoder::decode_batch: null input");
        }
        const std::size_t lanes = kernel->lanes();
        const std::size_t n = graph.num_cols;
        for (std::size_t first = 0; first < num_frames; first += lanes) {
            const std::size_t count = std::min(lanes, num_frames - first);
            detail::KernelIO io{};
            io.llr_in = llr + first * n;
            io.num_frames = static_cast<std::uint32_t>(count);
            io.bits_out = bits ? bits + first * n : nullptr;
            io.llr_out = llr_out ? llr_out + first * n : nullptr;
            io.results = results ? results + first : nullptr;
            kernel->decode(graph, params, workspace.data(), io);
        }
    }
};

Decoder::Decoder(SparseMatrix h, const DecoderConfig& config)
    : Decoder(std::make_shared<const SparseMatrix>(std::move(h)), config) {}

Decoder::Decoder(std::shared_ptr<const SparseMatrix> h, const DecoderConfig& config)
    : impl_(std::make_unique<Impl>(std::move(h), config)) {}

Decoder::~Decoder() = default;
Decoder::Decoder(Decoder&&) noexcept = default;
Decoder& Decoder::operator=(Decoder&&) noexcept = default;

DecodeResult Decoder::decode(const float* llr, std::uint8_t* bits) {
    DecodeResult result;
    impl_->decode_batch(llr, 1, bits, &result, nullptr);
    return result;
}

void Decoder::decode_batch(const float* llr, std::size_t num_frames, std::uint8_t* bits, DecodeResult* results,
                           float* llr_out) {
    impl_->decode_batch(llr, num_frames, bits, results, llr_out);
}

std::vector<DecodeResult> Decoder::decode_batch(const std::vector<float>& llr, std::vector<std::uint8_t>& bits) {
    const std::size_t frames = llr.size() / n();
    if (frames * n() != llr.size()) {
        throw std::invalid_argument("Decoder::decode_batch: input size is not a multiple of n()");
    }
    bits.resize(llr.size());
    std::vector<DecodeResult> results(frames);
    impl_->decode_batch(llr.data(), frames, bits.data(), results.data(), nullptr);
    return results;
}

std::size_t Decoder::n() const noexcept { return impl_->h->cols(); }
std::size_t Decoder::num_checks() const noexcept { return impl_->h->rows(); }
std::size_t Decoder::batch_size() const noexcept { return impl_->kernel->lanes(); }
SimdBackend Decoder::backend() const noexcept { return impl_->backend; }
const DecoderConfig& Decoder::config() const noexcept { return impl_->config; }
const SparseMatrix& Decoder::matrix() const noexcept { return *impl_->h; }
const std::shared_ptr<const SparseMatrix>& Decoder::shared_matrix() const noexcept { return impl_->h; }

}  // namespace ldpc
