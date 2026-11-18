#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ldpc/decoder.hpp"
#include "ldpc/dvb.hpp"
#include "ldpc/encoder.hpp"
#include "ldpc/parallel_decoder.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ldpc;

namespace {

const Precision kPrecisions[] = {Precision::Float32, Precision::Int16, Precision::Int8};
const SimdBackend kBackends[] = {SimdBackend::Scalar, SimdBackend::AVX2, SimdBackend::AVX512, SimdBackend::NEON};

std::shared_ptr<const SparseMatrix> short_dvb_code() {
    static const auto h = std::make_shared<const SparseMatrix>(
        dvb_parity_check_matrix(DvbStandard::S2, DvbFrameSize::Short, "1/2"));
    return h;
}

struct Output {
    std::vector<std::uint8_t> bits;
    std::vector<float> llr;
    std::vector<DecodeResult> results;
};

template <class DecoderT>
Output run(DecoderT& decoder, const std::vector<float>& llr) {
    const std::size_t frames = llr.size() / decoder.n();
    Output out{std::vector<std::uint8_t>(llr.size()), std::vector<float>(llr.size()),
               std::vector<DecodeResult>(frames)};
    decoder.decode_batch(llr.data(), frames, out.bits.data(), out.results.data(), out.llr.data());
    return out;
}

std::string label(SimdBackend b, Precision p) { return std::string(to_string(b)) + "/" + to_string(p); }

}  // namespace

TEST_CASE("decoder: corrects noisy DVB-S2 frames on every backend and precision") {
    const auto h = short_dvb_code();
    const Encoder encoder(*h);
    const std::size_t frames = 70;  // deliberately not a multiple of any SIMD width
    const auto data = test::make_noisy_frames(encoder, frames, 3.0, 42);
    for (SimdBackend backend : kBackends) {
        if (!is_backend_available(backend)) continue;
        for (Precision precision : kPrecisions) {
            DecoderConfig config;
            config.backend = backend;
            config.precision = precision;
            Decoder decoder(h, config);
            CHECK(decoder.backend() == backend);
            const Output out = run(decoder, data.llr);
            for (std::size_t f = 0; f < frames; ++f) {
                if (!out.results[f].converged) test::fail(__FILE__, __LINE__, label(backend, precision) + " failed");
                CHECK(out.results[f].iterations >= 1);
            }
            CHECK(out.bits == data.codewords);
        }
    }
}

TEST_CASE("decoder: SIMD backends are bit-exact with the scalar reference") {
    const auto h = short_dvb_code();
    const Encoder encoder(*h);
    // Low SNR so that some frames fail and every code path is exercised.
    const auto data = test::make_noisy_frames(encoder, 67, 0.9, 7);
    for (Precision precision : kPrecisions) {
        for (Algorithm algorithm : {Algorithm::NormalizedMinSum, Algorithm::OffsetMinSum}) {
            DecoderConfig config;
            config.precision = precision;
            config.algorithm = algorithm;
            config.max_iterations = 20;
            config.backend = SimdBackend::Scalar;
            Decoder reference_decoder(h, config);
            const Output reference = run(reference_decoder, data.llr);
            for (SimdBackend backend : kBackends) {
                if (backend == SimdBackend::Scalar || !is_backend_available(backend)) continue;
                config.backend = backend;
                Decoder decoder(h, config);
                const Output out = run(decoder, data.llr);
                if (out.bits != reference.bits || out.llr != reference.llr)
                    test::fail(__FILE__, __LINE__, label(backend, precision) + " differs from scalar");
                for (std::size_t f = 0; f < out.results.size(); ++f) {
                    CHECK_EQ(out.results[f].iterations, reference.results[f].iterations);
                    CHECK_EQ(out.results[f].converged, reference.results[f].converged);
                }
            }
        }
    }
}

TEST_CASE("decoder: generic alist-style code") {
    const auto h = test::random_regular_code(1008, 504, 3, 5);
    const Encoder encoder(h);
    const auto data = test::make_noisy_frames(encoder, 40, 4.0, 9);
    for (Precision precision : kPrecisions) {
        DecoderConfig config;
        config.precision = precision;
        Decoder decoder(h, config);
        std::vector<std::uint8_t> bits;
        const auto results = decoder.decode_batch(data.llr, bits);
        CHECK_EQ(results.size(), 40u);
        std::size_t converged = 0;
        for (const auto& r : results) converged += r.converged;
        CHECK(converged >= 38u);  // random code with short cycles: allow rare failures
    }
}

TEST_CASE("decoder: noiseless input converges after one iteration") {
    const auto h = short_dvb_code();
    std::vector<float> llr(h->cols(), 4.0f);  // all-zero codeword
    std::vector<std::uint8_t> bits(h->cols(), 1);
    Decoder decoder(h);
    const DecodeResult r = decoder.decode(llr.data(), bits.data());
    CHECK(r.converged);
    CHECK_EQ(r.iterations, 1);
    for (auto b : bits) CHECK_EQ(int(b), 0);
}

TEST_CASE("decoder: iteration limits") {
    const auto h = short_dvb_code();
    const Encoder encoder(*h);
    const auto data = test::make_noisy_frames(encoder, 3, 3.0, 1);
    DecoderConfig config;
    config.early_termination = false;
    config.max_iterations = 12;
    Decoder decoder(h, config);
    const Output out = run(decoder, data.llr);
    for (const auto& r : out.results) {
        CHECK_EQ(r.iterations, 12);
        CHECK(r.converged);
    }

    config.max_iterations = 0;  // plain hard decision of the channel values
    Decoder hard(h, config);
    const Output raw = run(hard, data.llr);
    for (std::size_t i = 0; i < data.llr.size(); ++i) CHECK_EQ(int(raw.bits[i]), int(data.llr[i] < 0));
    CHECK_EQ(raw.results[0].iterations, 0);
}

TEST_CASE("decoder: invalid configuration is rejected") {
    const auto h = short_dvb_code();
    DecoderConfig config;
    config.max_iterations = -1;
    CHECK_THROWS_AS(Decoder(h, config), std::invalid_argument);
    config = DecoderConfig{};
    config.normalization_factor = 1.5f;
    CHECK_THROWS_AS(Decoder(h, config), std::invalid_argument);
    config = DecoderConfig{};
    config.offset = -1.0f;
    CHECK_THROWS_AS(Decoder(h, config), std::invalid_argument);
    CHECK_THROWS_AS(Decoder(SparseMatrix{}), std::invalid_argument);
    for (SimdBackend backend : kBackends) {
        if (is_backend_available(backend)) continue;
        config = DecoderConfig{};
        config.backend = backend;
        CHECK_THROWS_AS(Decoder(h, config), std::invalid_argument);
    }
    Decoder decoder(h);
    std::vector<std::uint8_t> bits;
    CHECK_THROWS_AS(decoder.decode_batch(std::vector<float>(h->cols() + 1), bits), std::invalid_argument);
}

TEST_CASE("parallel_decoder: matches the single-threaded decoder") {
    const auto h = short_dvb_code();
    const Encoder encoder(*h);
    const auto data = test::make_noisy_frames(encoder, 203, 1.2, 99);
    for (Precision precision : {Precision::Float32, Precision::Int8}) {
        DecoderConfig config;
        config.precision = precision;
        config.max_iterations = 25;
        Decoder single(h, config);
        const Output expected = run(single, data.llr);
        for (unsigned threads : {1u, 2u, 3u, 8u}) {
            ParallelDecoder parallel(h, config, threads);
            CHECK_EQ(parallel.num_threads(), threads);
            CHECK_EQ(parallel.batch_size(), single.batch_size());
            for (int repeat = 0; repeat < 3; ++repeat) {  // pool reuse
                const Output out = run(parallel, data.llr);
                CHECK(out.bits == expected.bits);
                CHECK(out.llr == expected.llr);
                for (std::size_t f = 0; f < out.results.size(); ++f) {
                    CHECK_EQ(out.results[f].iterations, expected.results[f].iterations);
                }
            }
        }
    }
}

TEST_CASE("parallel_decoder: empty batch and default thread count") {
    ParallelDecoder decoder(short_dvb_code());
    CHECK(decoder.num_threads() >= 1u);
    decoder.decode_batch(nullptr, 0, nullptr);
    std::vector<std::uint8_t> bits;
    CHECK(decoder.decode_batch(std::vector<float>{}, bits).empty());
}
