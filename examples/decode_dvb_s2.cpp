// Minimal example: decode noisy DVB-S2 frames with the multi-threaded decoder.

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "ldpc/ldpc.hpp"

int main() {
    // 1. Get a parity-check matrix: a built-in standard code or any alist file
    //    (ldpc::read_alist_file("code.alist")).
    const ldpc::SparseMatrix h = ldpc::dvb_parity_check_matrix(ldpc::DvbStandard::S2, ldpc::DvbFrameSize::Normal, "1/2");

    // 2. Configure the decoder.
    ldpc::DecoderConfig config;
    config.precision = ldpc::Precision::Int8;  // fastest; Float32 is most accurate
    config.max_iterations = 30;

    // 3. Create codewords and channel LLRs (BPSK over AWGN, Es/N0 = 2 dB).
    const ldpc::Encoder encoder(h);
    const std::size_t n = encoder.n();
    const std::size_t frames = 256;
    const float sigma = std::sqrt(0.5f / std::pow(10.0f, 0.2f));
    std::mt19937 rng(1);
    std::normal_distribution<float> noise(0.0f, sigma);
    std::vector<std::uint8_t> info(encoder.k()), codewords(frames * n);
    std::vector<float> llr(frames * n);
    for (std::size_t f = 0; f < frames; ++f) {
        for (auto& b : info) b = rng() & 1u;
        encoder.encode(info.data(), &codewords[f * n]);
        for (std::size_t i = 0; i < n; ++i) {
            const float y = (codewords[f * n + i] ? -1.0f : 1.0f) + noise(rng);
            llr[f * n + i] = 2.0f * y / (sigma * sigma);  // positive => bit 0
        }
    }

    // 4. Decode on all cores. Use ldpc::Decoder for a single thread.
    ldpc::ParallelDecoder decoder(h, config);
    std::vector<std::uint8_t> decoded;
    const std::vector<ldpc::DecodeResult> results = decoder.decode_batch(llr, decoded);

    std::size_t converged = 0;
    for (const auto& r : results) converged += r.converged;
    std::cout << "backend " << ldpc::to_string(decoder.backend()) << ", " << decoder.num_threads() << " threads x "
              << decoder.batch_size() << " lanes\n"
              << converged << "/" << frames << " frames converged, output "
              << (decoded == codewords ? "matches" : "differs from") << " the transmitted codewords\n";
    return decoded == codewords ? 0 : 1;
}
