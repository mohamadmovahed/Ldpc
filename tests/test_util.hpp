#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "ldpc/encoder.hpp"
#include "ldpc/sparse_matrix.hpp"

namespace test {

/// Random column-weight-3 code (Gallager-like), used for generic-matrix tests.
inline ldpc::SparseMatrix random_regular_code(std::size_t n, std::size_t m, unsigned col_weight, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint32_t> pick(0, static_cast<std::uint32_t>(m - 1));
    std::vector<std::vector<ldpc::SparseMatrix::Index>> rows(m);
    for (std::size_t c = 0; c < n; ++c) {
        std::vector<std::uint32_t> chosen;
        while (chosen.size() < col_weight) {
            const std::uint32_t r = pick(rng);
            bool dup = false;
            for (auto x : chosen) dup |= x == r;
            if (!dup) chosen.push_back(r);
        }
        for (auto r : chosen) rows[r].push_back(static_cast<ldpc::SparseMatrix::Index>(c));
    }
    return ldpc::SparseMatrix::from_rows(n, rows);
}

/// Encodes `frames` random codewords and returns BPSK/AWGN channel LLRs.
struct NoisyFrames {
    std::vector<std::uint8_t> codewords;
    std::vector<float> llr;
};

inline NoisyFrames make_noisy_frames(const ldpc::Encoder& encoder, std::size_t frames, double ebn0_db,
                                     unsigned seed) {
    std::mt19937 rng(seed);
    std::bernoulli_distribution coin(0.5);
    const double rate = static_cast<double>(encoder.k()) / static_cast<double>(encoder.n());
    const double sigma = std::sqrt(1.0 / (2.0 * rate * std::pow(10.0, ebn0_db / 10.0)));
    std::normal_distribution<double> noise(0.0, sigma);

    NoisyFrames out;
    out.codewords.resize(frames * encoder.n());
    out.llr.resize(frames * encoder.n());
    std::vector<std::uint8_t> info(encoder.k());
    for (std::size_t f = 0; f < frames; ++f) {
        for (auto& b : info) b = coin(rng);
        encoder.encode(info.data(), out.codewords.data() + f * encoder.n());
        for (std::size_t i = 0; i < encoder.n(); ++i) {
            const double x = out.codewords[f * encoder.n() + i] ? -1.0 : 1.0;
            out.llr[f * encoder.n() + i] = static_cast<float>(2.0 * (x + noise(rng)) / (sigma * sigma));
        }
    }
    return out;
}

}  // namespace test
