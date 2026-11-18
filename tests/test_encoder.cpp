#include <random>
#include <stdexcept>

#include "ldpc/encoder.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

TEST_CASE("encoder: gaussian elimination on a generic code") {
    const auto h = test::random_regular_code(204, 102, 3, 11);
    const ldpc::Encoder encoder(h);
    CHECK_EQ(encoder.n(), 204u);
    CHECK(encoder.k() >= 102u);  // rank may be deficient
    std::mt19937 rng(3);
    std::vector<std::uint8_t> info(encoder.k()), cw(encoder.n()), back(encoder.k());
    for (int trial = 0; trial < 20; ++trial) {
        for (auto& b : info) b = rng() & 1u;
        encoder.encode(info.data(), cw.data());
        CHECK(h.is_codeword(cw));
        encoder.extract_info(cw.data(), back.data());
        CHECK(back == info);
    }
}

TEST_CASE("encoder: rank-deficient matrix") {
    // Row 2 = row 0 + row 1, so rank 2 and k = 6 - 2 = 4.
    const auto h = ldpc::SparseMatrix::from_rows(6, {{0, 1, 2}, {2, 3, 4}, {0, 1, 3, 4}});
    const ldpc::Encoder encoder(h);
    CHECK_EQ(encoder.k(), 4u);
    for (unsigned v = 0; v < 16; ++v) {
        std::vector<std::uint8_t> info = {std::uint8_t(v & 1), std::uint8_t(v >> 1 & 1), std::uint8_t(v >> 2 & 1),
                                          std::uint8_t(v >> 3 & 1)};
        CHECK(h.is_codeword(encoder.encode(info)));
    }
}

TEST_CASE("encoder: wrong input size throws") {
    const auto h = test::random_regular_code(60, 30, 3, 1);
    const ldpc::Encoder encoder(h);
    CHECK_THROWS_AS(encoder.encode(std::vector<std::uint8_t>(encoder.k() + 1)), std::invalid_argument);
}
