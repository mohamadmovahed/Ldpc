#include <stdexcept>

#include "ldpc/sparse_matrix.hpp"
#include "test_framework.hpp"

using ldpc::SparseMatrix;

namespace {

// Hamming(7,4) parity-check matrix.
SparseMatrix hamming() { return SparseMatrix::from_rows(7, {{0, 1, 2, 4}, {1, 2, 3, 5}, {0, 1, 3, 6}}); }

}  // namespace

TEST_CASE("sparse_matrix: dimensions and views") {
    const SparseMatrix h = hamming();
    CHECK_EQ(h.rows(), 3u);
    CHECK_EQ(h.cols(), 7u);
    CHECK_EQ(h.nnz(), 12u);
    CHECK_EQ(h.max_row_degree(), 4u);
    CHECK_EQ(h.max_col_degree(), 3u);
    CHECK_EQ(h.col_degree(1), 3u);
    CHECK_EQ(h.col_degree(6), 1u);
    CHECK(h.contains(2, 6));
    CHECK(!h.contains(2, 5));
    const auto col1 = h.col(1);
    CHECK_EQ(col1.size(), 3u);
    CHECK_EQ(col1[0], 0u);
    CHECK_EQ(col1[2], 2u);
}

TEST_CASE("sparse_matrix: rows are sorted and entries equivalent") {
    const SparseMatrix a = SparseMatrix::from_rows(4, {{3, 0}, {2, 1}});
    const SparseMatrix b = SparseMatrix::from_entries(2, 4, {{1, 1}, {0, 3}, {1, 2}, {0, 0}});
    CHECK(a == b);
    CHECK_EQ(a.row(0)[0], 0u);
    CHECK_EQ(a.row(0)[1], 3u);
}

TEST_CASE("sparse_matrix: invalid input is rejected") {
    CHECK_THROWS_AS(SparseMatrix::from_rows(3, {{0, 3}}), std::invalid_argument);
    CHECK_THROWS_AS(SparseMatrix::from_rows(3, {{1, 1}}), std::invalid_argument);
    CHECK_THROWS_AS(SparseMatrix::from_entries(1, 3, {{1, 0}}), std::invalid_argument);
}

TEST_CASE("sparse_matrix: syndrome and codeword check") {
    const SparseMatrix h = hamming();
    const std::vector<std::uint8_t> zero(7, 0);
    CHECK(h.is_codeword(zero));
    // Data 1000 -> parity bits 1 0 1.
    const std::vector<std::uint8_t> cw = {1, 0, 0, 0, 1, 0, 1};
    CHECK(h.is_codeword(cw));
    std::vector<std::uint8_t> bad = cw;
    bad[3] ^= 1;
    CHECK(!h.is_codeword(bad));
    const auto s = h.syndrome(bad.data());
    CHECK_EQ(int(s[0]), 0);
    CHECK_EQ(int(s[1]), 1);
    CHECK_EQ(int(s[2]), 1);
    CHECK_THROWS_AS(h.is_codeword(std::vector<std::uint8_t>(6)), std::invalid_argument);
}
