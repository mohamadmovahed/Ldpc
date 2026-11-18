#include "structure.hpp"

#include <algorithm>
#include <numeric>

namespace ldpc {
namespace detail {

namespace {

constexpr std::size_t kDvbGroupSize = 360;

bool is_dvb_quasi_cyclic(const SparseMatrix& h) {
    const std::size_t m = h.rows();
    const std::size_t n = h.cols();
    if (m == 0 || m >= n || m % kDvbGroupSize != 0 || (n - m) % kDvbGroupSize != 0 || !has_accumulator_parity(h)) {
        return false;
    }
    const std::size_t k = n - m;
    const std::size_t q = m / kDvbGroupSize;
    std::vector<SparseMatrix::Index> shifted;
    for (std::size_t c = 0; c + 1 < k; ++c) {
        if (c % kDvbGroupSize == kDvbGroupSize - 1) {
            continue;  // last bit of a group
        }
        const auto current = h.col(c);
        const auto next = h.col(c + 1);
        if (current.size() != next.size()) {
            return false;
        }
        shifted.clear();
        for (auto r : current) {
            shifted.push_back(static_cast<SparseMatrix::Index>((r + q) % m));
        }
        std::sort(shifted.begin(), shifted.end());
        if (!std::equal(shifted.begin(), shifted.end(), next.begin())) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool has_accumulator_parity(const SparseMatrix& h) noexcept {
    const std::size_t m = h.rows();
    const std::size_t n = h.cols();
    if (m == 0 || m >= n) {
        return false;
    }
    const std::size_t k = n - m;
    for (std::size_t i = 0; i < m; ++i) {
        const auto col = h.col(k + i);
        const bool last = i + 1 == m;
        if (col.size() != (last ? 1u : 2u) || col[0] != i || (!last && col[1] != i + 1)) {
            return false;
        }
    }
    return true;
}

std::vector<std::uint32_t> layered_check_order(const SparseMatrix& h) {
    std::vector<std::uint32_t> order(h.rows());
    if (is_dvb_quasi_cyclic(h)) {
        const std::size_t q = h.rows() / kDvbGroupSize;
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = static_cast<std::uint32_t>((i % kDvbGroupSize) * q + i / kDvbGroupSize);
        }
    } else {
        std::iota(order.begin(), order.end(), 0u);
    }
    return order;
}

}  // namespace detail
}  // namespace ldpc
