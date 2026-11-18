#pragma once

#include <cstdint>
#include <vector>

#include "ldpc/sparse_matrix.hpp"

namespace ldpc {
namespace detail {

/// True if H = [A | P] with P dual-diagonal: parity column i (i.e. column
/// k + i) covers checks i and i + 1, and the last one only check m - 1.
bool has_accumulator_parity(const SparseMatrix& h) noexcept;

/// Order in which the layered decoder visits the checks.
///
/// For DVB-style codes (accumulator parity and information columns that are
/// quasi-cyclic in groups of 360 with row shift q = m / 360) the checks are
/// visited as i, i + q, i + 2q, ...: consecutive checks then touch adjacent
/// variables, which greatly improves cache locality. Other matrices keep
/// their natural order.
std::vector<std::uint32_t> layered_check_order(const SparseMatrix& h);

}  // namespace detail
}  // namespace ldpc
