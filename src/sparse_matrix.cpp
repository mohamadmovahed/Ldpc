#include "ldpc/sparse_matrix.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ldpc {

namespace {

void check_size_fits(std::size_t value, const char* what) {
    if (value >= std::numeric_limits<SparseMatrix::Index>::max()) {
        throw std::invalid_argument(std::string("SparseMatrix: too many ") + what);
    }
}

}  // namespace

SparseMatrix SparseMatrix::from_rows(std::size_t num_cols, const std::vector<std::vector<Index>>& rows) {
    check_size_fits(rows.size(), "rows");
    check_size_fits(num_cols, "columns");

    SparseMatrix m;
    m.num_rows_ = rows.size();
    m.num_cols_ = num_cols;
    m.row_ptr_.assign(1, 0);
    m.row_ptr_.reserve(rows.size() + 1);

    std::vector<Index> sorted;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        sorted = rows[r];
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            if (sorted[i] >= num_cols) {
                throw std::invalid_argument("SparseMatrix: column index " + std::to_string(sorted[i]) +
                                            " out of range in row " + std::to_string(r));
            }
            if (i > 0 && sorted[i] == sorted[i - 1]) {
                throw std::invalid_argument("SparseMatrix: duplicate column " + std::to_string(sorted[i]) +
                                            " in row " + std::to_string(r));
            }
        }
        m.row_idx_.insert(m.row_idx_.end(), sorted.begin(), sorted.end());
        check_size_fits(m.row_idx_.size(), "non-zero entries");
        m.row_ptr_.push_back(static_cast<Index>(m.row_idx_.size()));
    }
    m.build_columns();
    return m;
}

SparseMatrix SparseMatrix::from_entries(std::size_t num_rows, std::size_t num_cols,
                                        const std::vector<std::pair<Index, Index>>& entries) {
    std::vector<std::vector<Index>> rows(num_rows);
    for (const auto& e : entries) {
        if (e.first >= num_rows) {
            throw std::invalid_argument("SparseMatrix: row index " + std::to_string(e.first) + " out of range");
        }
        rows[e.first].push_back(e.second);
    }
    return from_rows(num_cols, rows);
}

void SparseMatrix::build_columns() {
    col_ptr_.assign(num_cols_ + 1, 0);
    for (Index c : row_idx_) {
        ++col_ptr_[c + 1];
    }
    for (std::size_t c = 0; c < num_cols_; ++c) {
        col_ptr_[c + 1] += col_ptr_[c];
    }
    col_idx_.resize(row_idx_.size());
    std::vector<Index> fill(col_ptr_.begin(), col_ptr_.end() - 1);
    // Rows are visited in increasing order, so every column list ends up sorted.
    for (std::size_t r = 0; r < num_rows_; ++r) {
        for (Index i = row_ptr_[r]; i < row_ptr_[r + 1]; ++i) {
            col_idx_[fill[row_idx_[i]]++] = static_cast<Index>(r);
        }
    }
}

SparseMatrix::IndexRange SparseMatrix::row(std::size_t r) const noexcept {
    return {row_idx_.data() + row_ptr_[r], row_idx_.data() + row_ptr_[r + 1]};
}

SparseMatrix::IndexRange SparseMatrix::col(std::size_t c) const noexcept {
    return {col_idx_.data() + col_ptr_[c], col_idx_.data() + col_ptr_[c + 1]};
}

std::size_t SparseMatrix::max_row_degree() const noexcept {
    std::size_t best = 0;
    for (std::size_t r = 0; r < num_rows_; ++r) {
        best = std::max(best, row_degree(r));
    }
    return best;
}

std::size_t SparseMatrix::max_col_degree() const noexcept {
    std::size_t best = 0;
    for (std::size_t c = 0; c < num_cols_; ++c) {
        best = std::max(best, col_degree(c));
    }
    return best;
}

bool SparseMatrix::contains(std::size_t r, std::size_t c) const noexcept {
    const IndexRange range = row(r);
    return std::binary_search(range.begin(), range.end(), static_cast<Index>(c));
}

std::vector<std::uint8_t> SparseMatrix::syndrome(const std::uint8_t* bits) const {
    std::vector<std::uint8_t> s(num_rows_, 0);
    for (std::size_t r = 0; r < num_rows_; ++r) {
        std::uint8_t parity = 0;
        for (Index c : row(r)) {
            parity ^= static_cast<std::uint8_t>(bits[c] != 0);
        }
        s[r] = parity;
    }
    return s;
}

bool SparseMatrix::is_codeword(const std::uint8_t* bits) const noexcept {
    for (std::size_t r = 0; r < num_rows_; ++r) {
        std::uint8_t parity = 0;
        for (Index c : row(r)) {
            parity ^= static_cast<std::uint8_t>(bits[c] != 0);
        }
        if (parity) {
            return false;
        }
    }
    return true;
}

bool SparseMatrix::is_codeword(const std::vector<std::uint8_t>& bits) const {
    if (bits.size() != num_cols_) {
        throw std::invalid_argument("SparseMatrix::is_codeword: expected " + std::to_string(num_cols_) + " bits");
    }
    return is_codeword(bits.data());
}

bool operator==(const SparseMatrix& a, const SparseMatrix& b) noexcept {
    return a.num_rows_ == b.num_rows_ && a.num_cols_ == b.num_cols_ && a.row_ptr_ == b.row_ptr_ &&
           a.row_idx_ == b.row_idx_;
}

}  // namespace ldpc
