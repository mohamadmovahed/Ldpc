#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ldpc {

/// Binary sparse matrix (e.g. an LDPC parity-check matrix H).
///
/// Stores both a row-major (CSR) and a column-major (CSC) view so that
/// check-node and variable-node neighbourhoods are available in O(1).
/// Indices inside every row / column are sorted and unique.
class SparseMatrix {
public:
    using Index = std::uint32_t;

    /// Read-only view of the indices of one row or column.
    class IndexRange {
    public:
        IndexRange(const Index* first, const Index* last) noexcept : first_(first), last_(last) {}
        const Index* begin() const noexcept { return first_; }
        const Index* end() const noexcept { return last_; }
        std::size_t size() const noexcept { return static_cast<std::size_t>(last_ - first_); }
        bool empty() const noexcept { return first_ == last_; }
        Index operator[](std::size_t i) const noexcept { return first_[i]; }

    private:
        const Index* first_;
        const Index* last_;
    };

    SparseMatrix() = default;

    /// Builds a matrix from the column indices of every row.
    /// Throws std::invalid_argument on out-of-range or duplicate indices.
    static SparseMatrix from_rows(std::size_t num_cols, const std::vector<std::vector<Index>>& rows);

    /// Builds a matrix from a list of (row, col) positions of the non-zero entries.
    /// Throws std::invalid_argument on out-of-range or duplicate positions.
    static SparseMatrix from_entries(std::size_t num_rows, std::size_t num_cols,
                                     const std::vector<std::pair<Index, Index>>& entries);

    std::size_t rows() const noexcept { return num_rows_; }
    std::size_t cols() const noexcept { return num_cols_; }
    /// Number of non-zero entries (= Tanner graph edges).
    std::size_t nnz() const noexcept { return row_idx_.size(); }

    /// Column indices of the non-zeros in row r.
    IndexRange row(std::size_t r) const noexcept;
    /// Row indices of the non-zeros in column c.
    IndexRange col(std::size_t c) const noexcept;

    std::size_t row_degree(std::size_t r) const noexcept { return row_ptr_[r + 1] - row_ptr_[r]; }
    std::size_t col_degree(std::size_t c) const noexcept { return col_ptr_[c + 1] - col_ptr_[c]; }
    std::size_t max_row_degree() const noexcept;
    std::size_t max_col_degree() const noexcept;

    bool contains(std::size_t r, std::size_t c) const noexcept;

    /// Raw CSR arrays: row_offsets() has rows()+1 entries, row_indices() has nnz() entries.
    const std::vector<Index>& row_offsets() const noexcept { return row_ptr_; }
    const std::vector<Index>& row_indices() const noexcept { return row_idx_; }
    /// Raw CSC arrays: col_offsets() has cols()+1 entries, col_indices() has nnz() entries.
    const std::vector<Index>& col_offsets() const noexcept { return col_ptr_; }
    const std::vector<Index>& col_indices() const noexcept { return col_idx_; }

    /// Computes the syndrome H * bits (mod 2). `bits` holds cols() values, each 0 or 1
    /// (any non-zero value counts as 1). Returns rows() values in {0, 1}.
    std::vector<std::uint8_t> syndrome(const std::uint8_t* bits) const;
    /// True if H * bits == 0 (mod 2).
    bool is_codeword(const std::uint8_t* bits) const noexcept;
    bool is_codeword(const std::vector<std::uint8_t>& bits) const;

    friend bool operator==(const SparseMatrix& a, const SparseMatrix& b) noexcept;
    friend bool operator!=(const SparseMatrix& a, const SparseMatrix& b) noexcept { return !(a == b); }

private:
    void build_columns();

    std::size_t num_rows_ = 0;
    std::size_t num_cols_ = 0;
    std::vector<Index> row_ptr_{0};
    std::vector<Index> row_idx_;
    std::vector<Index> col_ptr_{0};
    std::vector<Index> col_idx_;
};

}  // namespace ldpc
