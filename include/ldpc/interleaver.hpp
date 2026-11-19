#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ldpc {

/// Block (column/row) bit interleaver of DVB-S2/S2X (EN 302 307-1 clause 5.3.3, EN 302 307-2
/// clause 5.3.3): codeword bits are written column-wise into `columns` columns of n/columns
/// rows and read out row-wise, the columns of each row in `read_order` (read_order[0] is the
/// column read first; DVB-S2 8PSK 3/5 uses {2, 1, 0}, S2X MODCODs their table 9a/9b
/// pattern).
///
/// Thread safety: immutable after construction.
class BlockInterleaver {
public:
    /// @param n codeword length (a multiple of read_order.size()).
    /// @param read_order a permutation of 0 .. columns-1.
    /// Throws std::invalid_argument otherwise.
    BlockInterleaver(std::size_t n, std::vector<int> read_order);

    std::size_t n() const noexcept { return perm_.size(); }
    std::size_t columns() const noexcept { return columns_; }

    /// Codeword order -> transmission order (n bytes each).
    void interleave(const std::uint8_t* codeword, std::uint8_t* out) const;
    /// Transmission order -> codeword order, for soft values (LLRs).
    void deinterleave(const float* in, float* codeword) const;
    /// Transmission order -> codeword order, for bits.
    void deinterleave(const std::uint8_t* in, std::uint8_t* codeword) const;

    /// perm()[i] = codeword index of the i-th transmitted bit.
    const std::vector<std::uint32_t>& perm() const noexcept { return perm_; }

private:
    std::size_t columns_ = 0;
    std::vector<std::uint32_t> perm_;
};

}  // namespace ldpc
