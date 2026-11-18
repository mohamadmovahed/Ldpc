#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ldpc/sparse_matrix.hpp"

namespace ldpc {

/// Systematic encoder for an arbitrary parity-check matrix.
///
/// If H = [A | P] with P dual-diagonal (DVB-S2/S2X/T2, IEEE 802.11n-style
/// accumulators, ...) a linear-time accumulator encoder is used and the
/// information bits are the first k() codeword bits. Otherwise H is reduced
/// by Gaussian elimination over GF(2), which costs O(m^2 n / 64) time and
/// O(m * k / 8) memory, so it is only practical for n up to roughly 20000.
class Encoder {
public:
    explicit Encoder(const SparseMatrix& h);
    ~Encoder();
    Encoder(Encoder&&) noexcept;
    Encoder& operator=(Encoder&&) noexcept;
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    std::size_t n() const noexcept;  ///< codeword length
    std::size_t k() const noexcept;  ///< information bits (n - rank(H))

    /// Encodes k() information bits (0/1) into n() codeword bits.
    void encode(const std::uint8_t* info, std::uint8_t* codeword) const;
    std::vector<std::uint8_t> encode(const std::vector<std::uint8_t>& info) const;

    /// Codeword positions holding the information bits, in order.
    const std::vector<SparseMatrix::Index>& info_positions() const noexcept;
    /// Copies the k() information bits out of a codeword.
    void extract_info(const std::uint8_t* codeword, std::uint8_t* info) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ldpc
