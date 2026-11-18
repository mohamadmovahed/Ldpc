#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ldpc {
namespace detail {

/// Minimal packed GF(2) vector used by the Gaussian-elimination encoder.
class BitVector {
public:
    explicit BitVector(std::size_t size) : words_((size + 63) / 64, 0) {}

    void set(std::size_t i) noexcept { words_[i / 64] |= std::uint64_t{1} << (i % 64); }
    bool test(std::size_t i) const noexcept { return (words_[i / 64] >> (i % 64)) & 1u; }

    void xor_with(const BitVector& other) noexcept {
        for (std::size_t w = 0; w < words_.size(); ++w) {
            words_[w] ^= other.words_[w];
        }
    }

    /// Inner product over GF(2).
    std::uint8_t dot(const BitVector& other) const noexcept {
        std::uint64_t acc = 0;
        for (std::size_t w = 0; w < words_.size(); ++w) {
            acc ^= words_[w] & other.words_[w];
        }
        return static_cast<std::uint8_t>(parity(acc));
    }

private:
    static unsigned parity(std::uint64_t x) noexcept {
        x ^= x >> 32;
        x ^= x >> 16;
        x ^= x >> 8;
        x ^= x >> 4;
        x ^= x >> 2;
        x ^= x >> 1;
        return static_cast<unsigned>(x & 1u);
    }

    std::vector<std::uint64_t> words_;
};

}  // namespace detail
}  // namespace ldpc
