#include "ldpc/interleaver.hpp"

#include <algorithm>
#include <stdexcept>

namespace ldpc {

BlockInterleaver::BlockInterleaver(std::size_t n, std::vector<int> read_order) : columns_(read_order.size()) {
    if (columns_ == 0 || n == 0 || n % columns_ != 0) {
        throw std::invalid_argument("interleaver: n must be a non-zero multiple of the column count");
    }
    std::vector<int> sorted = read_order;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] != static_cast<int>(i)) {
            throw std::invalid_argument("interleaver: read_order must be a permutation of 0..columns-1");
        }
    }
    const std::size_t rows = n / columns_;
    perm_.resize(n);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t j = 0; j < columns_; ++j) {
            perm_[r * columns_ + j] = static_cast<std::uint32_t>(static_cast<std::size_t>(read_order[j]) * rows + r);
        }
    }
}

void BlockInterleaver::interleave(const std::uint8_t* codeword, std::uint8_t* out) const {
    for (std::size_t i = 0; i < perm_.size(); ++i) out[i] = codeword[perm_[i]];
}

void BlockInterleaver::deinterleave(const float* in, float* codeword) const {
    for (std::size_t i = 0; i < perm_.size(); ++i) codeword[perm_[i]] = in[i];
}

void BlockInterleaver::deinterleave(const std::uint8_t* in, std::uint8_t* codeword) const {
    for (std::size_t i = 0; i < perm_.size(); ++i) codeword[perm_[i]] = in[i];
}

}  // namespace ldpc
