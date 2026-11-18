#include "ldpc/encoder.hpp"

#include <stdexcept>
#include <string>

#include "bit_vector.hpp"
#include "structure.hpp"

namespace ldpc {

namespace {

using Index = SparseMatrix::Index;

/// Strategy interface: one implementation per encoding method.
class EncoderStrategy {
public:
    virtual ~EncoderStrategy() = default;
    virtual void encode(const std::uint8_t* info, std::uint8_t* codeword) const = 0;
};

/// Linear-time encoder for accumulator (IRA / DVB-style) codes.
class AccumulatorEncoder final : public EncoderStrategy {
public:
    explicit AccumulatorEncoder(const SparseMatrix& h) : h_(h), k_(h.cols() - h.rows()) {}

    void encode(const std::uint8_t* info, std::uint8_t* codeword) const override {
        for (std::size_t i = 0; i < k_; ++i) {
            codeword[i] = info[i] ? 1 : 0;
        }
        std::uint8_t parity = 0;
        for (std::size_t r = 0; r < h_.rows(); ++r) {
            for (Index c : h_.row(r)) {
                if (c >= k_) {
                    break;  // indices are sorted, the rest are parity bits
                }
                parity ^= codeword[c];
            }
            codeword[k_ + r] = parity;
        }
    }

private:
    const SparseMatrix& h_;
    std::size_t k_;
};

/// Generic encoder based on the reduced row echelon form of H over GF(2).
class GaussianEncoder final : public EncoderStrategy {
public:
    explicit GaussianEncoder(const SparseMatrix& h) : n_(h.cols()) {
        std::vector<detail::BitVector> rows;
        rows.reserve(h.rows());
        for (std::size_t r = 0; r < h.rows(); ++r) {
            detail::BitVector bits(n_);
            for (Index c : h.row(r)) {
                bits.set(c);
            }
            rows.push_back(std::move(bits));
        }

        // Pivot on columns from the right so that, for H = [A | B] with B
        // invertible, the information bits are the leading positions.
        std::vector<bool> is_pivot(n_, false);
        std::size_t rank = 0;
        for (std::size_t c = n_; c-- > 0 && rank < rows.size();) {
            std::size_t pivot = rank;
            while (pivot < rows.size() && !rows[pivot].test(c)) {
                ++pivot;
            }
            if (pivot == rows.size()) {
                continue;
            }
            std::swap(rows[rank], rows[pivot]);
            for (std::size_t r = 0; r < rows.size(); ++r) {
                if (r != rank && rows[r].test(c)) {
                    rows[r].xor_with(rows[rank]);
                }
            }
            pivot_cols_.push_back(static_cast<Index>(c));
            is_pivot[c] = true;
            ++rank;
        }

        for (std::size_t c = 0; c < n_; ++c) {
            if (!is_pivot[c]) {
                info_positions_.push_back(static_cast<Index>(c));
            }
        }

        // Row i of the RREF yields: x[pivot_i] = sum over info positions j of R[i][j] * u_j.
        equations_.reserve(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            detail::BitVector eq(info_positions_.size());
            for (std::size_t j = 0; j < info_positions_.size(); ++j) {
                if (rows[i].test(info_positions_[j])) {
                    eq.set(j);
                }
            }
            equations_.push_back(std::move(eq));
        }
    }

    const std::vector<Index>& info_positions() const noexcept { return info_positions_; }

    void encode(const std::uint8_t* info, std::uint8_t* codeword) const override {
        detail::BitVector u(info_positions_.size());
        for (std::size_t j = 0; j < info_positions_.size(); ++j) {
            const std::uint8_t bit = info[j] ? 1 : 0;
            codeword[info_positions_[j]] = bit;
            if (bit) {
                u.set(j);
            }
        }
        for (std::size_t i = 0; i < pivot_cols_.size(); ++i) {
            codeword[pivot_cols_[i]] = equations_[i].dot(u);
        }
    }

private:
    std::size_t n_;
    std::vector<Index> pivot_cols_;
    std::vector<Index> info_positions_;
    std::vector<detail::BitVector> equations_;
};

}  // namespace

struct Encoder::Impl {
    SparseMatrix h;
    std::vector<Index> info_positions;
    std::unique_ptr<EncoderStrategy> strategy;
};

Encoder::Encoder(const SparseMatrix& h) : impl_(std::make_unique<Impl>()) {
    if (h.cols() == 0) {
        throw std::invalid_argument("Encoder: empty parity-check matrix");
    }
    impl_->h = h;
    if (detail::has_accumulator_parity(impl_->h)) {
        const std::size_t k = h.cols() - h.rows();
        for (std::size_t i = 0; i < k; ++i) {
            impl_->info_positions.push_back(static_cast<Index>(i));
        }
        impl_->strategy = std::make_unique<AccumulatorEncoder>(impl_->h);
    } else {
        auto gaussian = std::make_unique<GaussianEncoder>(impl_->h);
        impl_->info_positions = gaussian->info_positions();
        impl_->strategy = std::move(gaussian);
    }
}

Encoder::~Encoder() = default;
Encoder::Encoder(Encoder&&) noexcept = default;
Encoder& Encoder::operator=(Encoder&&) noexcept = default;

std::size_t Encoder::n() const noexcept { return impl_->h.cols(); }
std::size_t Encoder::k() const noexcept { return impl_->info_positions.size(); }

void Encoder::encode(const std::uint8_t* info, std::uint8_t* codeword) const {
    impl_->strategy->encode(info, codeword);
}

std::vector<std::uint8_t> Encoder::encode(const std::vector<std::uint8_t>& info) const {
    if (info.size() != k()) {
        throw std::invalid_argument("Encoder::encode: expected " + std::to_string(k()) + " information bits, got " +
                                    std::to_string(info.size()));
    }
    std::vector<std::uint8_t> codeword(n());
    encode(info.data(), codeword.data());
    return codeword;
}

const std::vector<SparseMatrix::Index>& Encoder::info_positions() const noexcept { return impl_->info_positions; }

void Encoder::extract_info(const std::uint8_t* codeword, std::uint8_t* info) const {
    const auto& positions = impl_->info_positions;
    for (std::size_t j = 0; j < positions.size(); ++j) {
        info[j] = codeword[positions[j]];
    }
}

}  // namespace ldpc
