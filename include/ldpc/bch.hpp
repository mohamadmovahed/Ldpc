#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ldpc/dvb.hpp"

namespace ldpc {

/// Outcome of BchCode::decode().
struct BchDecodeResult {
    bool ok = true;      ///< false: more errors than the code can correct (input left unchanged)
    int corrected = 0;   ///< bits flipped (0 when ok and the word was already a codeword)
};

/// Binary, systematic, possibly shortened t-error-correcting BCH code over GF(2^m).
///
/// The generator polynomial is the least common multiple of the minimal polynomials of
/// alpha^1 ... alpha^2t, where alpha is a root of the primitive polynomial. Codewords are
/// sent highest degree first: bit 0 of a codeword is the coefficient of x^(n-1), the first
/// k() bits are the message, the last parity_bits() bits the remainder (the DVB-S2 layout,
/// EN 302 307-1 clause 5.3.1).
///
/// Bits are one byte each (0 or 1). Encoding and the error-free decode path cost one
/// table-driven polynomial division (a byte per step); errors are located with
/// Berlekamp-Massey and a Chien search.
///
/// Thread safety: a BchCode is immutable after construction; encode() and decode() may run
/// concurrently on one instance.
class BchCode {
public:
    /// @param m field degree (2..16); @param primitive_poly bit i = coefficient of x^i, degree m;
    /// @param t errors corrected (>= 1); @param n codeword length (shortened: <= 2^m - 1).
    /// Throws std::invalid_argument for an inconsistent code (n <= parity bits, more than 256
    /// parity bits, a non-primitive polynomial).
    BchCode(int m, std::uint32_t primitive_poly, int t, std::size_t n);
    ~BchCode();
    BchCode(BchCode&&) noexcept;
    BchCode& operator=(BchCode&&) noexcept;
    BchCode(const BchCode&) = delete;
    BchCode& operator=(const BchCode&) = delete;

    std::size_t n() const noexcept;            ///< codeword length (Nbch)
    std::size_t k() const noexcept;            ///< message length (Kbch)
    std::size_t parity_bits() const noexcept;  ///< n() - k() = degree of the generator
    int t() const noexcept;                    ///< correctable errors
    int m() const noexcept;                    ///< field degree

    /// Generator polynomial coefficients, index = degree (size parity_bits() + 1).
    const std::vector<std::uint8_t>& generator() const noexcept;

    /// Computes the parity bits of a k()-bit message (parity_bits() bits, highest degree first).
    void encode(const std::uint8_t* message, std::uint8_t* parity) const;
    /// Message followed by its parity: returns n() bits.
    std::vector<std::uint8_t> encode(const std::vector<std::uint8_t>& message) const;

    /// Corrects up to t() bit errors of an n()-bit codeword in place.
    BchDecodeResult decode(std::uint8_t* codeword) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// BCH parameters of one DVB-S2/S2X code (EN 302 307-1 tables 5a/5b, EN 302 307-2 tables 4/6).
struct DvbBchInfo {
    std::size_t kbch;  ///< BBFRAME length (BCH message bits)
    std::size_t nbch;  ///< BCH codeword length = LDPC message length k
    int t;             ///< errors corrected
    int m;             ///< field degree: 16 (normal frames) or 14 (short frames)
};

/// BCH parameters of a DVB-S2/S2X normal or short code (rates as in dvb_parity_check_matrix()).
/// Throws std::invalid_argument for unknown codes and for medium frames (VL-SNR, not supported)
/// and for the VL-SNR codes whose LDPC words are punctured (S2X normal 2/9, short 11/45, 4/15).
DvbBchInfo dvb_bch_info(DvbStandard standard, DvbFrameSize frame, const std::string& rate);

/// The BCH code of a DVB-S2/S2X normal or short code (see dvb_bch_info()).
BchCode dvb_bch_code(DvbStandard standard, DvbFrameSize frame, const std::string& rate);

}  // namespace ldpc
