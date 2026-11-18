#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ldpc/sparse_matrix.hpp"

namespace ldpc {

enum class DvbStandard {
    S2,   ///< ETSI EN 302 307-1 (DVB-S2)
    S2X,  ///< ETSI EN 302 307-2 (DVB-S2X); also accepts every DVB-S2 rate
    T2,   ///< ETSI EN 302 755 (DVB-T2)
};

enum class DvbFrameSize {
    Normal,  ///< n = 64800
    Medium,  ///< n = 32400 (DVB-S2X VL-SNR only)
    Short,   ///< n = 16200
};

/// Describes one built-in standard code.
struct DvbCodeInfo {
    DvbStandard standard;
    DvbFrameSize frame;
    std::string rate;  ///< Nominal rate as named in the standard, e.g. "1/2", "132/180".
    std::size_t n;     ///< Codeword length.
    std::size_t k;     ///< Number of information bits (the first k codeword bits).
};

/// Lists every built-in code.
std::vector<DvbCodeInfo> dvb_codes();

/// Builds the parity-check matrix of a DVB-S2/S2X/T2 LDPC code.
///
/// The matrix is H = [A | P] where A is the information part generated from
/// the standard's address tables and P is the (n-k)x(n-k) dual-diagonal
/// accumulator. Codewords are systematic: bits [0, k) carry the information.
///
/// `rate` is the nominal rate string from the standard ("1/2", "9/10",
/// "2/9", "90/180", ...). Throws std::invalid_argument for unknown codes.
SparseMatrix dvb_parity_check_matrix(DvbStandard standard, DvbFrameSize frame, const std::string& rate);

/// Returns the information for one code (throws std::invalid_argument if unknown).
DvbCodeInfo dvb_code_info(DvbStandard standard, DvbFrameSize frame, const std::string& rate);

const char* to_string(DvbStandard standard) noexcept;
const char* to_string(DvbFrameSize frame) noexcept;

}  // namespace ldpc
