#pragma once

#include <cstddef>
#include <cstdint>

#include "ldpc/dvb.hpp"

namespace ldpc {
namespace detail {

/// One DVB parity-bit address table (ETSI "Annex B/C" style).
///
/// Information bits are processed in groups of 360. Group g has degree
/// `degree[i]` for `count[i]` consecutive groups; its `degree` entries in
/// `positions` are the parity addresses x of the group's first bit. Bit j of
/// the group then connects to parity check (x + j * q) mod (n - k), where
/// q = (n - k) / 360.
struct DvbTable {
    DvbStandard standard;
    DvbFrameSize frame;
    const char* rate;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t num_degree_runs;
    const std::uint16_t* degree;
    const std::uint16_t* count;
    const std::uint16_t* positions;
    // Reference statistics used by the tests.
    std::uint32_t min_check_degree;
    std::uint32_t max_check_degree;
    std::uint32_t num_edges;
};

constexpr std::uint32_t kDvbGroupSize = 360;

extern const DvbTable kDvbTables[];
extern const std::size_t kDvbTableCount;

}  // namespace detail
}  // namespace ldpc
