#include "ldpc/dvb.hpp"

#include <stdexcept>
#include <string>

#include "dvb_tables.hpp"

namespace ldpc {

namespace {

using detail::DvbTable;
using detail::kDvbGroupSize;

const DvbTable* find_table(DvbStandard standard, DvbFrameSize frame, const std::string& rate) {
    auto lookup = [&](DvbStandard s) -> const DvbTable* {
        for (std::size_t i = 0; i < detail::kDvbTableCount; ++i) {
            const DvbTable& t = detail::kDvbTables[i];
            if (t.standard == s && t.frame == frame && rate == t.rate) {
                return &t;
            }
        }
        return nullptr;
    };
    const DvbTable* table = lookup(standard);
    if (!table && standard == DvbStandard::S2X) {
        table = lookup(DvbStandard::S2);  // DVB-S2X includes all DVB-S2 codes.
    }
    return table;
}

const DvbTable& require_table(DvbStandard standard, DvbFrameSize frame, const std::string& rate) {
    const DvbTable* table = find_table(standard, frame, rate);
    if (!table) {
        throw std::invalid_argument(std::string("no ") + to_string(standard) + " " + to_string(frame) +
                                    "-frame LDPC code with rate '" + rate + "'");
    }
    return *table;
}

DvbCodeInfo make_info(const DvbTable& t) {
    return {t.standard, t.frame, t.rate, t.n, t.k};
}

}  // namespace

std::vector<DvbCodeInfo> dvb_codes() {
    std::vector<DvbCodeInfo> codes;
    codes.reserve(detail::kDvbTableCount);
    for (std::size_t i = 0; i < detail::kDvbTableCount; ++i) {
        codes.push_back(make_info(detail::kDvbTables[i]));
    }
    return codes;
}

DvbCodeInfo dvb_code_info(DvbStandard standard, DvbFrameSize frame, const std::string& rate) {
    return make_info(require_table(standard, frame, rate));
}

SparseMatrix dvb_parity_check_matrix(DvbStandard standard, DvbFrameSize frame, const std::string& rate) {
    const DvbTable& t = require_table(standard, frame, rate);
    const std::uint32_t num_checks = t.n - t.k;
    const std::uint32_t q = num_checks / kDvbGroupSize;

    std::vector<std::vector<SparseMatrix::Index>> rows(num_checks);

    // Information part, generated group by group from the address table.
    const std::uint16_t* addresses = t.positions;
    std::uint32_t bit = 0;
    for (std::uint32_t run = 0; run < t.num_degree_runs; ++run) {
        const std::uint32_t degree = t.degree[run];
        for (std::uint32_t group = 0; group < t.count[run]; ++group) {
            for (std::uint32_t j = 0; j < kDvbGroupSize; ++j, ++bit) {
                for (std::uint32_t d = 0; d < degree; ++d) {
                    rows[(addresses[d] + j * q) % num_checks].push_back(bit);
                }
            }
            addresses += degree;
        }
    }

    // Parity part: dual-diagonal accumulator p_i = p_{i-1} xor (check i).
    for (std::uint32_t i = 0; i < num_checks; ++i) {
        if (i > 0) {
            rows[i].push_back(t.k + i - 1);
        }
        rows[i].push_back(t.k + i);
    }
    return SparseMatrix::from_rows(t.n, rows);
}

const char* to_string(DvbStandard standard) noexcept {
    switch (standard) {
        case DvbStandard::S2: return "DVB-S2";
        case DvbStandard::S2X: return "DVB-S2X";
        case DvbStandard::T2: return "DVB-T2";
    }
    return "unknown";
}

const char* to_string(DvbFrameSize frame) noexcept {
    switch (frame) {
        case DvbFrameSize::Normal: return "normal";
        case DvbFrameSize::Medium: return "medium";
        case DvbFrameSize::Short: return "short";
    }
    return "unknown";
}

}  // namespace ldpc
