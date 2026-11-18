#include <set>
#include <stdexcept>
#include <string>

#include "dvb_tables.hpp"
#include "ldpc/dvb.hpp"
#include "ldpc/encoder.hpp"
#include "test_framework.hpp"

using ldpc::DvbFrameSize;
using ldpc::DvbStandard;

TEST_CASE("dvb: catalogue covers DVB-S2, S2X and T2") {
    const auto codes = ldpc::dvb_codes();
    CHECK_EQ(codes.size(), 70u);
    std::set<std::string> keys;
    for (const auto& c : codes) {
        keys.insert(std::string(ldpc::to_string(c.standard)) + ldpc::to_string(c.frame) + c.rate);
        const std::size_t expected_n =
            c.frame == DvbFrameSize::Normal ? 64800 : (c.frame == DvbFrameSize::Medium ? 32400 : 16200);
        CHECK_EQ(c.n, expected_n);
        CHECK_EQ(c.k % 360, 0u);
    }
    CHECK_EQ(keys.size(), codes.size());  // no duplicates
}

TEST_CASE("dvb: every matrix matches its reference statistics") {
    for (std::size_t i = 0; i < ldpc::detail::kDvbTableCount; ++i) {
        const auto& t = ldpc::detail::kDvbTables[i];
        const auto h = ldpc::dvb_parity_check_matrix(t.standard, t.frame, t.rate);
        CHECK_EQ(h.cols(), t.n);
        CHECK_EQ(h.rows(), std::size_t{t.n - t.k});
        CHECK_EQ(h.nnz(), std::size_t{t.num_edges});
        std::size_t min_deg = h.row_degree(0), max_deg = 0;
        for (std::size_t r = 0; r < h.rows(); ++r) {
            min_deg = std::min(min_deg, h.row_degree(r));
            max_deg = std::max(max_deg, h.row_degree(r));
        }
        CHECK_EQ(min_deg, std::size_t{t.min_check_degree});
        CHECK_EQ(max_deg, std::size_t{t.max_check_degree});
    }
}

TEST_CASE("dvb: known code parameters") {
    const auto s2 = ldpc::dvb_code_info(DvbStandard::S2, DvbFrameSize::Normal, "1/2");
    CHECK_EQ(s2.k, 32400u);
    const auto s2_short = ldpc::dvb_code_info(DvbStandard::S2, DvbFrameSize::Short, "1/4");
    CHECK_EQ(s2_short.k, 3240u);  // short-frame nominal rates differ from k/n
    const auto s2x = ldpc::dvb_code_info(DvbStandard::S2X, DvbFrameSize::Normal, "132/180");
    CHECK_EQ(s2x.k, 47520u);
    const auto medium = ldpc::dvb_code_info(DvbStandard::S2X, DvbFrameSize::Medium, "1/3");
    CHECK_EQ(medium.n, 32400u);
    // DVB-S2X includes every DVB-S2 code.
    const auto inherited = ldpc::dvb_code_info(DvbStandard::S2X, DvbFrameSize::Normal, "9/10");
    CHECK(inherited.standard == DvbStandard::S2);
    CHECK_EQ(inherited.k, 58320u);
    CHECK_THROWS_AS(ldpc::dvb_parity_check_matrix(DvbStandard::S2, DvbFrameSize::Normal, "1/7"),
                    std::invalid_argument);
    CHECK_THROWS_AS(ldpc::dvb_parity_check_matrix(DvbStandard::S2, DvbFrameSize::Medium, "1/2"),
                    std::invalid_argument);
}

TEST_CASE("dvb: accumulator encoder produces valid codewords") {
    const auto h = ldpc::dvb_parity_check_matrix(DvbStandard::S2, DvbFrameSize::Short, "3/5");
    const ldpc::Encoder encoder(h);
    CHECK_EQ(encoder.k(), 9720u);
    std::vector<std::uint8_t> info(encoder.k());
    for (std::size_t i = 0; i < info.size(); ++i) info[i] = static_cast<std::uint8_t>((i * 7919u >> 3) & 1u);
    const auto cw = encoder.encode(info);
    CHECK(h.is_codeword(cw));
    for (std::size_t i = 0; i < info.size(); ++i) CHECK_EQ(int(cw[i]), int(info[i]));
}
