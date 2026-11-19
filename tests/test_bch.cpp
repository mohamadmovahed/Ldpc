#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "ldpc/bch.hpp"
#include "ldpc/interleaver.hpp"
#include "test_framework.hpp"

using ldpc::DvbFrameSize;
using ldpc::DvbStandard;

namespace {

using Poly = std::vector<std::uint8_t>;  // index = degree

Poly poly(std::initializer_list<int> degrees) {
    int top = 0;
    for (int d : degrees) top = std::max(top, d);
    Poly p(static_cast<std::size_t>(top + 1), 0);
    for (int d : degrees) p[static_cast<std::size_t>(d)] ^= 1;
    return p;
}

Poly multiply(const Poly& a, const Poly& b) {
    Poly out(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (std::size_t j = 0; j < b.size(); ++j) out[i + j] ^= b[j];
    return out;
}

std::vector<std::uint8_t> random_bits(std::size_t n, std::mt19937& rng) {
    std::vector<std::uint8_t> v(n);
    for (auto& b : v) b = static_cast<std::uint8_t>(rng() & 1U);
    return v;
}

}  // namespace

TEST_CASE("bch: DVB generators equal the products of the standard's polynomials") {
    // EN 302 307-1 table 6a (normal) g1..g12 and table 6b (short) g1..g12.
    const std::vector<Poly> normal = {
        poly({0, 2, 3, 5, 16}), poly({0, 1, 4, 5, 6, 8, 16}), poly({0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 16}),
        poly({0, 2, 4, 6, 9, 11, 12, 14, 16}), poly({0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 16}),
        poly({0, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16}), poly({0, 2, 5, 6, 8, 9, 10, 11, 13, 15, 16}),
        poly({0, 1, 2, 5, 6, 8, 9, 12, 13, 14, 16}), poly({0, 5, 7, 9, 10, 11, 16}),
        poly({0, 1, 2, 5, 7, 8, 10, 12, 13, 14, 16}), poly({0, 2, 3, 5, 9, 11, 12, 13, 16}),
        poly({0, 1, 5, 6, 7, 9, 11, 12, 16})};
    const std::vector<Poly> shorts = {
        poly({0, 1, 3, 5, 14}), poly({0, 6, 8, 11, 14}), poly({0, 1, 2, 6, 9, 10, 14}),
        poly({0, 4, 7, 8, 10, 12, 14}), poly({0, 2, 4, 6, 8, 9, 11, 13, 14}), poly({0, 3, 7, 8, 9, 13, 14}),
        poly({0, 2, 5, 6, 7, 10, 11, 13, 14}), poly({0, 5, 8, 9, 10, 11, 14}), poly({0, 1, 2, 3, 9, 10, 14}),
        poly({0, 3, 6, 9, 11, 12, 14}), poly({0, 4, 11, 12, 14}), poly({0, 1, 2, 3, 5, 6, 7, 8, 10, 13, 14})};
    for (const int t : {8, 10, 12}) {
        Poly expected{1};
        for (int i = 0; i < t; ++i) expected = multiply(expected, normal[static_cast<std::size_t>(i)]);
        const auto code = ldpc::dvb_bch_code(DvbStandard::S2, DvbFrameSize::Normal,
                                             t == 8 ? "8/9" : (t == 10 ? "2/3" : "1/2"));
        CHECK_EQ(code.t(), t);
        CHECK(code.generator() == expected);
    }
    Poly expected{1};
    for (const auto& p : shorts) expected = multiply(expected, p);
    const auto code = ldpc::dvb_bch_code(DvbStandard::S2, DvbFrameSize::Short, "1/2");
    CHECK(code.generator() == expected);
    CHECK_EQ(code.n(), 7200u);
    CHECK_EQ(code.k(), 7032u);
}

TEST_CASE("bch: encode, then correct up to t errors and refuse more") {
    std::mt19937 rng(7);
    for (const auto& [frame, rate] : std::vector<std::pair<DvbFrameSize, const char*>>{
             {DvbFrameSize::Normal, "9/10"}, {DvbFrameSize::Normal, "2/3"}, {DvbFrameSize::Normal, "3/5"},
             {DvbFrameSize::Short, "8/9"}, {DvbFrameSize::Short, "1/4"}}) {
        const auto code = ldpc::dvb_bch_code(DvbStandard::S2, frame, rate);
        const auto msg = random_bits(code.k(), rng);
        const auto cw = code.encode(msg);
        auto clean = cw;
        auto r = code.decode(clean.data());
        CHECK(r.ok);
        CHECK_EQ(r.corrected, 0);
        for (int errors = 1; errors <= code.t(); ++errors) {
            auto noisy = cw;
            std::uniform_int_distribution<std::size_t> pos(0, code.n() - 1);
            std::vector<std::size_t> flipped;
            while (static_cast<int>(flipped.size()) < errors) {
                const std::size_t p = pos(rng);
                if (std::find(flipped.begin(), flipped.end(), p) == flipped.end()) {
                    flipped.push_back(p);
                    noisy[p] ^= 1U;
                }
            }
            r = code.decode(noisy.data());
            CHECK(r.ok);
            CHECK_EQ(r.corrected, errors);
            CHECK(noisy == cw);
        }
        // t+2 errors: usually detected as uncorrectable (and then left unchanged).
        int refused = 0;
        for (int trial = 0; trial < 5; ++trial) {
            auto noisy = cw;
            for (int i = 0; i < code.t() + 2; ++i) noisy[(static_cast<std::size_t>(i) * 977 + trial * 13) % code.n()] ^= 1U;
            const auto before = noisy;
            r = code.decode(noisy.data());
            if (!r.ok) {
                ++refused;
                CHECK(noisy == before);
            }
        }
        CHECK(refused >= 3);
    }
}

TEST_CASE("bch: S2X codes and unsupported codes") {
    const auto info = ldpc::dvb_bch_info(DvbStandard::S2X, DvbFrameSize::Normal, "132/180");
    CHECK_EQ(info.kbch, 47328u);
    CHECK_EQ(info.nbch, 47520u);
    CHECK_EQ(ldpc::dvb_bch_info(DvbStandard::S2X, DvbFrameSize::Normal, "1/2").kbch, 32208u);  // S2 rate
    CHECK_EQ(ldpc::dvb_bch_info(DvbStandard::S2X, DvbFrameSize::Short, "7/15").nbch, 7560u);
    bool threw = false;
    try {
        static_cast<void>(ldpc::dvb_bch_info(DvbStandard::S2X, DvbFrameSize::Medium, "1/3"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        static_cast<void>(ldpc::dvb_bch_info(DvbStandard::S2, DvbFrameSize::Short, "9/10"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("interleaver: column write, row read in pattern order, and back") {
    // 8PSK, n = 12, rows = 4: rows read columns in order 1, 0, 2 (S2X pattern "102").
    ldpc::BlockInterleaver il(12, {1, 0, 2});
    std::vector<std::uint8_t> cw(12), out(12), back(12);
    for (std::size_t i = 0; i < 12; ++i) cw[i] = static_cast<std::uint8_t>(i);
    il.interleave(cw.data(), out.data());
    // column c holds codeword bits [4c, 4c+4); row 0 reads columns 1, 0, 2 -> 4, 0, 8.
    CHECK_EQ(int(out[0]), 4);
    CHECK_EQ(int(out[1]), 0);
    CHECK_EQ(int(out[2]), 8);
    CHECK_EQ(int(out[3]), 5);
    il.deinterleave(out.data(), back.data());
    CHECK(back == cw);
    std::vector<float> soft(12), soft_back(12);
    for (std::size_t i = 0; i < 12; ++i) soft[i] = static_cast<float>(out[i]);
    il.deinterleave(soft.data(), soft_back.data());
    for (std::size_t i = 0; i < 12; ++i) CHECK_EQ(soft_back[i], static_cast<float>(i));
    bool threw = false;
    try {
        ldpc::BlockInterleaver bad(10, {0, 1, 2});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("bch: a clean normal frame decodes quickly") {
    const auto code = ldpc::dvb_bch_code(DvbStandard::S2, DvbFrameSize::Normal, "9/10");
    std::mt19937 rng(3);
    auto cw = code.encode(random_bits(code.k(), rng));
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) CHECK(code.decode(cw.data()).ok);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::printf("  bch: 20 clean 58320-bit frames in %lld ms\n", static_cast<long long>(ms));
    CHECK(ms < 2000);
}
