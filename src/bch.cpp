#include "ldpc/bch.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace ldpc {

namespace {

constexpr int kMaxParityBits = 256;
constexpr int kWords = kMaxParityBits / 64;

/// Polynomial remainder register: bits [0, r) of a 256-bit value, bit i = coefficient of x^i.
using Reg = std::array<std::uint64_t, kWords>;

bool get_bit(const Reg& v, int i) { return ((v[static_cast<std::size_t>(i / 64)] >> (i % 64)) & 1U) != 0; }
void flip_bit(Reg& v, int i) { v[static_cast<std::size_t>(i / 64)] ^= std::uint64_t{1} << (i % 64); }

/// v <<= s (s < 64), keeping bits [0, r).
void shift_left(Reg& v, int s, int r) {
    if (s > 0) {
        for (int w = kWords - 1; w > 0; --w) {
            v[static_cast<std::size_t>(w)] =
                (v[static_cast<std::size_t>(w)] << s) | (v[static_cast<std::size_t>(w - 1)] >> (64 - s));
        }
        v[0] <<= s;
    }
    // Clear bits >= r.
    for (int w = 0; w < kWords; ++w) {
        const int lo = w * 64;
        if (lo >= r) {
            v[static_cast<std::size_t>(w)] = 0;
        } else if (lo + 64 > r) {
            v[static_cast<std::size_t>(w)] &= (std::uint64_t{1} << (r - lo)) - 1;
        }
    }
}

void xor_into(Reg& a, const Reg& b) {
    for (std::size_t w = 0; w < a.size(); ++w) a[w] ^= b[w];
}

/// Top 8 bits [r-8, r) of v as a byte (bit 7 = x^(r-1)).
unsigned top_byte(const Reg& v, int r) {
    const int lo = r - 8;  // bits [lo, lo + 8)
    const int w = lo / 64;
    const int off = lo % 64;
    std::uint64_t bits = v[static_cast<std::size_t>(w)] >> off;
    if (off > 56 && w + 1 < kWords) bits |= v[static_cast<std::size_t>(w + 1)] << (64 - off);
    return static_cast<unsigned>(bits & 0xFFU);
}

}  // namespace

struct BchCode::Impl {
    int m = 0;
    int t = 0;
    std::size_t n = 0;
    int r = 0;                   // parity bits = degree of g
    int field_size = 0;          // 2^m - 1
    std::vector<std::uint16_t> exp;  // exp[i] = alpha^i, i in [0, 2 * field_size)
    std::vector<int> log;            // log[x], x != 0
    std::vector<std::uint8_t> g;     // generator, index = degree
    Reg g_low{};                     // g without x^r
    std::array<Reg, 256> table{};    // (byte * x^r) mod g, the byte holding x^(r+7)..x^r

    std::uint16_t mul(std::uint16_t a, std::uint16_t b) const {
        if (a == 0 || b == 0) return 0;
        return exp[static_cast<std::size_t>(log[a] + log[b])];
    }
    std::uint16_t pow_alpha(long long e) const {
        long long k = e % field_size;
        if (k < 0) k += field_size;
        return exp[static_cast<std::size_t>(k)];
    }

    /// Feeds bits (highest degree first) into the remainder register.
    void feed(Reg& rem, const std::uint8_t* bits, std::size_t count) const {
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            unsigned byte = 0;
            for (int b = 0; b < 8; ++b) byte = (byte << 1) | (bits[i + static_cast<std::size_t>(b)] & 1U);
            const unsigned idx = top_byte(rem, r) ^ byte;
            shift_left(rem, 8, r);
            xor_into(rem, table[idx]);
        }
        for (; i < count; ++i) {
            const bool fb = ((bits[i] & 1U) != 0) != get_bit(rem, r - 1);
            shift_left(rem, 1, r);
            if (fb) xor_into(rem, g_low);
        }
    }
};

BchCode::BchCode(int m, std::uint32_t primitive_poly, int t, std::size_t n) : impl_(std::make_unique<Impl>()) {
    Impl& s = *impl_;
    if (m < 2 || m > 16 || t < 1 || ((primitive_poly >> m) & 1U) == 0 || (primitive_poly >> (m + 1)) != 0) {
        throw std::invalid_argument("bch: invalid field degree, t or primitive polynomial");
    }
    s.m = m;
    s.t = t;
    s.n = n;
    s.field_size = (1 << m) - 1;
    if (n == 0 || n > static_cast<std::size_t>(s.field_size)) {
        throw std::invalid_argument("bch: n must be in [1, 2^m - 1]");
    }
    // Field tables.
    s.exp.assign(static_cast<std::size_t>(2 * s.field_size), 0);
    s.log.assign(static_cast<std::size_t>(s.field_size + 1), -1);
    std::uint32_t x = 1;
    for (int i = 0; i < s.field_size; ++i) {
        if (s.log[x] != -1) throw std::invalid_argument("bch: the polynomial is not primitive");
        s.exp[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(x);
        s.log[x] = i;
        x <<= 1;
        if ((x >> m) & 1U) x ^= primitive_poly;
    }
    for (int i = s.field_size; i < 2 * s.field_size; ++i) {
        s.exp[static_cast<std::size_t>(i)] = s.exp[static_cast<std::size_t>(i - s.field_size)];
    }
    // Generator: product of the distinct minimal polynomials of alpha^1..alpha^2t.
    std::vector<bool> used(static_cast<std::size_t>(s.field_size), false);
    std::vector<std::uint8_t> g{1};
    for (int i = 1; i <= 2 * t; ++i) {
        if (used[static_cast<std::size_t>(i % s.field_size)]) continue;
        // Conjugacy class of i: i, 2i, 4i, ... mod (2^m - 1).
        std::vector<int> cls;
        int c = i % s.field_size;
        while (!used[static_cast<std::size_t>(c)]) {
            used[static_cast<std::size_t>(c)] = true;
            cls.push_back(c);
            c = (2 * c) % s.field_size;
        }
        // Minimal polynomial = prod (x - alpha^c) over the class; coefficients end in GF(2).
        std::vector<std::uint16_t> mp{1};
        for (const int e : cls) {
            const std::uint16_t root = s.exp[static_cast<std::size_t>(e)];
            std::vector<std::uint16_t> next(mp.size() + 1, 0);
            for (std::size_t d = 0; d < mp.size(); ++d) {
                next[d + 1] ^= mp[d];             // x * mp
                next[d] ^= s.mul(mp[d], root);    // root * mp (char 2: minus = plus)
            }
            mp = std::move(next);
        }
        std::vector<std::uint8_t> product(g.size() + mp.size() - 1, 0);
        for (std::size_t a = 0; a < g.size(); ++a) {
            if (g[a] == 0) continue;
            for (std::size_t b = 0; b < mp.size(); ++b) {
                if (mp[b] > 1) throw std::invalid_argument("bch: minimal polynomial not binary");
                product[a + b] ^= static_cast<std::uint8_t>(mp[b]);
            }
        }
        g = std::move(product);
    }
    s.g = g;
    s.r = static_cast<int>(g.size()) - 1;
    if (s.r > kMaxParityBits || s.r < 8 || static_cast<std::size_t>(s.r) >= n) {
        throw std::invalid_argument("bch: parity bits must be in [8, 256] and below n");
    }
    for (int i = 0; i < s.r; ++i) {
        if (g[static_cast<std::size_t>(i)] != 0) flip_bit(s.g_low, i);
    }
    // Byte table: T[b] = (b(x) * x^r) mod g, computed bit-serially.
    for (unsigned b = 0; b < 256; ++b) {
        Reg rem{};
        for (int bit = 7; bit >= 0; --bit) {
            const bool fb = (((b >> bit) & 1U) != 0) != get_bit(rem, s.r - 1);
            shift_left(rem, 1, s.r);
            if (fb) xor_into(rem, s.g_low);
        }
        s.table[b] = rem;
    }
}

BchCode::~BchCode() = default;
BchCode::BchCode(BchCode&&) noexcept = default;
BchCode& BchCode::operator=(BchCode&&) noexcept = default;

std::size_t BchCode::n() const noexcept { return impl_->n; }
std::size_t BchCode::k() const noexcept { return impl_->n - static_cast<std::size_t>(impl_->r); }
std::size_t BchCode::parity_bits() const noexcept { return static_cast<std::size_t>(impl_->r); }
int BchCode::t() const noexcept { return impl_->t; }
int BchCode::m() const noexcept { return impl_->m; }
const std::vector<std::uint8_t>& BchCode::generator() const noexcept { return impl_->g; }

void BchCode::encode(const std::uint8_t* message, std::uint8_t* parity) const {
    const Impl& s = *impl_;
    Reg rem{};
    s.feed(rem, message, k());
    // remainder of m(x) * x^r: feeding m and then r zeros equals feeding m into the
    // "premultiplied" register used here (feed() multiplies by x^r implicitly).
    for (int i = 0; i < s.r; ++i) parity[i] = get_bit(rem, s.r - 1 - i) ? 1 : 0;
}

std::vector<std::uint8_t> BchCode::encode(const std::vector<std::uint8_t>& message) const {
    if (message.size() != k()) throw std::invalid_argument("bch: message must have k() bits");
    std::vector<std::uint8_t> out(n());
    std::copy(message.begin(), message.end(), out.begin());
    encode(message.data(), out.data() + k());
    return out;
}

BchDecodeResult BchCode::decode(std::uint8_t* codeword) const {
    const Impl& s = *impl_;
    // c(x) mod g: feeding all n bits of a codeword into the premultiplied register gives
    // (c(x) * x^r) mod g, which is zero exactly when c(x) mod g is zero.
    Reg rem{};
    s.feed(rem, codeword, s.n);
    bool zero = true;
    for (const auto w : rem) zero = zero && w == 0;
    if (zero) return {true, 0};

    // The register holds R(x) = (c(x) x^r) mod g, so for a root beta of g:
    // R(beta) = c(beta) * beta^r. Syndrome S_j = c(alpha^j) = R(alpha^j) * alpha^(-j r).
    const int two_t = 2 * s.t;
    std::vector<std::uint16_t> syn(static_cast<std::size_t>(two_t + 1), 0);
    for (int j = 1; j <= two_t; ++j) {
        if (j % 2 == 0) {
            syn[static_cast<std::size_t>(j)] = s.mul(syn[static_cast<std::size_t>(j / 2)], syn[static_cast<std::size_t>(j / 2)]);
            continue;
        }
        std::uint16_t acc = 0;
        for (int i = 0; i < s.r; ++i) {
            if (get_bit(rem, i)) acc ^= s.pow_alpha(static_cast<long long>(j) * i);
        }
        syn[static_cast<std::size_t>(j)] = s.mul(acc, s.pow_alpha(-static_cast<long long>(j) * s.r));
    }
    bool all_zero = true;
    for (int j = 1; j <= two_t; ++j) all_zero = all_zero && syn[static_cast<std::size_t>(j)] == 0;
    if (all_zero) return {false, 0};  // remainder non-zero but no syndrome: cannot happen for valid g

    // Berlekamp-Massey.
    std::vector<std::uint16_t> lambda(static_cast<std::size_t>(two_t + 2), 0), prev = lambda, tmp;
    lambda[0] = 1;
    prev[0] = 1;
    int len = 0;
    int shift = 1;
    std::uint16_t prev_disc = 1;
    for (int k = 0; k < two_t; ++k) {
        std::uint16_t d = syn[static_cast<std::size_t>(k + 1)];
        for (int i = 1; i <= len; ++i) {
            d ^= s.mul(lambda[static_cast<std::size_t>(i)], syn[static_cast<std::size_t>(k + 1 - i)]);
        }
        if (d == 0) {
            ++shift;
            continue;
        }
        const std::uint16_t coef = s.mul(d, s.exp[static_cast<std::size_t>((s.field_size - s.log[prev_disc]) % s.field_size)]);
        tmp = lambda;
        for (std::size_t i = 0; i + static_cast<std::size_t>(shift) < lambda.size(); ++i) {
            lambda[i + static_cast<std::size_t>(shift)] ^= s.mul(coef, prev[i]);
        }
        if (2 * len <= k) {
            len = k + 1 - len;
            prev = tmp;
            prev_disc = d;
            shift = 1;
        } else {
            ++shift;
        }
    }
    if (len > s.t) return {false, 0};

    // Chien search over the n positions: degree e is in error iff lambda(alpha^-e) == 0.
    std::vector<std::uint16_t> term(static_cast<std::size_t>(len + 1));
    std::vector<std::uint16_t> step(static_cast<std::size_t>(len + 1));
    for (int i = 0; i <= len; ++i) {
        term[static_cast<std::size_t>(i)] = lambda[static_cast<std::size_t>(i)];
        step[static_cast<std::size_t>(i)] = s.pow_alpha(-static_cast<long long>(i));
    }
    std::vector<std::size_t> errors;
    for (std::size_t e = 0; e < s.n; ++e) {
        std::uint16_t sum = 0;
        for (int i = 0; i <= len; ++i) sum ^= term[static_cast<std::size_t>(i)];
        if (sum == 0) {
            errors.push_back(e);
            if (errors.size() > static_cast<std::size_t>(len)) break;
        }
        for (int i = 1; i <= len; ++i) {
            term[static_cast<std::size_t>(i)] = s.mul(term[static_cast<std::size_t>(i)], step[static_cast<std::size_t>(i)]);
        }
    }
    if (errors.size() != static_cast<std::size_t>(len)) return {false, 0};
    for (const std::size_t e : errors) codeword[s.n - 1 - e] ^= 1U;
    return {true, len};
}

namespace {

struct BchRow {
    DvbStandard standard;  // S2 rows serve S2X too
    DvbFrameSize frame;
    const char* rate;
    std::size_t kbch;
    std::size_t nbch;
    int t;
};

// EN 302 307-1 tables 5a/5b and EN 302 307-2 tables 4/6 (without the punctured VL-SNR codes).
constexpr BchRow kBchRows[] = {
    {DvbStandard::S2, DvbFrameSize::Normal, "1/4", 16008, 16200, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "1/3", 21408, 21600, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "2/5", 25728, 25920, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "1/2", 32208, 32400, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "3/5", 38688, 38880, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "2/3", 43040, 43200, 10},
    {DvbStandard::S2, DvbFrameSize::Normal, "3/4", 48408, 48600, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "4/5", 51648, 51840, 12},
    {DvbStandard::S2, DvbFrameSize::Normal, "5/6", 53840, 54000, 10},
    {DvbStandard::S2, DvbFrameSize::Normal, "8/9", 57472, 57600, 8},
    {DvbStandard::S2, DvbFrameSize::Normal, "9/10", 58192, 58320, 8},
    {DvbStandard::S2, DvbFrameSize::Short, "1/4", 3072, 3240, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "1/3", 5232, 5400, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "2/5", 6312, 6480, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "1/2", 7032, 7200, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "3/5", 9552, 9720, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "2/3", 10632, 10800, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "3/4", 11712, 11880, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "4/5", 12432, 12600, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "5/6", 13152, 13320, 12},
    {DvbStandard::S2, DvbFrameSize::Short, "8/9", 14232, 14400, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "13/45", 18528, 18720, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "9/20", 28968, 29160, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "90/180", 32208, 32400, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "96/180", 34368, 34560, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "11/20", 35448, 35640, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "100/180", 35808, 36000, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "104/180", 37248, 37440, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "26/45", 37248, 37440, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "18/30", 38688, 38880, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "28/45", 40128, 40320, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "23/36", 41208, 41400, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "116/180", 41568, 41760, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "20/30", 43008, 43200, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "124/180", 44448, 44640, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "25/36", 44808, 45000, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "128/180", 45888, 46080, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "13/18", 46608, 46800, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "132/180", 47328, 47520, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "22/30", 47328, 47520, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "135/180", 48408, 48600, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "140/180", 50208, 50400, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "7/9", 50208, 50400, 12},
    {DvbStandard::S2X, DvbFrameSize::Normal, "154/180", 55248, 55440, 12},
    {DvbStandard::S2X, DvbFrameSize::Short, "14/45", 4872, 5040, 12},
    {DvbStandard::S2X, DvbFrameSize::Short, "7/15", 7392, 7560, 12},
    {DvbStandard::S2X, DvbFrameSize::Short, "8/15", 8472, 8640, 12},
    {DvbStandard::S2X, DvbFrameSize::Short, "26/45", 9192, 9360, 12},
    {DvbStandard::S2X, DvbFrameSize::Short, "32/45", 11352, 11520, 12},
};

// g1 of tables 6a / 6b: the primitive polynomials of GF(2^16) and GF(2^14).
constexpr std::uint32_t kPoly16 = (1U << 16) | (1U << 5) | (1U << 3) | (1U << 2) | 1U;  // 1+x2+x3+x5+x16
constexpr std::uint32_t kPoly14 = (1U << 14) | (1U << 5) | (1U << 3) | (1U << 1) | 1U;  // 1+x+x3+x5+x14

}  // namespace

DvbBchInfo dvb_bch_info(DvbStandard standard, DvbFrameSize frame, const std::string& rate) {
    if (standard == DvbStandard::T2) {
        throw std::invalid_argument("bch: DVB-T2 BCH codes are not provided");
    }
    for (const BchRow& row : kBchRows) {
        const bool std_ok = row.standard == standard || (standard == DvbStandard::S2X && row.standard == DvbStandard::S2);
        if (std_ok && row.frame == frame && rate == row.rate) {
            return {row.kbch, row.nbch, row.t, frame == DvbFrameSize::Normal ? 16 : 14};
        }
    }
    throw std::invalid_argument(std::string("bch: no ") + to_string(standard) + " " + to_string(frame) +
                                "-frame BCH code for rate '" + rate + "' (medium frames and VL-SNR codes are not supported)");
}

BchCode dvb_bch_code(DvbStandard standard, DvbFrameSize frame, const std::string& rate) {
    const DvbBchInfo info = dvb_bch_info(standard, frame, rate);
    return BchCode(info.m, info.m == 16 ? kPoly16 : kPoly14, info.t, info.nbch);
}

}  // namespace ldpc
