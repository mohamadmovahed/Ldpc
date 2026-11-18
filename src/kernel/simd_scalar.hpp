#pragma once

// Portable traits: fixed-size arrays and plain loops, which compilers
// auto-vectorise for the baseline ISA (SSE2 on x86-64, NEON on AArch64).

#include <cstdint>
#include <cstring>

#include "layered_min_sum.hpp"

namespace ldpc {
namespace detail {
namespace {

template <class Derived, class TT, unsigned Lanes>
struct ScalarBase {
    using T = TT;
    static constexpr unsigned W = Lanes;
    struct V {
        T v[Lanes];
    };

    static V load(const T* p) noexcept {
        V r;
        std::memcpy(r.v, p, sizeof r.v);
        return r;
    }
    static void store(T* p, const V& a) noexcept { std::memcpy(p, a.v, sizeof a.v); }
    static V splat(T x) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = x;
        return r;
    }
    static V zero() noexcept { return splat(T(0)); }
    static V min(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = b.v[i] < a.v[i] ? b.v[i] : a.v[i];
        return r;
    }
    static V max(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = a.v[i] < b.v[i] ? b.v[i] : a.v[i];
        return r;
    }
    static V select_eq(const V& a, const V& b, const V& if_equal, const V& otherwise) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = a.v[i] == b.v[i] ? if_equal.v[i] : otherwise.v[i];
        return r;
    }
};

/// Fixed-point lanes (int8 / int16) with saturating arithmetic in [-kMax, kMax].
template <class TT, unsigned Lanes>
struct ScalarFixed : ScalarBase<ScalarFixed<TT, Lanes>, TT, Lanes> {
    using Base = ScalarBase<ScalarFixed<TT, Lanes>, TT, Lanes>;
    using T = TT;
    using V = typename Base::V;
    using Scale = std::int16_t;
    static constexpr unsigned W = Lanes;
    static constexpr T kMax = sizeof(T) == 1 ? T(127) : T(32767);

    static T saturate(int x) noexcept { return static_cast<T>(x > kMax ? kMax : (x < -kMax ? -kMax : x)); }
    static V add_sat(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = saturate(int(a.v[i]) + int(b.v[i]));
        return r;
    }
    static V sub_sat(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = saturate(int(a.v[i]) - int(b.v[i]));
        return r;
    }
    static V abs(const V& a) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = static_cast<T>(a.v[i] < 0 ? -a.v[i] : a.v[i]);
        return r;
    }
    static V bit_xor(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = static_cast<T>(a.v[i] ^ b.v[i]);
        return r;
    }
    static V bit_or(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = static_cast<T>(a.v[i] | b.v[i]);
        return r;
    }
    static V copy_sign(const V& magnitude, const V& sign) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = static_cast<T>(sign.v[i] < 0 ? -magnitude.v[i] : magnitude.v[i]);
        return r;
    }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    /// Matches x86 pmulhrsw / ARM sqrdmulh rounding: (x * f + 2^14) >> 15.
    static V scale(const V& a, Scale f) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = static_cast<T>((int(a.v[i]) * int(f) + (1 << 14)) >> 15);
        return r;
    }
    static V sub_floor0(const V& a, const V& offset) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) {
            const int d = int(a.v[i]) - int(offset.v[i]);
            r.v[i] = static_cast<T>(d > 0 ? d : 0);
        }
        return r;
    }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

struct ScalarF32 : ScalarBase<ScalarF32, float, 8> {
    using Scale = float;
    static constexpr float kMax = 3.0e38f;

    static std::uint32_t bits(float x) noexcept {
        std::uint32_t b;
        std::memcpy(&b, &x, sizeof b);
        return b;
    }
    static float from_bits(std::uint32_t b) noexcept {
        float x;
        std::memcpy(&x, &b, sizeof x);
        return x;
    }
    static V add_sat(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = a.v[i] + b.v[i];
        return r;
    }
    static V sub_sat(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = a.v[i] - b.v[i];
        return r;
    }
    static V abs(const V& a) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = from_bits(bits(a.v[i]) & 0x7fffffffu);
        return r;
    }
    static V bit_xor(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = from_bits(bits(a.v[i]) ^ bits(b.v[i]));
        return r;
    }
    static V bit_or(const V& a, const V& b) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = from_bits(bits(a.v[i]) | bits(b.v[i]));
        return r;
    }
    static V copy_sign(const V& magnitude, const V& sign) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = from_bits(bits(magnitude.v[i]) ^ (bits(sign.v[i]) & 0x80000000u));
        return r;
    }
    static Scale make_scale(float factor) noexcept { return factor; }
    static V scale(const V& a, Scale f) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) r.v[i] = a.v[i] * f;
        return r;
    }
    static V sub_floor0(const V& a, const V& offset) noexcept {
        V r;
        for (unsigned i = 0; i < W; ++i) {
            const float d = a.v[i] - offset.v[i];
            r.v[i] = d > 0.0f ? d : 0.0f;
        }
        return r;
    }
    static float quantize(float x, float) noexcept { return x; }
    static bool is_negative(float x) noexcept { return float_is_negative(x); }
};

using ScalarI16 = ScalarFixed<std::int16_t, 16>;
using ScalarI8 = ScalarFixed<std::int8_t, 32>;

}  // namespace
}  // namespace detail
}  // namespace ldpc
