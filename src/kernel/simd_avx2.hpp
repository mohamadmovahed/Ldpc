#pragma once

// AVX2 traits: 8 x float, 16 x int16, 32 x int8 lanes per 256-bit vector.

#include <immintrin.h>

#include <cstdint>

#include "layered_min_sum.hpp"

namespace ldpc {
namespace detail {
namespace {

struct Avx2F32 {
    using T = float;
    using V = __m256;
    using Scale = __m256;
    static constexpr unsigned W = 8;
    static constexpr float kMax = 3.0e38f;

    static V load(const T* p) noexcept { return _mm256_load_ps(p); }
    static void store(T* p, V a) noexcept { _mm256_store_ps(p, a); }
    static V splat(T x) noexcept { return _mm256_set1_ps(x); }
    static V zero() noexcept { return _mm256_setzero_ps(); }
    static V sign_mask() noexcept { return _mm256_set1_ps(-0.0f); }
    static V add_sat(V a, V b) noexcept { return _mm256_add_ps(a, b); }
    static V sub_sat(V a, V b) noexcept { return _mm256_sub_ps(a, b); }
    static V abs(V a) noexcept { return _mm256_andnot_ps(sign_mask(), a); }
    static V min(V a, V b) noexcept { return _mm256_min_ps(a, b); }
    static V max(V a, V b) noexcept { return _mm256_max_ps(a, b); }
    static V bit_xor(V a, V b) noexcept { return _mm256_xor_ps(a, b); }
    static V bit_or(V a, V b) noexcept { return _mm256_or_ps(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept {
        return _mm256_xor_ps(magnitude, _mm256_and_ps(sign, sign_mask()));
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return _mm256_blendv_ps(otherwise, if_equal, _mm256_cmp_ps(a, b, _CMP_EQ_OQ));
    }
    static Scale make_scale(float factor) noexcept { return _mm256_set1_ps(factor); }
    static V scale(V a, Scale f) noexcept { return _mm256_mul_ps(a, f); }
    static V sub_floor0(V a, V offset) noexcept { return _mm256_max_ps(_mm256_sub_ps(a, offset), zero()); }
    static T quantize(float x, float) noexcept { return x; }
    static bool is_negative(T x) noexcept { return float_is_negative(x); }
};

struct Avx2I16 {
    using T = std::int16_t;
    using V = __m256i;
    using Scale = std::int16_t;
    static constexpr unsigned W = 16;
    static constexpr T kMax = 32767;

    static V load(const T* p) noexcept { return _mm256_load_si256(reinterpret_cast<const __m256i*>(p)); }
    static void store(T* p, V a) noexcept { _mm256_store_si256(reinterpret_cast<__m256i*>(p), a); }
    static V splat(T x) noexcept { return _mm256_set1_epi16(x); }
    static V zero() noexcept { return _mm256_setzero_si256(); }
    static V add_sat(V a, V b) noexcept { return _mm256_max_epi16(_mm256_adds_epi16(a, b), splat(-kMax)); }
    static V sub_sat(V a, V b) noexcept { return _mm256_max_epi16(_mm256_subs_epi16(a, b), splat(-kMax)); }
    static V abs(V a) noexcept { return _mm256_abs_epi16(a); }
    static V min(V a, V b) noexcept { return _mm256_min_epi16(a, b); }
    static V max(V a, V b) noexcept { return _mm256_max_epi16(a, b); }
    static V bit_xor(V a, V b) noexcept { return _mm256_xor_si256(a, b); }
    static V bit_or(V a, V b) noexcept { return _mm256_or_si256(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept {
        // OR-ing 1 makes `sign` non-zero so vpsignw never zeroes the result.
        return _mm256_sign_epi16(magnitude, _mm256_or_si256(sign, splat(1)));
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return _mm256_blendv_epi8(otherwise, if_equal, _mm256_cmpeq_epi16(a, b));
    }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    static V scale(V a, Scale f) noexcept { return _mm256_mulhrs_epi16(a, splat(f)); }
    static V sub_floor0(V a, V offset) noexcept { return _mm256_max_epi16(_mm256_subs_epi16(a, offset), zero()); }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

struct Avx2I8 {
    using T = std::int8_t;
    using V = __m256i;
    using Scale = std::int16_t;
    static constexpr unsigned W = 32;
    static constexpr T kMax = 127;

    static V load(const T* p) noexcept { return _mm256_load_si256(reinterpret_cast<const __m256i*>(p)); }
    static void store(T* p, V a) noexcept { _mm256_store_si256(reinterpret_cast<__m256i*>(p), a); }
    static V splat(T x) noexcept { return _mm256_set1_epi8(x); }
    static V zero() noexcept { return _mm256_setzero_si256(); }
    static V add_sat(V a, V b) noexcept { return _mm256_max_epi8(_mm256_adds_epi8(a, b), splat(-kMax)); }
    static V sub_sat(V a, V b) noexcept { return _mm256_max_epi8(_mm256_subs_epi8(a, b), splat(-kMax)); }
    static V abs(V a) noexcept { return _mm256_abs_epi8(a); }
    static V min(V a, V b) noexcept { return _mm256_min_epi8(a, b); }
    static V max(V a, V b) noexcept { return _mm256_max_epi8(a, b); }
    static V bit_xor(V a, V b) noexcept { return _mm256_xor_si256(a, b); }
    static V bit_or(V a, V b) noexcept { return _mm256_or_si256(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept {
        return _mm256_sign_epi8(magnitude, _mm256_or_si256(sign, splat(1)));
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return _mm256_blendv_epi8(otherwise, if_equal, _mm256_cmpeq_epi8(a, b));
    }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    static V scale(V a, Scale f) noexcept {
        // Widen to int16, multiply with rounding, then narrow back in order.
        const __m256i factor = _mm256_set1_epi16(f);
        const __m256i lo = _mm256_mulhrs_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)), factor);
        const __m256i hi = _mm256_mulhrs_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)), factor);
        return _mm256_permute4x64_epi64(_mm256_packs_epi16(lo, hi), 0xD8);
    }
    static V sub_floor0(V a, V offset) noexcept { return _mm256_max_epi8(_mm256_subs_epi8(a, offset), zero()); }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

}  // namespace
}  // namespace detail
}  // namespace ldpc
