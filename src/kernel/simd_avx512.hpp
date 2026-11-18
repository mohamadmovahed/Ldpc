#pragma once

// AVX-512 (F + BW) traits: 16 x float, 32 x int16, 64 x int8 lanes per vector.

#include <immintrin.h>

#include <cstdint>

#include "layered_min_sum.hpp"

namespace ldpc {
namespace detail {
namespace {

struct Avx512F32 {
    using T = float;
    using V = __m512;
    using Scale = __m512;
    static constexpr unsigned W = 16;
    static constexpr float kMax = 3.0e38f;

    static __m512i as_int(V a) noexcept { return _mm512_castps_si512(a); }
    static V as_float(__m512i a) noexcept { return _mm512_castsi512_ps(a); }
    static __m512i sign_bits() noexcept { return _mm512_set1_epi32(static_cast<int>(0x80000000u)); }

    static V load(const T* p) noexcept { return _mm512_load_ps(p); }
    static void store(T* p, V a) noexcept { _mm512_store_ps(p, a); }
    static V splat(T x) noexcept { return _mm512_set1_ps(x); }
    static V zero() noexcept { return _mm512_setzero_ps(); }
    static V add_sat(V a, V b) noexcept { return _mm512_add_ps(a, b); }
    static V sub_sat(V a, V b) noexcept { return _mm512_sub_ps(a, b); }
    static V abs(V a) noexcept { return as_float(_mm512_andnot_si512(sign_bits(), as_int(a))); }
    static V min(V a, V b) noexcept { return _mm512_min_ps(a, b); }
    static V max(V a, V b) noexcept { return _mm512_max_ps(a, b); }
    static V bit_xor(V a, V b) noexcept { return as_float(_mm512_xor_si512(as_int(a), as_int(b))); }
    static V bit_or(V a, V b) noexcept { return as_float(_mm512_or_si512(as_int(a), as_int(b))); }
    static V copy_sign(V magnitude, V sign) noexcept {
        return as_float(_mm512_xor_si512(as_int(magnitude), _mm512_and_si512(as_int(sign), sign_bits())));
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ), otherwise, if_equal);
    }
    static Scale make_scale(float factor) noexcept { return _mm512_set1_ps(factor); }
    static V scale(V a, Scale f) noexcept { return _mm512_mul_ps(a, f); }
    static V sub_floor0(V a, V offset) noexcept { return _mm512_max_ps(_mm512_sub_ps(a, offset), zero()); }
    static T quantize(float x, float) noexcept { return x; }
    static bool is_negative(T x) noexcept { return float_is_negative(x); }
};

struct Avx512I16 {
    using T = std::int16_t;
    using V = __m512i;
    using Scale = std::int16_t;
    static constexpr unsigned W = 32;
    static constexpr T kMax = 32767;

    static V load(const T* p) noexcept { return _mm512_load_si512(p); }
    static void store(T* p, V a) noexcept { _mm512_store_si512(p, a); }
    static V splat(T x) noexcept { return _mm512_set1_epi16(x); }
    static V zero() noexcept { return _mm512_setzero_si512(); }
    static V add_sat(V a, V b) noexcept { return _mm512_max_epi16(_mm512_adds_epi16(a, b), splat(-kMax)); }
    static V sub_sat(V a, V b) noexcept { return _mm512_max_epi16(_mm512_subs_epi16(a, b), splat(-kMax)); }
    static V abs(V a) noexcept { return _mm512_abs_epi16(a); }
    static V min(V a, V b) noexcept { return _mm512_min_epi16(a, b); }
    static V max(V a, V b) noexcept { return _mm512_max_epi16(a, b); }
    static V bit_xor(V a, V b) noexcept { return _mm512_xor_si512(a, b); }
    static V bit_or(V a, V b) noexcept { return _mm512_or_si512(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept {
        return _mm512_mask_sub_epi16(magnitude, _mm512_movepi16_mask(sign), zero(), magnitude);
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return _mm512_mask_blend_epi16(_mm512_cmpeq_epi16_mask(a, b), otherwise, if_equal);
    }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    static V scale(V a, Scale f) noexcept { return _mm512_mulhrs_epi16(a, splat(f)); }
    static V sub_floor0(V a, V offset) noexcept { return _mm512_max_epi16(_mm512_subs_epi16(a, offset), zero()); }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

struct Avx512I8 {
    using T = std::int8_t;
    using V = __m512i;
    using Scale = std::int16_t;
    static constexpr unsigned W = 64;
    static constexpr T kMax = 127;

    static V load(const T* p) noexcept { return _mm512_load_si512(p); }
    static void store(T* p, V a) noexcept { _mm512_store_si512(p, a); }
    static V splat(T x) noexcept { return _mm512_set1_epi8(x); }
    static V zero() noexcept { return _mm512_setzero_si512(); }
    static V add_sat(V a, V b) noexcept { return _mm512_max_epi8(_mm512_adds_epi8(a, b), splat(-kMax)); }
    static V sub_sat(V a, V b) noexcept { return _mm512_max_epi8(_mm512_subs_epi8(a, b), splat(-kMax)); }
    static V abs(V a) noexcept { return _mm512_abs_epi8(a); }
    static V min(V a, V b) noexcept { return _mm512_min_epi8(a, b); }
    static V max(V a, V b) noexcept { return _mm512_max_epi8(a, b); }
    static V bit_xor(V a, V b) noexcept { return _mm512_xor_si512(a, b); }
    static V bit_or(V a, V b) noexcept { return _mm512_or_si512(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept {
        return _mm512_mask_sub_epi8(magnitude, _mm512_movepi8_mask(sign), zero(), magnitude);
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(a, b), otherwise, if_equal);
    }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    static V scale(V a, Scale f) noexcept {
        const __m512i factor = _mm512_set1_epi16(f);
        const __m512i lo = _mm512_mulhrs_epi16(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a)), factor);
        const __m512i hi = _mm512_mulhrs_epi16(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a, 1)), factor);
        return _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(lo)), _mm512_cvtsepi16_epi8(hi), 1);
    }
    static V sub_floor0(V a, V offset) noexcept { return _mm512_max_epi8(_mm512_subs_epi8(a, offset), zero()); }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

}  // namespace
}  // namespace detail
}  // namespace ldpc
