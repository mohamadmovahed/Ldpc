#pragma once

// AArch64 NEON traits: 4 x float, 8 x int16, 16 x int8 lanes per 128-bit vector.

#include <arm_neon.h>

#include <cstdint>

#include "layered_min_sum.hpp"

namespace ldpc {
namespace detail {
namespace {

struct NeonF32 {
    using T = float;
    using V = float32x4_t;
    using Scale = float;
    static constexpr unsigned W = 4;
    static constexpr float kMax = 3.0e38f;

    static uint32x4_t as_u(V a) noexcept { return vreinterpretq_u32_f32(a); }
    static V as_f(uint32x4_t a) noexcept { return vreinterpretq_f32_u32(a); }
    static uint32x4_t sign_bits() noexcept { return vdupq_n_u32(0x80000000u); }

    static V load(const T* p) noexcept { return vld1q_f32(p); }
    static void store(T* p, V a) noexcept { vst1q_f32(p, a); }
    static V splat(T x) noexcept { return vdupq_n_f32(x); }
    static V zero() noexcept { return vdupq_n_f32(0.0f); }
    static V add_sat(V a, V b) noexcept { return vaddq_f32(a, b); }
    static V sub_sat(V a, V b) noexcept { return vsubq_f32(a, b); }
    static V abs(V a) noexcept { return vabsq_f32(a); }
    static V min(V a, V b) noexcept { return vminq_f32(a, b); }
    static V max(V a, V b) noexcept { return vmaxq_f32(a, b); }
    static V bit_xor(V a, V b) noexcept { return as_f(veorq_u32(as_u(a), as_u(b))); }
    static V bit_or(V a, V b) noexcept { return as_f(vorrq_u32(as_u(a), as_u(b))); }
    static V copy_sign(V magnitude, V sign) noexcept {
        return as_f(veorq_u32(as_u(magnitude), vandq_u32(as_u(sign), sign_bits())));
    }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept {
        return vbslq_f32(vceqq_f32(a, b), if_equal, otherwise);
    }
    static Scale make_scale(float factor) noexcept { return factor; }
    static V scale(V a, Scale f) noexcept { return vmulq_n_f32(a, f); }
    static V sub_floor0(V a, V offset) noexcept { return vmaxq_f32(vsubq_f32(a, offset), zero()); }
    static T quantize(float x, float) noexcept { return x; }
    static bool is_negative(T x) noexcept { return float_is_negative(x); }
};

struct NeonI16 {
    using T = std::int16_t;
    using V = int16x8_t;
    using Scale = std::int16_t;
    static constexpr unsigned W = 8;
    static constexpr T kMax = 32767;

    static V load(const T* p) noexcept { return vld1q_s16(p); }
    static void store(T* p, V a) noexcept { vst1q_s16(p, a); }
    static V splat(T x) noexcept { return vdupq_n_s16(x); }
    static V zero() noexcept { return vdupq_n_s16(0); }
    static V add_sat(V a, V b) noexcept { return vmaxq_s16(vqaddq_s16(a, b), splat(-kMax)); }
    static V sub_sat(V a, V b) noexcept { return vmaxq_s16(vqsubq_s16(a, b), splat(-kMax)); }
    static V abs(V a) noexcept { return vabsq_s16(a); }
    static V min(V a, V b) noexcept { return vminq_s16(a, b); }
    static V max(V a, V b) noexcept { return vmaxq_s16(a, b); }
    static V bit_xor(V a, V b) noexcept { return veorq_s16(a, b); }
    static V bit_or(V a, V b) noexcept { return vorrq_s16(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept { return vbslq_s16(vcltzq_s16(sign), vnegq_s16(magnitude), magnitude); }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept { return vbslq_s16(vceqq_s16(a, b), if_equal, otherwise); }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    static V scale(V a, Scale f) noexcept { return vqrdmulhq_n_s16(a, f); }
    static V sub_floor0(V a, V offset) noexcept { return vmaxq_s16(vqsubq_s16(a, offset), zero()); }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

struct NeonI8 {
    using T = std::int8_t;
    using V = int8x16_t;
    using Scale = std::int16_t;
    static constexpr unsigned W = 16;
    static constexpr T kMax = 127;

    static V load(const T* p) noexcept { return vld1q_s8(p); }
    static void store(T* p, V a) noexcept { vst1q_s8(p, a); }
    static V splat(T x) noexcept { return vdupq_n_s8(x); }
    static V zero() noexcept { return vdupq_n_s8(0); }
    static V add_sat(V a, V b) noexcept { return vmaxq_s8(vqaddq_s8(a, b), splat(-kMax)); }
    static V sub_sat(V a, V b) noexcept { return vmaxq_s8(vqsubq_s8(a, b), splat(-kMax)); }
    static V abs(V a) noexcept { return vabsq_s8(a); }
    static V min(V a, V b) noexcept { return vminq_s8(a, b); }
    static V max(V a, V b) noexcept { return vmaxq_s8(a, b); }
    static V bit_xor(V a, V b) noexcept { return veorq_s8(a, b); }
    static V bit_or(V a, V b) noexcept { return vorrq_s8(a, b); }
    static V copy_sign(V magnitude, V sign) noexcept { return vbslq_s8(vcltzq_s8(sign), vnegq_s8(magnitude), magnitude); }
    static V select_eq(V a, V b, V if_equal, V otherwise) noexcept { return vbslq_s8(vceqq_s8(a, b), if_equal, otherwise); }
    static Scale make_scale(float factor) noexcept { return factor_q15(factor); }
    static V scale(V a, Scale f) noexcept {
        const int16x8_t lo = vqrdmulhq_n_s16(vmovl_s8(vget_low_s8(a)), f);
        const int16x8_t hi = vqrdmulhq_n_s16(vmovl_s8(vget_high_s8(a)), f);
        return vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi));
    }
    static V sub_floor0(V a, V offset) noexcept { return vmaxq_s8(vqsubq_s8(a, offset), zero()); }
    static T quantize(float x, float s) noexcept { return quantize_fixed<T>(x, s, kMax); }
    static bool is_negative(T x) noexcept { return x < 0; }
};

}  // namespace
}  // namespace detail
}  // namespace ldpc
