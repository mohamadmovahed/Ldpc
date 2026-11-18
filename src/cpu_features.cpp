#include "cpu_features.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LDPC_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace ldpc {
namespace detail {

namespace {

#ifdef LDPC_X86

struct CpuidRegisters {
    std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuidRegisters cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
    CpuidRegisters r;
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<std::uint32_t>(regs[0]);
    r.ebx = static_cast<std::uint32_t>(regs[1]);
    r.ecx = static_cast<std::uint32_t>(regs[2]);
    r.edx = static_cast<std::uint32_t>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

std::uint64_t xgetbv0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    std::uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (std::uint64_t{hi} << 32) | lo;
#endif
}

CpuFeatures detect() noexcept {
    CpuFeatures f;
    if (cpuid(0, 0).eax < 7) {
        return f;
    }
    const bool osxsave = (cpuid(1, 0).ecx >> 27) & 1u;
    if (!osxsave) {
        return f;
    }
    const std::uint64_t xcr0 = xgetbv0();
    const bool os_avx = (xcr0 & 0x6) == 0x6;         // XMM + YMM state
    const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;    // + opmask, ZMM_Hi256, Hi16_ZMM
    const CpuidRegisters leaf7 = cpuid(7, 0);
    f.avx2 = os_avx && ((leaf7.ebx >> 5) & 1u);
    f.avx512bw = os_avx512 && ((leaf7.ebx >> 16) & 1u) && ((leaf7.ebx >> 30) & 1u);
    return f;
}

#else

CpuFeatures detect() noexcept { return {}; }

#endif

}  // namespace

const CpuFeatures& cpu_features() noexcept {
    static const CpuFeatures features = detect();
    return features;
}

}  // namespace detail
}  // namespace ldpc
