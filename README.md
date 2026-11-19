# ldpc — fast LDPC decoding in C++17

A small, dependency-free C++17 library for decoding LDPC codes:

- **Any code** — load the parity-check matrix from an [alist](#alist-format) file or build one in code.
- **Built-in standard codes** — every LDPC code of **DVB-S2**, **DVB-S2X** (normal, medium and short frames) and **DVB-T2** (70 codes).
- **SIMD single-thread decoding** — layered min-sum with AVX-512, AVX2, NEON and portable kernels, selected at run time by CPU detection.
- **Multi-threaded decoding** — `ParallelDecoder` spreads batches over a persistent thread pool and gives the same results as the single-threaded decoder.
- **Fixed or floating point** — `Float32`, `Int16` and `Int8` arithmetic (`Int8` processes 32 frames per AVX2 vector, 64 per AVX-512 vector).
- **Encoder included** — linear-time for DVB-style accumulator codes, Gaussian elimination for anything else, so you can run simulations end to end.

```cpp
#include <ldpc/ldpc.hpp>

ldpc::SparseMatrix h = ldpc::read_alist_file("code.alist");
// or: ldpc::dvb_parity_check_matrix(ldpc::DvbStandard::S2X, ldpc::DvbFrameSize::Normal, "2/9");

ldpc::ParallelDecoder decoder(h);              // all cores, best SIMD backend
std::vector<std::uint8_t> bits;
auto results = decoder.decode_batch(llr, bits); // llr: frames * n floats
```

## Contents

- [Building](#building)
- [Using it in your project](#using-it-in-your-project)
- [API guide](#api-guide)
- [Performance](#performance)
- [Tuning the decoder](#tuning-the-decoder)
- [Command-line tool](#command-line-tool)
- [Design](#design)

## Building

Needs CMake 3.16 or newer and a C++17 compiler (GCC 8+, Clang 7+ or MSVC 2019+).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| CMake option          | Default                    | Effect                                   |
|-----------------------|----------------------------|------------------------------------------|
| `LDPC_ENABLE_AVX2`    | `ON` on x86                | build the AVX2 kernel                    |
| `LDPC_ENABLE_AVX512`  | `ON` on x86                | build the AVX-512 (F+BW) kernel          |
| `LDPC_ENABLE_NEON`    | `ON` on AArch64            | build the NEON kernel                    |
| `LDPC_BUILD_TESTS`    | `ON` if top-level project  | unit tests (`ldpc_tests`)                |
| `LDPC_BUILD_APPS`     | `ON` if top-level project  | `ldpc_tool` (simulate, benchmark, export) |
| `LDPC_BUILD_EXAMPLES` | `ON` if top-level project  | examples                                 |
| `LDPC_INSTALL`        | `ON` if top-level project  | install rules and CMake package          |

You don't need `-march=native`. Each SIMD kernel is compiled in its own
translation unit with only the flags that kernel needs, and the library picks
the best kernel the running CPU supports. One binary works on any x86-64
machine.

## Using it in your project

**As a subdirectory or with FetchContent** (tests, apps and examples are skipped automatically):

```cmake
include(FetchContent)
FetchContent_Declare(ldpc GIT_REPOSITORY https://github.com/mohamadmovahed/Ldpc.git GIT_TAG main)
FetchContent_MakeAvailable(ldpc)

target_link_libraries(my_app PRIVATE ldpc::ldpc)
```

**As an installed package:**

```sh
cmake --install build --prefix /opt/ldpc
```

```cmake
find_package(ldpc 1.0 REQUIRED)            # with CMAKE_PREFIX_PATH=/opt/ldpc
target_link_libraries(my_app PRIVATE ldpc::ldpc)
```

The public headers are under `include/ldpc/`. `#include <ldpc/ldpc.hpp>` pulls in all of them.

## API guide

### Parity-check matrices

```cpp
// From an alist file (MacKay format; zero padding and the row section are optional)
ldpc::SparseMatrix h = ldpc::read_alist_file("wifi_648_r12.alist");

// From a standard
ldpc::SparseMatrix s2 = ldpc::dvb_parity_check_matrix(ldpc::DvbStandard::S2,  ldpc::DvbFrameSize::Short,  "3/5");
ldpc::SparseMatrix vl = ldpc::dvb_parity_check_matrix(ldpc::DvbStandard::S2X, ldpc::DvbFrameSize::Medium, "1/3");

// From explicit rows / entries
ldpc::SparseMatrix small = ldpc::SparseMatrix::from_rows(7, {{0, 1, 2, 4}, {1, 2, 3, 5}, {0, 1, 3, 6}});

ldpc::write_alist_file("dvbs2_short_35.alist", s2);  // export for other tools
for (const auto& code : ldpc::dvb_codes()) { /* code.standard, code.frame, code.rate, code.n, code.k */ }
```

Built-in codes (rates are the nominal names used by the standards):

| Standard | Frame (n)      | Rates |
|----------|----------------|-------|
| DVB-S2   | normal (64800) | 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6 8/9 9/10 |
| DVB-S2   | short (16200)  | 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6 8/9 |
| DVB-S2X  | normal (64800) | 2/9 13/45 9/20 11/20 26/45 28/45 23/36 25/36 13/18 7/9 90/180 96/180 100/180 104/180 116/180 124/180 128/180 132/180 135/180 140/180 154/180 18/30 20/30 22/30 |
| DVB-S2X  | medium (32400) | 1/5 11/45 1/3 |
| DVB-S2X  | short (16200)  | 11/45 4/15 14/45 7/15 8/15 26/45 32/45 |
| DVB-T2   | normal (64800) | 1/2 3/5 2/3 3/4 4/5 5/6 |
| DVB-T2   | short (16200)  | 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6 |

`DvbStandard::S2X` also accepts every DVB-S2 rate, because DVB-S2X includes
them. DVB codewords are systematic: the first `k` bits are the information
bits. The BCH outer code and the bit interleaver are provided too (see below); scrambling,
framing and modulation are outside the scope of this library.

### Decoding

**LLR convention:** `llr = log(P(bit=0) / P(bit=1))`, so a positive value means "0".
For BPSK (`0 → +1`, `1 → −1`) over AWGN with noise variance σ², `llr = 2·y/σ²`.
A batch of frames is stored contiguously, so frame `f` is
`llr[f*n .. (f+1)*n)`. Hard decisions are written as one byte per bit (0 or 1).

```cpp
ldpc::DecoderConfig config;
config.precision      = ldpc::Precision::Int8;   // Float32 (default) | Int16 | Int8
config.max_iterations = 30;
config.backend        = ldpc::SimdBackend::Auto; // or Scalar / AVX2 / AVX512 / NEON

// Single thread
ldpc::Decoder decoder(h, config);
ldpc::DecodeResult r = decoder.decode(llr.data(), bits.data());   // one frame
decoder.decode_batch(llr.data(), num_frames, bits.data(),          // many frames
                     results.data() /* optional */, soft_out.data() /* optional APP LLRs */);

// All cores (or pass a thread count)
ldpc::ParallelDecoder parallel(h, config /*, num_threads */);
parallel.decode_batch(llr.data(), num_frames, bits.data(), results.data());
```

- `DecodeResult::converged` is true when the output satisfies every parity check. `iterations` is the number of iterations that frame used. With `early_termination` (the default), each frame stops as soon as it is a codeword.
- **Batching matters.** The SIMD kernels decode `batch_size()` frames at once, one per vector lane: 8/16/32 lanes on AVX2 and 16/32/64 on AVX-512 for Float32/Int16/Int8. A single frame takes about as long as a full batch. For throughput, pass frames in multiples of `batch_size()`, or many batches at once to `ParallelDecoder`.
- A `Decoder` owns scratch memory and is not thread-safe. Use one per thread, or a `ParallelDecoder`. Both classes can share one matrix through `std::shared_ptr<const SparseMatrix>`.
- Invalid configurations, and backends that this build or CPU doesn't support, throw `std::invalid_argument`. Check with `ldpc::is_backend_available()`.

### DVB-S2/S2X FEC around the LDPC code: BCH and bit interleaver

The DVB outer code and the bit interleaver are included, so the library covers the whole
DVB-S2/S2X FEC chain from BBFRAME to FECFRAME and back:

```cpp
// BCH (EN 302 307-1 clause 5.3.1; S2X normal and short codes, t = 8/10/12)
ldpc::BchCode bch = ldpc::dvb_bch_code(ldpc::DvbStandard::S2X, ldpc::DvbFrameSize::Normal, "132/180");
std::vector<std::uint8_t> nbch_bits = bch.encode(bbframe);   // Kbch bits -> Nbch bits (= LDPC k)
ldpc::BchDecodeResult r = bch.decode(decoded.data());         // corrects up to t errors in place

// Bit interleaver (clause 5.3.3): columns = bits per symbol, read order = the MODCOD's pattern
ldpc::BlockInterleaver il(64800, {1, 0, 2});                   // e.g. S2X 8PSK 25/36 ("102")
il.interleave(codeword.data(), tx_bits.data());
il.deinterleave(rx_llr.data(), codeword_llr.data());           // soft values back to codeword order
```

`BchCode` also works for any binary BCH code: `BchCode(m, primitive_polynomial, t, n)`.
The generator is built from the field definition (and the tests check it against the
standard's polynomial tables), decoding uses a table-driven remainder, Berlekamp-Massey and a
Chien search over the shortened code. Medium frames (VL-SNR) and the punctured VL-SNR codes
are not provided.

### Encoding

```cpp
ldpc::Encoder encoder(h);                         // picks the best strategy for h
std::vector<std::uint8_t> cw = encoder.encode(info);   // info.size() == encoder.k()
encoder.extract_info(decoded.data(), info_out.data()); // info bits back from a codeword
```

When H has a dual-diagonal (accumulator) parity part, as in DVB and similar
codes, encoding takes linear time and is systematic. For any other H the
encoder runs Gaussian elimination over GF(2). That is fine up to n ≈ 20000;
`info_positions()` then tells you where the information bits are.

## Performance

These are measured with `ldpc_tool simulate S2:normal:1/2 --iterations 10 --no-early-termination` on one core of a 2.1 GHz Xeon (4-vCPU cloud VM). The code is DVB-S2 n = 64800 with 226 799 edges, decoded with exactly 10 layered iterations. Throughput counts coded bits.

| Backend \ precision | Float32 | Int16 | Int8 |
|---------------------|--------:|------:|-----:|
| Scalar (portable)   | 6.5 Mb/s | 6.5 Mb/s | 43 Mb/s |
| AVX2                | 49 Mb/s | 76 Mb/s | 131 Mb/s |
| AVX-512             | 58 Mb/s | 98 Mb/s | 142 Mb/s |

With early termination, a frame stops when it converges, so in practice
throughput depends on SNR. At 2 dB above threshold, a DVB-S2 frame typically
needs 5–8 iterations. `ParallelDecoder` scales close to linearly with the
number of cores, as long as each call has enough frames to give every thread
at least one batch. On the same 4-vCPU VM, Int8 AVX2 reaches 132 / 234 / 350
Mb/s with 1 / 2 / 4 threads.

Error rate on DVB-S2 normal 1/2 (offset min-sum, at most 50 iterations,
FER at Eb/N0 = 0.9 / 1.0 dB):

| Precision | FER @ 0.9 dB | FER @ 1.0 dB |
|-----------|-------------:|-------------:|
| Float32   | 4.5e-2       | < 3e-4 (0 errors in 3000 frames) |
| Int16     | 4.3e-2       | < 3e-4 (0 errors in 3000 frames) |
| Int8      | 1.5e-1       | 1.0e-3       |

Int16 matches Float32. Int8 costs about 0.05–0.1 dB in the waterfall.

## Tuning the decoder

| Setting                | Default        | Notes |
|------------------------|----------------|-------|
| `algorithm`            | `OffsetMinSum` | `NormalizedMinSum` is also available |
| `offset`               | 0.5            | in LLR units. Low-rate codes (e.g. S2X 2/9) prefer about 0.25 |
| `normalization_factor` | 0.8            | used by `NormalizedMinSum` |
| `max_iterations`       | 50             | low-rate DVB-S2X codes benefit from 100 or more |
| `fixed_point_scale`    | Int16: 64, Int8: 2 | an input LLR `x` becomes `round(x·scale)`, saturated. The offset is quantized too, so choose `offset·scale` close to an integer |

Offset min-sum is the default because it is robust on codes with long
degree-2 chains (DVB, 802.11n, 5G NR). On those codes, normalized min-sum with
a small factor (for example 0.75) can stall with a few residual bit errors. An
independent flooding reference decoder shows the same behaviour, so this is a
property of the algorithm, not of this implementation. Use `ldpc_tool simulate`
to tune the settings for your code.

## Command-line tool

```text
ldpc_tool list                                   # all built-in codes
ldpc_tool export S2X:normal:2/9 s2x_29.alist     # write any built-in code as alist
ldpc_tool simulate <code> [options]              # BER/FER + throughput over BPSK/AWGN
```

`<code>` is either an alist path or `STD:FRAME:RATE`. For example:

```text
$ ldpc_tool simulate S2:short:1/2 --ebn0 1:3:0.5 --frames 2000 --threads 4 --precision int8
```

Run `ldpc_tool` with no arguments for all options: precision, backend,
algorithm, iterations, threads, Eb/N0 sweep and stopping on a frame-error
count.

## Design

```
include/ldpc/            public API
  sparse_matrix.hpp      CSR + CSC binary matrix
  alist.hpp              alist reader / writer
  dvb.hpp                DVB-S2 / S2X / T2 code catalogue
  decoder_config.hpp     configuration and result types
  decoder.hpp            single-threaded SIMD decoder
  parallel_decoder.hpp   multi-threaded decoder
  encoder.hpp            systematic encoder
  bch.hpp                BCH outer code (DVB-S2/S2X normal and short frames)
  interleaver.hpp        DVB block bit interleaver
src/
  kernel/kernel.hpp           plain-data interface between decoder and kernels
  kernel/layered_min_sum.hpp  the algorithm, written once against a SIMD "traits" type
  kernel/simd_*.hpp           traits per ISA (scalar, AVX2, AVX-512, NEON)
  kernel/kernel_*.cpp         one translation unit per ISA, with its own compiler flags
  kernel/dispatch.cpp         CPU detection and kernel selection
  structure.cpp               accumulator / quasi-cyclic detection, check scheduling
  dvb_tables.cpp              generated from the ETSI tables (tools/gen_dvb_tables.py)
```

- **Inter-frame SIMD.** Each vector lane decodes a different frame. The kernel therefore needs no gathers or shuffles and works for any H, regular or not. Messages are stored lane-interleaved (`value[edge * W + lane]`), so every access is a full aligned vector.
- **Layered offset/normalized min-sum.** Only the two smallest magnitudes and a sign parity are kept per check. An edge whose magnitude equals the minimum gets the second minimum, which is exact even when there are ties. Fixed-point arithmetic saturates symmetrically to `[-max, max]`.
- **One algorithm, many ISAs.** `layered_min_sum.hpp` is written against about 20 traits primitives. Every backend produces **bit-identical** results to the portable scalar reference, and the tests check this.
- **ISA isolation.** Kernels share only plain data with the rest of the library and live in anonymous namespaces. This avoids the ODR trap where code compiled with `-mavx512bw` leaks into callers running on older CPUs.
- **Cache-aware scheduling.** For DVB-style codes, which the decoder detects automatically, checks are processed as 360-row layers (`i, i+q, i+2q, …`). Consecutive checks then touch adjacent variables, which makes an iteration about 30% faster with the same convergence. Input and output are transposed in cache-sized blocks, and branches are avoided where LLR signs would make them unpredictable.

## Licence and attribution

The DVB parity-bit address tables in `src/dvb_tables.cpp` were generated from
the tables of [xdsopl/LDPC](https://github.com/xdsopl/LDPC) by Ahmet Inan and
Ron Economos (0BSD licence). Those were transcribed from ETSI EN 302 307-1,
EN 302 307-2 and EN 302 755. The generator checks every table against its
published check-node degree statistics.
