// ldpc_tool: list built-in codes, export them as alist and run BER/throughput
// simulations (BPSK over AWGN).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ldpc/ldpc.hpp"

namespace {

void print_usage() {
    std::cout <<
        R"(Usage:
  ldpc_tool list
  ldpc_tool export <code> <output.alist>
  ldpc_tool simulate <code> [options]

<code> is either an alist file or a DVB code "STD:FRAME:RATE", e.g.
  S2:normal:1/2   S2X:normal:132/180   S2X:medium:1/3   T2:short:3/5

simulate options:
  --ebn0 A[:B[:STEP]]   Eb/N0 range in dB                  (default 1.0)
  --frames N            frames per Eb/N0 point             (default 10000)
  --min-frame-errors N  stop a point after N frame errors  (default 0 = off)
  --precision P         float32 | int16 | int8             (default float32)
  --backend B           auto | scalar | avx2 | avx512 | neon (default auto)
  --algorithm A         oms | nms                          (default oms)
  --factor F            normalization factor for nms       (default 0.8)
  --offset O            offset for oms                     (default 0.5)
  --scale S             fixed-point LLR scale (0 = default)
  --iterations N        maximum iterations                 (default 50)
  --no-early-termination  always run all iterations
  --threads N           worker threads, 0 = all cores      (default 1)
  --seed N              random seed                        (default 1)
)";
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    for (;;) {
        const std::size_t pos = s.find(sep, start);
        parts.push_back(s.substr(start, pos - start));
        if (pos == std::string::npos) return parts;
        start = pos + 1;
    }
}

ldpc::SparseMatrix load_code(const std::string& spec) {
    const auto parts = split(spec, ':');
    if (parts.size() != 3) {
        return ldpc::read_alist_file(spec);
    }
    static const std::map<std::string, ldpc::DvbStandard> standards = {
        {"S2", ldpc::DvbStandard::S2}, {"S2X", ldpc::DvbStandard::S2X}, {"T2", ldpc::DvbStandard::T2}};
    static const std::map<std::string, ldpc::DvbFrameSize> frames = {
        {"normal", ldpc::DvbFrameSize::Normal}, {"medium", ldpc::DvbFrameSize::Medium},
        {"short", ldpc::DvbFrameSize::Short}};
    const auto standard = standards.find(parts[0]);
    const auto frame = frames.find(parts[1]);
    if (standard == standards.end() || frame == frames.end()) {
        throw std::invalid_argument("unknown DVB code '" + spec + "'");
    }
    return ldpc::dvb_parity_check_matrix(standard->second, frame->second, parts[2]);
}

template <class Enum>
Enum parse_enum(const std::string& value, const std::map<std::string, Enum>& choices, const char* what) {
    const auto it = choices.find(value);
    if (it == choices.end()) throw std::invalid_argument(std::string("invalid ") + what + " '" + value + "'");
    return it->second;
}

int cmd_list() {
    std::printf("%-8s %-7s %-8s %6s %6s\n", "standard", "frame", "rate", "n", "k");
    for (const auto& c : ldpc::dvb_codes()) {
        std::string standard = ldpc::to_string(c.standard);
        std::printf("%-8s %-7s %-8s %6zu %6zu\n", standard.substr(4).c_str(), ldpc::to_string(c.frame),
                    c.rate.c_str(), c.n, c.k);
    }
    return 0;
}

int cmd_export(const std::string& code, const std::string& path) {
    const auto h = load_code(code);
    ldpc::write_alist_file(path, h);
    std::cout << "wrote " << h.rows() << "x" << h.cols() << " matrix to " << path << '\n';
    return 0;
}

struct SimulationOptions {
    double ebn0_start = 1.0, ebn0_stop = 1.0, ebn0_step = 0.5;
    std::size_t frames = 10000;
    std::size_t min_frame_errors = 0;
    unsigned threads = 1;
    unsigned seed = 1;
    ldpc::DecoderConfig config;
};

SimulationOptions parse_simulation_options(int argc, char** argv, int first) {
    SimulationOptions o;
    for (int i = first; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-early-termination") {
            o.config.early_termination = false;
            continue;
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--ebn0") {
            const auto p = split(value, ':');
            o.ebn0_start = o.ebn0_stop = std::stod(p[0]);
            if (p.size() > 1) o.ebn0_stop = std::stod(p[1]);
            if (p.size() > 2) o.ebn0_step = std::stod(p[2]);
        } else if (arg == "--frames") {
            o.frames = std::stoul(value);
        } else if (arg == "--min-frame-errors") {
            o.min_frame_errors = std::stoul(value);
        } else if (arg == "--precision") {
            o.config.precision = parse_enum<ldpc::Precision>(
                value, {{"float32", ldpc::Precision::Float32}, {"int16", ldpc::Precision::Int16},
                        {"int8", ldpc::Precision::Int8}}, "precision");
        } else if (arg == "--backend") {
            o.config.backend = parse_enum<ldpc::SimdBackend>(
                value, {{"auto", ldpc::SimdBackend::Auto}, {"scalar", ldpc::SimdBackend::Scalar},
                        {"avx2", ldpc::SimdBackend::AVX2}, {"avx512", ldpc::SimdBackend::AVX512},
                        {"neon", ldpc::SimdBackend::NEON}}, "backend");
        } else if (arg == "--algorithm") {
            o.config.algorithm = parse_enum<ldpc::Algorithm>(
                value, {{"nms", ldpc::Algorithm::NormalizedMinSum}, {"oms", ldpc::Algorithm::OffsetMinSum}},
                "algorithm");
        } else if (arg == "--factor") {
            o.config.normalization_factor = std::stof(value);
        } else if (arg == "--offset") {
            o.config.offset = std::stof(value);
        } else if (arg == "--scale") {
            o.config.fixed_point_scale = std::stof(value);
        } else if (arg == "--iterations") {
            o.config.max_iterations = std::stoi(value);
        } else if (arg == "--threads") {
            o.threads = static_cast<unsigned>(std::stoul(value));
        } else if (arg == "--seed") {
            o.seed = static_cast<unsigned>(std::stoul(value));
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    return o;
}

int cmd_simulate(const std::string& code, const SimulationOptions& o) {
    auto h = std::make_shared<const ldpc::SparseMatrix>(load_code(code));
    const ldpc::Encoder encoder(*h);
    ldpc::ParallelDecoder decoder(h, o.config, o.threads);
    const std::size_t n = encoder.n();
    const std::size_t k = encoder.k();
    const double rate = static_cast<double>(k) / static_cast<double>(n);

    std::printf("code: n=%zu k=%zu rate=%.4f edges=%zu | %s %s %s, %u thread(s) x %zu lanes, max %d iterations\n",
                n, k, rate, h->nnz(), ldpc::to_string(decoder.backend()), ldpc::to_string(o.config.precision),
                ldpc::to_string(o.config.algorithm), decoder.num_threads(), decoder.batch_size(),
                o.config.max_iterations);
    std::printf("%7s %10s %12s %12s %9s %12s %12s\n", "Eb/N0", "frames", "BER", "FER", "avg.iter", "coded Mb/s",
                "info Mb/s");

    // Frames per decoder call: enough work for every thread.
    const std::size_t chunk = decoder.batch_size() * decoder.num_threads() * 4;
    std::mt19937_64 rng(o.seed);
    std::bernoulli_distribution coin(0.5);
    std::vector<std::uint8_t> info(k), codewords(chunk * n), decoded(chunk * n);
    std::vector<float> llr(chunk * n);
    std::vector<ldpc::DecodeResult> results(chunk);

    for (double ebn0 = o.ebn0_start; ebn0 <= o.ebn0_stop + 1e-9; ebn0 += o.ebn0_step) {
        const double sigma = std::sqrt(1.0 / (2.0 * rate * std::pow(10.0, ebn0 / 10.0)));
        std::normal_distribution<float> noise(0.0f, static_cast<float>(sigma));
        const float llr_gain = static_cast<float>(2.0 / (sigma * sigma));

        std::size_t frames = 0, frame_errors = 0, bit_errors = 0, iterations = 0;
        double seconds = 0.0;
        while (frames < o.frames && (o.min_frame_errors == 0 || frame_errors < o.min_frame_errors)) {
            const std::size_t count = std::min(chunk, o.frames - frames);
            for (std::size_t f = 0; f < count; ++f) {
                for (auto& b : info) b = coin(rng);
                std::uint8_t* cw = codewords.data() + f * n;
                encoder.encode(info.data(), cw);
                for (std::size_t i = 0; i < n; ++i) {
                    llr[f * n + i] = llr_gain * ((cw[i] ? -1.0f : 1.0f) + noise(rng));
                }
            }
            const auto start = std::chrono::steady_clock::now();
            decoder.decode_batch(llr.data(), count, decoded.data(), results.data());
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            for (std::size_t f = 0; f < count; ++f) {
                std::size_t errors = 0;
                for (auto pos : encoder.info_positions()) {
                    errors += decoded[f * n + pos] != codewords[f * n + pos];
                }
                bit_errors += errors;
                frame_errors += errors != 0;
                iterations += static_cast<std::size_t>(results[f].iterations);
            }
            frames += count;
        }
        std::printf("%7.2f %10zu %12.4e %12.4e %9.2f %12.1f %12.1f\n", ebn0, frames,
                    static_cast<double>(bit_errors) / static_cast<double>(frames * k),
                    static_cast<double>(frame_errors) / static_cast<double>(frames),
                    static_cast<double>(iterations) / static_cast<double>(frames),
                    static_cast<double>(frames * n) / seconds * 1e-6, static_cast<double>(frames * k) / seconds * 1e-6);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "list") return cmd_list();
        if (command == "export" && argc == 4) return cmd_export(argv[2], argv[3]);
        if (command == "simulate" && argc >= 3) return cmd_simulate(argv[2], parse_simulation_options(argc, argv, 3));
        print_usage();
        return command.empty() || command == "--help" || command == "-h" ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
