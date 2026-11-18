#pragma once

// Tiny dependency-free test harness.

#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace test {

struct Case {
    const char* name;
    std::function<void()> body;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

struct Failure {
    std::string message;
};

inline void fail(const char* file, int line, const std::string& message) {
    std::ostringstream os;
    os << file << ':' << line << ": " << message;
    throw Failure{os.str()};
}

}  // namespace test

#define TEST_CONCAT_INNER(a, b) a##b
#define TEST_CONCAT(a, b) TEST_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                                                   \
    static void TEST_CONCAT(test_fn_, __LINE__)();                                        \
    static const test::Registrar TEST_CONCAT(test_reg_, __LINE__)(name, &TEST_CONCAT(test_fn_, __LINE__)); \
    static void TEST_CONCAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                 \
    do {                                                            \
        if (!(cond)) test::fail(__FILE__, __LINE__, "CHECK(" #cond ") failed"); \
    } while (0)

#define CHECK_EQ(a, b)                                                                     \
    do {                                                                                   \
        const auto& test_a_ = (a);                                                         \
        const auto& test_b_ = (b);                                                         \
        if (!(test_a_ == test_b_)) {                                                       \
            std::ostringstream test_os_;                                                   \
            test_os_ << "CHECK_EQ(" #a ", " #b ") failed: " << test_a_ << " != " << test_b_; \
            test::fail(__FILE__, __LINE__, test_os_.str());                                \
        }                                                                                  \
    } while (0)

#define CHECK_THROWS_AS(expr, type)                                                     \
    do {                                                                                \
        bool test_thrown_ = false;                                                      \
        try {                                                                           \
            (void)(expr);                                                               \
        } catch (const type&) {                                                         \
            test_thrown_ = true;                                                        \
        }                                                                               \
        if (!test_thrown_) test::fail(__FILE__, __LINE__, #expr " did not throw " #type); \
    } while (0)
