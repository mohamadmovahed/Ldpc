#include <cstring>
#include <exception>
#include <iostream>

#include "test_framework.hpp"

// Usage: ldpc_tests [substring-filter]
int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int passed = 0;
    int failed = 0;
    for (const auto& c : test::registry()) {
        if (filter && !std::strstr(c.name, filter)) {
            continue;
        }
        try {
            c.body();
            ++passed;
            std::cout << "[ PASS ] " << c.name << '\n';
        } catch (const test::Failure& f) {
            ++failed;
            std::cout << "[ FAIL ] " << c.name << "\n         " << f.message << '\n';
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[ FAIL ] " << c.name << "\n         unexpected exception: " << e.what() << '\n';
        }
    }
    std::cout << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
