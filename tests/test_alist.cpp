#include <sstream>

#include "ldpc/alist.hpp"
#include "test_framework.hpp"
#include "test_util.hpp"

using ldpc::SparseMatrix;

namespace {

const char* kHammingAlist =
    "7 3\n"
    "3 4\n"
    "2 3 2 2 1 1 1\n"
    "4 4 4\n"
    "1 3 0\n1 2 3\n1 2 0\n2 3 0\n1 0 0\n2 0 0\n3 0 0\n"
    "1 2 3 5\n2 3 4 6\n1 2 4 7\n";

}  // namespace

TEST_CASE("alist: parse padded file") {
    std::istringstream in(kHammingAlist);
    const SparseMatrix h = ldpc::read_alist(in);
    CHECK_EQ(h.cols(), 7u);
    CHECK_EQ(h.rows(), 3u);
    CHECK(h == SparseMatrix::from_rows(7, {{0, 1, 2, 4}, {1, 2, 3, 5}, {0, 1, 3, 6}}));
}

TEST_CASE("alist: parse unpadded file without row section") {
    std::istringstream in("7 3\n3 4\n2 3 2 2 1 1 1\n4 4 4\n1 3\n1 2 3\n1 2\n2 3\n1\n2\n3\n");
    const SparseMatrix h = ldpc::read_alist(in);
    CHECK(h == SparseMatrix::from_rows(7, {{0, 1, 2, 4}, {1, 2, 3, 5}, {0, 1, 3, 6}}));
}

TEST_CASE("alist: write/read round trip") {
    const SparseMatrix h = test::random_regular_code(120, 60, 3, 7);
    std::stringstream buffer;
    ldpc::write_alist(buffer, h);
    CHECK(ldpc::read_alist(buffer) == h);
}

TEST_CASE("alist: malformed files are rejected") {
    const char* bad[] = {
        "",                                                             // empty
        "7 3\n3 4\n",                                                   // truncated
        "2 1\n1 2\n1 1\n2\n1\n5\n",                                     // row index out of range
        "2 1\n1 2\n1 1\n3\n1\n1\n",                                     // row degree mismatch
        "2 1\n1 2\n1 1\n2\n1\n1\n1 1\n",                                // row section disagrees
        "2 1\n1 1\n2 1\n2\n1 1\n1\n",                                   // degree above maximum
    };
    for (const char* text : bad) {
        std::istringstream in(text);
        CHECK_THROWS_AS(ldpc::read_alist(in), ldpc::AlistError);
    }
    CHECK_THROWS_AS(ldpc::read_alist_file("/nonexistent/file.alist"), ldpc::AlistError);
}
