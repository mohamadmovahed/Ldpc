#include "ldpc/alist.hpp"

#include <algorithm>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

namespace ldpc {

namespace {

/// Reads whitespace-separated non-negative integers and reports precise errors.
class TokenReader {
public:
    explicit TokenReader(std::istream& in) : in_(in) {}

    long next(const char* what) {
        long value = 0;
        if (!(in_ >> value)) {
            throw AlistError(std::string("alist: failed to read ") + what);
        }
        if (value < 0) {
            throw AlistError(std::string("alist: negative value for ") + what);
        }
        return value;
    }

    /// Reads the next non-zero value (zeros are padding in alist files).
    long next_nonzero(const char* what) {
        long value = 0;
        do {
            value = next(what);
        } while (value == 0);
        return value;
    }

    bool at_end() {
        in_ >> std::ws;
        return in_.eof();
    }

private:
    std::istream& in_;
};

std::vector<std::size_t> read_degrees(TokenReader& reader, std::size_t count, std::size_t max_degree,
                                      const char* what) {
    std::vector<std::size_t> degrees(count);
    for (auto& d : degrees) {
        d = static_cast<std::size_t>(reader.next(what));
        if (d > max_degree) {
            throw AlistError(std::string("alist: ") + what + " exceeds the declared maximum");
        }
    }
    return degrees;
}

/// Reads `degrees.size()` 1-based index lists, each with values in [1, limit].
std::vector<std::vector<SparseMatrix::Index>> read_lists(TokenReader& reader, const std::vector<std::size_t>& degrees,
                                                         std::size_t limit, const char* what) {
    std::vector<std::vector<SparseMatrix::Index>> lists(degrees.size());
    for (std::size_t i = 0; i < degrees.size(); ++i) {
        lists[i].reserve(degrees[i]);
        for (std::size_t j = 0; j < degrees[i]; ++j) {
            const long v = reader.next_nonzero(what);
            if (static_cast<std::size_t>(v) > limit) {
                throw AlistError(std::string("alist: ") + what + " index " + std::to_string(v) + " out of range");
            }
            lists[i].push_back(static_cast<SparseMatrix::Index>(v - 1));
        }
    }
    return lists;
}

}  // namespace

SparseMatrix read_alist(std::istream& in) {
    TokenReader reader(in);
    const auto n = static_cast<std::size_t>(reader.next("number of columns"));
    const auto m = static_cast<std::size_t>(reader.next("number of rows"));
    if (n == 0 || m == 0) {
        throw AlistError("alist: matrix dimensions must be positive");
    }
    const auto max_col_deg = static_cast<std::size_t>(reader.next("maximum column degree"));
    const auto max_row_deg = static_cast<std::size_t>(reader.next("maximum row degree"));
    const auto col_deg = read_degrees(reader, n, max_col_deg, "column degree");
    const auto row_deg = read_degrees(reader, m, max_row_deg, "row degree");

    const auto cols = read_lists(reader, col_deg, m, "column entry");

    std::vector<std::vector<SparseMatrix::Index>> rows_from_cols(m);
    for (std::size_t c = 0; c < n; ++c) {
        for (auto r : cols[c]) {
            rows_from_cols[r].push_back(static_cast<SparseMatrix::Index>(c));
        }
    }

    SparseMatrix h;
    try {
        h = SparseMatrix::from_rows(n, rows_from_cols);
    } catch (const std::invalid_argument& e) {
        throw AlistError(std::string("alist: invalid column section: ") + e.what());
    }

    for (std::size_t r = 0; r < m; ++r) {
        if (h.row_degree(r) != row_deg[r]) {
            throw AlistError("alist: row " + std::to_string(r + 1) + " degree does not match the column lists");
        }
    }

    // The row section is redundant; validate it when present.
    if (!reader.at_end()) {
        auto rows = read_lists(reader, row_deg, n, "row entry");
        for (std::size_t r = 0; r < m; ++r) {
            std::sort(rows[r].begin(), rows[r].end());
            const auto expected = h.row(r);
            if (!std::equal(rows[r].begin(), rows[r].end(), expected.begin(), expected.end())) {
                throw AlistError("alist: row " + std::to_string(r + 1) + " disagrees with the column lists");
            }
        }
    }
    return h;
}

SparseMatrix read_alist_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw AlistError("alist: cannot open '" + path + "'");
    }
    return read_alist(file);
}

namespace {

void write_index_list(std::ostream& out, SparseMatrix::IndexRange range, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
        if (i) {
            out << ' ';
        }
        out << (i < range.size() ? range[i] + 1 : 0);
    }
    out << '\n';
}

}  // namespace

void write_alist(std::ostream& out, const SparseMatrix& h) {
    const std::size_t max_col = h.max_col_degree();
    const std::size_t max_row = h.max_row_degree();
    out << h.cols() << ' ' << h.rows() << '\n' << max_col << ' ' << max_row << '\n';
    for (std::size_t c = 0; c < h.cols(); ++c) {
        out << (c ? " " : "") << h.col_degree(c);
    }
    out << '\n';
    for (std::size_t r = 0; r < h.rows(); ++r) {
        out << (r ? " " : "") << h.row_degree(r);
    }
    out << '\n';
    for (std::size_t c = 0; c < h.cols(); ++c) {
        write_index_list(out, h.col(c), max_col);
    }
    for (std::size_t r = 0; r < h.rows(); ++r) {
        write_index_list(out, h.row(r), max_row);
    }
    if (!out) {
        throw AlistError("alist: write failed");
    }
}

void write_alist_file(const std::string& path, const SparseMatrix& h) {
    std::ofstream file(path);
    if (!file) {
        throw AlistError("alist: cannot create '" + path + "'");
    }
    write_alist(file, h);
}

}  // namespace ldpc
