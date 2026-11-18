#pragma once

#include <iosfwd>
#include <stdexcept>
#include <string>

#include "ldpc/sparse_matrix.hpp"

namespace ldpc {

/// Thrown when an alist file cannot be read or is malformed.
class AlistError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Parses a parity-check matrix in MacKay's alist format:
///
///     N M                      (columns = code length, rows = checks)
///     max_col_degree max_row_degree
///     <N column degrees>
///     <M row degrees>
///     <N lines: 1-based row indices of each column, optionally 0-padded>
///     <M lines: 1-based column indices of each row, optionally 0-padded>
///
/// The row section is optional; when present it must agree with the columns.
SparseMatrix read_alist(std::istream& in);
SparseMatrix read_alist_file(const std::string& path);

/// Writes `h` in alist format (zero-padded, as produced by MacKay's tools).
void write_alist(std::ostream& out, const SparseMatrix& h);
void write_alist_file(const std::string& path, const SparseMatrix& h);

}  // namespace ldpc
