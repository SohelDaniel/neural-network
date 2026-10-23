// Matrix arithmetic: element-wise ops, cache-friendly products and transposes.
#include "matrix.h"
#include <stdexcept>
#include <string>

// ---------- constructors ----------

matrix::matrix(std::size_t r, std::size_t c, float fill)
	: data(r * c, fill), rows(r), cols(c) {}

matrix::matrix(std::size_t r, float fill)
	: data(r, fill), rows(r), cols(1) {}

// ---------- helper ----------

// shape checks are O(1) next to O(n) work, so they stay on in release builds
static void check_same_size(const matrix& a, const matrix& b, const char* op) {
	if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) {
		throw std::invalid_argument(std::string(op) + ": size mismatch");
	}
}

// ---------- matrix-matrix operators ----------

matrix matrix::operator+(const matrix& other) const {
	matrix result = *this;
	return result += other;
}

matrix matrix::operator-(const matrix& other) const {
	matrix result = *this;
	return result -= other;
}

matrix matrix::operator*(const matrix& other) const {
	if (cols != other.rows) {
		throw std::invalid_argument("operator*: inner dimensions don't match");
	}
	matrix result(rows, other.cols);
	// i-k-j loop order: the inner loop walks both rows in a straight line
	for (std::size_t i = 0; i < rows; ++i) {
		float* out = &result.data[i * other.cols];
		for (std::size_t k = 0; k < cols; ++k) {
			const float a = data[i * cols + k];
			const float* b = &other.data[k * other.cols];
			for (std::size_t j = 0; j < other.cols; ++j) {
				out[j] += a * b[j];
			}
		}
	}
	return result;
}

// ---------- matrix-scalar operators ----------

matrix matrix::operator+(float s) const {
	matrix result = *this;
	for (float& v : result.data) v += s;
	return result;
}

matrix matrix::operator-(float s) const {
	return *this + (-s);
}

matrix matrix::operator*(float s) const {
	matrix result = *this;
	return result *= s;
}

// ---------- in-place operators ----------

matrix& matrix::operator+=(const matrix& other) {
	check_same_size(*this, other, "operator+=");
	for (std::size_t i = 0; i < data.size(); ++i) data[i] += other.data[i];
	return *this;
}

matrix& matrix::operator-=(const matrix& other) {
	check_same_size(*this, other, "operator-=");
	for (std::size_t i = 0; i < data.size(); ++i) data[i] -= other.data[i];
	return *this;
}

matrix& matrix::operator*=(float s) {
	for (float& v : data) v *= s;
	return *this;
}

matrix& matrix::add_scaled(const matrix& other, float s) {
	check_same_size(*this, other, "add_scaled");
	for (std::size_t i = 0; i < data.size(); ++i) data[i] += s * other.data[i];
	return *this;
}

// ---------- needed functions ----------

matrix matrix::transpose() const {
	matrix result(cols, rows);
	for (std::size_t i = 0; i < rows; ++i) {
		for (std::size_t j = 0; j < cols; ++j) {
			result.data[j * rows + i] = data[i * cols + j];
		}
	}
	return result;
}

matrix matrix::hadamard(const matrix& other) const {
	check_same_size(*this, other, "hadamard");
	matrix result = *this;
	for (std::size_t i = 0; i < data.size(); ++i) result.data[i] *= other.data[i];
	return result;
}

// ---------- transposed products ----------

matrix multiply_transposed_a(const matrix& a, const matrix& b) {
	if (a.get_rows() != b.get_rows()) {
		throw std::invalid_argument("multiply_transposed_a: row counts don't match");
	}
	const std::size_t n = a.get_rows(), m = a.get_cols(), p = b.get_cols();
	matrix result(m, p);
	// (aᵀb)(i, j) = Σ_k a(k, i) b(k, j); k outermost reads both inputs row by row
	for (std::size_t k = 0; k < n; ++k) {
		for (std::size_t i = 0; i < m; ++i) {
			const float aki = a(k, i);
			for (std::size_t j = 0; j < p; ++j) {
				result(i, j) += aki * b(k, j);
			}
		}
	}
	return result;
}

matrix outer(const matrix& a, const matrix& b) {
	if (!a.is_vec() || !b.is_vec()) {
		throw std::invalid_argument("outer: both inputs must be column vectors");
	}
	matrix result(a.get_rows(), b.get_rows());
	for (std::size_t i = 0; i < a.get_rows(); ++i) {
		const float ai = a(i);
		for (std::size_t j = 0; j < b.get_rows(); ++j) {
			result(i, j) = ai * b(j);
		}
	}
	return result;
}
