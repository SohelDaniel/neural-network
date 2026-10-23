// Dense row-major float matrix; a column vector is just a matrix with one column.
#pragma once
#include <cassert>
#include <cstddef>
#include <vector>

class matrix {
public:
	// rule of zero: std::vector already gives correct copy and move semantics
	matrix() = default;
	matrix(std::size_t r, std::size_t c, float fill = 0.0f);
	explicit matrix(std::size_t r, float fill = 0.0f); // column vector

	std::size_t get_rows() const { return rows; }
	std::size_t get_cols() const { return cols; }
	std::size_t size() const { return data.size(); }
	bool is_vec() const { return cols == 1; }

	// element access, row-major: (r, c) lives at data[r * cols + c]
	// asserts only, so release builds pay nothing in the hot loops
	float& operator()(std::size_t r, std::size_t c) {
		assert(r < rows && c < cols);
		return data[r * cols + c];
	}
	float operator()(std::size_t r, std::size_t c) const {
		assert(r < rows && c < cols);
		return data[r * cols + c];
	}
	float& operator()(std::size_t i) {
		assert(i < data.size());
		return data[i];
	}
	float operator()(std::size_t i) const {
		assert(i < data.size());
		return data[i];
	}

	// basic operators
	matrix operator+(const matrix& other) const;
	matrix operator+(float s) const;
	matrix operator-(const matrix& other) const;
	matrix operator-(float s) const;
	matrix operator*(const matrix& other) const;
	matrix operator*(float s) const;

	// in-place versions, no allocation
	matrix& operator+=(const matrix& other);
	matrix& operator-=(const matrix& other);
	matrix& operator*=(float s);
	matrix& add_scaled(const matrix& other, float s); // this += s * other

	// needed functions
	matrix transpose() const;
	matrix hadamard(const matrix& other) const;

private:
	std::vector<float> data;
	std::size_t rows = 0;
	std::size_t cols = 0;
};

// products with a transposed operand, without building the transpose
matrix multiply_transposed_a(const matrix& a, const matrix& b); // aᵀ · b
matrix outer(const matrix& a, const matrix& b);                 // a · bᵀ for column vectors
