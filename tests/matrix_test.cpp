// Small unit tests for the matrix class: products, transposes, hadamard and in-place ops.
#include "matrix.h"
#include <cmath>
#include <iostream>

static int failures = 0;

#define CHECK(cond)                                                              \
	do {                                                                         \
		if (!(cond)) {                                                           \
			std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond "\n"; \
			++failures;                                                          \
		}                                                                        \
	} while (0)

static matrix from(std::size_t r, std::size_t c, std::initializer_list<float> values) {
	matrix m(r, c, 0.0f);
	std::size_t i = 0;
	for (float v : values) m(i / c, i % c) = v, ++i;
	return m;
}

static bool equal(const matrix& a, const matrix& b, float tol = 1e-6f) {
	if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
	for (std::size_t i = 0; i < a.size(); ++i) {
		if (std::fabs(a(i) - b(i)) > tol) return false;
	}
	return true;
}

static void test_multiply() {
	const matrix a = from(2, 3, {1, 2, 3, 4, 5, 6});
	const matrix b = from(3, 2, {7, 8, 9, 10, 11, 12});
	CHECK(equal(a * b, from(2, 2, {58, 64, 139, 154})));

	const matrix v = from(3, 1, {1, 0, -1});
	CHECK(equal(a * v, from(2, 1, {-2, -2})));

	bool threw = false;
	try { (void)(a * a); } catch (const std::invalid_argument&) { threw = true; }
	CHECK(threw);
}

static void test_transpose() {
	const matrix a = from(2, 3, {1, 2, 3, 4, 5, 6});
	const matrix t = a.transpose();
	CHECK(equal(t, from(3, 2, {1, 4, 2, 5, 3, 6})));
	CHECK(equal(t.transpose(), a));
}

static void test_transposed_products() {
	const matrix w = from(2, 3, {1, 2, 3, 4, 5, 6});
	const matrix d = from(2, 1, {1, -1});
	const matrix x = from(3, 1, {2, 0, 1});
	CHECK(equal(multiply_transposed_a(w, d), w.transpose() * d)); // Wᵀ·δ
	CHECK(equal(outer(d, x), d * x.transpose()));                 // δ·xᵀ
}

static void test_hadamard() {
	const matrix a = from(2, 2, {1, 2, 3, 4});
	const matrix b = from(2, 2, {0, -1, 2, 0.5f});
	CHECK(equal(a.hadamard(b), from(2, 2, {0, -2, 6, 2})));
}

static void test_in_place_and_copy() {
	matrix a = from(2, 2, {1, 2, 3, 4});
	const matrix b = from(2, 2, {1, 1, 1, 1});
	matrix c = a; // copies are deep
	a += b;
	CHECK(equal(a, from(2, 2, {2, 3, 4, 5})));
	CHECK(equal(c, from(2, 2, {1, 2, 3, 4})));
	a -= b;
	a *= 2.0f;
	CHECK(equal(a, from(2, 2, {2, 4, 6, 8})));
	a.add_scaled(b, -0.5f);
	CHECK(equal(a, from(2, 2, {1.5f, 3.5f, 5.5f, 7.5f})));
	CHECK(equal(a + 1.0f - 1.0f, a));
}

int main() {
	test_multiply();
	test_transpose();
	test_transposed_products();
	test_hadamard();
	test_in_place_and_copy();
	if (failures) {
		std::cout << failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "all matrix tests passed\n";
}
