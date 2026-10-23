// Activations, softmax + cross-entropy, their derivatives, and the SGD update rules.
#pragma once
#include "matrix.h"
#include <cmath>
#include <random>
#include <stdexcept>

enum class activation { none, relu };
constexpr float eps = 1e-6f; // keeps log() finite when a probability rounds to 0

// notation:
// z = w·x + b        pre-activation
// o = z of the last layer (logits)
// ypred = softmax(o)
// L = cross_entropy(y, ypred)

// He init: N(0, 2 / fan_in) keeps the activation variance steady through ReLU layers
inline void he_initialization(matrix& weights, matrix& bias, std::mt19937& gen) {
	const std::size_t input_size = weights.get_cols();
	const std::size_t output_size = weights.get_rows();
	std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_size));

	for (std::size_t i = 0; i < output_size; ++i) {
		bias(i) = 0.0f;
		for (std::size_t j = 0; j < input_size; ++j) {
			weights(i, j) = dist(gen);
		}
	}
}

// ---------- forward pass: x -> relu -> ... -> o -> softmax -> ypred -> L ----------

inline matrix relu(const matrix& z) {
	if (!z.is_vec()) throw std::invalid_argument("relu: input is not a vector");
	matrix result(z.get_rows());
	for (std::size_t i = 0; i < z.get_rows(); ++i) {
		result(i) = (z(i) > 0.0f) ? z(i) : 0.0f;
	}
	return result;
}

inline matrix softmax(const matrix& o) {
	if (!o.is_vec()) throw std::invalid_argument("softmax: input is not a vector");
	// subtract max for numerical stability: softmax(o) = softmax(o - m), and exp() can't overflow
	float m = o(0);
	for (std::size_t i = 1; i < o.get_rows(); ++i) {
		if (o(i) > m) m = o(i);
	}
	matrix ypred(o.get_rows());
	float sum = 0.0f;
	for (std::size_t i = 0; i < o.get_rows(); ++i) {
		ypred(i) = std::exp(o(i) - m);
		sum += ypred(i);
	}
	// sum >= 1 because the max term is e^0, so the division is safe
	ypred *= 1.0f / sum;
	return ypred;
}

// L = -Σ y_i log(ypred_i)
inline float loss(const matrix& y, const matrix& ypred) {
	if (!ypred.is_vec() || y.get_rows() != ypred.get_rows()) {
		throw std::invalid_argument("loss: size mismatch");
	}
	float l = 0.0f;
	for (std::size_t i = 0; i < ypred.get_rows(); ++i) {
		l += y(i) * std::log(ypred(i) + eps);
	}
	return -l;
}

// ---------- backward pass ----------

// relu'(z) = 1 if z > 0 else 0
inline matrix drelu(const matrix& z) {
	if (!z.is_vec()) throw std::invalid_argument("drelu: input is not a vector");
	matrix result(z.get_rows());
	for (std::size_t i = 0; i < z.get_rows(); ++i) {
		result(i) = (z(i) > 0.0f) ? 1.0f : 0.0f;
	}
	return result;
}

// fused softmax + cross-entropy: ∂L/∂o = ypred - y
inline matrix dl_o(const matrix& ypred, const matrix& y) {
	if (!ypred.is_vec() || ypred.get_rows() != y.get_rows()) {
		throw std::invalid_argument("dl_o: predicted output size doesn't match label size");
	}
	return ypred - y;
}

// ∂L/∂W = δ · xᵀ
inline matrix dl_dw(const matrix& blame, const matrix& x) {
	return outer(blame, x);
}

// ∂L/∂x = Wᵀ · δ, the blame handed to the previous layer
inline matrix dl_dx(const matrix& blame, const matrix& w) {
	if (!blame.is_vec()) throw std::invalid_argument("dl_dx: blame is not a vector");
	return multiply_transposed_a(w, blame);
}

// ∂L/∂b = δ
inline matrix dl_db(const matrix& blame) {
	return blame;
}

// SGD: θ ← θ - η ∂L/∂θ, done in place
inline void update_weights(matrix& w, const matrix& dw, float learning_rate) {
	w.add_scaled(dw, -learning_rate);
}

inline void update_bias(matrix& b, const matrix& db, float learning_rate) {
	if (!b.is_vec()) throw std::invalid_argument("update_bias: bias is not a vector");
	b.add_scaled(db, -learning_rate);
}
