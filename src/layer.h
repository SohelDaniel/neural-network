// One fully connected layer, z = act(w·x + b), with its own cached inputs and gradients.
#pragma once
#include "activations.h"
#include "matrix.h"
#include <random>

class layer {
public:
	layer(std::size_t input_size, std::size_t output_size, activation act, std::mt19937& gen);

	matrix w;
	matrix b;
	matrix dw;
	matrix db;

	matrix forward(const matrix& in);
	// takes ∂L/∂output, fills dw/db, returns ∂L/∂input
	// (the first layer can skip that last product, nobody consumes it)
	matrix backward(const matrix& blame, bool need_input_blame = true);
	void step(float learning_rate);

private:
	matrix x;     // input, cached for dw
	matrix z_pre; // w·x + b, cached for relu'
	activation act_ = activation::none;
};
