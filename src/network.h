// A stack of layers: ReLU hidden layers, a linear output layer and softmax on top.
#pragma once
#include "layer.h"
#include "matrix.h"
#include <random>
#include <vector>

class network {
public:
	// sizes = {input, hidden..., output}, e.g. {784, 128, 64, 10}
	network(const std::vector<std::size_t>& sizes, std::mt19937& gen);

	matrix generate(const matrix& input);                 // forward pass, returns ypred
	void backward(const matrix& y, const matrix& ypred); // fills every layer's dw, db
	void update(float learning_rate);                     // one SGD step on every layer

	std::vector<layer>& get_layers() { return layers; }

private:
	std::vector<layer> layers;
};
