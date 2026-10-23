// Runs the layers forward to a softmax output and backpropagates the loss through them.
#include "network.h"
#include <stdexcept>

network::network(const std::vector<std::size_t>& sizes, std::mt19937& gen) {
	if (sizes.size() < 2) {
		throw std::invalid_argument("network: need at least an input and an output size");
	}
	layers.reserve(sizes.size() - 1);
	for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
		// the last layer stays linear: softmax is applied in generate()
		activation act = (i + 2 == sizes.size()) ? activation::none : activation::relu;
		layers.emplace_back(sizes[i], sizes[i + 1], act, gen);
	}
}

matrix network::generate(const matrix& input) {
	matrix current = input;
	for (layer& l : layers) {
		current = l.forward(current);
	}
	return softmax(current);
}

void network::backward(const matrix& y, const matrix& ypred) {
	// softmax and cross-entropy are differentiated together: ∂L/∂o = ypred - y
	matrix blame = dl_o(ypred, y);
	for (std::size_t i = layers.size(); i-- > 0;) {
		blame = layers[i].backward(blame, i > 0);
	}
}

void network::update(float learning_rate) {
	for (layer& l : layers) {
		l.step(learning_rate);
	}
}
