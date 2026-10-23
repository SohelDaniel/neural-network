// Forward, backward and SGD step for a single fully connected layer.
#include "layer.h"
#include <stdexcept>

layer::layer(std::size_t input_size, std::size_t output_size, activation act, std::mt19937& gen)
	: w(output_size, input_size), b(output_size), act_(act) {
	he_initialization(w, b, gen);
}

matrix layer::forward(const matrix& in) {
	if (!in.is_vec() || in.get_rows() != w.get_cols()) {
		throw std::invalid_argument("layer::forward: wrong input size");
	}
	x = in;
	z_pre = w * x;
	z_pre += b;
	return (act_ == activation::relu) ? relu(z_pre) : z_pre;
}

matrix layer::backward(const matrix& blame, bool need_input_blame) {
	// δ = ∂L/∂z_pre = blame ⊙ act'(z_pre)
	matrix dz = (act_ == activation::relu) ? blame.hadamard(drelu(z_pre)) : blame;
	dw = dl_dw(dz, x);
	db = dl_db(dz);
	return need_input_blame ? dl_dx(dz, w) : matrix();
}

void layer::step(float learning_rate) {
	update_weights(w, dw, learning_rate);
	update_bias(b, db, learning_rate);
}
