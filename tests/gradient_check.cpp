// Checks backprop against central finite differences on the real 784 → 128 → 64 → 10 network.
#include "network.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// in float32, L ≈ 2.3 has a resolution of 2.4e-7, so (L(θ+h) - L(θ-h)) / 2h moves
// in steps of ~1.2e-5; below noise_floor the relative error becomes an absolute one
constexpr float h = 1e-2f;
constexpr float tolerance = 1e-3f;
constexpr float noise_floor = 3e-2f;
constexpr int samples_per_layer = 8;

struct probe {
	float loss = 0.0f;
	std::vector<bool> active; // which hidden ReLU units fired
};

static probe run(network& net, const matrix& x, const matrix& y) {
	std::vector<layer>& layers = net.get_layers();
	probe p;
	matrix current = x;
	for (std::size_t l = 0; l < layers.size(); ++l) {
		current = layers[l].forward(current);
		if (l + 1 == layers.size()) break;
		for (std::size_t k = 0; k < current.size(); ++k) p.active.push_back(current(k) > 0.0f);
	}
	p.loss = loss(y, softmax(current));
	return p;
}

int main(int argc, char** argv) {
	const unsigned seed = (argc > 1) ? unsigned(std::stoul(argv[1])) : 7;
	std::mt19937 gen(seed);
	network net({784, 128, 64, 10}, gen);
	std::vector<layer>& layers = net.get_layers();

	// image-like input: about half the pixels are background (exactly 0)
	matrix x(784);
	std::uniform_real_distribution<float> pixel(0.0f, 1.0f);
	for (std::size_t i = 0; i < x.size(); ++i) x(i) = (pixel(gen) < 0.5f) ? 0.0f : pixel(gen);
	matrix y(10);
	y(3) = 1.0f;

	// each layer's input, to skip weights fed by a zero
	std::vector<matrix> inputs{x};
	for (std::size_t l = 0; l + 1 < layers.size(); ++l) inputs.push_back(layers[l].forward(inputs[l]));

	net.backward(y, net.generate(x));

	int checked = 0, failed = 0, kinks = 0;
	float worst = 0.0f;
	std::cout << std::scientific << std::setprecision(3);
	for (std::size_t l = 0; l < layers.size(); ++l) {
		layer& lay = layers[l];
		std::uniform_int_distribution<std::size_t> row(0, lay.w.get_rows() - 1);
		std::uniform_int_distribution<std::size_t> col(0, lay.w.get_cols() - 1);

		for (int s = 0; s < samples_per_layer;) {
			const std::size_t i = row(gen), j = col(gen);
			const float analytic = lay.dw(i, j);
			// zero input or a dead ReLU unit: the gradient is exactly 0, nothing to check
			if (inputs[l](j) == 0.0f || analytic == 0.0f) continue;

			const float original = lay.w(i, j);
			lay.w(i, j) = original + h;
			const probe plus = run(net, x, y);
			lay.w(i, j) = original - h;
			const probe minus = run(net, x, y);
			lay.w(i, j) = original;
			// a ReLU switching on/off inside [θ-h, θ+h] makes the difference quotient meaningless
			if (plus.active != minus.active) {
				++kinks;
				continue;
			}
			++s;
			const float numeric = (plus.loss - minus.loss) / (2.0f * h);

			const float rel = std::fabs(analytic - numeric) /
			                  std::fmax(std::fabs(analytic) + std::fabs(numeric), noise_floor);
			const bool ok = rel < tolerance;
			worst = std::fmax(worst, rel);
			++checked;
			if (!ok) ++failed;
			std::cout << (ok ? "  ok  " : "  FAIL") << " layer " << l << " w(" << i << "," << j << ")"
			          << "  analytic " << std::setw(10) << analytic << "  numeric " << std::setw(10) << numeric
			          << "  rel " << rel << "\n";
		}
	}

	std::cout << "seed " << seed << ": " << checked - failed << "/" << checked << " gradients within " << tolerance
	          << " (worst " << worst << ", " << kinks << " resampled across a ReLU kink)\n";
	std::cout << (failed ? "gradient check FAILED\n" : "gradient check passed\n");
	return failed ? 1 : 0;
}
