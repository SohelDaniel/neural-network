// Trains the 784 → 128 → 64 → 10 network on MNIST with plain per-sample SGD.
#include "mnist.h"
#include "network.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

struct config {
	int epochs = 10;
	float learning_rate = 0.01f;
	float decay = 0.9f;          // lr for epoch e is learning_rate · decay^(e-1)
	unsigned seed = 42;
	bool augment = false;        // random ±2px shifts of each training image
	int max_shift = 2;
	std::size_t val_size = 10000; // held out from the 60k training images
	bool keep_best = true;        // test the epoch with the best validation accuracy
	std::string data_dir = "data";
};

static void print_usage(const char* prog) {
	const config d;
	std::cout << "usage: " << prog << " [options]\n"
	          << "  --epochs N       training epochs (default " << d.epochs << ")\n"
	          << "  --lr X           initial learning rate (default " << d.learning_rate << ")\n"
	          << "  --decay X        per-epoch learning-rate multiplier, 1 = off (default " << d.decay << ")\n"
	          << "  --seed N         random seed for init, shuffling and augmentation (default " << d.seed << ")\n"
	          << "  --augment        train on randomly shifted images (±" << d.max_shift << "px)\n"
	          << "  --no-augment     train on the original images (default)\n"
	          << "  --data DIR       directory with the MNIST files (default " << d.data_dir << ")\n"
	          << "  -h, --help       show this message\n";
}

static config parse_args(int argc, char** argv) {
	config cfg;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		auto value = [&]() -> std::string {
			if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
			return argv[++i];
		};
		if (arg == "--epochs") cfg.epochs = std::stoi(value());
		else if (arg == "--lr") cfg.learning_rate = std::stof(value());
		else if (arg == "--decay") cfg.decay = std::stof(value());
		else if (arg == "--seed") cfg.seed = unsigned(std::stoul(value()));
		else if (arg == "--augment") cfg.augment = true;
		else if (arg == "--no-augment") cfg.augment = false;
		else if (arg == "--data") cfg.data_dir = value();
		else if (arg == "-h" || arg == "--help") {
			print_usage(argv[0]);
			std::exit(0);
		} else throw std::invalid_argument("unknown option " + arg + " (try --help)");
	}
	if (cfg.epochs < 1) throw std::invalid_argument("--epochs must be at least 1");
	if (cfg.learning_rate <= 0.0f) throw std::invalid_argument("--lr must be positive");
	if (cfg.decay <= 0.0f || cfg.decay > 1.0f) throw std::invalid_argument("--decay must be in (0, 1]");
	return cfg;
}

static std::size_t argmax(const matrix& v) {
	std::size_t best = 0;
	for (std::size_t i = 1; i < v.get_rows(); ++i) {
		if (v(i) > v(best)) best = i;
	}
	return best;
}

static float evaluate(network& net, const std::vector<matrix>& images,
                      const std::vector<std::uint8_t>& labels) {
	std::size_t correct = 0;
	for (std::size_t i = 0; i < images.size(); ++i) {
		if (argmax(net.generate(images[i])) == labels[i]) ++correct;
	}
	return 100.0f * correct / images.size();
}

int main(int argc, char** argv) {
	try {
		const config cfg = parse_args(argc, argv);
		using clock = std::chrono::steady_clock;

		std::cout << "loading data from " << cfg.data_dir << "/ ...\n";
		auto train_x = load_images(cfg.data_dir + "/train-images-idx3-ubyte");
		auto train_y = load_labels(cfg.data_dir + "/train-labels-idx1-ubyte");
		auto test_x = load_images(cfg.data_dir + "/t10k-images-idx3-ubyte");
		auto test_y = load_labels(cfg.data_dir + "/t10k-labels-idx1-ubyte");
		if (cfg.val_size >= train_x.size()) throw std::invalid_argument("validation split too large");

		// the last val_size training images become the validation set
		const std::size_t n_train = train_x.size() - cfg.val_size;
		std::vector<matrix> val_x(train_x.begin() + n_train, train_x.end());
		std::vector<std::uint8_t> val_y(train_y.begin() + n_train, train_y.end());
		train_x.resize(n_train);
		train_y.resize(n_train);
		std::cout << train_x.size() << " train / " << val_x.size() << " validation / "
		          << test_x.size() << " test images\n";

		// one generator for everything, so a seed reproduces the whole run
		std::mt19937 gen(cfg.seed);
		network net({mnist_side * mnist_side, 128, 64, mnist_classes}, gen);

		matrix out = net.generate(train_x[0]);
		float sum = 0.0f;
		for (std::size_t i = 0; i < out.get_rows(); ++i) sum += out(i);
		std::cout << "untrained: guesses " << argmax(out) << ", real label " << int(train_y[0])
		          << ", probs sum to " << sum << "\n";
		std::cout << "epochs " << cfg.epochs << ", lr " << cfg.learning_rate << ", decay " << cfg.decay
		          << ", seed " << cfg.seed << ", augment " << (cfg.augment ? "on" : "off") << "\n\n";

		std::vector<std::size_t> order(train_x.size());
		std::iota(order.begin(), order.end(), 0);
		std::uniform_int_distribution<int> shift(-cfg.max_shift, cfg.max_shift);

		network best = net;
		float best_acc = -1.0f;
		int best_epoch = 0;
		float learning_rate = cfg.learning_rate;
		double total_seconds = 0.0;

		std::cout << std::fixed;
		for (int e = 1; e <= cfg.epochs; ++e) {
			const auto start = clock::now();
			std::shuffle(order.begin(), order.end(), gen);
			double total_loss = 0.0;

			for (std::size_t n = 0; n < order.size(); ++n) {
				const std::size_t i = order[n];
				const matrix y = one_hot(train_y[i]);
				const matrix ypred = cfg.augment
					? net.generate(shift_image(train_x[i], shift(gen), shift(gen)))
					: net.generate(train_x[i]);
				total_loss += loss(y, ypred);
				net.backward(y, ypred);
				net.update(learning_rate);

				if ((n + 1) % 10000 == 0) {
					std::cout << "  epoch " << e << ": " << n + 1 << "/" << order.size() << "\r" << std::flush;
				}
			}
			const double seconds = std::chrono::duration<double>(clock::now() - start).count();
			total_seconds += seconds;

			const float val_acc = evaluate(net, val_x, val_y);
			const bool improved = val_acc > best_acc;
			if (improved || !cfg.keep_best) {
				best = net;
				best_acc = val_acc;
				best_epoch = e;
			}
			std::cout << "epoch " << std::setw(2) << e << ": lr " << std::setprecision(5) << learning_rate
			          << ", avg loss " << std::setprecision(4) << total_loss / order.size()
			          << ", val accuracy " << std::setprecision(2) << val_acc << "%"
			          << ", " << std::setprecision(2) << seconds << " s" << (improved ? "  *" : "") << "\n";
			learning_rate *= cfg.decay;
		}

		std::cout << "\naverage epoch time " << std::setprecision(2) << total_seconds / cfg.epochs << " s\n";
		std::cout << "best validation accuracy " << best_acc << "% (epoch " << best_epoch << ")\n";
		std::cout << "test accuracy " << evaluate(best, test_x, test_y) << "%\n";
	} catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
		return 1;
	}
}
