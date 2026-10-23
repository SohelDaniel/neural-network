// Reads MNIST's big-endian IDX format into column vectors.
#include "mnist.h"
#include <fstream>
#include <stdexcept>

namespace {

struct dataset_info {
	std::uint32_t count = 0;
	std::uint32_t rows = 0;
	std::uint32_t cols = 0;
	std::size_t input_size() const { return std::size_t(rows) * cols; }
};

// IDX stores numbers big-endian, so assemble the 4 bytes manually
std::uint32_t read_be_u32(std::ifstream& f) {
	unsigned char b[4];
	f.read(reinterpret_cast<char*>(b), 4);
	if (!f) throw std::runtime_error("unexpected end of file");
	return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) |
	       (std::uint32_t(b[2]) << 8) | std::uint32_t(b[3]);
}

std::ifstream open(const std::string& path, std::uint32_t magic, const char* kind) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("could not open " + path + " (run scripts/download_mnist.sh)");
	if (read_be_u32(f) != magic) throw std::runtime_error(path + " is not an IDX " + kind + " file");
	return f;
}

} // namespace

std::vector<matrix> load_images(const std::string& path) {
	std::ifstream f = open(path, 2051, "image");
	dataset_info info;
	info.count = read_be_u32(f);
	info.rows = read_be_u32(f);
	info.cols = read_be_u32(f);

	const std::size_t size = info.input_size();
	std::vector<unsigned char> buf(size * info.count);
	f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
	if (!f) throw std::runtime_error("unexpected end of " + path);

	std::vector<matrix> images;
	images.reserve(info.count);
	for (std::size_t n = 0; n < info.count; ++n) {
		matrix img(size);
		for (std::size_t i = 0; i < size; ++i) {
			img(i) = buf[n * size + i] / 255.0f;
		}
		images.push_back(std::move(img));
	}
	return images;
}

std::vector<std::uint8_t> load_labels(const std::string& path) {
	std::ifstream f = open(path, 2049, "label");
	const std::uint32_t count = read_be_u32(f);
	std::vector<std::uint8_t> labels(count);
	f.read(reinterpret_cast<char*>(labels.data()), count);
	if (!f) throw std::runtime_error("unexpected end of " + path);
	return labels;
}

matrix one_hot(std::uint8_t label) {
	matrix y(mnist_classes);
	y(label) = 1.0f;
	return y;
}

matrix shift_image(const matrix& img, int dx, int dy) {
	const int side = int(mnist_side);
	matrix out(img.get_rows());
	for (int r = 0; r < side; ++r) {
		const int src_r = r - dy;
		if (src_r < 0 || src_r >= side) continue;
		for (int c = 0; c < side; ++c) {
			const int src_c = c - dx;
			if (src_c < 0 || src_c >= side) continue;
			out(std::size_t(r * side + c)) = img(std::size_t(src_r * side + src_c));
		}
	}
	return out;
}
