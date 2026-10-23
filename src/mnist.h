// Loading the MNIST IDX files, plus label encoding and shift augmentation.
#pragma once
#include "matrix.h"
#include <cstdint>
#include <string>
#include <vector>

constexpr std::size_t mnist_side = 28;
constexpr std::size_t mnist_classes = 10;

std::vector<matrix> load_images(const std::string& path);      // 784x1 vectors scaled to [0, 1]
std::vector<std::uint8_t> load_labels(const std::string& path);

matrix one_hot(std::uint8_t label);

// moves the image dx pixels right and dy pixels down, filling the gap with background
matrix shift_image(const matrix& img, int dx, int dy);
