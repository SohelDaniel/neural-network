# neural-network

A neural network from scratch in C++, no ML libraries. A small multilayer perceptron with its own
matrix class, hand-written backpropagation and plain SGD, trained on MNIST handwritten digits.
Everything from the IDX file reader to the gradient check is in this repository and uses only the
C++17 standard library.

## Results

| configuration | epochs | best validation acc. | test acc. | time / epoch |
|---|---|---|---|---|
| default (`./build/nn`) | 10 | 98.11% | **98.33%** | 4.68 s |
| with `--augment` | 10 | 98.06% (epoch 9) | 98.08% | 5.30 s |

Single run with seed 42 on an Apple M5 (macOS 26.4, Apple clang 21, `-O3`, single thread).
Time per epoch covers one pass over the 50,000 training images. The test set is evaluated only once,
on the epoch with the best validation accuracy.

## Architecture

```
 input        hidden 1       hidden 2       output
 784    ──▶   128 ReLU  ──▶   64 ReLU  ──▶   10 linear ──▶ softmax ──▶ ŷ
 (28×28)      W₁: 128×784    W₂: 64×128     W₃: 10×64
```

109,386 parameters, He-initialized, trained one sample at a time with cross-entropy loss.

## The math

**Forward pass.** For each layer $l$, with $a^{(0)} = x$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \mathrm{ReLU}\big(z^{(l)}\big) = \max\big(0, z^{(l)}\big)$$

The last layer stays linear and its output $o = z^{(L)}$ (the logits) goes through softmax:

$$\hat{y}_i = \mathrm{softmax}(o)_i = \frac{e^{o_i - m}}{\sum_j e^{o_j - m}}, \qquad m = \max_j o_j$$

**Loss.** Cross-entropy against the one-hot label $y$:

$$L = -\sum_i y_i \log \hat{y}_i$$

**Backpropagation.** Softmax and cross-entropy are differentiated together, which collapses to

$$\delta^{(L)} = \frac{\partial L}{\partial o} = \hat{y} - y$$

The code calls $\partial L / \partial a^{(l)}$ the *blame*. Each layer turns the blame it gets into its error
$\delta$, its gradients, and the blame it passes back:

$$\delta^{(l)} = \frac{\partial L}{\partial a^{(l)}} \odot \mathrm{ReLU}'\big(z^{(l)}\big), \qquad
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \, a^{(l-1)\top}, \qquad
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}, \qquad
\frac{\partial L}{\partial a^{(l-1)}} = W^{(l)\top} \delta^{(l)}$$

**SGD update** with learning rate $\eta_e = \eta_0 \cdot \gamma^{\,e-1}$ at epoch $e$:

$$W \leftarrow W - \eta_e \frac{\partial L}{\partial W}, \qquad b \leftarrow b - \eta_e \frac{\partial L}{\partial b}$$

## Project structure

```
neural-network/
├── CMakeLists.txt            build: nn_core library, nn trainer, tests (Release by default)
├── src/
│   ├── matrix.h / .cpp       row-major float matrix: products, transposes, in-place ops
│   ├── activations.h         ReLU, softmax, cross-entropy, derivatives, He init, SGD updates
│   ├── layer.h / .cpp        fully connected layer: forward, backward, step
│   ├── network.h / .cpp      stack of layers: generate, backward, update
│   ├── mnist.h / .cpp        IDX file loader, one-hot labels, shift augmentation
│   └── main.cpp              training loop, validation, command-line options
├── tests/
│   ├── matrix_test.cpp       unit tests for multiply, transpose, hadamard, in-place ops
│   └── gradient_check.cpp    backprop vs. central finite differences
├── scripts/
│   └── download_mnist.sh     fetches the MNIST files into data/
└── LICENSE                   MIT
```

## Quick start

```sh
git clone https://github.com/SohelDaniel/neural-network.git
cd neural-network

./scripts/download_mnist.sh          # ~55 MB into data/

cmake -S . -B build                  # Release (-O3) unless you pass -DCMAKE_BUILD_TYPE
cmake --build build

./build/nn                           # train with the defaults
(cd build && ctest --output-on-failure)   # matrix tests + gradient check
```

No CMake? clang++ (or g++) works directly:

```sh
clang++ -std=c++17 -O3 -DNDEBUG -Isrc src/*.cpp -o nn
```

Run it from the repository root so the default `data/` path resolves, or pass `--data`.

### Options

| option | default | meaning |
|---|---|---|
| `--epochs N` | 10 | training epochs |
| `--lr X` | 0.01 | initial learning rate $\eta_0$ |
| `--decay X` | 0.9 | per-epoch learning-rate multiplier $\gamma$; `1` turns decay off |
| `--seed N` | 42 | seed for initialization, shuffling and augmentation |
| `--augment` / `--no-augment` | off | train on images shifted by up to ±2 px |
| `--data DIR` | `data` | directory with the four MNIST files |
| `-h`, `--help` | | print the options |

The last 10,000 training images are held out for validation, so the network trains on 50,000.
After every epoch it is scored on the validation set and the best one is kept for the final test.

## Design notes

- **Fused softmax + cross-entropy gradient.** Differentiating softmax on its own gives a full Jacobian,
  and chaining it with $\partial L/\partial \hat{y} = -y/\hat{y}$ divides by probabilities that can be tiny.
  Done together, everything cancels to $\hat{y} - y$: cheaper, and no division at all.
- **He initialization.** Weights are drawn from $\mathcal{N}(0,\, 2/n_\text{in})$. ReLU zeroes about half its
  inputs, and the factor 2 makes up for it, so activations neither die out nor blow up from layer to
  layer at the start of training.
- **Max subtraction in softmax.** $\mathrm{softmax}(o) = \mathrm{softmax}(o - m)$ for any constant $m$.
  Using $m = \max_j o_j$ keeps every exponent $\le 0$, so `exp` can't overflow, and the sum is at least 1,
  so the division is safe.
- **Row-major storage.** Element $(r, c)$ sits at `data[r * cols + c]`. The product uses i-k-j loop order so
  the inner loop runs along contiguous rows of both operands. $W^\top\delta$ and $\delta x^\top$ have their own
  helpers (`multiply_transposed_a`, `outer`) that read $W$ row by row instead of building a transposed copy.
- **Few allocations in the hot path.** The SGD step updates weights in place (`add_scaled`), and the
  input layer skips computing $W^\top\delta$ because nothing would use it.
- **Checks.** Element access is checked with `assert` only, so Release builds pay nothing. Shape checks on
  whole-matrix operations cost $O(1)$ next to $O(n)$ work, so they always throw on mismatch.
- **Reproducible.** One `std::mt19937`, seeded from `--seed`, drives initialization, shuffling and
  augmentation. The same seed gives the same run.

### Gradient check

`tests/gradient_check.cpp` builds the real 784 → 128 → 64 → 10 network and compares the analytic
$\partial L/\partial W$ against $\big(L(\theta + h) - L(\theta - h)\big) / 2h$ for 8 weights in every layer.
It skips weights whose input is zero or whose unit is dead (their gradient is exactly zero), and
resamples when the $\pm h$ step flips a ReLU on or off, since a finite difference across a kink means nothing.
The tolerance is $10^{-3}$ relative error. In float32, $L \approx 2.3$ only resolves to about $2.4\times10^{-7}$,
so the error is measured against a floor of $3\times10^{-2}$ rather than against tiny gradients.

## Roadmap

- **Convolutional network**: conv + pooling layers, which are a better fit for images than a dense layer on raw pixels.
- **Mini-batches**: matrix-matrix products instead of matrix-vector ones, for much better throughput.
- **Momentum / Adam**: optimizers that keep state per parameter, in place of plain SGD.

## License

[MIT](LICENSE) © 2026 Sohel Daniel
