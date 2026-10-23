#!/usr/bin/env bash
# Downloads the four MNIST IDX files into data/ (or the directory given as $1).
set -euo pipefail

MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DIR"

# name and expected size after gunzip, to catch truncated downloads
FILES=(
	"train-images-idx3-ubyte 47040016"
	"train-labels-idx1-ubyte 60008"
	"t10k-images-idx3-ubyte 7840016"
	"t10k-labels-idx1-ubyte 10008"
)

file_size() { wc -c < "$1" | tr -d ' '; }

for entry in "${FILES[@]}"; do
	read -r name size <<< "$entry"
	out="$DIR/$name"
	if [[ -f "$out" && "$(file_size "$out")" == "$size" ]]; then
		echo "have $name"
		continue
	fi
	echo "fetching $name"
	curl -fL --retry 3 --progress-bar "$MIRROR/$name.gz" -o "$out.gz"
	gunzip -f "$out.gz"
	if [[ "$(file_size "$out")" != "$size" ]]; then
		echo "error: $name has the wrong size, try again" >&2
		exit 1
	fi
done
echo "MNIST is ready in $DIR"
