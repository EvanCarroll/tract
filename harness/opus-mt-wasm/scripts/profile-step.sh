#!/usr/bin/env bash
# Per-node profile of one decoder-with-past step, using the tract CLI built from this checkout.
# usage: profile-step.sh [model.onnx] [past length] [source length] [extra tract args...]
# Prints the most expensive nodes; the full dump goes to target/profile.txt.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
model=${1:-$here/models/decoder_with_past_model_quantized.onnx}
past=${2:-10}
src=${3:-21}
layers=${LAYERS:-6} heads=${HEADS:-8} head_dim=${HEAD_DIM:-64}

tract=$root/target/release/tract
[ -x "$tract" ] || cargo build --release -p tract-cli --manifest-path "$root/Cargo.toml"

inputs=(-i "encoder_attention_mask:1,$src,i64" -i "input_ids:1,1,i64")
for ((l = 0; l < layers; l++)); do
    for k in key value; do
        inputs+=(-i "past_key_values.$l.decoder.$k:1,$heads,$past,$head_dim,f32")
        inputs+=(-i "past_key_values.$l.encoder.$k:1,$heads,$src,$head_dim,f32")
    done
done

mkdir -p "$here/target"
# --onnx-ignore-output-shapes: the CLI cannot bind the export's symbols before analysis.
"$tract" "$model" --onnx-ignore-value-info --onnx-ignore-output-shapes "${inputs[@]}" -O "${@:4}" \
    dump --profile --cost --allow-random-input > "$here/target/profile.txt" 2>&1
sed 's/\x1b\[[0-9;]*m//g' "$here/target/profile.txt" | grep 'ms/i' | sort -k1 -n -r | head -12 | cut -c1-170
