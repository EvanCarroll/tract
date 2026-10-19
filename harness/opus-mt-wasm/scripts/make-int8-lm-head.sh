#!/usr/bin/env bash
# Builds models-int8/: the split decoders with the output projection rewritten to integer form
# (see src/bin/rewrite_lm_head.rs), the rest linked from models/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models-int8
for f in decoder_model_quantized decoder_with_past_model_quantized; do
    cargo run --release --quiet --bin rewrite_lm_head -- "models/$f.onnx" "models-int8/$f.onnx"
done
for f in encoder_model_quantized.onnx config.json generation_config.json tokenizer.json; do
    ln -sfn "../models/$f" "models-int8/$f"
done
ln -sfn ../models-int8 web/models-int8
