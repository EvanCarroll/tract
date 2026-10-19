#!/usr/bin/env bash
# Downloads everything the harness measures, into gitignored paths:
#   models/       the tract inputs (split q8 decoders, tokenizer with its null normalizer removed)
#   web/hub/      the Transformers.js baseline's model tree (merged q8 decoder, original tokenizer)
#   web/vendor/   the baseline runtime: Transformers.js 4.2.0 + the onnxruntime-web pair it loads
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=Xenova/opus-mt-en-de
REV=1ca130c44c4c5441ef16d48aae521a424ab644f7
TRANSFORMERS_TGZ=https://registry.npmjs.org/@huggingface/transformers/-/transformers-4.2.0.tgz
TRANSFORMERS_SHA512=8BRCoBMH0XsWaEIamuR0LrJGAfftgHAfb2Vrffy0VKlSAE/MnUJ5/h/zTfEP3fDIft+nk7TqB8xXEyABGitBjQ==
ORT_TGZ=https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.26.0-dev.20260416-b7804b056c.tgz
ORT_SHA512=MD6Ss4GSpQBo6zqoJzyT9LRbKYs7x/JVN23FT24EcEvlqF4VuzPOeH6X38orZPKHQDbprn7K+SBpu0/mj2CQiw==

hf() { # <path in repo> <destination>
    [ -s "$2" ] || curl -fsSL -o "$2" "https://huggingface.co/$REPO/resolve/$REV/$1"
}

npm_tarball() { # <url> <sha512 base64> <destination dir>
    local tgz
    tgz=$(mktemp)
    curl -fsSL -o "$tgz" "$1"
    [ "$(openssl dgst -sha512 -binary "$tgz" | base64 -w0)" = "$2" ] || { echo "integrity mismatch: $1" >&2; exit 1; }
    mkdir -p "$3"
    tar xzf "$tgz" -C "$3"
    rm "$tgz"
}

mkdir -p models web/hub/$REPO/onnx web/vendor
for f in encoder_model_quantized decoder_model_quantized decoder_with_past_model_quantized decoder_model_merged_quantized; do
    hf "onnx/$f.onnx" "models/$f.onnx"
done
for f in config.json generation_config.json tokenizer_config.json; do
    hf "$f" "models/$f"
done
hf tokenizer.json web/hub/$REPO/tokenizer.json

# The Rust `tokenizers` crate rejects a `Precompiled` normalizer whose charsmap is null, which
# Transformers.js treats as no normalization; drop it.
python3 -I - web/hub/$REPO/tokenizer.json models/tokenizer.json <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
if t.get("normalizer", {}) and t["normalizer"].get("type") == "Precompiled" and t["normalizer"].get("precompiled_charsmap") is None:
    t["normalizer"] = None
json.dump(t, open(sys.argv[2], "w"))
EOF

for f in config.json generation_config.json tokenizer_config.json; do
    ln -sfn "../../../../models/$f" "web/hub/$REPO/$f"
done
for f in encoder_model_quantized decoder_model_merged_quantized; do
    ln -sfn "../../../../../models/$f.onnx" "web/hub/$REPO/onnx/$f.onnx"
done
ln -sfn ../models web/models

if [ ! -s web/vendor/transformers.min.js ]; then
    tmp=$(mktemp -d)
    npm_tarball "$TRANSFORMERS_TGZ" "$TRANSFORMERS_SHA512" "$tmp/transformers"
    npm_tarball "$ORT_TGZ" "$ORT_SHA512" "$tmp/ort"
    cp "$tmp/transformers/package/dist/transformers.min.js" web/vendor/
    cp "$tmp/ort/package/dist/ort-wasm-simd-threaded.mjs" "$tmp/ort/package/dist/ort-wasm-simd-threaded.wasm" web/vendor/
    rm -rf "$tmp"
fi

ls -lL models web/vendor
