#!/usr/bin/env bash
# Builds the harness for the browser into web/pkg and prints the download size of both runtimes.
set -euo pipefail
cd "$(dirname "$0")/.."

# getrandom 0.3 (via tokenizers) needs the cfg as well as its `wasm_js` feature.
RUSTFLAGS='-C target-feature=+simd128 --cfg getrandom_backend="wasm_js"' \
    cargo build --release --lib --target wasm32-unknown-unknown

# wasm-bindgen's CLI must match the crate version Cargo resolved exactly.
version=$(cargo metadata --format-version 1 | python3 -I -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "wasm-bindgen"))')
bindgen=wasm-bindgen
if [ "$(wasm-bindgen --version 2>/dev/null | cut -d' ' -f2)" != "$version" ]; then
    cargo install --locked --quiet wasm-bindgen-cli --version "$version" --root target/tools
    bindgen=target/tools/bin/wasm-bindgen
fi
rm -rf web/pkg
"$bindgen" --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/tract_opus_mt.wasm

wasm=web/pkg/tract_opus_mt_bg.wasm
if command -v wasm-opt >/dev/null; then
    wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
        --enable-mutable-globals --enable-reference-types --enable-multivalue --strip-debug --strip-producers \
        "$wasm" -o "$wasm.opt"
    mv "$wasm.opt" "$wasm"
else
    echo "wasm-opt not found: sizes below are unoptimized" >&2
fi

size() { # <label> <files...>
    local label=$1 raw=0 gz=0 br=0
    shift
    for f in "$@"; do
        raw=$((raw + $(stat -c%s "$f")))
        gz=$((gz + $(gzip -9c "$f" | wc -c)))
        br=$((br + $(brotli -c -q 11 "$f" | wc -c)))
    done
    printf '%-14s raw %6.2f MB   gzip %5.2f MB   brotli %5.2f MB\n' "$label" \
        "$(bc -l <<< "$raw/1048576")" "$(bc -l <<< "$gz/1048576")" "$(bc -l <<< "$br/1048576")"
}
size tract "$wasm" web/pkg/tract_opus_mt.js
[ -d web/vendor ] && size baseline web/vendor/transformers.min.js web/vendor/ort-wasm-simd-threaded.mjs web/vendor/ort-wasm-simd-threaded.wasm
