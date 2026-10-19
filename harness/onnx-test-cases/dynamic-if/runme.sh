#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

for branch in then else
do
    $TRACT_RUN model.onnx --onnx-ignore-output-shapes -O run --input-from-npz $branch.npz \
        --assert-output-bundle $branch.npz
done
