# opus-mt in browser WASM: parity with onnxruntime-web

Read the root [`AGENTS.md`](../../AGENTS.md) first; its rules govern every change to tract
itself. This file covers only the parity work. It is the plan of record: when you measure
something new or finish an item, update the tables and the plan here in the same commit.

## Goal

Chattyness translates chat messages in the browser, in a Web Worker, with Helsinki-NLP opus-mt
models (Marian) through Transformers.js and onnxruntime-web. The aim is to replace that runtime
with tract compiled to WASM, without losing on any of:

- **time per generated token**,
- **peak WASM heap**,
- **runtime download size** (what the worker fetches besides the model),
- **model download size**,
- **output**: the same translation, token for token.

Tract only has to beat or match the baseline below; a win on one axis does not buy a loss on
another.

## The baseline

What chattyness ships, reproduced by `web/index.html?mode=ort`:

- `@huggingface/transformers` 4.2.0, bundling the onnxruntime-web JS;
- onnxruntime-web `1.26.0-dev.20260416-b7804b056c`, the plain `ort-wasm-simd-threaded`
  `.mjs`/`.wasm` pair (CPU `wasm` execution provider only);
- one thread: the page is not cross-origin isolated;
- `graphOptimizationLevel: "basic"`: the extended level rejects these q8 exports;
- `Xenova/opus-mt-en-de` at revision `1ca130c4`, `dtype: "q8"`: `encoder_model_quantized.onnx`
  plus the **merged** `decoder_model_merged_quantized.onnx`;
- greedy decoding (`num_beams: 1`), padding token banned.

## Targets

Measured 2026-10-19 with this harness in Chrome (Playwright), x86-64, 8 cores, otherwise idle
machine. Sizes in MiB. ms per token is the page's median over both sentences, warm-up excluded.

| Configuration | ms/token | peak heap | load | output |
|---|---|---|---|---|
| **baseline** (`mode=ort`) | **23.4** | **389** | 0.8 s | reference |
| baseline, `mode=ort-noqdq` | 14 | 742 | | same |
| tract, `mode=tract` | 61.2 | 594 | 2.1 s | same |
| tract, `&fold=1` | 30.8 | 594 | 2.8 s | same |
| tract, `&fold=1&models=models-int8` | 37.8 | 367 | 2.3 s | **differs** ("gebrochen") |
| tract, `&merged=1` | 60.6 | 482 | 2.2 s | same |
| tract, `&merged=1&fold=1` | 28.3 | 482 | 2.7 s | same |

The `ort-noqdq` row was measured on 2026-10-18 with an earlier page that averaged all
repetitions. The `merged` rows were measured on a loaded machine, back to back with
`mode=ort` (22.3 ms/token, 382 MiB) and `&fold=1` (28.3 ms/token, 594 MiB): compare them with
those, not with the rows above.

| Download | Baseline | tract |
|---|---|---|
| runtime (brotli / gzip / raw) | **2.11 / 3.31 / 12.90** | 2.62 / 4.24 / 15.22 |
| model | **101** (encoder 47 + merged decoder 54) | 101 with `&merged=1`; 151 split (decoders 104) |

Parity means every tract cell at or below the bold baseline cell, with the same output.

Expected output, both runtimes, greedy:

| Input | Output |
|---|---|
| Hello, how are you doing today? | Hallo, wie geht es dir heute? |
| I think the new update broke the chat window, can someone restart the server before the meeting starts? | Ich denke, das neue Update hat das Chatfenster kaputt gemacht, kann jemand den Server neu starten, bevor das Meeting beginnt? |

## Running the harness

Everything runs from this directory. The crate is its own workspace; it depends on this
checkout's `onnx` crate by path, so a change anywhere in tract is measured by rebuilding here.

```sh
./scripts/fetch.sh            # models + baseline runtime (integrity-checked), gitignored
cargo run --release --bin translate -- [--merged] [--fold]  # native: load, ms/token, peak RSS
cargo run --release --bin mem -- [--merged] [--fold]        # native: heap live/peak per load stage
./scripts/profile-step.sh     # per-node profile of one decoder step (tract CLI)
./scripts/make-int8-lm-head.sh  # models-int8/: output projection rewritten to integer form
./scripts/build-wasm.sh       # web/pkg + size table, tract vs baseline
./scripts/serve.sh            # http://127.0.0.1:8765
```

Then open, one page load per configuration:

- `index.html?mode=ort`: the baseline;
- `index.html?mode=ort-noqdq`: the baseline folding every `DequantizeLinear`;
- `index.html?mode=tract`, adding `&merged=1`, `&fold=1` and/or `&models=models-int8`.

The page prints each translation, then a JSON summary: load time, median ms per token
(warm-up repetition excluded), peak heap, outputs. Drive it with the Playwright MCP server
(Chrome): navigate, wait for the text `DONE`, read `#out`.

### Measurement rules

- **Nothing else running.** A concurrent `cargo build` once inflated the baseline from 23 to
  52-60 ms/token. Check `pgrep -a 'cargo|rustc'` first. Native numbers are noisier than browser
  ones for the same reason; trust the browser.
- **One configuration per page load.** A WASM heap never shrinks, so a second runtime in the
  same page measures the first one's high-water mark too.
- **Browser numbers are the targets.** Native runs are for iteration and profiling only:
  native tract is 2-4× faster than the same code in WASM.
- **Compare outputs, not just speed.** A change that alters a translation is a regression,
  whatever it gains.
- **Transformers.js under Node is not the baseline**: it takes its Node path and never touches
  the WASM backend.

## What we know

### The model and how tract loads it
- The **merged** decoder loads (`--merged`, `&merged=1`). Its `use_cache_branch` `If` has
  branches whose outputs disagree: present decoder key/values `D` against `P+1`, and the cached
  branch returns dummy `[0, 8, 1, 64]` constants for the encoder key/values, so even the batch
  dim differs. `If::infer_facts` joins the branches and mints a symbol per disagreeing dim;
  `IfThenElse` carries them in `runtime_dims` and the plan binds them from the taken branch's
  outputs. Every weight reaches the bodies as an outer input: `IfThenElse` declutter pulls
  constant inputs into the bodies before decluttering them, and codegen plans both bodies once
  (`OptIfThenElse`). Covered by `harness/onnx-test-cases/dynamic-if`. Branches of different
  rank (the sonos/tract#2695 reproducer) are still refused.
- The merged decoder's first step still takes past key/values: the harness feeds an empty
  decoder past and a zero encoder past of the source length (pinned to the same symbol as the
  mask). Later steps' encoder present outputs are the dummies, so the first step's are kept.
- The export names the source length twice (`encoder_sequence_length`,
  `encoder_sequence_length_out`); tract cannot prove them equal. `set_input_facts` pins one
  symbol for both. The CLI cannot bind symbols before analysis (sonos/tract#2650), hence
  `--onnx-ignore-output-shapes` in `profile-step.sh`.
- The Rust `tokenizers` crate rejects the Xenova `tokenizer.json` (`Precompiled` normalizer
  with a null charsmap); `fetch.sh` removes it. Transformers.js reads the original.

### Speed
- Unfolded, 85% of a decoder step is the tied output projection weight
  `model.shared.weight_transposed` (58101 × 512, uint8): `DequantizeLinear` (39 ms native) and
  matmul packing (47 ms native), recomputed every token. `PropConst`
  (`core/src/optim/prop_const.rs`) refuses folds whose output outgrows `max(input, 4 MB)`.
- onnxruntime-web does not fold it either: at `basic` level its `ConstantFolding` skips
  `DequantizeLinear` (`graph_transformer_utils.cc:259`, `constant_folding.cc:207` at the vendored
  commit). With `session.disable_quant_qdq` it folds every one: 14 ms/token, 742 MB.
- `fold_all_consts` (in `src/lib.rs`, no tract change) folds after `into_optimized`: native
  64 → 12 ms/token, WASM 61 → 31 ms/token, identical output.
- In integer form (`rewrite_lm_head`), the same weight costs a u8→i32 cast, a column sum and a
  pack every step; the matmul itself is 4.9 ms native. Folded: native 8.5 ms/token with the
  reference output, WASM 37.8 ms/token with a **different** second sentence ("kaputt gemacht" →
  "gebrochen"). The same model gives different words natively and in WASM, so an output check
  only counts in the browser.

### Memory
- Merged, each branch packs its own copy of the weights it uses (about 25 MiB of duplication
  natively), yet the merged decoder is still below the split pair: `mem --merged --fold` peaks at
  370 MiB and settles at 248, against 388 and 276 split. In WASM the load peak drops from 594 to
  482 MiB, the same as unfolded merged: the per-step f32 dequantization of the output
  projection reaches it too.
- Steady state without the fold is about the size of the model files: the int8 weights stay int8.
- Loading holds the file bytes, the decoded protobuf and the `InferenceModel` at once, about 3×
  the file, transiently. In WASM the heap never shrinks, so transients become the footprint.
- **In WASM the load transient is the peak**, not the fold: with and without the fold the heap
  ends at 594 MiB, set while loading the first decoder (146 → 481 MiB).
- The f32 fold adds about 117 MiB, and while folding the f32 tensor and its packed copy are
  both alive. `reuse_folds` hands the first decoder's folded tensors to the second, so it is
  paid once: native steady state 276 MiB with the fold (`mem --fold`).
- `Loader` takes one file at a time. Passing all three at once, and folding in both decoders,
  peaked at 812 MiB.

### Size
- `tokenizers` alone, built for WASM, is 2.3 MiB raw (0.53 MiB brotli); the rest of the tract
  worker is about the size of the onnxruntime `.wasm`. Transformers.js's JS tokenizer is far
  smaller.

### Ruled out
- `candle-onnx` 0.11: candle has no int8 dtype; the q8 encoder fails at load
  (`unsupported 'value' data-type Int8`). Its graph interpreter re-evaluates every node per call.
- `ort` + `ort-web`: runs onnxruntime-web from Rust through its JS API; same engine, so no gain
  on any axis here. Telemetry to pyke is on by default.

## Plan to parity

In order. Each item says what it should buy; measure before and after with the rules above,
and record the result in the tables.

1. ~~**Merged decoder: dynamic `If` with differing branch shapes.**~~ Done: model download
   151 → 101 MiB, WASM peak 594 → 482 MiB, same speed and output. Left over: the two branches
   pack the shared weights twice; and `IfThenElse` has no NNEF form and no `set_symbols`.
   Make `&merged=1` the default configuration once measured on an idle machine.
2. **Load memory: the WASM peak today.** Loading a part holds its file bytes, the decoded
   protobuf and the `InferenceModel` at once. Decode initializers without copying them again
   into tensors (protobuf `raw_data` straight into tensor storage), and release each
   initializer's protobuf as it is consumed. Verify with `mem` (the per-stage peaks) and the
   page's per-part heap lines. Buys: up to about 2× the largest file of peak heap.
3. **Speed: the remaining 31 → 23 ms/token.** `profile-step.sh` profiles the step without the
   fold, so skip the output projection's `DequantizeLinear` and pack nodes (the fold removes
   them) and look at what remains. Find where WASM loses to onnxruntime and attack the top
   nodes. Candidates: the
   encoder per message (the short sentence costs more per token on both runtimes), attention
   matmuls with a growing past, `linalg/src/wasm` kernels.
4. **Fold without the f32 spike.** Pack the output projection straight from its uint8 form
   rather than materializing the 117 MiB f32 tensor first, or fold the `DequantizeLinear` and
   the pack as one step. Matters once item 2 brings the load peak below the fold's.
5. **Make the large fold a tract decision, not a harness pass.** An opt-in budget for
   `PropConst`, or a rule that folds a weight whose consumer is a matmul pack. The harness pass
   walks the model by hand, which the root rules forbid inside tract.
6. **Native/WASM divergence on the integer path.** `models-int8` gives the reference output
   natively and a different word in WASM. Find out whether that is a near-tie in the logits or
   a WASM kernel computing something different (compare the logits of both builds at the step
   that diverges). A kernel bug blocks every integer-path idea.
7. **Speed without f32 weights.** A matmul taking f32 activations and int8 weights
   dequantized inside the kernel keeps the weight at 30 MiB without quantizing activations
   (which changed the output). Then the f32 fold, and items 4-5, may not be needed at all.
8. **Runtime size.** The 0.5 MiB brotli gap is the `tokenizers` crate. Options: tokenize in JS
   in the worker and pass ids; a slimmer Unigram-only tokenizer; `opt-level = "s"` and measure
   the speed cost.
9. **Second model.** Only opus-mt-en-de has been measured. Before calling parity, run an opus-mt
   multi-target model (`Xenova/opus-mt-en-ROMANCE`, which needs a `>>lang<<` token) and
   `Xenova/m2m100_418M` (1024 wide, 16 heads, about 5× slower in the baseline): neither has been
   loaded in tract yet.

## Harness layout

| Path | What it is |
|---|---|
| `src/lib.rs` | `Loader`, `Translator`, `set_input_facts`, `fold_all_consts`, `reuse_folds`; the wasm-bindgen surface |
| `src/bin/translate.rs` | native benchmark |
| `src/bin/mem.rs` | counting allocator, heap per load stage |
| `src/bin/rewrite_lm_head.rs` | ONNX surgery: output projection to integer form |
| `web/index.html` | the browser benchmark, both runtimes |
| `scripts/` | fetch, build, profile, serve |

The harness reaches into tract internals (`TypedModelPatch`, `Const`, `tract_onnx::pb`) on
purpose: it prototypes passes before they move into tract. It is not an example and does not
follow the `api/rs`-only rule for examples.
