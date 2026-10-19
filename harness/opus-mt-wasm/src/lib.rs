//! Greedy opus-mt translation on tract, built natively and for `wasm32-unknown-unknown`, so the
//! same code is measured on both against the Transformers.js baseline in `web/index.html`.
//!
//! The model is the Xenova q8 export split into encoder, first-step decoder and decoder with past;
//! the merged decoder does not load (see `AGENTS.md`).

pub mod error;

use std::collections::HashMap;
use std::sync::Arc;

use tract_onnx::prelude::*;
use tract_onnx::tract_core::ops::konst::Const;
use wasm_bindgen::prelude::*;

pub use error::Error;

/// The sentences every benchmark translates; the expected outputs are in `AGENTS.md`.
pub const SENTENCES: [&str; 2] = [
    "Hello, how are you doing today?",
    "I think the new update broke the chat window, can someone restart the server before the meeting starts?",
];

const MAX_NEW_TOKENS: usize = 512;

type Plan = Arc<TypedRunnableModel>;

/// One of the three files of a split seq2seq export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Encoder,
    Decoder,
    DecoderWithPast,
}

impl Part {
    pub const ALL: [Part; 3] = [Part::Encoder, Part::Decoder, Part::DecoderWithPast];

    pub fn file(self) -> &'static str {
        match self {
            Part::Encoder => "encoder_model_quantized.onnx",
            Part::Decoder => "decoder_model_quantized.onnx",
            Part::DecoderWithPast => "decoder_with_past_model_quantized.onnx",
        }
    }

    fn from_index(ix: u8) -> Result<Part, Error> {
        Part::ALL.get(ix as usize).copied().ok_or(Error::UnknownPart(ix))
    }
}

/// The few `config.json` fields the input facts and the decode loop need.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub hidden: usize,
    pub heads: usize,
    pub decoder_start: i64,
    pub eos: usize,
    pub pad: usize,
}

impl ModelConfig {
    pub fn from_json(json: &str) -> Result<ModelConfig, Error> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let get = |key: &'static str| {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or(Error::Config(key))
        };
        Ok(ModelConfig {
            hidden: get("d_model")?,
            heads: get("decoder_attention_heads")?,
            decoder_start: get("decoder_start_token_id")? as i64,
            eos: get("eos_token_id")?,
            pad: get("pad_token_id")?,
        })
    }
}

/// Pins every input to the export's own symbols, with one symbol for the two names the export
/// gives the source length (`encoder_sequence_length` and `encoder_sequence_length_out`), which
/// tract cannot otherwise prove equal.
pub fn set_input_facts(
    model: &mut InferenceModel,
    part: Part,
    cfg: &ModelConfig,
) -> Result<(), Error> {
    let b = model.symbols.sym("batch_size").to_dim();
    let s = model.symbols.sym("encoder_sequence_length").to_dim();
    let p = model.symbols.sym("past_decoder_sequence_length").to_dim();
    let d = model.symbols.sym("decoder_sequence_length").to_dim();
    let heads = cfg.heads.to_dim();
    let head_dim = (cfg.hidden / cfg.heads).to_dim();
    let names = io_names(model)?.0;
    for (ix, name) in names.iter().enumerate() {
        let fact: InferenceFact = match name.as_str() {
            "input_ids" if part == Part::Encoder => i64::fact([b.clone(), s.clone()]).into(),
            "attention_mask" | "encoder_attention_mask" => i64::fact([b.clone(), s.clone()]).into(),
            "input_ids" => i64::fact([b.clone(), d.clone()]).into(),
            "encoder_hidden_states" => {
                f32::fact([b.clone(), s.clone(), cfg.hidden.to_dim()]).into()
            }
            n if n.contains(".encoder.") => {
                f32::fact([b.clone(), heads.clone(), s.clone(), head_dim.clone()]).into()
            }
            n if n.contains(".decoder.") => {
                f32::fact([b.clone(), heads.clone(), p.clone(), head_dim.clone()]).into()
            }
            n => return Err(Error::UnexpectedInput(n.to_string())),
        };
        model.set_input_fact(ix, fact)?;
    }
    Ok(())
}

/// Input node names and output labels, in model order.
pub fn io_names(model: &InferenceModel) -> Result<(Vec<String>, Vec<String>), Error> {
    let inputs = model.input_outlets()?.iter().map(|o| model.node(o.node).name.clone()).collect();
    let outputs = model
        .output_outlets()?
        .iter()
        .map(|o| {
            model
                .outlet_label(*o)
                .map(str::to_string)
                .unwrap_or_else(|| model.node(o.node).name.clone())
        })
        .collect();
    Ok((inputs, outputs))
}

/// Replaces every node whose inputs are all constant by its value, whatever the size: the folds
/// `PropConst` declines because the output outgrows the inputs (a dequantized or packed weight).
/// Returns the final folded tensors by the name of the node they replaced.
pub fn fold_all_consts(model: &mut TypedModel) -> Result<HashMap<String, Arc<Tensor>>, Error> {
    let mut folded = HashMap::new();
    while let Some(patch) = next_fold(model, &mut folded)? {
        patch.apply(model)?;
    }
    model.compact()?;
    folded.retain(|name, _| model.nodes.iter().any(|n| n.name == format!("{name}.folded")));
    Ok(folded)
}

fn next_fold(
    model: &TypedModel,
    folded: &mut HashMap<String, Arc<Tensor>>,
) -> Result<Option<TypedModelPatch>, Error> {
    for id in model.eval_order()? {
        let node = model.node(id);
        if node.op_is::<Const>() || node.inputs.is_empty() || node.outputs.len() != 1 {
            continue;
        }
        let facts = model.node_input_facts(id)?;
        let Some(inputs) = facts
            .iter()
            .map(|f| f.konst.clone().map(|k| k.into_tvalue()))
            .collect::<Option<TVec<_>>>()
        else {
            continue;
        };
        let Ok(Some(mut res)) = node.op.eval_out_of_plan(inputs) else { continue };
        let tensor = res.remove(0).into_arc_tensor();
        folded.insert(node.name.clone(), tensor.clone());
        let exotic = model.outlet_fact(OutletId::new(id, 0))?.exotic_fact.clone();
        let mut patch = TypedModelPatch::default();
        let wire = patch.wire_node(
            format!("{}.folded", node.name),
            Const::new_with_opt_exotic_fact(tensor, exotic)?,
            &[],
        )?[0];
        patch.shunt_outside(model, OutletId::new(id, 0), wire)?;
        return Ok(Some(patch));
    }
    Ok(None)
}

/// Splices tensors folded in a sibling model into this one by node name, without evaluating:
/// both decoders carry the same tied output projection.
pub fn reuse_folds(
    model: &mut TypedModel,
    folds: &HashMap<String, Arc<Tensor>>,
) -> Result<usize, Error> {
    let mut reused = 0;
    for (name, tensor) in folds {
        let Some(id) = model.nodes.iter().position(|n| &n.name == name) else { continue };
        let fact = model.outlet_fact(OutletId::new(id, 0))?.clone();
        if fact.shape.as_concrete() != Some(tensor.shape())
            || fact.datum_type != tensor.datum_type()
        {
            continue;
        }
        let mut patch = TypedModelPatch::default();
        let wire = patch.wire_node(
            format!("{name}.folded"),
            Const::new_with_opt_exotic_fact(tensor.clone(), fact.exotic_fact.clone())?,
            &[],
        )?[0];
        patch.shunt_outside(model, OutletId::new(id, 0), wire)?;
        patch.apply(model)?;
        reused += 1;
    }
    model.compact()?;
    Ok(reused)
}

struct Loaded {
    plan: Plan,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

fn load(
    bytes: &[u8],
    part: Part,
    cfg: &ModelConfig,
    fold: bool,
    shared: &mut HashMap<String, Arc<Tensor>>,
) -> Result<Loaded, Error> {
    let mut model = tract_onnx::onnx().with_ignore_value_info(true).model_for_read(&mut &*bytes)?;
    set_input_facts(&mut model, part, cfg)?;
    let (inputs, outputs) = io_names(&model)?;
    let mut typed = model.into_optimized()?;
    if fold {
        reuse_folds(&mut typed, shared)?;
        shared.extend(fold_all_consts(&mut typed)?);
    }
    Ok(Loaded { plan: typed.into_runnable()?, inputs, outputs })
}

/// Loads the three parts one at a time, so only one file's bytes are in memory at once. With
/// `fold`, the decoder with past reuses the first decoder's folded weights.
#[wasm_bindgen]
pub struct Loader {
    cfg: ModelConfig,
    fold: bool,
    shared: HashMap<String, Arc<Tensor>>,
    parts: [Option<Loaded>; 3],
}

#[wasm_bindgen]
impl Loader {
    #[wasm_bindgen(constructor)]
    pub fn js_new(config_json: &str, fold: bool) -> Result<Loader, JsError> {
        Loader::new(config_json, fold).map_err(js)
    }

    /// `part`: 0 encoder, 1 decoder, 2 decoder with past.
    #[wasm_bindgen(js_name = add)]
    pub fn js_add(&mut self, part: u8, bytes: Vec<u8>) -> Result<(), JsError> {
        Part::from_index(part).and_then(|p| self.add(p, &bytes)).map_err(js)
    }

    #[wasm_bindgen(js_name = finish)]
    pub fn js_finish(self, tokenizer_json: &str) -> Result<Translator, JsError> {
        self.finish(tokenizer_json).map_err(js)
    }
}

impl Loader {
    pub fn new(config_json: &str, fold: bool) -> Result<Loader, Error> {
        Ok(Loader {
            cfg: ModelConfig::from_json(config_json)?,
            fold,
            shared: HashMap::new(),
            parts: [None, None, None],
        })
    }

    pub fn add(&mut self, part: Part, bytes: &[u8]) -> Result<(), Error> {
        self.parts[part as usize] =
            Some(load(bytes, part, &self.cfg, self.fold, &mut self.shared)?);
        Ok(())
    }

    pub fn finish(self, tokenizer_json: &str) -> Result<Translator, Error> {
        let [enc, dec, dec_past] = self.parts;
        Ok(Translator {
            tok: tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes())?,
            cfg: self.cfg,
            enc: enc.ok_or(Error::MissingPart("encoder"))?,
            dec: dec.ok_or(Error::MissingPart("decoder"))?,
            dec_past: dec_past.ok_or(Error::MissingPart("decoder with past"))?,
        })
    }
}

/// A finished translation.
pub struct Translation {
    pub text: String,
    pub tokens: usize,
}

#[wasm_bindgen]
pub struct Translator {
    tok: tokenizers::Tokenizer,
    cfg: ModelConfig,
    enc: Loaded,
    dec: Loaded,
    dec_past: Loaded,
}

#[wasm_bindgen]
impl Translator {
    /// Returns `{"text": .., "tokens": n}`.
    #[wasm_bindgen(js_name = translate)]
    pub fn js_translate(&self, text: &str) -> Result<String, JsError> {
        let t = self.translate(text).map_err(js)?;
        Ok(serde_json::json!({ "text": t.text, "tokens": t.tokens }).to_string())
    }
}

impl Translator {
    /// Greedy decoding with the padding token banned, as Marian's generation config asks.
    pub fn translate(&self, text: &str) -> Result<Translation, Error> {
        let ids: Vec<i64> =
            self.tok.encode(text, true)?.get_ids().iter().map(|&i| i as i64).collect();
        let n = ids.len();
        let input_ids = tensor1(&ids).into_shape(&[1, n])?.into_tvalue();
        let mask = Tensor::from_shape(&[1, n], &vec![1i64; n])?.into_tvalue();
        let hidden = self.enc.plan.run(tvec![input_ids, mask.clone()])?.remove(0);
        let mut out: Vec<u32> = vec![];
        let mut next = self.cfg.decoder_start;
        let mut past: HashMap<String, TValue> = HashMap::new();
        for step in 0..MAX_NEW_TOKENS {
            let token = Tensor::from_shape(&[1, 1], &[next])?.into_tvalue();
            let dec = if step == 0 { &self.dec } else { &self.dec_past };
            let inputs = dec
                .inputs
                .iter()
                .map(|name| match name.as_str() {
                    "input_ids" => Ok(token.clone()),
                    "encoder_attention_mask" => Ok(mask.clone()),
                    "encoder_hidden_states" => Ok(hidden.clone()),
                    n => past
                        .get(&n.replace("past_key_values", "present"))
                        .cloned()
                        .ok_or_else(|| Error::UnexpectedInput(n.to_string())),
                })
                .collect::<Result<TVec<_>, _>>()?;
            let res = dec.plan.run(inputs)?;
            let mut logits = None;
            for (name, value) in dec.outputs.iter().zip(res) {
                if name == "logits" {
                    logits = Some(value);
                } else if name.starts_with("present") {
                    past.insert(name.clone(), value);
                }
            }
            let logits = logits.ok_or(Error::MissingPart("logits output"))?;
            let view = logits.to_plain_array_view::<f32>()?;
            let last = view.index_axis(tract_ndarray::Axis(1), view.shape()[1] - 1);
            let best = last
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != self.cfg.pad)
                .fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a })
                .0;
            if best == self.cfg.eos {
                break;
            }
            out.push(best as u32);
            next = best as i64;
        }
        Ok(Translation { text: self.tok.decode(&out, true)?, tokens: out.len() })
    }
}

fn js(e: Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}
