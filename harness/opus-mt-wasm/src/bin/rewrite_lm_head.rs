//! Rewrites `MatMul(x, DequantizeLinear(Wq, s, zp))` on the output projection into the integer
//! form the other layers use, `DynamicQuantizeLinear(x)` -> `MatMulInteger` -> `Cast` ->
//! `Mul(x_s * s)`, so the weight never exists as f32. This quantizes the activations of the last
//! layer, which the original export does not, so outputs may differ.

use std::path::PathBuf;

use clap::Parser;
use prost::Message;
use tract_onnx::pb::AttributeProto;
use tract_onnx::pb::ModelProto;
use tract_onnx::pb::NodeProto;
use tract_onnx::pb::attribute_proto::AttributeType;
use tract_onnx::pb::tensor_proto::DataType;
use tract_opus_mt::Error;

const LM_HEAD: &str = "/lm_head/MatMul";

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn node(op: &str, name: &str, input: &[&str], output: &[&str]) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        name: name.into(),
        input: input.iter().map(|s| s.to_string()).collect(),
        output: output.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let mut model = ModelProto::decode(&*std::fs::read(&args.input)?)?;
    let graph = model.graph.as_mut().ok_or_else(|| Error::Rewrite("model has no graph".into()))?;
    let mm = graph
        .node
        .iter()
        .position(|n| n.name == LM_HEAD)
        .ok_or_else(|| Error::Rewrite(format!("no {LM_HEAD} node")))?;
    let weight = graph.node[mm].input[1].clone();
    let dq = graph
        .node
        .iter()
        .position(|n| n.op_type == "DequantizeLinear" && n.output[0] == weight)
        .ok_or_else(|| {
            Error::Rewrite(format!("{LM_HEAD} weight is not a DequantizeLinear output"))
        })?;
    let [wq, ws, wzp] = [0, 1, 2].map(|i| graph.node[dq].input[i].clone());
    let x = graph.node[mm].input[0].clone();
    let y = graph.node[mm].output[0].clone();
    let mut cast = node("Cast", "lm_int8/cast", &["lm_int8/acc"], &["lm_int8/accf"]);
    cast.attribute.push(AttributeProto {
        name: "to".into(),
        r#type: AttributeType::Int as i32,
        i: DataType::Float as i64,
        ..Default::default()
    });
    let replacement = vec![
        node(
            "DynamicQuantizeLinear",
            "lm_int8/quantize_x",
            &[&x],
            &["lm_int8/xq", "lm_int8/xs", "lm_int8/xzp"],
        ),
        node(
            "MatMulInteger",
            "lm_int8/matmul",
            &["lm_int8/xq", &wq, "lm_int8/xzp", &wzp],
            &["lm_int8/acc"],
        ),
        cast,
        node("Mul", "lm_int8/scale", &["lm_int8/xs", &ws], &["lm_int8/s"]),
        node("Mul", "lm_int8/out", &["lm_int8/accf", "lm_int8/s"], &[&y]),
    ];
    graph.node.splice(mm..=mm, replacement);
    if graph.node.iter().all(|n| !n.input.contains(&weight)) {
        graph.node.retain(|n| n.output.first() != Some(&weight));
    }
    std::fs::write(&args.output, model.encode_to_vec())?;
    println!("{} -> {}", args.input.display(), args.output.display());
    Ok(())
}
