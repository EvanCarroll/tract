use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::tract_core::ops::element_wise::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Cast", cast);
    reg.insert("CastLike", cast_like);
}

fn cast(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut to = node.get_attr::<DatumType>("to")?;
    let integer = to.is_integer();
    if to == i64::datum_type() {
        to = TDim::datum_type();
    }
    if integer && is_product_with_zero(ctx, &node.input[0])? {
        return Ok((expand(IntegerCastOfZeroProduct { to }), vec![]));
    }
    Ok((expand(Cast { to }), vec![]))
}

/// Whether `name` is a product with an all-zero initializer, seen through shape-only ops.
fn is_product_with_zero(ctx: &ParsingContext, name: &str) -> TractResult<bool> {
    let Some(graph) = ctx.parent_graphs.last() else {
        return Ok(false);
    };
    let mut name = name.to_string();
    loop {
        let Some(producer) = graph.node.iter().find(|n| n.output.first() == Some(&name)) else {
            return Ok(false);
        };
        match &*producer.op_type {
            "Squeeze" | "Unsqueeze" | "Reshape" | "Flatten" | "Identity" => {
                name = producer.input[0].clone()
            }
            "Mul" => {
                for input in &producer.input {
                    let Some(init) = graph.initializer.iter().find(|t| &t.name == input) else {
                        continue;
                    };
                    let factor = ctx.load_tensor(init)?.cast_to::<f64>()?.into_owned();
                    let factor = factor.try_as_plain_ram()?;
                    let factor = factor.as_slice::<f64>()?;
                    if !factor.is_empty() && factor.iter().all(|x| *x == 0.0) {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
            _ => return Ok(false),
        }
    }
}

/// An integer cast of a product with zero. A finite product casts to 0 and ONNX leaves the
/// cast of a non-finite one undefined, so the output is 0 whatever the other factor holds: its
/// value is known as soon as its shape is, which keeps shapes computed from it static.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct IntegerCastOfZeroProduct {
    to: DatumType,
}

impl IntegerCastOfZeroProduct {
    fn zeros(&self, shape: &[usize]) -> TractResult<Tensor> {
        Ok(Tensor::zero::<i64>(shape)?.cast_to_dt(self.to)?.into_owned())
    }
}

impl Expansion for IntegerCastOfZeroProduct {
    fn name(&self) -> StaticName {
        "onnx.Cast(ZeroProduct)".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.to.bex())?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        s.given(&inputs[0].shape, move |s, shape| {
            if let Ok(shape) =
                shape.iter().map(|d| d.to_usize()).collect::<Result<TVec<usize>, _>>()
            {
                s.equals(&outputs[0].value, self.zeros(&shape)?.into_arc_tensor())?;
            }
            Ok(())
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let shape = model.outlet_fact(inputs[0])?.shape.as_concrete().map(|s| s.to_vec());
        match shape {
            Some(shape) => Ok(tvec!(model.add_const(prefix, self.zeros(&shape)?)?)),
            None => model.wire_node(prefix, tract_core::ops::cast::cast(self.to), &[inputs[0]]),
        }
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Cast {
    to: DatumType,
}

impl Cast {
    fn is_string_parse(&self, from: DatumType) -> bool {
        from == String::datum_type() && self.to == f32::datum_type()
    }
}

impl Expansion for Cast {
    fn name(&self) -> StaticName {
        "onnx.Cast".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.to.bex())?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let from = model.outlet_fact(inputs[0])?.datum_type;
        if self.is_string_parse(from) {
            model.wire_node(
                prefix,
                ElementWiseOp(Box::new(StringParse { to: self.to }), None),
                &[inputs[0]],
            )
        } else {
            model.wire_node(prefix, tract_core::ops::cast::cast(self.to), &[inputs[0]])
        }
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct StringParse {
    to: DatumType,
}

impl ElementWiseMiniOp for StringParse {
    fn name(&self) -> String {
        "onnx.Cast(StringParse)".into()
    }

    fn output_type(&self, _input_type: DatumType) -> Option<DatumType> {
        Some(self.to)
    }

    fn eval_out_of_place(&self, t: &Tensor, _out_dt: Option<DatumType>) -> TractResult<Tensor> {
        unsafe {
            let mut output = Tensor::uninitialized::<f32>(t.shape())?;
            let output_slice = output.as_slice_mut_unchecked();
            let input = t.as_slice_unchecked::<String>();
            for i in 0..input.len() {
                output_slice[i] = match &*input[i] {
                    "-INF" => f32::NEG_INFINITY,
                    "INF" | "+INF" => f32::INFINITY,
                    v => v.parse()?,
                };
            }
            Ok(output)
        }
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}

fn cast_like(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(CastLike), vec![]))
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct CastLike;

impl Expansion for CastLike {
    fn name(&self) -> StaticName {
        "CastLike".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[1])?.datum_type;
        model.wire_node(prefix, tract_core::ops::cast::cast(dt), &[inputs[0]])
    }
}

#[cfg(test)]
mod tests {
    use crate::pb::*;
    use prost::Message;
    use tract_hir::internal::*;

    fn value_info(name: &str, elem_type: tensor_proto::DataType, dims: &[i64]) -> ValueInfoProto {
        let dim = dims
            .iter()
            .map(|d| tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*d)),
            })
            .collect();
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: elem_type as i32,
                    shape: Some(TensorShapeProto { dim }),
                })),
            }),
            doc_string: String::new(),
        }
    }

    fn node(op_type: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            name: output.to_string(),
            op_type: op_type.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..NodeProto::default()
        }
    }

    fn int64_initializer(name: &str, dims: &[i64], values: &[i64]) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            dims: dims.to_vec(),
            data_type: tensor_proto::DataType::Int64 as i32,
            int64_data: values.to_vec(),
            ..TensorProto::default()
        }
    }

    // Expand(y, [3 + int64(x * 0), 2]): the expand shape depends on the input x only through
    // a product with zero, which must not stop the output shape from being static.
    #[test]
    fn integer_cast_of_a_product_with_zero_is_a_static_zero() -> TractResult<()> {
        let mut cast = node("Cast", &["squeezed"], "zero_i64");
        cast.attribute.push(AttributeProto {
            name: "to".to_string(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: tensor_proto::DataType::Int64 as i64,
            ..AttributeProto::default()
        });
        let mut concat = node("Concat", &["rows_1d", "cols"], "shape");
        concat.attribute.push(AttributeProto {
            name: "axis".to_string(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: 0,
            ..AttributeProto::default()
        });
        let graph = GraphProto {
            node: vec![
                node("Mul", &["x", "zero"], "x_times_zero"),
                node("Squeeze", &["x_times_zero"], "squeezed"),
                cast,
                node("Add", &["three", "zero_i64"], "rows"),
                node("Unsqueeze", &["rows", "axes"], "rows_1d"),
                concat,
                node("Expand", &["y", "shape"], "out"),
            ],
            name: "g".to_string(),
            initializer: vec![
                TensorProto {
                    name: "zero".to_string(),
                    dims: vec![1, 1],
                    data_type: tensor_proto::DataType::Float as i32,
                    float_data: vec![0.0],
                    ..TensorProto::default()
                },
                int64_initializer("three", &[], &[3]),
                int64_initializer("axes", &[1], &[0]),
                int64_initializer("cols", &[1], &[2]),
            ],
            input: vec![
                value_info("x", tensor_proto::DataType::Float, &[1, 1]),
                value_info("y", tensor_proto::DataType::Float, &[2]),
            ],
            output: vec![value_info("out", tensor_proto::DataType::Float, &[3, 2])],
            ..GraphProto::default()
        };
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 18 }],
            graph: Some(graph),
            ..ModelProto::default()
        };
        let mut buf = vec![];
        model.encode(&mut buf).unwrap();
        let runnable =
            crate::onnx().model_for_read(&mut &*buf)?.into_optimized()?.into_runnable()?;

        let out = runnable
            .run(tvec!(tensor2(&[[f32::NAN]]).into_tvalue(), tensor1(&[1f32, 2.]).into_tvalue()))?;
        assert_eq!(*out[0], tensor2(&[[1f32, 2.], [1., 2.], [1., 2.]]));
        Ok(())
    }
}
