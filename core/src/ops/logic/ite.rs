use crate::internal::*;
use crate::ops::konst::Const;
use crate::optim::OptimizerSession;

/// Runs one of two bodies on a subset of its inputs, depending on the boolean scalar input 0.
///
/// `then_input_mapping[i]` (resp. `else_input_mapping`) is the outer input slot feeding body
/// input `i`; slot 0 is the condition. Both bodies must produce as many outputs, of the same
/// datum type and rank. A dim the bodies disagree on must be listed in `runtime_dims` as
/// `(output, axis, symbol)`: the output fact carries the symbol, and the plan binds it from the
/// tensor the taken branch produces. A constant condition inlines the taken body at declutter.
#[derive(Debug, Clone, Default)]
pub struct IfThenElse {
    pub then_body: TypedModel,
    pub then_input_mapping: Vec<usize>,
    pub else_body: TypedModel,
    pub else_input_mapping: Vec<usize>,
    pub runtime_dims: Vec<(usize, usize, Symbol)>,
    pub decluttered: bool,
}

impl PartialEq for IfThenElse {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
impl Eq for IfThenElse {}

impl IfThenElse {
    fn declutter_body(
        &self,
        session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        rule_if!(!self.decluttered);
        let mut new = self.clone();
        session.optimize(&mut new.then_body)?;
        session.optimize(&mut new.else_body)?;
        new.decluttered = true;
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?))
    }

    fn declutter_const_inputs(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let facts = model.node_input_facts(node.id)?;
        let konsts: Vec<Option<Arc<Tensor>>> = facts
            .iter()
            .map(|f| f.konst.clone())
            .enumerate()
            .map(|(s, k)| k.filter(|_| s > 0))
            .collect();
        rule_if!(konsts.iter().any(Option::is_some));
        let mut new = self.clone();
        for (body, mapping) in [
            (&mut new.then_body, &mut new.then_input_mapping),
            (&mut new.else_body, &mut new.else_input_mapping),
        ] {
            let mut kept = vec![];
            let mut kept_mapping = vec![];
            for (outlet, &slot) in body.inputs.clone().iter().zip(mapping.iter()) {
                if let Some(konst) = &konsts[slot] {
                    let op = Const::new(konst.clone())?;
                    body.nodes[outlet.node].outputs[0].fact = op.output_facts(&[])?.remove(0);
                    body.nodes[outlet.node].op = Box::new(op);
                } else {
                    kept.push(*outlet);
                    kept_mapping.push(slot);
                }
            }
            body.inputs = kept;
            *mapping = kept_mapping;
            body.refresh_output_facts()?;
        }
        new.decluttered = false;
        Self::rewire_without_unused_slots(new, model, node)
    }

    fn declutter_unused_inputs(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let is_dead = |body: &TypedModel, outlet: &OutletId| {
            body.outlet_successors(*outlet).is_empty() && !body.outputs.contains(outlet)
        };
        let referenced: Vec<usize> =
            self.then_input_mapping.iter().chain(&self.else_input_mapping).copied().collect();
        rule_if!(
            self.then_body.inputs.iter().any(|o| is_dead(&self.then_body, o))
                || self.else_body.inputs.iter().any(|o| is_dead(&self.else_body, o))
                || (1..node.inputs.len()).any(|s| !referenced.contains(&s))
        );
        let mut new = self.clone();
        for (body, mapping) in [
            (&mut new.then_body, &mut new.then_input_mapping),
            (&mut new.else_body, &mut new.else_input_mapping),
        ] {
            let (kept, kept_mapping): (Vec<OutletId>, Vec<usize>) = body
                .inputs
                .iter()
                .zip(mapping.iter())
                .filter(|(o, _)| !is_dead(body, o))
                .map(|(o, s)| (*o, *s))
                .unzip();
            body.inputs = kept;
            *mapping = kept_mapping;
            body.compact()?;
        }
        Self::rewire_without_unused_slots(new, model, node)
    }

    fn rewire_without_unused_slots(
        mut op: IfThenElse,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut used: Vec<usize> =
            op.then_input_mapping.iter().chain(&op.else_input_mapping).copied().collect();
        used.push(0);
        used.sort();
        used.dedup();
        let renumber = |s: &usize| used.iter().position(|u| u == s).unwrap();
        op.then_input_mapping = op.then_input_mapping.iter().map(renumber).collect();
        op.else_input_mapping = op.else_input_mapping.iter().map(renumber).collect();
        let inputs: TVec<OutletId> = used.iter().map(|&s| node.inputs[s]).collect();
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &inputs, op)?))
    }

    fn declutter_const_condition(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let Some(cond) = &model.outlet_fact(node.inputs[0])?.konst else { return Ok(None) };
        let cond = cond.cast_to_scalar::<bool>()?;
        let (body, input_mapping) = if cond {
            (&self.then_body, &self.then_input_mapping)
        } else {
            (&self.else_body, &self.else_input_mapping)
        };
        let mut inner_mapping: HashMap<OutletId, OutletId> = HashMap::default();
        let mut patch = TypedModelPatch::default();
        for (input_ix, outlet) in tract_itertools::izip!(input_mapping, body.input_outlets()?) {
            let tap = patch.tap_model(model, node.inputs[*input_ix])?;
            inner_mapping.insert(*outlet, tap);
        }
        for node in body.eval_order()? {
            if Graph::is_source(&body.node(node).op) {
                continue;
            }
            let node_inputs =
                body.node(node).inputs.iter().map(|o| inner_mapping[o]).collect::<TVec<_>>();
            let node_outputs =
                patch.wire_node(&body.node(node).name, &body.node(node).op, &node_inputs)?;
            for (slot_ix, outlet) in node_outputs.iter().enumerate() {
                inner_mapping.insert((node, slot_ix).into(), *outlet);
            }
        }
        for (ix, output) in body.outputs.iter().enumerate() {
            patch.shunt_outside(model, OutletId::new(node.id, ix), inner_mapping[output])?;
        }
        Ok(Some(patch))
    }
}

fn output_facts(
    inputs: &[&TypedFact],
    then_body: &TypedModel,
    then_input_mapping: &[usize],
    else_body: &TypedModel,
    else_input_mapping: &[usize],
    runtime_dims: &[(usize, usize, Symbol)],
) -> TractResult<TVec<TypedFact>> {
    ensure!(inputs[0].datum_type == bool::datum_type());
    ensure!(inputs[0].shape.volume() == 1.to_dim());
    ensure!(then_body.inputs.len() == then_input_mapping.len());
    ensure!(else_body.inputs.len() == else_input_mapping.len());
    ensure!(then_body.outputs.len() == else_body.outputs.len());
    let mut facts = tvec!();
    for ix in 0..then_body.outputs.len() {
        let then_fact = then_body.output_fact(ix)?;
        let else_fact = else_body.output_fact(ix)?;
        ensure!(
            then_fact.datum_type == else_fact.datum_type && then_fact.rank() == else_fact.rank(),
            "IfThenElse output {ix}: branches disagree on type or rank ({then_fact:?} vs {else_fact:?})"
        );
        let mut shape = tvec!();
        for (axis, (t, e)) in then_fact.shape.iter().zip(else_fact.shape.iter()).enumerate() {
            if let Some((_, _, sym)) = runtime_dims.iter().find(|(o, a, _)| *o == ix && *a == axis)
            {
                shape.push(sym.to_dim());
            } else {
                ensure!(t == e, "IfThenElse output {ix} axis {axis}: branches give {t} and {e}");
                shape.push(t.clone());
            }
        }
        facts.push(then_fact.datum_type.fact(shape));
    }
    Ok(facts)
}

impl Op for IfThenElse {
    fn name(&self) -> StaticName {
        "IfThenElse".into()
    }

    fn mints_runtime_symbols(&self) -> bool {
        !self.runtime_dims.is_empty()
    }

    op_as_typed_op!();
}

impl TypedOp for IfThenElse {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        output_facts(
            inputs,
            &self.then_body,
            &self.then_input_mapping,
            &self.else_body,
            &self.else_input_mapping,
            &self.runtime_dims,
        )
    }

    fn declutter_with_session(
        &self,
        session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(patch) = self.declutter_const_condition(model, node)? {
            return Ok(Some(patch));
        }
        if let Some(patch) = self.declutter_const_inputs(model, node)? {
            return Ok(Some(patch));
        }
        if let Some(patch) = self.declutter_unused_inputs(model, node)? {
            return Ok(Some(patch));
        }
        self.declutter_body(session, model, node)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let op = OptIfThenElse {
            then_plan: SimplePlan::new(self.then_body.clone().into_optimized()?)?,
            then_input_mapping: self.then_input_mapping.clone(),
            else_plan: SimplePlan::new(self.else_body.clone().into_optimized()?)?,
            else_input_mapping: self.else_input_mapping.clone(),
            runtime_dims: self.runtime_dims.clone(),
        };
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, op)?))
    }

    as_op!();
}

impl EvalOp for IfThenElse {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let cond = inputs[0].cast_to_scalar::<bool>()?;
        let (input_mapping, body) = if cond {
            (&self.then_input_mapping, &self.then_body)
        } else {
            (&self.else_input_mapping, &self.else_body)
        };
        let inputs: TVec<TValue> = input_mapping.iter().map(|&ix| inputs[ix].clone()).collect();
        body.clone().into_runnable()?.run(inputs)
    }
}

/// [`IfThenElse`] after codegen: both bodies optimized and planned once, each branch's state
/// spawned on first use and kept for the following turns.
#[derive(Debug, Clone)]
pub struct OptIfThenElse {
    pub then_plan: Arc<TypedSimplePlan>,
    pub then_input_mapping: Vec<usize>,
    pub else_plan: Arc<TypedSimplePlan>,
    pub else_input_mapping: Vec<usize>,
    pub runtime_dims: Vec<(usize, usize, Symbol)>,
}

impl PartialEq for OptIfThenElse {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
impl Eq for OptIfThenElse {}

impl Op for OptIfThenElse {
    fn name(&self) -> StaticName {
        "OptIfThenElse".into()
    }

    fn mints_runtime_symbols(&self) -> bool {
        !self.runtime_dims.is_empty()
    }

    op_as_typed_op!();
}

impl TypedOp for OptIfThenElse {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        output_facts(
            inputs,
            self.then_plan.model(),
            &self.then_input_mapping,
            self.else_plan.model(),
            &self.else_input_mapping,
            &self.runtime_dims,
        )
    }

    as_op!();
}

impl EvalOp for OptIfThenElse {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(OptIfThenElseState { then_state: None, else_state: None })))
    }
}

#[derive(Debug, Clone)]
struct OptIfThenElseState {
    then_state: Option<TypedSimpleState>,
    else_state: Option<TypedSimpleState>,
}

impl OpState for OptIfThenElseState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<OptIfThenElse>().context("Wrong op for OptIfThenElse state")?;
        let (plan, input_mapping, state) = if inputs[0].cast_to_scalar::<bool>()? {
            (&op.then_plan, &op.then_input_mapping, &mut self.then_state)
        } else {
            (&op.else_plan, &op.else_input_mapping, &mut self.else_state)
        };
        if state.is_none() {
            *state = Some(plan.spawn()?);
        }
        let inputs: TVec<TValue> = input_mapping.iter().map(|&ix| inputs[ix].clone()).collect();
        state.as_mut().unwrap().run(inputs)
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        bail!("OptIfThenElse is not lane-aware: its bodies are nested states")
    }
}
