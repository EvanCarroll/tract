use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::tract_core::tract_data::half::slice::HalfFloatSliceExt;
use tract_nnef::tract_core::tract_linalg::element_wise::ElementWise;
use tract_nnef::tract_core::tract_linalg::routines::Func;
use tract_nnef::tract_ndarray::{
    Array2, Array4, ArrayView2, ArrayView4, ArrayViewMut2, CowArray, Ix2, s,
};

tract_core::declare_knob!(
    TRACT_FLASH_SDPA_ST,
    bool,
    false,
    "Force single-threaded flash SDPA (disable per-head rayon parallelism)."
);

/// Longest K/V sequence attended as a single tile, its softmax taken in one pass
/// rather than rescaled block by block.
const ONE_TILE_MAX_KV_LEN: usize = 256;

/// Tract operator wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct FlashSdpaOp {
    pub causal: bool,
    pub scale: Option<f32>,
}
impl Eq for FlashSdpaOp {}

impl Op for FlashSdpaOp {
    fn name(&self) -> StaticName {
        "FlashSDPA".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("causal={}, scale={:?}", self.causal, self.scale)])
    }

    op_as_typed_op!();
}

impl TypedOp for FlashSdpaOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (q, k, v, opt_m) = match inputs.len() {
            3 => (inputs[0], inputs[1], inputs[2], None),
            4 => (inputs[0], inputs[1], inputs[2], Some(inputs[3])),
            _ => bail!("FlashSDPA expects 3 or 4 inputs (Q,K,V, optional mask), got {inputs:?}"),
        };

        // dtype checks
        ensure!(q.datum_type.is_float(), "Q must be floating point");
        ensure!(k.datum_type.is_float(), "K must be floating point");
        ensure!(v.datum_type.is_float(), "V must be floating point");
        if let Some(m) = opt_m {
            ensure!(m.datum_type.is_float(), "Mask must be floating point");
        }

        // rank checks
        ensure!(
            q.rank() == k.rank()
                && q.rank() == v.rank()
                && opt_m.as_ref().is_none_or(|m| m.rank() == q.rank()),
            "Q, K, V, mask must have the same rank, found {}, {}, {}, {:?} resp.",
            q.rank(),
            k.rank(),
            v.rank(),
            opt_m.as_ref().map(|m| m.rank().to_string())
        );
        ensure!(
            q.rank() == 3 || q.rank() == 4,
            "Inputs must be of rank 3 or 4, found {}",
            q.rank()
        );

        let one = 1.to_dim();

        let ((bq, hq, _tq, dq), (bk, hkv, tk, dk), (bv, hkv2, tk2, dv)) = if q.rank() == 4 {
            (
                (&q.shape[0], &q.shape[1], &q.shape[2], &q.shape[3]),
                (&k.shape[0], &k.shape[1], &k.shape[2], &k.shape[3]),
                (&v.shape[0], &v.shape[1], &v.shape[2], &v.shape[3]),
            )
        } else {
            (
                (&q.shape[0], &one, &q.shape[1], &q.shape[2]),
                (&k.shape[0], &one, &k.shape[1], &k.shape[2]),
                (&v.shape[0], &one, &v.shape[1], &v.shape[2]),
            )
        };

        ensure!(bq == bk && bq == bv, "Batch dims must match for Q/K/V");
        ensure!(hkv == hkv2, "K/V head counts must match");
        ensure!(tk == tk2, "K/V lengths must match");
        ensure!(dq == dk && dq == dv, "Head dims (D) must match across Q/K/V");

        // If heads are fully known, check GQA divisibility (best-effort).
        if let (Ok(hq), Ok(hkv)) = (hq.to_usize(), hkv.to_usize()) {
            ensure!(hq % hkv == 0, "num_q_heads must be a multiple of num_kv_heads for GQA");
        }

        // Output has same shape and dtype as Q.
        Ok(tvec!(q.without_value()))
    }

    as_op!();
}

impl EvalOp for FlashSdpaOp {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3 || inputs.len() == 4);
        let q = &inputs[0];

        let input_dt = q.datum_type();
        let is_3d_case = q.rank() == 3;
        let widened: TVec<Option<Tensor>> = inputs
            .iter()
            .map(|t| {
                (t.datum_type() != f16::datum_type() && t.datum_type() != f32::datum_type())
                    .then(|| Ok(t.cast_to::<f32>()?.into_owned()))
                    .transpose()
            })
            .collect::<TractResult<_>>()?;
        let views: TVec<FloatView> = inputs
            .iter()
            .zip(&widened)
            .map(|(t, w)| FloatView::of(w.as_ref().unwrap_or(t)))
            .collect::<TractResult<_>>()?;
        let (q, k, v) = (views[0], views[1], views[2]);

        let [batch_size, _, _, head_dim] = q.shape();
        let [bk, num_kv_heads, kv_len, hd_k] = k.shape();
        ensure!(batch_size == bk, "Batch mismatch between Q and K");
        ensure!(head_dim == hd_k, "Head dim mismatch between Q and K");
        ensure!(num_kv_heads == v.shape()[1], "K/V head mismatch");
        ensure!(kv_len == v.shape()[2] && head_dim == v.shape()[3], "K/V shape mismatch");

        let mut out =
            self.flash_attention_gqa_impl(q, k, v, views.get(3).copied(), false)?.into_dyn();
        if is_3d_case {
            out.index_axis_inplace(tract_ndarray::Axis(1), 0);
        }

        Ok(tvec!(out.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

impl FlashSdpaOp {
    /// Flash Attention forward with Grouped-Query Attention (GQA).
    ///
    /// Shapes:
    ///   Q: (batch_size, num_q_heads, query_len, head_dim)
    ///   K: (batch_size, num_kv_heads, kv_len,   head_dim)
    ///   V: (batch_size, num_kv_heads, kv_len,   head_dim)
    /// Returns O: (batch_size, num_q_heads, query_len, head_dim)
    ///
    /// GQA mapping: each query head qh maps to a key/value head index kh = qh / group_size
    /// where group_size = num_q_heads / num_kv_heads (must divide exactly).
    pub fn flash_attention_gqa(
        &self,
        q: ArrayView4<f32>,
        k: ArrayView4<f32>,
        v: ArrayView4<f32>,
        mask: Option<ArrayView4<f32>>,
    ) -> TractResult<Array4<f32>> {
        self.flash_attention_gqa_impl(
            FloatView::F32(q),
            FloatView::F32(k),
            FloatView::F32(v),
            mask.map(FloatView::F32),
            false,
        )
    }

    fn flash_attention_gqa_impl(
        &self,
        q: FloatView,
        k: FloatView,
        v: FloatView,
        mask: Option<FloatView>,
        pv_strided: bool,
    ) -> TractResult<Array4<f32>> {
        let [batch_size, num_q_heads, q_len, head_dim] = q.shape();
        let num_kv_heads = k.shape()[1];
        let group_size = num_q_heads / num_kv_heads;

        let mut out = Array4::<f32>::zeros((batch_size, num_q_heads, q_len, head_dim));

        // One independent task per (batch, q-head). Heads share only read-only Q/K/V
        // and write disjoint output slices, so this is embarrassingly parallel.
        let tasks: Vec<(usize, usize)> =
            (0..batch_size).flat_map(|b| (0..num_q_heads).map(move |qh| (b, qh))).collect();
        let exp = Func::Exp.ew_f32()?;
        let compute = |&(b, qh): &(usize, usize)| {
            let kvh = qh / group_size;
            let (qs, ks, vs) = (q.head(b, qh), k.head(b, kvh), v.head(b, kvh));
            let mask = mask.as_ref().map(|m| m.head(b, qh));
            let mask = mask.as_ref().map(|m| m.view());
            self.attend_one_head(qs.view(), ks.view(), vs.view(), mask, &*exp, pv_strided)
        };

        let results: Vec<Array2<f32>> = if Self::should_thread(tasks.len()) {
            #[cfg(not(target_family = "wasm"))]
            {
                use rayon::prelude::*;
                tasks.par_iter().map(compute).collect::<TractResult<_>>()?
            }
            #[cfg(target_family = "wasm")]
            {
                tasks.iter().map(compute).collect::<TractResult<_>>()?
            }
        } else {
            tasks.iter().map(compute).collect::<TractResult<_>>()?
        };

        for (&(b, qh), head_out) in tasks.iter().zip(results) {
            out.slice_mut(s!(b, qh, .., ..)).assign(&head_out);
        }
        Ok(out)
    }

    /// Whether to spread the per-head tasks across cores (rayon's global pool).
    /// Off for a single task, on wasm (no thread spawn), or when `TRACT_FLASH_SDPA_ST` is set.
    fn should_thread(num_tasks: usize) -> bool {
        num_tasks > 1 && cfg!(not(target_family = "wasm")) && !TRACT_FLASH_SDPA_ST.get()
    }

    /// Flash-attention for a single (batch, q-head) pair, given that head's
    /// `[q_len, head_dim]` Q, `[kv_len, head_dim]` K and V, and `[q_len, kv_len]`
    /// mask. Returns O of shape `[q_len, head_dim]`. Pure function of its read-only
    /// views, so it is safe to call concurrently for distinct heads.
    fn attend_one_head(
        &self,
        q: ArrayView2<f32>,
        k: ArrayView2<f32>,
        v: ArrayView2<f32>,
        mask: Option<ArrayView2<f32>>,
        exp: &dyn ElementWise<f32>,
        pv_strided: bool,
    ) -> TractResult<Array2<f32>> {
        let (q_len, head_dim) = q.dim();
        let kv_len = k.dim().0;
        let scale = self.scale.unwrap_or((head_dim as f32).recip().sqrt());

        let block_kv_len = if kv_len <= ONE_TILE_MAX_KV_LEN { kv_len.max(1) } else { 32 };
        let block_q_len = 32;

        let mut out = Array2::<f32>::zeros((q_len, head_dim));
        let mut l = vec![0f32; q_len];
        let mut m = vec![f32::NEG_INFINITY; q_len];
        for kbix in 0..kv_len.div_ceil(block_kv_len) {
            for qbix in 0..q_len.div_ceil(block_q_len) {
                let kv_range = (kbix * block_kv_len)..((kbix + 1) * block_kv_len).min(kv_len);
                let q_range = (qbix * block_q_len)..((qbix + 1) * block_q_len).min(q_len);
                if let Some(mask) = mask
                    && mask
                        .slice(s!(q_range.clone(), kv_range.clone()))
                        .iter()
                        .all(|x| *x < -65503.0)
                {
                    continue;
                }
                let m = &mut m[q_range.clone()];
                let l = &mut l[q_range.clone()];
                let qblock: ArrayView2<f32> = q.slice(s!(q_range.clone(), ..));
                let kblock: ArrayView2<f32> = k.slice(s!(kv_range.clone(), ..));
                let vblock: ArrayView2<f32> = v.slice(s!(kv_range.clone(), ..));
                let mut oblock: ArrayViewMut2<f32> = out.slice_mut(s!(q_range.clone(), ..));
                // Sij <- QiKTj
                let mut s = qblock.dot(&kblock.t()) * scale;
                if let Some(mask) = mask {
                    s += &mask.slice(s!(q_range.clone(), kv_range.clone()));
                } else if self.causal {
                    let mask =
                        Array2::from_elem((q_range.len(), kv_range.len()), f32::NEG_INFINITY);
                    let mask = mask.triu(
                        q_range.start as isize - kv_range.start as isize - q_len as isize
                            + kv_len as isize
                            + 1,
                    );
                    s += &mask;
                };
                let tile_m: Vec<f32> = s
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().copied().map(float_ord::FloatOrd).max().unwrap().0)
                    .collect_vec();
                for (row_ix, max) in tile_m.iter().enumerate() {
                    if max.is_finite() {
                        s.row_mut(row_ix).iter_mut().for_each(|x| *x -= max);
                    }
                }
                // Sij <- exp(Sij * scale - max_of_row)
                exp.run(s.as_slice_mut().context("score tile is contiguous")?)?;
                let tile_l = s.sum_axis(tract_ndarray::Axis(1)).insert_axis(tract_ndarray::Axis(1));
                // m_new = max(maxes, row_maxs)
                let m_new = (0..q_range.len()).map(|i| m[i].max(tile_m[i])).collect_vec();
                // l_new = exp(m[i] - m_new[i]) * l[i] + exp(tile_m[i] - m_new[i]) * tile_l[i]
                let l_new = (0..q_range.len())
                    .map(|i| {
                        (m[i] - m_new[i]).exp() * l[i]
                            + (tile_m[i] - m_new[i]).exp() * tile_l[(i, 0)]
                    })
                    .collect_vec();
                // P·V as one contiguous tile GEMM (s: [q,kv] · vblock: [kv,d]) instead of
                // `head_dim` strided column dots; then rescale each row with the online-softmax
                // factors. `pv_strided` keeps the old per-column path for A/B benching only.
                let sv_tile = if pv_strided { None } else { Some(s.dot(&vblock)) };
                for i in 0..q_range.len() {
                    let r_l_new = l_new[i].recip();
                    let mul_o = ((m[i] - m_new[i]).exp()) * l[i] * r_l_new;
                    let mul_sv = ((tile_m[i] - m_new[i]).exp()) * r_l_new;
                    if let Some(sv_tile) = &sv_tile {
                        for j in 0..head_dim {
                            oblock[(i, j)] = oblock[(i, j)] * mul_o + sv_tile[(i, j)] * mul_sv;
                        }
                    } else {
                        for j in 0..head_dim {
                            let sv = s.row(i).dot(&vblock.column(j));
                            oblock[(i, j)] = oblock[(i, j)] * mul_o + sv * mul_sv;
                        }
                    }
                }
                l.copy_from_slice(&l_new);
                m.copy_from_slice(&m_new);
            }
        }
        Ok(out)
    }
}

/// A 4-D Q, K, V or mask operand, in f32 or still in f16: each head's trailing
/// 2-D slice is widened by the task attending that head, so an f16 operand is
/// never converted whole. A rank-3 operand gets a head axis of 1; batch and
/// head axes of length 1 broadcast.
#[derive(Clone, Copy)]
enum FloatView<'a> {
    F32(ArrayView4<'a, f32>),
    F16(ArrayView4<'a, f16>),
}

impl<'a> FloatView<'a> {
    fn of(t: &'a Tensor) -> TractResult<Self> {
        let s = t.shape();
        let shape = match *s {
            [b, x, y] => (b, 1, x, y),
            [b, h, x, y] => (b, h, x, y),
            _ => bail!("FlashSDPA operands have rank 3 or 4, got {s:?}"),
        };
        Ok(if t.datum_type() == f16::datum_type() {
            FloatView::F16(t.to_plain_array_view::<f16>()?.into_shape_with_order(shape)?)
        } else {
            FloatView::F32(t.to_plain_array_view::<f32>()?.into_shape_with_order(shape)?)
        })
    }

    fn shape(&self) -> [usize; 4] {
        let s = match self {
            FloatView::F32(t) => t.shape(),
            FloatView::F16(t) => t.shape(),
        };
        [s[0], s[1], s[2], s[3]]
    }

    fn head(&self, b: usize, h: usize) -> CowArray<'a, f32, Ix2> {
        let [nb, nh, _, _] = self.shape();
        let (b, h) = (b.min(nb - 1), h.min(nh - 1));
        match self {
            FloatView::F32(t) => t.slice_move(s!(b, h, .., ..)).into(),
            FloatView::F16(t) => {
                let head = t.slice_move(s!(b, h, .., ..));
                let mut wide = Array2::<f32>::zeros(head.dim());
                match (head.as_slice(), wide.as_slice_mut()) {
                    (Some(src), Some(dst)) => src.convert_to_f32_slice(dst),
                    _ => wide.zip_mut_with(&head, |w, x| *w = x.to_f32()),
                }
                wide.into()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::Array4;

    fn rng(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn flash_sdpa_pv_matches_naive() -> TractResult<()> {
        let (b, h, sq, sk, d) = (1usize, 2, 16, 24, 8); // non-square, multi-head
        let scale = 1.0f32 / (d as f32).sqrt();
        let qv = rng(b * h * sq * d, 1);
        let kv = rng(b * h * sk * d, 2);
        let vv = rng(b * h * sk * d, 3);
        let q = Array4::from_shape_vec((b, h, sq, d), qv.clone())?;
        let k = Array4::from_shape_vec((b, h, sk, d), kv.clone())?;
        let v = Array4::from_shape_vec((b, h, sk, d), vv.clone())?;
        let op = FlashSdpaOp { causal: false, scale: None };
        let got = op.flash_attention_gqa(q.view(), k.view(), v.view(), None)?;
        let gv: Vec<f32> = got.iter().copied().collect();
        let mut want = vec![0f32; b * h * sq * d];
        for bh in 0..b * h {
            for i in 0..sq {
                let mut sc = vec![0f32; sk];
                for j in 0..sk {
                    let mut a = 0f32;
                    for dd in 0..d {
                        a += qv[(bh * sq + i) * d + dd] * kv[(bh * sk + j) * d + dd];
                    }
                    sc[j] = a * scale;
                }
                let m = sc.iter().copied().fold(f32::MIN, f32::max);
                let mut sum = 0f32;
                for x in sc.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for x in sc.iter_mut() {
                    *x /= sum;
                }
                for e in 0..d {
                    let mut a = 0f32;
                    for j in 0..sk {
                        a += sc[j] * vv[(bh * sk + j) * d + e];
                    }
                    want[(bh * sq + i) * d + e] = a;
                }
            }
        }
        let max_abs = gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
        println!("flash_sdpa PV-gemm max_abs={max_abs:.6}");
        ensure!(max_abs < 1e-5, "flash_sdpa PV mismatch: {max_abs}");
        Ok(())
    }

    #[test]
    fn flash_sdpa_eval_f16_matches_f32() -> TractResult<()> {
        let (b, hq, hkv, sq, sk, d) = (1usize, 4, 2, 1, 33, 8);
        let qv = rng(b * hq * sq * d, 4);
        let kv = rng(b * hkv * sk * d, 5);
        let vv = rng(b * hkv * sk * d, 6);
        let q = Tensor::from_shape(&[b, hq, sq, d], &qv)?;
        let k = Tensor::from_shape(&[b, hkv, sk, d], &kv)?;
        let v = Tensor::from_shape(&[b, hkv, sk, d], &vv)?;
        let op = FlashSdpaOp { causal: true, scale: None };
        let f32_out = op
            .eval(
                &EvalContext::out_of_plan(),
                tvec!(q.clone().into_tvalue(), k.clone().into_tvalue(), v.clone().into_tvalue()),
            )?
            .remove(0);
        let (q16, k16, v16) = (
            q.cast_to::<f16>()?.into_owned(),
            k.cast_to::<f16>()?.into_owned(),
            v.cast_to::<f16>()?.into_owned(),
        );
        let f16_out = op
            .eval(
                &EvalContext::out_of_plan(),
                tvec!(q16.into_tvalue(), k16.into_tvalue(), v16.into_tvalue()),
            )?
            .remove(0);
        ensure!(f16_out.datum_type() == f16::datum_type());
        let f16_as_f32 = f16_out.cast_to::<f32>()?;
        let got = f32_out.to_plain_array_view::<f32>()?;
        let via_f16 = f16_as_f32.to_plain_array_view::<f32>()?;
        let max_abs =
            got.iter().zip(via_f16.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        ensure!(max_abs < 5e-3, "f16 flash eval diverges from f32: {max_abs}");
        Ok(())
    }

    #[test]
    fn flash_sdpa_f16_mask_matches_f32_mask() -> TractResult<()> {
        let (b, h, sq, sk, d) = (2usize, 3, 40, 37, 8);
        let q = Tensor::from_shape(&[b, h, sq, d], &rng(b * h * sq * d, 7))?;
        let k = Tensor::from_shape(&[b, h, sk, d], &rng(b * h * sk * d, 8))?;
        let v = Tensor::from_shape(&[b, h, sk, d], &rng(b * h * sk * d, 9))?;
        let mask16 = Tensor::from_shape(&[1, h, sq, sk], &rng(h * sq * sk, 10))?
            .cast_to::<f16>()?
            .into_owned();
        let mask32 = mask16.cast_to::<f32>()?.into_owned();
        let op = FlashSdpaOp { causal: false, scale: None };
        let run = |mask: &Tensor| -> TractResult<Tensor> {
            Ok(op
                .eval(
                    &EvalContext::out_of_plan(),
                    tvec!(
                        q.clone().into_tvalue(),
                        k.clone().into_tvalue(),
                        v.clone().into_tvalue(),
                        mask.clone().into_tvalue()
                    ),
                )?
                .remove(0)
                .into_tensor())
        };
        run(&mask32)?.close_enough(&run(&mask16)?, Approximation::Exact)
    }

    // A/B the P·V step (tile GEMM vs strided per-column dot) across the microbench
    // shape AND the real Llama-3.2-1B PP512 prefill shape (causal, GQA 32q/8kv).
    //   cargo test -p tract-transformers --release bench_flash_sdpa_pv -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_flash_sdpa_pv() -> TractResult<()> {
        use std::time::Instant;
        // (label, hq, hkv, sq, sk, d, causal)
        let shapes = [
            (
                "microbench    non-causal Hq8  Hkv8 S512 D64",
                8usize,
                8usize,
                512usize,
                512usize,
                64usize,
                false,
            ),
            ("llama-3.2-1B  causal     Hq32 Hkv8 S512 D64", 32, 8, 512, 512, 64, true),
        ];
        for (label, hq, hkv, sq, sk, d, causal) in shapes {
            let q = Array4::from_shape_vec((1, hq, sq, d), rng(hq * sq * d, 1))?;
            let k = Array4::from_shape_vec((1, hkv, sk, d), rng(hkv * sk * d, 2))?;
            let v = Array4::from_shape_vec((1, hkv, sk, d), rng(hkv * sk * d, 3))?;
            let op = FlashSdpaOp { causal, scale: None };
            let bench = |strided: bool| -> f64 {
                for _ in 0..2 {
                    std::hint::black_box(
                        op.flash_attention_gqa_impl(
                            FloatView::F32(q.view()),
                            FloatView::F32(k.view()),
                            FloatView::F32(v.view()),
                            None,
                            strided,
                        )
                        .unwrap(),
                    );
                }
                let mut best = f64::MAX;
                for _ in 0..8 {
                    let t = Instant::now();
                    for _ in 0..3 {
                        std::hint::black_box(
                            op.flash_attention_gqa_impl(
                                FloatView::F32(q.view()),
                                FloatView::F32(k.view()),
                                FloatView::F32(v.view()),
                                None,
                                strided,
                            )
                            .unwrap(),
                        );
                    }
                    best = best.min(t.elapsed().as_secs_f64() / 3.0);
                }
                best * 1e3
            };
            let strided = bench(true);
            let gemm = bench(false);
            println!(
                "{label}: strided {strided:7.3} ms -> gemm {gemm:7.3} ms  ({:.2}x)  [per-layer; x16 layers = {:.1} -> {:.1} ms total SDPA]",
                strided / gemm,
                strided * 16.0,
                gemm * 16.0,
            );
        }
        Ok(())
    }
}
