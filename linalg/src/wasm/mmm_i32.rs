#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::Scaler;
use crate::mmm::FusedKerSpec;

/// Applies every non-`AddMatMul` fused op to a 4-row i32 tile of `NR` columns, `ab[row][col]`,
/// on the bit-exact scalar quant path (q_scale/q_shr/q_shl). O(4·NR) per tile, negligible
/// against the K loop. Returns `Some(status)` when the kernel must stop (`Done`, or an
/// unsupported tile with 1), `None` to go on with the next op; `AddMatMul` is left to the
/// kernel and answers `None` untouched.
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn fused_i32_4xn<const NR: usize>(
    ab: &mut [[i32; NR]; 4],
    spec: &FusedKerSpec<i32>,
) -> Option<isize> {
    use crate::ScaleShiftAndRound;
    unsafe {
        match *spec {
            FusedKerSpec::Done => return Some(0),
            FusedKerSpec::Clear => *ab = [[0i32; NR]; 4],
            FusedKerSpec::LoadTile(col_major, _row_major) => {
                for row in 0..4 {
                    for col in 0..NR {
                        ab[row][col] = *col_major.add(col * 4 + row);
                    }
                }
            }
            FusedKerSpec::ScalarAdd(a) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] += a;
                    }
                }
            }
            FusedKerSpec::ScalarMul(a) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] *= a;
                    }
                }
            }
            FusedKerSpec::ScalarMin(m) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].min(m);
                    }
                }
            }
            FusedKerSpec::ScalarMax(m) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].max(m);
                    }
                }
            }
            FusedKerSpec::ScalarSub(m) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = m - ab[i][j];
                    }
                }
            }
            FusedKerSpec::ScalarSubF(m) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] -= m;
                    }
                }
            }
            FusedKerSpec::LeakyRelu(a) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = if ab[i][j] > 0 { ab[i][j] } else { a * ab[i][j] };
                    }
                }
            }
            FusedKerSpec::PerRowMin(m) => {
                for i in 0..4 {
                    let v = *m.add(i);
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].min(v);
                    }
                }
            }
            FusedKerSpec::PerRowMax(m) => {
                for i in 0..4 {
                    let v = *m.add(i);
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].max(v);
                    }
                }
            }
            FusedKerSpec::PerRowAdd(m) => {
                for i in 0..4 {
                    let v = *m.add(i);
                    for j in 0..NR {
                        ab[i][j] += v;
                    }
                }
            }
            FusedKerSpec::PerRowMul(m) => {
                for i in 0..4 {
                    let v = *m.add(i);
                    for j in 0..NR {
                        ab[i][j] *= v;
                    }
                }
            }
            FusedKerSpec::PerRowSub(m) => {
                for i in 0..4 {
                    let v = *m.add(i);
                    for j in 0..NR {
                        ab[i][j] = v - ab[i][j];
                    }
                }
            }
            FusedKerSpec::PerRowSubF(m) => {
                for i in 0..4 {
                    let v = *m.add(i);
                    for j in 0..NR {
                        ab[i][j] -= v;
                    }
                }
            }
            FusedKerSpec::PerColMin(m) => {
                let c = std::slice::from_raw_parts(m, NR);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].min(c[j]);
                    }
                }
            }
            FusedKerSpec::PerColMax(m) => {
                let c = std::slice::from_raw_parts(m, NR);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].max(c[j]);
                    }
                }
            }
            FusedKerSpec::PerColAdd(m) => {
                let c = std::slice::from_raw_parts(m, NR);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] += c[j];
                    }
                }
            }
            FusedKerSpec::PerColMul(m) => {
                let c = std::slice::from_raw_parts(m, NR);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] *= c[j];
                    }
                }
            }
            FusedKerSpec::PerColSub(m) => {
                let c = std::slice::from_raw_parts(m, NR);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = c[j] - ab[i][j];
                    }
                }
            }
            FusedKerSpec::PerColSubF(m) => {
                let c = std::slice::from_raw_parts(m, NR);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] -= c[j];
                    }
                }
            }
            FusedKerSpec::AddRowColProducts(rows, cols) => {
                for i in 0..4 {
                    let r = *rows.add(i);
                    for j in 0..NR {
                        ab[i][j] += r * *cols.add(j);
                    }
                }
            }
            FusedKerSpec::AddUnicast(other) => {
                for i in 0..4 {
                    for j in 0..NR {
                        let p = other.ptr.offset(
                            other.row_byte_stride * i as isize + other.col_byte_stride * j as isize,
                        );
                        let v = match other.item_size {
                            1 => *(p as *const i8) as i32,
                            4 => *(p as *const i32),
                            _ => return Some(1),
                        };
                        ab[i][j] += v;
                    }
                }
            }
            FusedKerSpec::ShiftLeft(shift) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].q_shl(shift);
                    }
                }
            }
            FusedKerSpec::RoundingShiftRight(shift, rp) => {
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].q_shr(shift, rp);
                    }
                }
            }
            FusedKerSpec::QScale(shift, rp, mult) => {
                let s = Scaler::from_fuse_params(shift, rp, mult);
                for i in 0..4 {
                    for j in 0..NR {
                        ab[i][j] = ab[i][j].q_scale(s);
                    }
                }
            }
            FusedKerSpec::AddMatMul { .. } => {}
            FusedKerSpec::Store(tile) => match tile.item_size {
                1 => {
                    for i in 0..4 {
                        for j in 0..NR {
                            let loc = tile.ptr.offset(
                                tile.row_byte_stride * i as isize
                                    + tile.col_byte_stride * j as isize,
                            ) as *mut u8;
                            *loc = ab[i][j] as u8;
                        }
                    }
                }
                4 => {
                    for i in 0..4 {
                        for j in 0..NR {
                            let loc = tile.ptr.offset(
                                tile.row_byte_stride * i as isize
                                    + tile.col_byte_stride * j as isize,
                            ) as *mut i32;
                            *loc = ab[i][j];
                        }
                    }
                }
                _ => return Some(1),
            },
        }
    }
    None
}

// Wasm SIMD int8 -> i32 matmul kernel (4x4). Under +relaxed-simd the AddMatMul
// K-loop uses i32x4.relaxed_dot_i8x16_i7x16_add with B sign-split so both dot
// operands stay in the i7 range where the instruction is deterministic (see the
// kernel body); without relaxed-simd it uses widening i8->i32 + i32x4 mul/add
// (an extmul/SMLAL-style outer product). The quant epilogue + fuse ops reuse
// the bit-exact scalar path of `fused_i32_4xn`. Both paths are bit-identical to
// generic_i32_4x4; selected for i8 matmul because it is written for wasm and the generic
// scalar kernel is not.
#[inline(never)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn kernel_i32_4x4(mut pnl: *const FusedKerSpec<i32>) -> isize {
    use std::arch::wasm32::*;
    unsafe {
        let mut ab = [[0i32; 4]; 4];
        while !pnl.is_null() {
            if let FusedKerSpec::AddMatMul { k, pa, pb, packing } = *pnl {
                if packing == 1 {
                    let a = pa as *const i8;
                    let b = pb as *const i8;
                    let mut acc = [
                        v128_load(ab[0].as_ptr() as *const v128),
                        v128_load(ab[1].as_ptr() as *const v128),
                        v128_load(ab[2].as_ptr() as *const v128),
                        v128_load(ab[3].as_ptr() as *const v128),
                    ];
                    // PackedI8K4 (K=4-inner): per 4-K block, one B v128 load is
                    // shared across the 4 rows; each row broadcasts its 4 K bytes
                    // (a[kb*16 + m*4 ..]) against b[kb*16 + n*4 + kr]. Tail K (k%4)
                    // is zero-padded by the packer.
                    //
                    // relaxed_dot's 2nd operand is i7: lanes outside [0, 127] are
                    // read as signed by some engines (ARM SDOT) and unsigned by
                    // others (x86 pmaddubsw/vpdpbusd). B is full i8, so it is
                    // sign-split as b = (b & 0x7f) - ((b >> 7) << 7); both parts
                    // are i7-safe (which also keeps every pmaddubsw i16 pair-sum
                    // below saturation), each feeds its own dot, and the sign
                    // planes fold back after the K-loop as acc -= acc_hi << 7.
                    // Bit-exact on every engine.
                    #[cfg(target_feature = "relaxed-simd")]
                    {
                        let mut acc_hi = [i32x4_splat(0); 4];
                        for kb in 0..k.div_ceil(4) {
                            let b_all = v128_load(b.add(kb * 16) as *const v128);
                            let b_lo = v128_and(b_all, u8x16_splat(0x7f));
                            let b_hi = u8x16_shr(b_all, 7);
                            for (m, (acc_m, hi_m)) in
                                acc.iter_mut().zip(acc_hi.iter_mut()).enumerate()
                            {
                                let a4 = (a.add(kb * 16 + m * 4) as *const i32).read_unaligned();
                                let a4 = i32x4_splat(a4);
                                *acc_m = i32x4_relaxed_dot_i8x16_i7x16_add(a4, b_lo, *acc_m);
                                *hi_m = i32x4_relaxed_dot_i8x16_i7x16_add(a4, b_hi, *hi_m);
                            }
                        }
                        for (acc_m, hi_m) in acc.iter_mut().zip(acc_hi) {
                            *acc_m = i32x4_sub(*acc_m, i32x4_shl(hi_m, 7));
                        }
                    }
                    // Deterministic fallback (no relaxed-simd): standard PackedFormat
                    // K-major (4 i8 per k), widening outer-product accumulate.
                    #[cfg(not(target_feature = "relaxed-simd"))]
                    for ik in 0..k {
                        let bw = v128_load32_zero(b.add(4 * ik) as *const u32);
                        let bw = i16x8_extend_low_i8x16(bw);
                        let bw = i32x4_extend_low_i16x8(bw);
                        let ar = a.add(4 * ik);
                        acc[0] = i32x4_add(acc[0], i32x4_mul(i32x4_splat(*ar.add(0) as i32), bw));
                        acc[1] = i32x4_add(acc[1], i32x4_mul(i32x4_splat(*ar.add(1) as i32), bw));
                        acc[2] = i32x4_add(acc[2], i32x4_mul(i32x4_splat(*ar.add(2) as i32), bw));
                        acc[3] = i32x4_add(acc[3], i32x4_mul(i32x4_splat(*ar.add(3) as i32), bw));
                    }
                    v128_store(ab[0].as_mut_ptr() as *mut v128, acc[0]);
                    v128_store(ab[1].as_mut_ptr() as *mut v128, acc[1]);
                    v128_store(ab[2].as_mut_ptr() as *mut v128, acc[2]);
                    v128_store(ab[3].as_mut_ptr() as *mut v128, acc[3]);
                } else if packing == 0 {
                    // i32 x i32, K-major (scalar; rare path).
                    let a = pa as *const i32;
                    let b = pb as *const i32;
                    for ik in 0..k {
                        for i in 0..4 {
                            let av = *a.add(4 * ik + i);
                            for j in 0..4 {
                                ab[i][j] += av * *b.add(4 * ik + j);
                            }
                        }
                    }
                } else {
                    return 1;
                }
            } else if let Some(status) = fused_i32_4xn(&mut ab, &*pnl) {
                return status;
            }
            pnl = pnl.add(1);
        }
    }
    0
}

/// Wasm SIMD int8 -> i32 matrix-vector kernel (4x1), the n = 1 companion of `wasm_i32_4x4`.
/// Its A packing is the 4x4's, so a matmul with a symbolic n packs its weights once and
/// core pairs the two kernels, this one taking every n = 1 call. Under +relaxed-simd each
/// 4-K block of A (I8K4, 4 rows × 4 K) is one `relaxed_dot` against the 4 B bytes, B
/// sign-split as in the 4x4. Without it A is K-major (4 rows per k): a shuffle brings each
/// row's K pairs together and `i32x4_dot_i16x8` multiplies them exactly in i16. Integer
/// accumulation is exact, so both paths are bit-identical to generic_i32_4x1.
#[inline(never)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn kernel_i32_4x1(mut pnl: *const FusedKerSpec<i32>) -> isize {
    use std::arch::wasm32::*;
    unsafe {
        let mut ab = [[0i32; 1]; 4];
        while !pnl.is_null() {
            if let FusedKerSpec::AddMatMul { k, pa, pb, packing } = *pnl {
                if packing == 1 {
                    let a = pa as *const i8;
                    let b = pb as *const i8;
                    let mut acc = v128_load(ab.as_ptr() as *const v128);
                    #[cfg(target_feature = "relaxed-simd")]
                    {
                        let mut acc_hi = i32x4_splat(0);
                        for kb in 0..k.div_ceil(4) {
                            let a16 = v128_load(a.add(kb * 16) as *const v128);
                            let b4 = i32x4_splat((b.add(kb * 4) as *const i32).read_unaligned());
                            let b_lo = v128_and(b4, u8x16_splat(0x7f));
                            let b_hi = u8x16_shr(b4, 7);
                            acc = i32x4_relaxed_dot_i8x16_i7x16_add(a16, b_lo, acc);
                            acc_hi = i32x4_relaxed_dot_i8x16_i7x16_add(a16, b_hi, acc_hi);
                        }
                        acc = i32x4_sub(acc, i32x4_shl(acc_hi, 7));
                        v128_store(ab.as_mut_ptr() as *mut v128, acc);
                    }
                    #[cfg(not(target_feature = "relaxed-simd"))]
                    {
                        let mut acc_23 = i32x4_splat(0);
                        for kb in 0..k / 4 {
                            let a16 = v128_load(a.add(kb * 16) as *const v128);
                            let a16 = i8x16_shuffle::<
                                0,
                                4,
                                1,
                                5,
                                2,
                                6,
                                3,
                                7,
                                8,
                                12,
                                9,
                                13,
                                10,
                                14,
                                11,
                                15,
                            >(a16, a16);
                            let b4 = v128_load32_zero(b.add(kb * 4) as *const u32);
                            let b4 = i16x8_extend_low_i8x16(b4);
                            let b01 = i32x4_shuffle::<0, 0, 0, 0>(b4, b4);
                            let b23 = i32x4_shuffle::<1, 1, 1, 1>(b4, b4);
                            acc = i32x4_add(acc, i32x4_dot_i16x8(i16x8_extend_low_i8x16(a16), b01));
                            acc_23 = i32x4_add(
                                acc_23,
                                i32x4_dot_i16x8(i16x8_extend_high_i8x16(a16), b23),
                            );
                        }
                        v128_store(ab.as_mut_ptr() as *mut v128, i32x4_add(acc, acc_23));
                        for ik in k / 4 * 4..k {
                            let bk = *b.add(ik) as i32;
                            for (i, row) in ab.iter_mut().enumerate() {
                                row[0] += *a.add(4 * ik + i) as i32 * bk;
                            }
                        }
                    }
                } else if packing == 0 {
                    let a = pa as *const i32;
                    let b = pb as *const i32;
                    for ik in 0..k {
                        let bk = *b.add(ik);
                        for (i, row) in ab.iter_mut().enumerate() {
                            row[0] += *a.add(4 * ik + i) * bk;
                        }
                    }
                } else {
                    return 1;
                }
            } else if let Some(status) = fused_i32_4xn(&mut ab, &*pnl) {
                return status;
            }
            pnl = pnl.add(1);
        }
    }
    0
}

// i8i8 packing for the wasm i32 kernels, `r` rows or columns per panel. Under
// +relaxed-simd the kernels use the `i32x4_relaxed_dot_i8x16_i7x16_add` SDOT-analog,
// which wants 4 contiguous K per mn-lane → PackedI8K4 (K=4-inner). Without
// relaxed-simd they want K-major → standard PackedFormat. Both are picked at
// compile time so the kernels' AddMatMul and the packer always agree.
#[cfg(target_feature = "relaxed-simd")]
fn wasm_i8_packing(r: usize) -> impl crate::mmm::MMMInputFormat {
    crate::pack::PackedI8K4::new(r)
}
#[cfg(not(target_feature = "relaxed-simd"))]
fn wasm_i8_packing(r: usize) -> impl crate::mmm::MMMInputFormat {
    use crate::pack::Packing;
    i8::packing(r)
}

bail_stub!(wasm32; unsafe fn kernel_i32_4x4(*const FusedKerSpec<i32>) -> isize);
bail_stub!(wasm32; unsafe fn kernel_i32_4x1(*const FusedKerSpec<i32>) -> isize);

MMMRustKernel!(wasm32; kernel_i32_4x4 => wasm_i32_4x4<i32>(4,4)
    packing[1] = i8i8 => |k| k.with_packing(wasm_i8_packing(4), wasm_i8_packing(4));

    store(i8)
);

MMMRustKernel!(wasm32; kernel_i32_4x1 => wasm_i32_4x1<i32>(4,1)
    packing[1] = i8i8 => |k| k.with_packing(wasm_i8_packing(4), wasm_i8_packing(1));

    store(i8)
);
