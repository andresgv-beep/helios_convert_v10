//! HQS v4 (`hq44k_g16`): bloques afines de 16 pesos con escala y mínimo por grupo.
//!
//! Superbloque de 256 pesos, 152 bytes:
//! - `[0..2]` d y `[2..4]` dmin, FP16 little-endian.
//! - `[4..24]` flujo de bits little-endian: escala del grupo g (5 bits) en el bit 5g y
//!   mínimo del grupo g (5 bits) en el bit 80 + 5g.
//! - `[24..152]` códigos de 4 bits sin signo; el peso k está en el byte k/2, nibble bajo
//!   si k es par.
//!
//! Valor: d·escala[g]·código − dmin·mínimo[g], con g = k/16. Mismo diseño que
//! `kernels/hqs_v4_gemv.cuh` en Hector-engine.
//!
//! Codificador: error ponderado por imp·√(σ²+x²) (σ² = 2·media(x²) del superbloque;
//! `imp` es la importancia por columna, 1 si no hay), 37 escalas candidatas por grupo con
//! mínimos cuadrados ponderados, escalas y mínimos a 5 bits con base FP16 buscada entre 13
//! factores y un refinado local ±1. Validado en Gemma 4 12B el 28/09/2026: MLP en v4 con la
//! atención en HQ53 da KL 0,513/0,555 frente a 0,728/0,708 del estable (repo hqs_v4).
use anyhow::{ensure, Result};
use half::f16;
use rayon::prelude::*;

pub const DTYPE: &str = "hq44k_g16";
pub const BLOCK_BYTES: usize = 152;
const HEADER_BYTES: usize = 24;
const GROUP: usize = 16;
const GROUPS: usize = 256 / GROUP;
const NMAX: f64 = 15.0;
const LEVELS: f64 = 31.0;

fn fp16(v: f64) -> f64 { f16::from_f64(v).to_f64() }

/// Mejor (escala, mínimo ≥ 0) de un grupo para x ≈ a·q − m, con error ponderado.
fn fit_group(x: &[f64], w: &[f64]) -> (f64, f64) {
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (xmax - xmin).max(1e-12);
    let (mut best, mut ba, mut bm) = (f64::INFINITY, span / NMAX, -xmin);
    let mut q = [0f64; GROUP];
    for k in -18..=18 {
        let iscale = (NMAX + 0.05 * k as f64) / span;
        let (mut sw, mut sq, mut sqq, mut sx, mut sqx) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..GROUP {
            q[i] = ((x[i] - xmin) * iscale).round_ties_even().clamp(0.0, NMAX);
            sw += w[i]; sq += w[i] * q[i]; sqq += w[i] * q[i] * q[i]; sx += w[i] * x[i]; sqx += w[i] * q[i] * x[i];
        }
        let det = sw * sqq - sq * sq;
        let (mut a, mut m) = if det > 1e-20 { ((sw * sqx - sq * sx) / det, (sq * sqx - sqq * sx) / det) } else { (span / NMAX, -xmin) };
        m = m.max(0.0); a = a.max(1e-12);
        let err: f64 = (0..GROUP).map(|i| { let e = a * q[i] - m - x[i]; w[i] * e * e }).sum();
        if err < best { best = err; ba = a; bm = m; }
    }
    (ba, bm)
}

/// Enteros [0, 31] con una escala base FP16, búsqueda ponderada.
fn quant_levels(v: &[f64; GROUPS], sw: &[f64; GROUPS]) -> (f64, [f64; GROUPS]) {
    let top = v.iter().cloned().fold(0.0, f64::max).max(1e-12);
    let (mut best, mut bd) = (f64::INFINITY, fp16(top / LEVELS));
    for s in 0..13 {
        let d = fp16(top / LEVELS * (0.85 + 0.025 * s as f64));
        if d <= 0.0 { continue; }
        let err: f64 = (0..GROUPS).map(|g| { let l = (v[g] / d).round_ties_even().clamp(0.0, LEVELS); let e = l * d - v[g]; sw[g] * e * e }).sum();
        if err < best { best = err; bd = d; }
    }
    let mut l = [0f64; GROUPS];
    if bd > 0.0 { for g in 0..GROUPS { l[g] = (v[g] / bd).round_ties_even().clamp(0.0, LEVELS); } }
    (bd, l)
}

fn put5(dst: &mut [u8], bitpos: usize, value: u32) {
    let v = value << (bitpos % 8);
    dst[bitpos / 8] |= v as u8;
    if bitpos % 8 + 5 > 8 { dst[bitpos / 8 + 1] |= (v >> 8) as u8; }
}

fn encode_block(x: &[f32], h: &[f32], out: &mut [u8]) {
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f64>() / 256.0;
    let w: Vec<f64> = (0..256).map(|i| h[i] as f64 * (sigma2 + x[i] * x[i]).sqrt()).collect();
    let (mut a, mut m, mut sw) = ([0f64; GROUPS], [0f64; GROUPS], [0f64; GROUPS]);
    for g in 0..GROUPS {
        let r = g * GROUP..(g + 1) * GROUP;
        (a[g], m[g]) = fit_group(&x[r.clone()], &w[r.clone()]);
        sw[g] = w[r].iter().sum();
    }
    let (d, mut la) = quant_levels(&a, &sw);
    let (dm, mut lm) = quant_levels(&m, &sw);
    let recon = |g: usize, la: f64, lm: f64, codes: &mut [u8; GROUP]| -> f64 {
        let (aa, mm) = (d * la, dm * lm);
        let mut err = 0.0;
        for i in 0..GROUP {
            let k = g * GROUP + i;
            let q = if aa > 0.0 { ((x[k] + mm) / aa).round_ties_even().clamp(0.0, NMAX) } else { 0.0 };
            codes[i] = q as u8;
            let e = aa * q - mm - x[k];
            err += w[k] * e * e;
        }
        err
    };
    let mut codes = [0u8; 256];
    for g in 0..GROUPS {
        let mut best_codes = [0u8; GROUP];
        let mut best = recon(g, la[g], lm[g], &mut best_codes);
        // Refinado local: vecinos de la escala y el mínimo ya cuantizados.
        for da in [-1.0, 0.0, 1.0] {
            for dmv in [-1.0, 0.0, 1.0] {
                if da == 0.0 && dmv == 0.0 { continue; }
                let (la2, lm2) = ((la[g] + da).clamp(0.0, LEVELS), (lm[g] + dmv).clamp(0.0, LEVELS));
                let mut c = [0u8; GROUP];
                let e = recon(g, la2, lm2, &mut c);
                if e < best { best = e; best_codes = c; la[g] = la2; lm[g] = lm2; }
            }
        }
        codes[g * GROUP..(g + 1) * GROUP].copy_from_slice(&best_codes);
    }
    out.fill(0);
    out[0..2].copy_from_slice(&f16::from_f64(d).to_le_bytes());
    out[2..4].copy_from_slice(&f16::from_f64(dm).to_le_bytes());
    for g in 0..GROUPS { put5(&mut out[4..HEADER_BYTES], g * 5, la[g] as u32); }
    for g in 0..GROUPS { put5(&mut out[4..HEADER_BYTES], 80 + g * 5, lm[g] as u32); }
    for j in 0..128 { out[HEADER_BYTES + j] = codes[2 * j] | (codes[2 * j + 1] << 4); }
}

/// Cuantiza una matriz fila a fila (`values.len()` múltiplo de `cols`, `cols` de 256).
/// `importance` tiene una entrada por columna, o está vacía para usar 1.
pub fn quantize(values: &[f32], cols: usize, importance: &[f32]) -> Result<Vec<u8>> {
    ensure!(cols > 0 && cols % 256 == 0, "hq44k requires rows of whole 256-value blocks");
    ensure!(values.len() % cols == 0, "values are not a whole number of rows");
    ensure!(importance.is_empty() || importance.len() == cols, "importance must have one value per column");
    ensure!(importance.iter().all(|v| v.is_finite() && *v >= 0.0), "importance must be finite and non-negative");
    let ones = vec![1.0f32; cols];
    let h = if importance.is_empty() { &ones[..] } else { importance };
    let blocks = values.len() / 256;
    let mut out = vec![0u8; blocks * BLOCK_BYTES];
    out.par_chunks_mut(BLOCK_BYTES).enumerate().for_each(|(i, dst)| {
        let col = (i * 256) % cols;
        encode_block(&values[i * 256..i * 256 + 256], &h[col..col + 256], dst);
    });
    Ok(out)
}

/// Decodificador de referencia, el mismo cálculo que el kernel de dequant.
pub fn dequantize(bytes: &[u8], count: usize) -> Result<Vec<f32>> {
    ensure!(count % 256 == 0 && bytes.len() == count / 256 * BLOCK_BYTES, "hq44k size mismatch");
    let mut out = vec![0f32; count];
    for (b, dst) in bytes.chunks_exact(BLOCK_BYTES).zip(out.chunks_exact_mut(256)) {
        let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dm = f16::from_le_bytes([b[2], b[3]]).to_f32();
        let field = |p: usize| ((b[4 + p / 8] as u32 | (b[5 + p / 8] as u32) << 8) >> (p % 8)) & 31;
        for g in 0..GROUPS {
            let (a, m) = (d * field(g * 5) as f32, dm * field(80 + g * 5) as f32);
            for i in 0..GROUP {
                let k = g * GROUP + i;
                let c = b[HEADER_BYTES + k / 2];
                dst[k] = a * (if k % 2 == 0 { c & 15 } else { c >> 4 }) as f32 - m;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_close_and_layout_is_exact() {
        let cols = 512;
        let values: Vec<f32> = (0..cols * 3).map(|i| ((i as f32 * 0.37).sin() * 0.02) + if i % 97 == 0 { 0.08 } else { 0.0 }).collect();
        let packed = quantize(&values, cols, &[]).unwrap();
        assert_eq!(packed.len(), values.len() / 256 * BLOCK_BYTES);
        let back = dequantize(&packed, values.len()).unwrap();
        let (num, den) = values.iter().zip(&back).fold((0f64, 0f64), |(n, d), (x, y)| (n + ((x - y) as f64).powi(2), d + (*x as f64).powi(2)));
        assert!((num / den).sqrt() < 0.08, "relative error {}", (num / den).sqrt());
        // Un bloque constante: mínimo y códigos coherentes con el formato.
        let flat = vec![-0.5f32; 256];
        let b = quantize(&flat, 256, &[]).unwrap();
        let y = dequantize(&b, 256).unwrap();
        assert!(y.iter().all(|v| (v + 0.5).abs() < 2e-3));
    }

    #[test]
    fn importance_is_validated() {
        let values = vec![0.1f32; 256];
        assert!(quantize(&values, 256, &[1.0; 128]).is_err());
        assert!(quantize(&values, 256, &vec![f32::NAN; 256]).is_err());
        assert!(quantize(&values, 128, &[]).is_err());
    }
}
