// src/hqs/hq8.rs
// ============================================================================
// HQ8K — escalón de 8 bits de HQS (2026-11-02)
// ============================================================================
//
// Sustituye a HQ6.2K, que gastaba 2 bits por peso en metadatos. Medido sobre
// pesos reales de Gemma 4 26B, HQ8K da 0,56 % de error L2 relativo, igual que
// Q8_0 de llama.cpp (0,555 %), frente al 1,11 % de HQ6.2K con casi los mismos
// bits.
//
// Superbloque de 256 pesos, 272 bytes (8,5 bits/peso), alineado a 16 bytes:
//   [0..2)    d      f16 LE, paso máximo del superbloque
//   [2..10)   q[8]   u8 por grupo de 32 pesos; paso del grupo = d * q / 255
//   [10..16)  reservado, cero
//   [16..272) 256 códigos i8 simétricos en [-127, 127]; peso = código * paso
//
// q se redondea hacia arriba y d se lleva al f16 inmediatamente superior si el
// redondeo lo dejó corto: el paso nunca recorta el máximo del grupo.

use super::common::{pad_to_superblock, EPS, SUPER_BLOCK_SIZE};
use half::f16;
use rayon::prelude::*;

pub const HQ8K_HEADER_SIZE: usize = 16;
pub const HQ8K_BLOCK_SIZE: usize = HQ8K_HEADER_SIZE + SUPER_BLOCK_SIZE; // 272
const GROUP: usize = 32;
const GROUPS: usize = SUPER_BLOCK_SIZE / GROUP;

/// Redondeo con empates alejándose de cero, como `f32::round`.
fn code(x: f32, step: f32) -> i8 {
    if step <= EPS { 0 } else { (x / step).round().clamp(-127.0, 127.0) as i8 }
}

fn f16_at_least(x: f32) -> f16 {
    let h = f16::from_f32(x);
    if h.to_f32() >= x { h } else { f16::from_bits(h.to_bits() + 1) }
}

fn quantize_block(block: &[f32]) -> [u8; HQ8K_BLOCK_SIZE] {
    let mut ideal = [0.0f32; GROUPS];
    for (g, group) in block.chunks_exact(GROUP).enumerate() {
        ideal[g] = group.iter().fold(0.0f32, |m, x| m.max(x.abs())) / 127.0;
    }
    let peak = ideal.iter().copied().fold(0.0f32, f32::max);
    let mut out = [0u8; HQ8K_BLOCK_SIZE];
    if peak <= EPS {
        return out;
    }
    let d_h = f16_at_least(peak);
    let d = d_h.to_f32();
    out[0..2].copy_from_slice(&d_h.to_le_bytes());
    for g in 0..GROUPS {
        let q = ((ideal[g] / d) * 255.0).ceil().clamp(1.0, 255.0) as u8;
        out[2 + g] = q;
        let step = d * q as f32 / 255.0;
        for i in 0..GROUP {
            out[HQ8K_HEADER_SIZE + g * GROUP + i] = code(block[g * GROUP + i], step) as u8;
        }
    }
    out
}

pub fn quantize_hq8k(data: &[f32]) -> Vec<u8> {
    let padded = pad_to_superblock(data);
    padded
        .par_chunks(SUPER_BLOCK_SIZE)
        .flat_map_iter(|chunk| {
            let clean: Vec<f32> = chunk.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
            quantize_block(&clean)
        })
        .collect()
}

pub fn dequantize_hq8k(data: &[u8], numel: usize) -> Vec<f32> {
    assert_eq!(data.len() % HQ8K_BLOCK_SIZE, 0, "invalid HQ8K byte length");
    let mut output = Vec::with_capacity(data.len() / HQ8K_BLOCK_SIZE * SUPER_BLOCK_SIZE);
    for block in data.chunks_exact(HQ8K_BLOCK_SIZE) {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for g in 0..GROUPS {
            let step = d * block[2 + g] as f32 / 255.0;
            for i in 0..GROUP {
                output.push(block[HQ8K_HEADER_SIZE + g * GROUP + i] as i8 as f32 * step);
            }
        }
    }
    output.truncate(numel);
    output
}

pub fn hq8k_size(numel: usize) -> usize {
    (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE * HQ8K_BLOCK_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Datos deterministas con escala variable por grupo y algún valor extremo.
    fn sample(n: usize) -> Vec<f32> {
        (0..n).map(|i| {
            let x = i as f32;
            let base = (x * 0.731).sin() * 0.02 + (x * 0.0137).cos() * 0.01;
            if i % 97 == 0 { base * 9.0 } else { base * (1.0 + (i / 32 % 5) as f32) }
        }).collect()
    }

    fn relative_l2(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum();
        (num / den).sqrt()
    }

    #[test]
    fn layout_and_size() {
        assert_eq!(HQ8K_BLOCK_SIZE, 272);
        assert_eq!(hq8k_size(256), 272);
        assert_eq!(hq8k_size(257), 544);
        let bytes = quantize_hq8k(&sample(512));
        assert_eq!(bytes.len(), 544);
        for block in bytes.chunks_exact(HQ8K_BLOCK_SIZE) {
            assert!(block[10..16].iter().all(|b| *b == 0), "reserved bytes must be zero");
            assert!(block[2..10].iter().all(|q| *q >= 1), "every group keeps a nonzero step");
        }
    }

    #[test]
    fn roundtrip_error_matches_measured_format() {
        let data = sample(256 * 64);
        let back = dequantize_hq8k(&quantize_hq8k(&data), data.len());
        let error = relative_l2(&back, &data);
        assert!(error < 0.007, "relative L2 {error}");
    }

    #[test]
    fn never_clips_the_group_maximum() {
        let data = sample(256 * 16);
        let bytes = quantize_hq8k(&data);
        let back = dequantize_hq8k(&bytes, data.len());
        for (b, block) in bytes.chunks_exact(HQ8K_BLOCK_SIZE).enumerate() {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for g in 0..GROUPS {
                let step = d * block[2 + g] as f32 / 255.0;
                for i in 0..GROUP {
                    let at = b * SUPER_BLOCK_SIZE + g * GROUP + i;
                    assert!((back[at] - data[at]).abs() <= step * 0.5 + 1e-9, "clipped at {at}");
                }
            }
        }
    }

    #[test]
    fn zeros_and_padding() {
        let zeros = vec![0.0f32; 300];
        let bytes = quantize_hq8k(&zeros);
        assert!(bytes.iter().all(|b| *b == 0));
        assert_eq!(dequantize_hq8k(&bytes, 300), zeros);
        let data = sample(300);
        let back = dequantize_hq8k(&quantize_hq8k(&data), 300);
        assert_eq!(back.len(), 300);
        assert!(relative_l2(&back, &data) < 0.007);
    }
}
