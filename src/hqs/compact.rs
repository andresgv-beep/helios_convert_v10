// src/hqs/compact.rs
// ============================================================================
// HQ3.1K / HQ4.1K / HQ5.1K / HQ4.2K / HQ5.2K — formatos compactos
// ============================================================================
//
// El header clásico de HQ4K/HQ5K gasta 128 B por superbloque (32 grupos ×
// min/scale fp16) — 4 bits/peso solo de metadatos. El compacto meta-cuantiza
// los parámetros de grupo a 4 bits contra rangos fp16 del superbloque:
//
//   Layout (40 bytes, matching hqs_common.cuh del engine):
//     [0..2)   d_scale:  fp16   (scale máximo del superbloque)
//     [2..4)   d_min:    fp16   (rango de mins: max_min - min_base)
//     [4..6)   min_base: fp16   (min mínimo del superbloque)
//     [6..8)   _pad
//     [8..24)  q_scale[32]: 4-bit packed (grupo par → nibble ALTO)
//     [24..40) q_min[32]:   4-bit packed (grupo par → nibble ALTO)
//
//   Decode del kernel (decode_compact_group):
//     scale_g = d_scale * (q_s / 15)
//     min_g   = min_base + d_min * (q_m / 15)
//     peso    = q * (scale_g / Q_MAX) + min_g
//
// CRÍTICO: los elementos se recuantizan contra los parámetros EFECTIVOS
// (tras redondear d_scale/d_min/min_base a f16 y los q_s/q_m a 4 bits),
// no contra los del grid search — así el encoder y el kernel ven exactamente
// los mismos números.
//
// Bits/peso totales: HQ3.1K = 136 B/256 = 4.25
//                     HQ4.1K = 168 B/256 = 5.25
//                     HQ5.1K = 200 B/256 = 6.25
// (vs 8.0 y 9.0 de los formatos clásicos)

use rayon::prelude::*;
use half::f16;
use crate::hqs::common::*;
use crate::hqs::grid_search::*;

pub const COMPACT_HEADER_SIZE: usize = 40;
pub const HQ31K_PAYLOAD: usize = 96;
pub const HQ31K_BLOCK_SIZE: usize = COMPACT_HEADER_SIZE + HQ31K_PAYLOAD; // 136
pub const HQ41K_BLOCK_SIZE: usize = COMPACT_HEADER_SIZE + HQ4K_PAYLOAD; // 168
pub const HQ51K_BLOCK_SIZE: usize = COMPACT_HEADER_SIZE + HQ5K_PAYLOAD; // 200
// HQ4.2K/HQ5.2K: cuantización simétrica por grupos de 8. El paso de cada
// grupo se meta-cuantiza a 5 bits; 32 pasos ocupan 20 B y d_step fp16 ocupa
// 2 B. Se redondea el header a 24 B para mantener payload y superbloques
// alineados a 8 bytes.
pub const SYMMETRIC_HEADER_SIZE: usize = 24;
pub const HQ42K_BLOCK_SIZE: usize = SYMMETRIC_HEADER_SIZE + HQ4K_PAYLOAD; // 152
pub const HQ52K_BLOCK_SIZE: usize = SYMMETRIC_HEADER_SIZE + HQ5K_PAYLOAD; // 184
// HQ6.2K conserva los grupos de 8, pero cada q_scale/q_min ocupa un byte.
// 72 B header + 192 B payload = 264 B = 8,25 bpw.
pub const HQ62K_HEADER_SIZE: usize = 72;
pub const HQ62K_PAYLOAD: usize = 192;
pub const HQ62K_BLOCK_SIZE: usize = HQ62K_HEADER_SIZE + HQ62K_PAYLOAD; // 264

const EPS: f32 = 1e-8;

fn pack_5bit(values: &[u8], output: &mut [u8]) {
    output.fill(0);
    for (index, value) in values.iter().enumerate() {
        let bit = index * 5;
        let byte = bit / 8;
        let shift = bit % 8;
        let packed = (*value as u16 & 0x1f) << shift;
        output[byte] |= packed as u8;
        if shift > 3 {
            output[byte + 1] |= (packed >> 8) as u8;
        }
    }
}

fn unpack_5bit(input: &[u8], index: usize) -> u8 {
    let bit = index * 5;
    let byte = bit / 8;
    let shift = bit % 8;
    let word = input[byte] as u16 | ((input.get(byte + 1).copied().unwrap_or(0) as u16) << 8);
    ((word >> shift) & 0x1f) as u8
}

fn quantize_superblock_symmetric(block: &[f32; SUPER_BLOCK_SIZE], bits: u8) -> Vec<u8> {
    let q_negative = (1u16 << (bits - 1)) as f32;
    let q_positive = q_negative - 1.0;
    let payload_size = SUPER_BLOCK_SIZE * bits as usize / 8;
    let mut ideal_steps = [0.0f32; NUM_GROUPS];
    let mut max_step = 0.0f32;
    for (group_index, group) in block.chunks_exact(GROUP_SIZE).enumerate() {
        let positive = group.iter().copied().fold(0.0f32, f32::max);
        let negative = -group.iter().copied().fold(0.0f32, f32::min);
        let step = (positive / q_positive).max(negative / q_negative);
        ideal_steps[group_index] = step;
        max_step = max_step.max(step);
    }

    let d_step_h = f16::from_f32(max_step);
    let d_step = d_step_h.to_f32();
    let mut q_steps = [0u8; NUM_GROUPS];
    let mut effective_steps = [0.0f32; NUM_GROUPS];
    for group in 0..NUM_GROUPS {
        let q = if d_step > EPS {
            ((ideal_steps[group] / d_step) * 31.0).round().clamp(1.0, 31.0) as u8
        } else {
            0
        };
        q_steps[group] = q;
        effective_steps[group] = d_step * q as f32 / 31.0;
    }

    let mut output = vec![0u8; SYMMETRIC_HEADER_SIZE + payload_size];
    output[0..2].copy_from_slice(&d_step_h.to_le_bytes());
    pack_5bit(&q_steps, &mut output[4..SYMMETRIC_HEADER_SIZE]);

    if bits == 4 {
        for pair in 0..SUPER_BLOCK_SIZE / 2 {
            let mut codes = [0u8; 2];
            for lane in 0..2 {
                let index = pair * 2 + lane;
                let step = effective_steps[index / GROUP_SIZE];
                let q = if step > EPS {
                    (block[index] / step).round().clamp(-q_negative, q_positive) as i16
                } else { 0 };
                codes[lane] = (q + q_negative as i16) as u8;
            }
            output[SYMMETRIC_HEADER_SIZE + pair] = (codes[0] << 4) | codes[1];
        }
    } else {
        for group in 0..NUM_GROUPS {
            let mut packed = 0u64;
            let step = effective_steps[group];
            for lane in 0..GROUP_SIZE {
                let index = group * GROUP_SIZE + lane;
                let q = if step > EPS {
                    (block[index] / step).round().clamp(-q_negative, q_positive) as i16
                } else { 0 };
                packed |= ((q + q_negative as i16) as u64 & 0x1f) << (lane * 5);
            }
            let offset = SYMMETRIC_HEADER_SIZE + group * 5;
            for byte in 0..5 { output[offset + byte] = (packed >> (byte * 8)) as u8; }
        }
    }
    output
}

fn quantize_symmetric_internal(data: &[f32], bits: u8) -> Vec<u8> {
    let padded = pad_to_superblock(data);
    padded.par_chunks(SUPER_BLOCK_SIZE)
        .flat_map_iter(|chunk| {
            let block: &[f32; SUPER_BLOCK_SIZE] = chunk.try_into().unwrap();
            quantize_superblock_symmetric(block, bits)
        })
        .collect()
}

fn dequantize_symmetric(data: &[u8], numel: usize, bits: u8) -> Vec<f32> {
    let block_size = if bits == 4 { HQ42K_BLOCK_SIZE } else { HQ52K_BLOCK_SIZE };
    let q_negative = (1u16 << (bits - 1)) as i16;
    let mut output = Vec::with_capacity(data.len() / block_size * SUPER_BLOCK_SIZE);
    for block in data.chunks_exact(block_size) {
        let d_step = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut steps = [0.0f32; NUM_GROUPS];
        for group in 0..NUM_GROUPS {
            steps[group] = d_step * unpack_5bit(&block[4..SYMMETRIC_HEADER_SIZE], group) as f32 / 31.0;
        }
        if bits == 4 {
            for index in 0..SUPER_BLOCK_SIZE {
                let byte = block[SYMMETRIC_HEADER_SIZE + index / 2];
                let code = if index % 2 == 0 { byte >> 4 } else { byte & 0x0f };
                output.push((code as i16 - q_negative) as f32 * steps[index / GROUP_SIZE]);
            }
        } else {
            for group in 0..NUM_GROUPS {
                let offset = SYMMETRIC_HEADER_SIZE + group * 5;
                let mut packed = 0u64;
                for byte in 0..5 { packed |= (block[offset + byte] as u64) << (byte * 8); }
                for lane in 0..GROUP_SIZE {
                    let code = ((packed >> (lane * 5)) & 0x1f) as i16;
                    output.push((code - q_negative) as f32 * steps[group]);
                }
            }
        }
    }
    output.truncate(numel);
    output
}

pub fn quantize_hq42k(data: &[f32]) -> Vec<u8> { quantize_symmetric_internal(data, 4) }
pub fn quantize_hq52k(data: &[f32]) -> Vec<u8> { quantize_symmetric_internal(data, 5) }
pub fn dequantize_hq42k(data: &[u8], numel: usize) -> Vec<f32> { dequantize_symmetric(data, numel, 4) }
pub fn dequantize_hq52k(data: &[u8], numel: usize) -> Vec<f32> { dequantize_symmetric(data, numel, 5) }

/// Codifica el header compacto y devuelve los parámetros EFECTIVOS por grupo
/// (los que el kernel reconstruirá al decodificar).
fn encode_compact_header(
    group_params: &[GroupParams; NUM_GROUPS],
) -> ([u8; COMPACT_HEADER_SIZE], [GroupParams; NUM_GROUPS]) {
    let mut max_scale = 0.0f32;
    let mut min_min = f32::INFINITY;
    let mut max_min = f32::NEG_INFINITY;
    for gp in group_params.iter() {
        max_scale = max_scale.max(gp.scale);
        min_min = min_min.min(gp.min);
        max_min = max_min.max(gp.min);
    }
    if !min_min.is_finite() {
        min_min = 0.0;
        max_min = 0.0;
    }

    // Redondear a f16 ANTES de derivar nada: el kernel leerá estos f16.
    let d_scale_h = f16::from_f32(max_scale);
    let min_base_h = f16::from_f32(min_min);
    let d_scale_f = d_scale_h.to_f32();
    let min_base_f = min_base_h.to_f32();
    let d_min_h = f16::from_f32(max_min - min_base_f);
    let d_min_f = d_min_h.to_f32();

    let mut header = [0u8; COMPACT_HEADER_SIZE];
    header[0..2].copy_from_slice(&d_scale_h.to_le_bytes());
    header[2..4].copy_from_slice(&d_min_h.to_le_bytes());
    header[4..6].copy_from_slice(&min_base_h.to_le_bytes());
    // [6..8) pad = 0

    let mut effective = [GroupParams::default(); NUM_GROUPS];

    for g in 0..NUM_GROUPS {
        let gp = &group_params[g];

        let mut q_s = if d_scale_f > EPS {
            ((gp.scale / d_scale_f) * 15.0).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        // Un grupo con rango real no debe colapsar a scale 0
        if gp.scale > EPS && q_s == 0 {
            q_s = 1;
        }

        let q_m = if d_min_f > EPS {
            (((gp.min - min_base_f) / d_min_f) * 15.0).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };

        // Empaquetado: grupo par → nibble ALTO (matching decode_compact_group)
        let s_idx = 8 + g / 2;
        let m_idx = 24 + g / 2;
        if g % 2 == 0 {
            header[s_idx] |= q_s << 4;
            header[m_idx] |= q_m << 4;
        } else {
            header[s_idx] |= q_s & 0x0F;
            header[m_idx] |= q_m & 0x0F;
        }

        effective[g] = GroupParams {
            scale: d_scale_f * (q_s as f32) / 15.0,
            min: min_base_f + d_min_f * (q_m as f32) / 15.0,
        };
    }

    (header, effective)
}

fn decode_compact_group_cpu(block: &[u8], group_idx: usize, q_max: f32) -> (f32, f32) {
    let d_scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let d_min = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let min_base = f16::from_le_bytes([block[4], block[5]]).to_f32();
    let s_packed = block[8 + group_idx / 2];
    let q_s = if group_idx % 2 == 0 { s_packed >> 4 } else { s_packed & 0x0F };
    let m_packed = block[24 + group_idx / 2];
    let q_m = if group_idx % 2 == 0 { m_packed >> 4 } else { m_packed & 0x0F };
    let scale = d_scale * (q_s as f32 / 15.0);
    let min_f = min_base + d_min * (q_m as f32 / 15.0);
    (min_f, scale / q_max)
}

/// Cuantiza los 256 elementos contra los parámetros efectivos.
fn quantize_indices(
    block: &[f32; SUPER_BLOCK_SIZE],
    effective: &[GroupParams; NUM_GROUPS],
    q_max: f32,
) -> [u8; SUPER_BLOCK_SIZE] {
    let mut q_indices = [0u8; SUPER_BLOCK_SIZE];
    for g in 0..NUM_GROUPS {
        let gp = &effective[g];
        let start = g * GROUP_SIZE;
        for i in 0..GROUP_SIZE {
            let q = if gp.scale > EPS {
                ((block[start + i] - gp.min) / gp.scale * q_max)
                    .round()
                    .clamp(0.0, q_max) as u8
            } else {
                0
            };
            q_indices[start + i] = q;
        }
    }
    q_indices
}

// ============================================================================
// HQ3.1K
// ============================================================================

fn quantize_superblock_hq31k(block: &[f32; SUPER_BLOCK_SIZE], use_mse: bool) -> Vec<u8> {
    let config = GridConfig::hq3k();
    let group_params = if use_mse {
        optimize_superblock(block, &config)
    } else {
        fast_superblock(block)
    };

    let (header, effective) = encode_compact_header(&group_params);
    let q_indices = quantize_indices(block, &effective, 7.0);

    let mut output = vec![0u8; HQ31K_BLOCK_SIZE];
    output[..COMPACT_HEADER_SIZE].copy_from_slice(&header);

    // Cada grupo natural de 8 pesos ocupa exactamente 24 bits = 3 bytes.
    // LSB-first, igual que HQ5.1K: q[k] empieza en el bit k*3.
    for chunk_idx in 0..(SUPER_BLOCK_SIZE / 8) {
        let base = chunk_idx * 8;
        let mut bits: u32 = 0;
        for k in 0..8 {
            bits |= (q_indices[base + k] as u32 & 0x07) << (k * 3);
        }
        let byte_idx = COMPACT_HEADER_SIZE + chunk_idx * 3;
        output[byte_idx] = (bits & 0xFF) as u8;
        output[byte_idx + 1] = ((bits >> 8) & 0xFF) as u8;
        output[byte_idx + 2] = ((bits >> 16) & 0xFF) as u8;
    }

    output
}

pub fn quantize_hq31k(data: &[f32], use_mse: bool) -> Vec<u8> {
    quantize_compact_internal(data, use_mse, HQ31K_BLOCK_SIZE, quantize_superblock_hq31k)
}

pub fn dequantize_hq31k(data: &[u8], numel: usize) -> Vec<f32> {
    if data.is_empty() {
        return vec![0.0; numel];
    }
    assert_eq!(data.len() % HQ31K_BLOCK_SIZE, 0, "invalid HQ3.1K byte length");

    let num_blocks = data.len() / HQ31K_BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_blocks * SUPER_BLOCK_SIZE);
    for b in 0..num_blocks {
        let block = &data[b * HQ31K_BLOCK_SIZE..(b + 1) * HQ31K_BLOCK_SIZE];
        for g in 0..NUM_GROUPS {
            let (min_f, scoeff) = decode_compact_group_cpu(block, g, 7.0);
            let byte_idx = COMPACT_HEADER_SIZE + g * 3;
            let bits = (block[byte_idx] as u32)
                | ((block[byte_idx + 1] as u32) << 8)
                | ((block[byte_idx + 2] as u32) << 16);
            for i in 0..GROUP_SIZE {
                let q = ((bits >> (i * 3)) & 0x07) as f32;
                output.push(q * scoeff + min_f);
            }
        }
    }
    output.truncate(numel);
    output
}

// ============================================================================
// HQ4.1K
// ============================================================================

fn quantize_superblock_hq41k(block: &[f32; SUPER_BLOCK_SIZE], use_mse: bool) -> Vec<u8> {
    let config = GridConfig::hq4k();
    let group_params = if use_mse {
        optimize_superblock(block, &config)
    } else {
        fast_superblock(block)
    };

    let (header, effective) = encode_compact_header(&group_params);
    let q_indices = quantize_indices(block, &effective, 15.0);

    let mut output = vec![0u8; HQ41K_BLOCK_SIZE];
    output[..COMPACT_HEADER_SIZE].copy_from_slice(&header);

    // Payload idéntico a HQ4K: par → nibble alto, impar → nibble bajo
    for i in 0..HQ4K_PAYLOAD {
        let even = q_indices[i * 2] & 0x0F;
        let odd = q_indices[i * 2 + 1] & 0x0F;
        output[COMPACT_HEADER_SIZE + i] = (even << 4) | odd;
    }

    output
}

pub fn quantize_hq41k(data: &[f32], use_mse: bool) -> Vec<u8> {
    quantize_compact_internal(data, use_mse, HQ41K_BLOCK_SIZE, quantize_superblock_hq41k)
}

// ============================================================================
// HQ5.1K
// ============================================================================

fn quantize_superblock_hq51k(block: &[f32; SUPER_BLOCK_SIZE], use_mse: bool) -> Vec<u8> {
    let config = GridConfig::hq5k();
    let group_params = if use_mse {
        optimize_superblock(block, &config)
    } else {
        fast_superblock(block)
    };

    let (header, effective) = encode_compact_header(&group_params);
    let q_indices = quantize_indices(block, &effective, 31.0);

    let mut output = vec![0u8; HQ51K_BLOCK_SIZE];
    output[..COMPACT_HEADER_SIZE].copy_from_slice(&header);

    // Payload idéntico a HQ5K: 8 valores de 5 bits LSB-first = 5 bytes
    for chunk_idx in 0..(SUPER_BLOCK_SIZE / 8) {
        let base = chunk_idx * 8;
        let mut bits: u64 = 0;
        for k in 0..8 {
            bits |= (q_indices[base + k] as u64 & 0x1F) << (k * 5);
        }
        let byte_idx = COMPACT_HEADER_SIZE + chunk_idx * 5;
        for k in 0..5 {
            output[byte_idx + k] = ((bits >> (k * 8)) & 0xFF) as u8;
        }
    }

    output
}

pub fn quantize_hq51k(data: &[f32], use_mse: bool) -> Vec<u8> {
    quantize_compact_internal(data, use_mse, HQ51K_BLOCK_SIZE, quantize_superblock_hq51k)
}

// ============================================================================
// HQ6.2K — 6-bit payload + 8-bit metadata
// ============================================================================

fn encode_hq62k_header(
    group_params: &[GroupParams; NUM_GROUPS],
) -> ([u8; HQ62K_HEADER_SIZE], [GroupParams; NUM_GROUPS]) {
    let mut max_scale = 0.0f32;
    let mut min_min = f32::INFINITY;
    let mut max_min = f32::NEG_INFINITY;
    for gp in group_params.iter() {
        max_scale = max_scale.max(gp.scale);
        min_min = min_min.min(gp.min);
        max_min = max_min.max(gp.min);
    }
    if !min_min.is_finite() {
        min_min = 0.0;
        max_min = 0.0;
    }

    let d_scale_h = f16::from_f32(max_scale);
    let min_base_h = f16::from_f32(min_min);
    let d_scale = d_scale_h.to_f32();
    let min_base = min_base_h.to_f32();
    let d_min_h = f16::from_f32(max_min - min_base);
    let d_min = d_min_h.to_f32();

    let mut header = [0u8; HQ62K_HEADER_SIZE];
    header[0..2].copy_from_slice(&d_scale_h.to_le_bytes());
    header[2..4].copy_from_slice(&d_min_h.to_le_bytes());
    header[4..6].copy_from_slice(&min_base_h.to_le_bytes());

    let mut effective = [GroupParams::default(); NUM_GROUPS];
    for g in 0..NUM_GROUPS {
        let gp = group_params[g];
        let mut q_scale = if d_scale > EPS {
            ((gp.scale / d_scale) * 255.0).round().clamp(0.0, 255.0) as u8
        } else {
            0
        };
        if gp.scale > EPS && q_scale == 0 {
            q_scale = 1;
        }
        let q_min = if d_min > EPS {
            (((gp.min - min_base) / d_min) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        } else {
            0
        };
        header[8 + g] = q_scale;
        header[40 + g] = q_min;
        effective[g] = GroupParams {
            scale: d_scale * q_scale as f32 / 255.0,
            min: min_base + d_min * q_min as f32 / 255.0,
        };
    }

    (header, effective)
}

fn quantize_superblock_hq62k(block: &[f32; SUPER_BLOCK_SIZE], use_mse: bool) -> Vec<u8> {
    let config = GridConfig::hq6k();
    let group_params = if use_mse {
        optimize_superblock(block, &config)
    } else {
        fast_superblock(block)
    };
    let (header, effective) = encode_hq62k_header(&group_params);
    let q_indices = quantize_indices(block, &effective, 63.0);

    let mut output = vec![0u8; HQ62K_BLOCK_SIZE];
    output[..HQ62K_HEADER_SIZE].copy_from_slice(&header);
    for group in 0..NUM_GROUPS {
        let base = group * GROUP_SIZE;
        let mut bits = 0u64;
        for i in 0..GROUP_SIZE {
            bits |= (q_indices[base + i] as u64 & 0x3f) << (i * 6);
        }
        let byte_offset = HQ62K_HEADER_SIZE + group * 6;
        for i in 0..6 {
            output[byte_offset + i] = ((bits >> (i * 8)) & 0xff) as u8;
        }
    }
    output
}

pub fn quantize_hq62k(data: &[f32], use_mse: bool) -> Vec<u8> {
    quantize_compact_internal(data, use_mse, HQ62K_BLOCK_SIZE, quantize_superblock_hq62k)
}

pub fn dequantize_hq62k(data: &[u8], numel: usize) -> Vec<f32> {
    if data.is_empty() {
        return vec![0.0; numel];
    }
    assert_eq!(data.len() % HQ62K_BLOCK_SIZE, 0, "invalid HQ6.2K byte length");
    let mut output = Vec::with_capacity(data.len() / HQ62K_BLOCK_SIZE * SUPER_BLOCK_SIZE);
    for block in data.chunks_exact(HQ62K_BLOCK_SIZE) {
        let d_scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let d_min = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let min_base = f16::from_le_bytes([block[4], block[5]]).to_f32();
        for group in 0..NUM_GROUPS {
            let scale = d_scale * block[8 + group] as f32 / 255.0;
            let min = min_base + d_min * block[40 + group] as f32 / 255.0;
            let coefficient = scale / 63.0;
            let byte_offset = HQ62K_HEADER_SIZE + group * 6;
            let mut bits = 0u64;
            for i in 0..6 {
                bits |= (block[byte_offset + i] as u64) << (i * 8);
            }
            for i in 0..GROUP_SIZE {
                let q = ((bits >> (i * 6)) & 0x3f) as f32;
                output.push(q * coefficient + min);
            }
        }
    }
    output.truncate(numel);
    output
}

// ============================================================================
// DRIVER COMÚN
// ============================================================================

fn quantize_compact_internal(
    data: &[f32],
    use_mse: bool,
    block_size: usize,
    quantize_fn: fn(&[f32; SUPER_BLOCK_SIZE], bool) -> Vec<u8>,
) -> Vec<u8> {
    let padded = pad_to_superblock(data);
    let num_blocks = padded.len() / SUPER_BLOCK_SIZE;
    if num_blocks == 0 {
        return Vec::new();
    }

    let results: Vec<Vec<u8>> = (0..num_blocks)
        .into_par_iter()
        .map(|b| {
            let start = b * SUPER_BLOCK_SIZE;
            let mut block = [0.0f32; SUPER_BLOCK_SIZE];
            for i in 0..SUPER_BLOCK_SIZE {
                let val = padded[start + i];
                block[i] = if val.is_finite() { val } else { 0.0 };
            }
            quantize_fn(&block, use_mse)
        })
        .collect();

    let mut output = Vec::with_capacity(num_blocks * block_size);
    for block_data in results {
        output.extend(block_data);
    }
    output
}

pub fn hq41k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ41K_BLOCK_SIZE
}

pub fn hq31k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ31K_BLOCK_SIZE
}

pub fn hq51k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ51K_BLOCK_SIZE
}

pub fn hq42k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ42K_BLOCK_SIZE
}

pub fn hq52k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ52K_BLOCK_SIZE
}

pub fn hq62k_size(numel: usize) -> usize {
    let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    num_blocks * HQ62K_BLOCK_SIZE
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_layout_and_roundtrip_are_finite() {
        let values: Vec<f32> = (0..(SUPER_BLOCK_SIZE + 17))
            .map(|i| ((i as f32 * 0.173).sin() * 0.2) + ((i % 11) as f32 - 5.0) * 0.01)
            .collect();
        let q4 = quantize_hq42k(&values);
        let q5 = quantize_hq52k(&values);
        assert_eq!(q4.len(), 2 * HQ42K_BLOCK_SIZE);
        assert_eq!(q5.len(), 2 * HQ52K_BLOCK_SIZE);
        let d4 = dequantize_hq42k(&q4, values.len());
        let d5 = dequantize_hq52k(&q5, values.len());
        assert!(d4.iter().all(|x| x.is_finite()));
        assert!(d5.iter().all(|x| x.is_finite()));
        let reference_energy = values.iter().map(|x| x * x).sum::<f32>();
        let nrmse = |decoded: &[f32]| {
            (values.iter().zip(decoded).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
                / reference_energy).sqrt()
        };
        let error4 = nrmse(&d4);
        let error5 = nrmse(&d5);
        assert!(error4 < 0.12, "HQ4.2K NRMSE too high: {error4}");
        assert!(error5 < 0.06, "HQ5.2K NRMSE too high: {error5}");
        assert!(error5 < error4, "HQ5.2K should improve on HQ4.2K");

        let zeros = vec![0.0; SUPER_BLOCK_SIZE];
        assert!(dequantize_hq42k(&quantize_hq42k(&zeros), zeros.len())
            .iter().all(|&x| x == 0.0));
        assert!(dequantize_hq52k(&quantize_hq52k(&zeros), zeros.len())
            .iter().all(|&x| x == 0.0));
    }

    /// Réplica exacta del decode_compact_group del kernel CUDA
    fn decode_compact_group_ref(block: &[u8], group_idx: usize, q_max: f32) -> (f32, f32) {
        let d_scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let d_min = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let min_base = f16::from_le_bytes([block[4], block[5]]).to_f32();

        let s_packed = block[8 + group_idx / 2];
        let q_s = if group_idx % 2 == 0 { s_packed >> 4 } else { s_packed & 0x0F };
        let m_packed = block[24 + group_idx / 2];
        let q_m = if group_idx % 2 == 0 { m_packed >> 4 } else { m_packed & 0x0F };

        let scale = d_scale * (q_s as f32 / 15.0);
        let min_f = min_base + d_min * (q_m as f32 / 15.0);
        (min_f, scale / q_max)
    }

    #[test]
    fn test_hq31k_layout_and_roundtrip() {
        let mut data = vec![0.0f32; SUPER_BLOCK_SIZE + 17];
        for i in 0..data.len() {
            let g = (i / GROUP_SIZE) as f32;
            data[i] = (i as f32 * 0.51).sin() * (0.03 + g * 0.01) + g * 0.04 - 0.7;
        }

        let encoded = quantize_hq31k(&data, false);
        assert_eq!(HQ31K_BLOCK_SIZE, 136);
        assert_eq!(encoded.len(), 2 * HQ31K_BLOCK_SIZE);
        assert_eq!(hq31k_size(data.len()), encoded.len());

        let decoded = dequantize_hq31k(&encoded, data.len());
        assert_eq!(decoded.len(), data.len());
        let sum_sq: f64 = data.iter().map(|x| (*x as f64).powi(2)).sum();
        let sum_sq_err: f64 = data.iter().zip(decoded.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2)).sum();
        let rel_err = (sum_sq_err / sum_sq).sqrt();
        assert!(rel_err < 0.25, "HQ3.1K rel err too high: {}", rel_err);
    }

    #[test]
    fn test_hq41k_roundtrip_error() {
        // Superbloque sintético con rangos variados por grupo
        let mut data = vec![0.0f32; SUPER_BLOCK_SIZE];
        for i in 0..SUPER_BLOCK_SIZE {
            let g = (i / GROUP_SIZE) as f32;
            data[i] = (i as f32 * 0.37).sin() * (0.01 + g * 0.05) + g * 0.1 - 0.8;
        }

        let encoded = quantize_hq41k(&data, true);
        assert_eq!(encoded.len(), HQ41K_BLOCK_SIZE);

        // Decodificar como el kernel y medir error relativo
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq = 0.0f64;
        for g in 0..NUM_GROUPS {
            let (min_f, scoeff) = decode_compact_group_ref(&encoded, g, 15.0);
            for i in 0..GROUP_SIZE {
                let idx = g * GROUP_SIZE + i;
                let byte = encoded[COMPACT_HEADER_SIZE + idx / 2];
                let q = if idx % 2 == 0 { byte >> 4 } else { byte & 0x0F };
                let decoded = q as f32 * scoeff + min_f;
                sum_sq_err += ((decoded - data[idx]) as f64).powi(2);
                sum_sq += (data[idx] as f64).powi(2);
            }
        }
        let rel_err = (sum_sq_err / sum_sq).sqrt();
        assert!(rel_err < 0.15, "HQ4.1K rel err too high: {}", rel_err);
    }

    #[test]
    fn test_hq51k_roundtrip_error() {
        let mut data = vec![0.0f32; SUPER_BLOCK_SIZE];
        for i in 0..SUPER_BLOCK_SIZE {
            let g = (i / GROUP_SIZE) as f32;
            data[i] = (i as f32 * 0.73).cos() * (0.02 + g * 0.03) + g * 0.07 - 0.5;
        }

        let encoded = quantize_hq51k(&data, true);
        assert_eq!(encoded.len(), HQ51K_BLOCK_SIZE);

        let mut sum_sq_err = 0.0f64;
        let mut sum_sq = 0.0f64;
        for g in 0..NUM_GROUPS {
            let (min_f, scoeff) = decode_compact_group_ref(&encoded, g, 31.0);
            for i in 0..GROUP_SIZE {
                let idx = g * GROUP_SIZE + i;
                let bit_off = idx * 5;
                let byte0 = bit_off / 8;
                let sh = bit_off % 8;
                let raw = (encoded[COMPACT_HEADER_SIZE + byte0] as u16)
                    | ((encoded.get(COMPACT_HEADER_SIZE + byte0 + 1).copied().unwrap_or(0) as u16) << 8);
                let q = ((raw >> sh) & 0x1F) as u8;
                let decoded = q as f32 * scoeff + min_f;
                sum_sq_err += ((decoded - data[idx]) as f64).powi(2);
                sum_sq += (data[idx] as f64).powi(2);
            }
        }
        let rel_err = (sum_sq_err / sum_sq).sqrt();
        assert!(rel_err < 0.08, "HQ5.1K rel err too high: {}", rel_err);
    }

    #[test]
    fn test_hq62k_layout_and_roundtrip() {
        let mut data = vec![0.0f32; SUPER_BLOCK_SIZE + 19];
        for i in 0..data.len() {
            let group = (i / GROUP_SIZE) as f32;
            data[i] = (i as f32 * 0.117).sin() * (0.01 + group * 0.003)
                + (group % 11.0) * 0.002;
        }
        let encoded = quantize_hq62k(&data, false);
        assert_eq!(HQ62K_HEADER_SIZE, 72);
        assert_eq!(HQ62K_BLOCK_SIZE, 264);
        assert_eq!(encoded.len(), 2 * HQ62K_BLOCK_SIZE);
        assert_eq!(hq62k_size(data.len()), encoded.len());
        let decoded = dequantize_hq62k(&encoded, data.len());
        let error = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>();
        let energy = data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        assert!((error / energy).sqrt() < 0.03);
    }
}
