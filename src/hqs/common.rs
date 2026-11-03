// src/hqs/common.rs
// ============================================================================
// HQS COMMON v6 - NUCLEAR: Grupos de 8 elementos
// ============================================================================
//
// La bomba nuclear de precisión:
// - GROUP_SIZE: 8 (era 16, era 32)
// - NUM_GROUPS: 32 (era 16, era 8)
// - Header: 128 bytes (32 grupos × 4 bytes)
//
// Tamaños:
//   HQ4K: 128 header + 128 payload = 256 bytes (1:2 vs FP16)
//   HQ5K: 128 header + 160 payload = 288 bytes (1:1.78 vs FP16)
//
// ============================================================================

use half::f16;

pub const SUPER_BLOCK_SIZE: usize = 256;
pub const GROUP_SIZE: usize = 8;    // NUCLEAR: era 16
pub const NUM_GROUPS: usize = 32;   // NUCLEAR: era 16
pub const EPS: f32 = 1e-7;

// Header: 32 grupos × 4 bytes = 128 bytes
pub const HEADER_SIZE: usize = 128;

pub const HQ4K_PAYLOAD: usize = 128;  // 4 bits × 256 / 2
pub const HQ4K_BLOCK_SIZE: usize = HEADER_SIZE + HQ4K_PAYLOAD; // 256

pub const HQ5K_PAYLOAD: usize = 160;  // 5 bits × 256
pub const HQ5K_BLOCK_SIZE: usize = HEADER_SIZE + HQ5K_PAYLOAD; // 288

// HQ3K placeholder (not implemented in v6)
pub const HQ3K_BLOCK_SIZE: usize = 192;

#[derive(Debug, Clone, Copy)]
pub struct GroupParams {
    pub min: f32,
    pub scale: f32,
}

impl Default for GroupParams {
    fn default() -> Self {
        Self { min: 0.0, scale: 1.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GridSearchResult {
    pub min: f32,
    pub scale: f32,
    pub mse: f32,
}

impl Default for GridSearchResult {
    fn default() -> Self {
        Self { min: 0.0, scale: 1.0, mse: f32::INFINITY }
    }
}

pub fn pad_to_superblock(data: &[f32]) -> Vec<f32> {
    let remainder = data.len() % SUPER_BLOCK_SIZE;
    if remainder == 0 {
        data.to_vec()
    } else {
        let padding = SUPER_BLOCK_SIZE - remainder;
        let mut result = data.to_vec();
        result.extend(std::iter::repeat(0.0f32).take(padding));
        result
    }
}

#[inline]
pub fn compute_group_params(group: &[f32]) -> GroupParams {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    
    for &val in group.iter() {
        if val < min { min = val; }
        if val > max { max = val; }
    }
    
    let scale = (max - min).max(EPS);
    
    let min_f16 = f16::from_f32(min).to_f32();
    let scale_f16 = f16::from_f32(scale).to_f32().max(EPS);
    
    GroupParams {
        min: min_f16,
        scale: scale_f16,
    }
}

pub fn encode_header(group_params: &[GroupParams; NUM_GROUPS]) -> [u8; HEADER_SIZE] {
    let mut header = [0u8; HEADER_SIZE];
    
    for g in 0..NUM_GROUPS {
        let offset = g * 4;
        
        let min_bytes = f16::from_f32(group_params[g].min).to_le_bytes();
        header[offset] = min_bytes[0];
        header[offset + 1] = min_bytes[1];
        
        let scale_bytes = f16::from_f32(group_params[g].scale).to_le_bytes();
        header[offset + 2] = scale_bytes[0];
        header[offset + 3] = scale_bytes[1];
    }
    
    header
}

pub fn decode_header(header: &[u8; HEADER_SIZE]) -> [GroupParams; NUM_GROUPS] {
    let mut group_params = [GroupParams::default(); NUM_GROUPS];
    
    for g in 0..NUM_GROUPS {
        let offset = g * 4;
        
        let min = f16::from_le_bytes([header[offset], header[offset + 1]]).to_f32();
        let scale = f16::from_le_bytes([header[offset + 2], header[offset + 3]]).to_f32();
        
        group_params[g] = GroupParams {
            min,
            scale: scale.max(EPS),
        };
    }
    
    group_params
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_header_roundtrip() {
        let mut group_params = [GroupParams::default(); NUM_GROUPS];
        for g in 0..NUM_GROUPS {
            group_params[g] = GroupParams {
                min: -1.0 + g as f32 * 0.05,
                scale: 0.5 + g as f32 * 0.02,
            };
        }
        
        let header = encode_header(&group_params);
        let decoded = decode_header(&header);
        
        for g in 0..NUM_GROUPS {
            assert!((decoded[g].min - group_params[g].min).abs() < 0.01);
            assert!((decoded[g].scale - group_params[g].scale).abs() < 0.01);
        }
    }
    
    #[test]
    fn test_sizes() {
        assert_eq!(HEADER_SIZE, 128);
        assert_eq!(HQ4K_BLOCK_SIZE, 256);
        assert_eq!(HQ5K_BLOCK_SIZE, 288);
        assert_eq!(NUM_GROUPS * GROUP_SIZE, SUPER_BLOCK_SIZE);
    }
}
