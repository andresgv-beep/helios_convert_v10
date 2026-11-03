// src/hqs/grid_search.rs
// ============================================================================
// HQS GRID SEARCH v6 - NUCLEAR
// ============================================================================

use rayon::prelude::*;
use half::f16;
use crate::hqs::common::*;

#[derive(Debug, Clone, Copy)]
pub struct GridConfig {
    pub bits: u8,
}

impl GridConfig {
    pub fn hq3k() -> Self {
        Self { bits: 3 }
    }

    pub fn hq4k() -> Self {
        Self { bits: 4 }
    }
    
    pub fn hq5k() -> Self {
        Self { bits: 5 }
    }

    pub fn hq6k() -> Self {
        Self { bits: 6 }
    }
    
    #[inline]
    pub fn q_max(&self) -> f32 {
        ((1u32 << self.bits) - 1) as f32
    }
}

/// Grid search para grupos de 8 elementos
/// Con grupos tan pequeños, min/max directo + ±4 ULP debería ser suficiente
pub fn optimize_group(group: &[f32], config: &GridConfig) -> GroupParams {
    let q_max = config.q_max();
    
    // Con solo 8 elementos, min/max directo es óptimo
    let mut min_raw = f32::INFINITY;
    let mut max_raw = f32::NEG_INFINITY;
    for &val in group.iter() {
        if val < min_raw { min_raw = val; }
        if val > max_raw { max_raw = val; }
    }
    let scale_raw = (max_raw - min_raw).max(EPS);
    
    let min_f16 = f16::from_f32(min_raw);
    let scale_f16 = f16::from_f32(scale_raw);
    
    let mut best_mse = f32::INFINITY;
    let mut best_min = min_f16.to_f32();
    let mut best_scale = scale_f16.to_f32().max(EPS);
    
    // Grid search ±4 ULP (suficiente para grupos pequeños)
    let search_range: i16 = 4;
    
    for min_delta in -search_range..=search_range {
        let min_bits = min_f16.to_bits() as i32 + min_delta as i32;
        if min_bits < 0 { continue; }
        let test_min = f16::from_bits(min_bits as u16).to_f32();
        
        for scale_delta in -search_range..=search_range {
            let scale_bits = scale_f16.to_bits() as i32 + scale_delta as i32;
            if scale_bits <= 0 { continue; }
            let test_scale = f16::from_bits(scale_bits as u16).to_f32();
            if test_scale < EPS { continue; }
            
            let mut mse = 0.0f32;
            for &val in group.iter() {
                let q = ((val - test_min) / test_scale * q_max).round().clamp(0.0, q_max);
                let recon = test_min + q / q_max * test_scale;
                let diff = val - recon;
                mse += diff * diff;
            }
            mse /= group.len() as f32;
            
            if mse < best_mse {
                best_mse = mse;
                best_min = test_min;
                best_scale = test_scale;
            }
        }
    }
    
    GroupParams {
        min: best_min,
        scale: best_scale,
    }
}

pub fn optimize_superblock(
    block: &[f32; SUPER_BLOCK_SIZE],
    config: &GridConfig,
) -> [GroupParams; NUM_GROUPS] {
    let results: Vec<GroupParams> = (0..NUM_GROUPS)
        .into_par_iter()
        .map(|g| {
            let start = g * GROUP_SIZE;
            let end = start + GROUP_SIZE;
            optimize_group(&block[start..end], config)
        })
        .collect();
    
    let mut params = [GroupParams::default(); NUM_GROUPS];
    for (g, p) in results.into_iter().enumerate() {
        params[g] = p;
    }
    
    params
}

pub fn fast_superblock(block: &[f32; SUPER_BLOCK_SIZE]) -> [GroupParams; NUM_GROUPS] {
    let mut params = [GroupParams::default(); NUM_GROUPS];
    
    for g in 0..NUM_GROUPS {
        let start = g * GROUP_SIZE;
        let end = start + GROUP_SIZE;
        params[g] = compute_group_params(&block[start..end]);
    }
    
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    
    #[test]
    fn test_optimize_vs_fast() {
        let mut rng = rand::thread_rng();
        let mut block = [0.0f32; SUPER_BLOCK_SIZE];
        for i in 0..SUPER_BLOCK_SIZE {
            block[i] = rng.gen_range(-2.0..2.0);
        }
        
        let config = GridConfig::hq5k();
        let q_max = config.q_max();
        
        let fast = fast_superblock(&block);
        let optimized = optimize_superblock(&block, &config);
        
        // Calculate MSE for both
        let mut fast_mse = 0.0f32;
        let mut opt_mse = 0.0f32;
        
        for g in 0..NUM_GROUPS {
            let start = g * GROUP_SIZE;
            for i in 0..GROUP_SIZE {
                let val = block[start + i];
                
                // Fast
                let q_fast = ((val - fast[g].min) / fast[g].scale * q_max).round().clamp(0.0, q_max);
                let r_fast = fast[g].min + q_fast / q_max * fast[g].scale;
                fast_mse += (val - r_fast).powi(2);
                
                // Optimized
                let q_opt = ((val - optimized[g].min) / optimized[g].scale * q_max).round().clamp(0.0, q_max);
                let r_opt = optimized[g].min + q_opt / q_max * optimized[g].scale;
                opt_mse += (val - r_opt).powi(2);
            }
        }
        
        fast_mse /= SUPER_BLOCK_SIZE as f32;
        opt_mse /= SUPER_BLOCK_SIZE as f32;
        
        println!("Fast MSE: {:.6}, Optimized MSE: {:.6}", fast_mse, opt_mse);
        assert!(opt_mse <= fast_mse + 1e-6);
    }
}
