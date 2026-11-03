// src/hqs/mod.rs
// ============================================================================
// HQS - HELIOS Quantization System
// ============================================================================

pub mod common;
pub mod grid_search;
pub mod compact;
pub mod hq8;

// Re-exports
pub use common::*;
pub use grid_search::GridConfig;
pub use compact::{quantize_hq31k, dequantize_hq31k, quantize_hq41k, quantize_hq51k,
                  quantize_hq42k, dequantize_hq42k, quantize_hq52k, dequantize_hq52k,
                  quantize_hq62k, dequantize_hq62k, hq31k_size, hq41k_size, hq42k_size, hq52k_size,
                  hq51k_size, hq62k_size, HQ31K_BLOCK_SIZE, HQ41K_BLOCK_SIZE,
                  HQ42K_BLOCK_SIZE, HQ51K_BLOCK_SIZE, HQ52K_BLOCK_SIZE, HQ62K_BLOCK_SIZE};
pub use hq8::{quantize_hq8k, dequantize_hq8k, hq8k_size, HQ8K_BLOCK_SIZE};

/// Formato de cuantización
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    FP16,
    HQ3K,
    HQ31K,  // compacto (header 40B) — 4.25 bits/peso
    HQ41K,  // v9.1.1: compacto (header 40B) — 5.25 bits/peso
    HQ51K,  // v9.1.1: compacto (header 40B) — 6.25 bits/peso
    // DESCARTADO el 2026-11-02: 2 bits/peso de metadatos y 1,11 % de error frente al
    // 0,56 % de HQ8K con casi los mismos bits. Solo se conserva para leer HNF antiguos
    // y para el laboratorio; ninguna conversión nueva debe emitirlo.
    HQ62K,  // 6-bit payload + metadata de 8 bits — 8.25 bpw
    HQ42K,  // simétrico, metadata de paso a 5 bits — 4.75 bpw
    HQ52K,  // simétrico, metadata de paso a 5 bits — 5.75 bpw
    HQ8K,   // simétrico, códigos i8 y subescala u8 por cada 32 — 8.5 bpw (src/hqs/hq8.rs)
}

impl QuantFormat {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "FP16" | "FLOAT16" => Some(Self::FP16),
            "HQ3K" | "3BIT" => Some(Self::HQ3K),
            // HQ4K/HQ5K (cabecera 128 B, 8,0 y 9,0 bpw) se retiraron el
            // 2026-08-01: ningun HNF los usaba y obligaban a implementar cada
            // mejora del cuantizador dos veces. Los nombres se conservan
            // apuntando al compacto para no romper lineas de comando: pedir
            // --quant HQ4K produce ahora HQ4.1K.
            "HQ4K" | "4BIT" => Some(Self::HQ41K),
            "HQ5K" | "5BIT" => Some(Self::HQ51K),
            "HQ31K" | "HQ3.1K" => Some(Self::HQ31K),
            "HQ41K" | "HQ4.1K" => Some(Self::HQ41K),
            "HQ51K" | "HQ5.1K" => Some(Self::HQ51K),
            "HQ42K" | "HQ4.2K" => Some(Self::HQ42K),
            "HQ52K" | "HQ5.2K" => Some(Self::HQ52K),
            "HQ8K" | "HQ8" => Some(Self::HQ8K),
            _ => None,
        }
    }

    /// Conservada por compatibilidad: los formatos ya son siempre compactos.
    pub fn to_compact(&self) -> Self { *self }

    pub fn code(&self) -> u8 {
        match self {
            Self::FP16 => 0x00,
            Self::HQ3K => 0x01,
            Self::HQ41K => 0x04,
            Self::HQ51K => 0x05,
            Self::HQ31K => 0x06,
            Self::HQ62K => 0x07,
            Self::HQ42K => 0x08,
            Self::HQ52K => 0x09,
            Self::HQ8K => 0x0A,
        }
    }

    pub fn bits(&self) -> u8 {
        match self {
            Self::FP16 => 16,
            Self::HQ3K => 3,
            Self::HQ31K => 3,
            Self::HQ41K => 4,
            Self::HQ51K => 5,
            Self::HQ62K => 6,
            Self::HQ42K => 4,
            Self::HQ52K => 5,
            Self::HQ8K => 8,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::FP16 => 2,  // 2 bytes per element
            Self::HQ3K => HQ3K_BLOCK_SIZE,
            Self::HQ31K => HQ31K_BLOCK_SIZE,
            Self::HQ41K => HQ41K_BLOCK_SIZE,
            Self::HQ51K => HQ51K_BLOCK_SIZE,
            Self::HQ62K => HQ62K_BLOCK_SIZE,
            Self::HQ42K => HQ42K_BLOCK_SIZE,
            Self::HQ52K => HQ52K_BLOCK_SIZE,
            Self::HQ8K => HQ8K_BLOCK_SIZE,
        }
    }

    /// Calcula bytes necesarios para N elementos
    pub fn size_for(&self, numel: usize) -> usize {
        match self {
            Self::FP16 => numel * 2,
            Self::HQ3K => {
                let num_blocks = (numel + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
                num_blocks * HQ3K_BLOCK_SIZE
            }
            Self::HQ31K => hq31k_size(numel),
            Self::HQ41K => hq41k_size(numel),
            Self::HQ51K => hq51k_size(numel),
            Self::HQ62K => hq62k_size(numel),
            Self::HQ42K => hq42k_size(numel),
            Self::HQ52K => hq52k_size(numel),
            Self::HQ8K => hq8k_size(numel),
        }
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FP16 => write!(f, "FP16"),
            Self::HQ3K => write!(f, "HQ3K"),
            Self::HQ31K => write!(f, "HQ31K"),
            Self::HQ41K => write!(f, "HQ41K"),
            Self::HQ51K => write!(f, "HQ51K"),
            Self::HQ62K => write!(f, "HQ62K"),
            Self::HQ42K => write!(f, "HQ42K"),
            Self::HQ52K => write!(f, "HQ52K"),
            Self::HQ8K => write!(f, "HQ8K"),
        }
    }
}

/// Cuantiza datos según el formato especificado
pub fn quantize(data: &[f32], format: QuantFormat, use_mse: bool) -> Vec<u8> {
    match format {
        QuantFormat::FP16 => {
            // Convertir a f16
            data.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        }
        QuantFormat::HQ3K => {
            // TODO: Implementar HQ3K
            unimplemented!("HQ3K not yet implemented")
        }
        QuantFormat::HQ31K => quantize_hq31k(data, use_mse),
        QuantFormat::HQ41K => quantize_hq41k(data, use_mse),
        QuantFormat::HQ51K => quantize_hq51k(data, use_mse),
        QuantFormat::HQ62K => quantize_hq62k(data, use_mse),
        QuantFormat::HQ42K => quantize_hq42k(data),
        QuantFormat::HQ52K => quantize_hq52k(data),
        QuantFormat::HQ8K => quantize_hq8k(data),
    }
}

/// Dequantiza datos según el formato
pub fn dequantize(data: &[u8], format: QuantFormat, numel: usize) -> Vec<f32> {
    match format {
        QuantFormat::FP16 => {
            data.chunks_exact(2)
                .map(|chunk| {
                    let bytes: [u8; 2] = [chunk[0], chunk[1]];
                    half::f16::from_le_bytes(bytes).to_f32()
                })
                .take(numel)
                .collect()
        }
        QuantFormat::HQ3K => {
            unimplemented!("HQ3K dequantize not yet implemented")
        }
        QuantFormat::HQ31K => dequantize_hq31k(data, numel),
        QuantFormat::HQ62K => dequantize_hq62k(data, numel),
        QuantFormat::HQ42K => dequantize_hq42k(data, numel),
        QuantFormat::HQ52K => dequantize_hq52k(data, numel),
        QuantFormat::HQ8K => dequantize_hq8k(data, numel),
        QuantFormat::HQ41K | QuantFormat::HQ51K => {
            unimplemented!("compact dequantize pendiente (usar kernels del engine)")
        }
    }
}

// Formatos de HQS v4: códec x.3 de GPT (atención y embedding) y el bloque afín v4 (MLP).
pub mod vx3;
pub mod v4;
