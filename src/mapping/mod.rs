// src/mapping/mod.rs
// ============================================================================
// MAPPING - Sistema de mapeo de tensores HELIOS
// ============================================================================

pub mod types;
pub mod traits;
pub mod factory;
pub mod qwen2;
pub mod llama;
pub mod clip;
pub mod phi;  // AÑADIDO
pub mod gemma4;
pub mod gemma4_moe;
pub mod gemma4_vision;
pub mod gemma4_unified_vision;
pub mod gemma4_unified_audio;

// Re-exports
pub use types::{BlockType, QuantHint, TensorCategory, TensorMapping};
pub use traits::ModelMapper;
pub use factory::{create_mapper, create_mapper_for_block, detect_architecture, load_config};

pub mod moe_inventory;
