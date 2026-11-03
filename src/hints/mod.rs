// src/hints/mod.rs
// ============================================================================
// HINTS - Genera execution_hints para HNFv9.1
// ============================================================================
//
// Genera DOS formatos:
//   [0xA] execution_hints     - JSON (obligatorio, compatibilidad)
//   [0xB] exec_hints_bin      - Binario (preferido, O(1) parsing)
//
// ============================================================================

pub mod binary;
pub mod gemma4_moe_v2;
pub mod gemma4_moe_payload;

use std::path::Path;
use anyhow::Result;
use serde_json::{json, Value};

pub use binary::{build_execution_hints_binary, ExecutionHintsBin, TextModelConfigBin, VisionModelConfigBin};

/// Lee config.json de HuggingFace y genera execution_hints
pub fn build_execution_hints(model_dir: impl AsRef<Path>) -> Result<Value> {
    let config_path = model_dir.as_ref().join("config.json");
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)?
    )?;
    
    // Extraer valores con defaults
    let arch = config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string();
    
    let num_hidden_layers = config.get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    
    let hidden_size = config.get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;
    
    let intermediate_size = config.get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(11008) as usize;
    
    let vocab_size = config.get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as usize;
    
    let num_attention_heads = config.get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    
    let num_key_value_heads = config.get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;
    
    let head_dim = config.get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    
    let rope_theta = config.get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);
    
    let rms_norm_eps = config.get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6);
    
    let max_position_embeddings = config.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;
    
    let tie_word_embeddings = config.get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    // Detectar attention_type
    let attention_type = if num_key_value_heads == num_attention_heads {
        "mha"
    } else if num_key_value_heads == 1 {
        "mqa"
    } else {
        "gqa"
    };
    
    // Detectar mlp_type y activation basado en arquitectura
    let (mlp_type, mlp_activation) = match arch.as_str() {
        "gemma" | "gemma2" => ("geglu", "gelu"),
        _ => ("swiglu", "silu"),
    };
    
    // Detectar norm_type
    let norm_type = if arch.contains("bert") || arch.contains("gpt2") {
        "layernorm"
    } else {
        "rmsnorm"
    };
    
    // Detectar rope_type
    let rope_type = if arch.contains("llama3") {
        "llama3"
    } else if arch.contains("phi") {
        "su"
    } else {
        "default"
    };
    
    // Detectar si tiene biases
    let attention_bias = config.get("attention_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    let mlp_bias = config.get("mlp_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    // Construir JSON
    let hints = json!({
        "arch": arch,
        "dtype": "bf16",
        
        // Dimensiones
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        
        // Attention
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "attention_type": attention_type,
        "attention_bias": attention_bias,
        "qkv_layout": "separate",
        "use_qk_norm": false,
        "parallel_attention": false,
        "kv_layout": "BHSD",
        
        // MLP
        "mlp_type": mlp_type,
        "mlp_activation": mlp_activation,
        "mlp_bias": mlp_bias,
        
        // Normalization
        "norm_type": norm_type,
        "norm_bias": false,
        "rms_norm_eps": rms_norm_eps,
        "pre_norm": true,
        "final_norm": true,
        
        // RoPE
        "rope_type": rope_type,
        "rope_theta": rope_theta,
        "rope_dim": head_dim,
        "rope_interleaved": false,
        
        // Embeddings
        "tie_word_embeddings": tie_word_embeddings,
        "embedding_bias": false,
        "lm_head_bias": false,
        
        // Inference
        "supports_paged_attention": true,
        "supports_flash_attention": true,
        "supports_sdpa": true,
        "max_position_embeddings": max_position_embeddings,
        
        // Memory
        "workspace_mb": 512,
        "kv_cache_mb_per_1k_tokens": 32,
        
        // Startup hints
        "startup": {
            "priority_tensors": [
                "token_embedding.weight",
                "final_norm.weight",
                "lm_head.weight",
                "layer0.*",
                "layer1.*",
                "layer2.*"
            ],
            "priority_blocks": [0, 10],
            "pinned_memory_mb": 2048,
            "streaming_enabled": true,
            "warmup_layers": 3
        }
    });
    
    Ok(hints)
}
