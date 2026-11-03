// src/mapping/phi.rs
// ============================================================================
// PHI MAPPER - Mapea tensores Phi3/Phi4 a nombres canónicos
// ============================================================================
//
// Soporta: Phi-3, Phi-3.5, Phi-4, Phi-4-mini
//
// Características especiales:
// - QKV fusionado (qkv_proj en vez de q/k/v separados)
// - Gate+Up fusionado (gate_up_proj en vez de gate/up separados)
// - Partial RoPE (partial_rotary_factor, típicamente 0.75)
// - LongRoPE scaling con long_factor[] y short_factor[]
// - Tied embeddings (sin lm_head separado)
// - GQA (num_key_value_heads < num_attention_heads)
//
// v9.0.5: Soporte completo para Phi-4-mini-instruct
//
// ============================================================================

use regex::Regex;
use serde_json::{json, Value};

use super::traits::ModelMapper;
use super::types::{TensorMapping, QuantHint, TensorCategory};

#[derive(Debug, Clone)]
pub struct PhiConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    // Phi-specific
    pub partial_rotary_factor: f64,
    pub original_max_position_embeddings: usize,
    // LongRoPE scaling
    pub rope_scaling: Option<LongRopeScaling>,
}

#[derive(Debug, Clone)]
pub struct LongRopeScaling {
    pub scaling_type: String,  // "longrope"
    pub long_factor: Vec<f64>,
    pub short_factor: Vec<f64>,
}

impl PhiConfig {
    pub fn from_json(config: &Value) -> Self {
        // Parse LongRoPE scaling if present
        let rope_scaling = config.get("rope_scaling")
            .and_then(|rs| {
                if rs.is_null() {
                    return None;
                }
                
                let scaling_type = rs.get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("longrope")
                    .to_string();
                
                let long_factor = rs.get("long_factor")
                    .and_then(|f| f.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect::<Vec<_>>())
                    .unwrap_or_default();
                
                let short_factor = rs.get("short_factor")
                    .and_then(|f| f.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect::<Vec<_>>())
                    .unwrap_or_default();
                
                if !long_factor.is_empty() {
                    Some(LongRopeScaling { 
                        scaling_type, 
                        long_factor, 
                        short_factor 
                    })
                } else {
                    None
                }
            });
        
        Self {
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(3072) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(8192) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(24) as usize,
            num_key_value_heads: config["num_key_value_heads"]
                .as_u64()
                .or(config["num_attention_heads"].as_u64())
                .unwrap_or(24) as usize,
            vocab_size: config["vocab_size"].as_u64().unwrap_or(200064) as usize,
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(131072) as usize,
            rope_theta: config["rope_theta"].as_f64().unwrap_or(10000.0),
            rms_norm_eps: config["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            tie_word_embeddings: config["tie_word_embeddings"].as_bool().unwrap_or(true),
            partial_rotary_factor: config["partial_rotary_factor"].as_f64().unwrap_or(0.75),
            original_max_position_embeddings: config["original_max_position_embeddings"]
                .as_u64().unwrap_or(4096) as usize,
            rope_scaling,
        }
    }
}

pub struct PhiMapper {
    config: PhiConfig,
    re_embed: Regex,
    re_lm_head: Regex,
    re_final_norm: Regex,
    // Phi usa QKV fusionado
    re_attn_qkv: Regex,
    re_attn_o: Regex,
    // Phi usa gate_up fusionado
    re_mlp_gate_up: Regex,
    re_mlp_down: Regex,
    re_input_norm: Regex,
    re_post_attn_norm: Regex,
}

impl PhiMapper {
    pub fn new(config: PhiConfig) -> Self {
        Self {
            config,
            re_embed: Regex::new(r"^model\.embed_tokens\.weight$").unwrap(),
            re_lm_head: Regex::new(r"^lm_head\.weight$").unwrap(),
            re_final_norm: Regex::new(r"^model\.norm\.weight$").unwrap(),
            // Phi usa qkv_proj fusionado (no q_proj, k_proj, v_proj separados)
            re_attn_qkv: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$").unwrap(),
            re_attn_o: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$").unwrap(),
            // Phi usa gate_up_proj fusionado (no gate_proj, up_proj separados)
            re_mlp_gate_up: Regex::new(r"^model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight$").unwrap(),
            re_mlp_down: Regex::new(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$").unwrap(),
            re_input_norm: Regex::new(r"^model\.layers\.(\d+)\.input_layernorm\.weight$").unwrap(),
            re_post_attn_norm: Regex::new(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$").unwrap(),
        }
    }
    
    pub fn from_json(config: &Value) -> Self {
        Self::new(PhiConfig::from_json(config))
    }
}

impl ModelMapper for PhiMapper {
    fn name(&self) -> &str {
        "phi"
    }
    
    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        if self.should_ignore(name) {
            return None;
        }
        
        // ═══════════════════════════════════════════════════════════════
        // EMBEDDINGS (FP16)
        // ═══════════════════════════════════════════════════════════════
        
        if self.re_embed.is_match(name) {
            return Some(TensorMapping::new(
                "token_embedding.weight",
                QuantHint::FP16,
                TensorCategory::Embedding,
            ));
        }
        
        // Phi usa tied embeddings, pero por si hay lm_head explícito
        if self.re_lm_head.is_match(name) {
            return Some(TensorMapping::new(
                "lm_head.weight",
                QuantHint::FP16,
                TensorCategory::LMHead,
            ));
        }
        
        if self.re_final_norm.is_match(name) {
            return Some(TensorMapping::new(
                "final_norm.weight",
                QuantHint::FP16,
                TensorCategory::Norm,
            ));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // ATTENTION (HQ5K) - QKV fusionado
        // ═══════════════════════════════════════════════════════════════
        
        // qkv_proj fusionado: shape [3 * num_heads * head_dim, hidden_size]
        // El runtime debe separar Q, K, V o usar directamente
        if let Some(caps) = self.re_attn_qkv.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.attn.qkv_proj.weight", layer),
                QuantHint::HQ5K,
                TensorCategory::Attention,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_attn_o.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.attn.o_proj.weight", layer),
                QuantHint::HQ5K,
                TensorCategory::Attention,
            ).with_layer(layer));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // MLP (HQ4K) - Gate+Up fusionado
        // ═══════════════════════════════════════════════════════════════
        
        // gate_up_proj fusionado: shape [2 * intermediate_size, hidden_size]
        // Primera mitad es gate, segunda mitad es up
        if let Some(caps) = self.re_mlp_gate_up.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.mlp.gate_up.weight", layer),
                QuantHint::HQ4K,
                TensorCategory::MLP,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_mlp_down.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.mlp.down.weight", layer),
                QuantHint::HQ4K,
                TensorCategory::MLP,
            ).with_layer(layer));
        }
        
        // ═══════════════════════════════════════════════════════════════
        // NORMS (FP16)
        // ═══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_input_norm.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.ln_attn_in.weight", layer),
                QuantHint::FP16,
                TensorCategory::Norm,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_post_attn_norm.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            return Some(TensorMapping::new(
                format!("layer{}.ln_attn_out.weight", layer),
                QuantHint::FP16,
                TensorCategory::Norm,
            ).with_layer(layer));
        }
        
        None
    }
    
    fn execution_hints(&self) -> Value {
        let c = &self.config;
        
        let attention_type = if c.num_key_value_heads == c.num_attention_heads {
            "mha"
        } else if c.num_key_value_heads == 1 {
            "mqa"
        } else {
            "gqa"
        };
        
        let head_dim = c.hidden_size / c.num_attention_heads;
        let rope_dim = ((head_dim as f64) * c.partial_rotary_factor) as usize;
        
        // Determinar rope_type
        let rope_type = match &c.rope_scaling {
            Some(rs) => rs.scaling_type.as_str(),
            None => "default",
        };
        
        let mut hints = json!({
            // IDENTIFICACIÓN (OBLIGATORIO)
            "arch": "phi",
            "dtype": "bf16",
            
            // DIMENSIONES (OBLIGATORIO)
            "num_hidden_layers": c.num_hidden_layers,
            "hidden_size": c.hidden_size,
            "intermediate_size": c.intermediate_size,
            "vocab_size": c.vocab_size,
            
            // ATTENTION (OBLIGATORIO) - QKV FUSIONADO
            "num_attention_heads": c.num_attention_heads,
            "num_key_value_heads": c.num_key_value_heads,
            "head_dim": head_dim,
            "attention_type": attention_type,
            "attention_bias": false,
            "qkv_layout": "fused",  // CRÍTICO: QKV fusionado
            "use_qk_norm": false,
            "parallel_attention": false,
            "kv_layout": "BHSD",
            
            // MLP (OBLIGATORIO) - GATE+UP FUSIONADO
            "mlp_type": "swiglu_fused",  // CRÍTICO: gate+up fusionado
            "mlp_activation": "silu",
            "mlp_bias": false,
            
            // NORMALIZATION (OBLIGATORIO)
            "norm_type": "rmsnorm",
            "norm_bias": false,
            "rms_norm_eps": c.rms_norm_eps,
            "pre_norm": true,
            "final_norm": true,
            
            // RoPE (OBLIGATORIO) - PARTIAL + LONGROPE
            "rope_type": rope_type,
            "rope_theta": c.rope_theta,
            "rope_dim": rope_dim,
            "rope_partial": true,  // CRÍTICO: solo 75% dimensiones
            "partial_rotary_factor": c.partial_rotary_factor,
            "rope_interleaved": false,
            
            // EMBEDDINGS (OBLIGATORIO)
            "tie_word_embeddings": c.tie_word_embeddings,
            "embedding_bias": false,
            "lm_head_bias": false,
            
            // CONTEXT
            "max_position_embeddings": c.max_position_embeddings,
            "original_max_position_embeddings": c.original_max_position_embeddings,
            
            // INFERENCE CAPABILITIES
            "supports_flash_attention": true,
            "supports_paged_attention": true,
            "supports_sdpa": true
        });
        
        // Añadir LongRoPE scaling si está presente
        if let Some(rs) = &c.rope_scaling {
            hints["rope_scaling"] = json!({
                "type": rs.scaling_type,
                "long_factor": rs.long_factor,
                "short_factor": rs.short_factor
            });
        }
        
        hints
    }
    
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
    
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
