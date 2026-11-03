// src/dictionary/mod.rs
// ============================================================================
// HELIOS DICTIONARY v9.0.2 - Única fuente de verdad para nombres canónicos
// ============================================================================
//
// Cualquier tensor que no esté aquí es INVÁLIDO y no debe escribirse.
//
// ============================================================================

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

pub const DICTIONARY_VERSION: &str = "9.0.2";

// ============================================================================
// PATRONES POR BLOQUE
// ============================================================================
// {N} = índice de capa (0, 1, 2, ...)
// {E} = índice de experto (0, 1, 2, ...)
// ============================================================================

pub const TEXT_MODEL_PATTERNS: &[&str] = &[
    // §2.1 EMBEDDINGS
    "token_embedding.weight",
    "lm_head.weight",
    "lm_head.bias",
    
    // §2.2 FINAL NORM
    "final_norm.weight",
    "final_norm.bias",
    
    // §2.3 ATTENTION (por capa)
    "layer{N}.attn.q_proj.weight",
    "layer{N}.attn.q_proj.bias",
    "layer{N}.attn.k_proj.weight",
    "layer{N}.attn.k_proj.bias",
    "layer{N}.attn.v_proj.weight",
    "layer{N}.attn.v_proj.bias",
    "layer{N}.attn.o_proj.weight",
    "layer{N}.attn.o_proj.bias",
    "layer{N}.attn.qkv_proj.weight",  // QKV fusionado (Phi3/4)
    "layer{N}.attn.qkv_proj.bias",
    "layer{N}.attn.q_norm.weight",
    "layer{N}.attn.k_norm.weight",
    
    // §2.4 LAYER NORMS (por capa)
    "layer{N}.ln_attn_in.weight",
    "layer{N}.ln_attn_in.bias",
    "layer{N}.ln_attn_out.weight",
    "layer{N}.ln_attn_out.bias",
    "layer{N}.ln_pre_ffn.weight",
    "layer{N}.ln_pre_ffn.bias",
    "layer{N}.ln_post_ffn.weight",
    "layer{N}.ln_post_ffn.bias",
    
    // §2.5 MLP DENSO (por capa)
    "layer{N}.mlp.gate.weight",
    "layer{N}.mlp.gate.bias",
    "layer{N}.mlp.up.weight",
    "layer{N}.mlp.up.bias",
    "layer{N}.mlp.down.weight",
    "layer{N}.mlp.down.bias",
    "layer{N}.mlp.gate_up.weight",  // Gate+Up fusionado (Phi3/4)
    "layer{N}.mlp.gate_up.bias",
    
    // Gemma 4 decoder norms/scalar and packed MoE (text block, no global router).
    "layer{N}.ln_attn_post.weight",
    "layer{N}.ln_mlp_in.weight",
    "layer{N}.ln_mlp_post.weight",
    "layer{N}.ln_moe_in.weight",
    "layer{N}.ln_mlp_branch_post.weight",
    "layer{N}.ln_moe_branch_post.weight",
    "layer{N}.layer_scalar",
    "layer{N}.moe.router.weight",
    "layer{N}.moe.router.input_scale",
    "layer{N}.moe.router.expert_scale",
    "layer{N}.moe.experts.gate_up.weight",
    "layer{N}.moe.experts.down.weight",

    // §2.6 MoE (por capa)
    "layer{N}.moe.gate.weight",
    "layer{N}.moe.gate.bias",
    "layer{N}.moe.experts.{E}.gate.weight",
    "layer{N}.moe.experts.{E}.up.weight",
    "layer{N}.moe.experts.{E}.down.weight",
];

pub const VISION_PATTERNS: &[&str] = &[
    // §3.1 PATCH EMBEDDING
    "vision.patch_embed.weight",
    "vision.patch_embed.bias",
    "vision.pos_embed.weight",
    "vision.class_embed.weight",
    "vision.cls_token",
    
    // §3.2 VISION ENCODER (por capa)
    "vision.layer{N}.attn.q_proj.weight",
    "vision.layer{N}.attn.q_proj.bias",
    "vision.layer{N}.attn.k_proj.weight",
    "vision.layer{N}.attn.k_proj.bias",
    "vision.layer{N}.attn.v_proj.weight",
    "vision.layer{N}.attn.v_proj.bias",
    "vision.layer{N}.attn.o_proj.weight",
    "vision.layer{N}.attn.o_proj.bias",
    "vision.layer{N}.ln1.weight",
    "vision.layer{N}.ln1.bias",
    "vision.layer{N}.ln2.weight",
    "vision.layer{N}.ln2.bias",
    "vision.layer{N}.mlp.fc1.weight",
    "vision.layer{N}.mlp.fc1.bias",
    "vision.layer{N}.mlp.fc2.weight",
    "vision.layer{N}.mlp.fc2.bias",
    
    // §3.3 VISION FINAL
    "vision.post_layernorm.weight",
    "vision.post_layernorm.bias",
    "vision.pre_layernorm.weight",
    "vision.pre_layernorm.bias",
    "vision.head.weight",
    "vision.head.bias",

    // §3.4 GEMMA 4 VISION
    "vision.patch_embed.input_proj.weight",
    "vision.patch_embed.position_embedding.weight",
    "vision.layer{N}.attn.q_proj.input_min",
    "vision.layer{N}.attn.q_proj.input_max",
    "vision.layer{N}.attn.q_proj.output_min",
    "vision.layer{N}.attn.q_proj.output_max",
    "vision.layer{N}.attn.k_proj.input_min",
    "vision.layer{N}.attn.k_proj.input_max",
    "vision.layer{N}.attn.k_proj.output_min",
    "vision.layer{N}.attn.k_proj.output_max",
    "vision.layer{N}.attn.v_proj.input_min",
    "vision.layer{N}.attn.v_proj.input_max",
    "vision.layer{N}.attn.v_proj.output_min",
    "vision.layer{N}.attn.v_proj.output_max",
    "vision.layer{N}.attn.o_proj.input_min",
    "vision.layer{N}.attn.o_proj.input_max",
    "vision.layer{N}.attn.o_proj.output_min",
    "vision.layer{N}.attn.o_proj.output_max",
    "vision.layer{N}.attn.q_norm.weight",
    "vision.layer{N}.attn.k_norm.weight",
    "vision.layer{N}.ln_attn_in.weight",
    "vision.layer{N}.ln_attn_post.weight",
    "vision.layer{N}.ln_mlp_in.weight",
    "vision.layer{N}.ln_mlp_post.weight",
    "vision.layer{N}.mlp.gate.weight",
    "vision.layer{N}.mlp.gate.input_min",
    "vision.layer{N}.mlp.gate.input_max",
    "vision.layer{N}.mlp.gate.output_min",
    "vision.layer{N}.mlp.gate.output_max",
    "vision.layer{N}.mlp.up.weight",
    "vision.layer{N}.mlp.up.input_min",
    "vision.layer{N}.mlp.up.input_max",
    "vision.layer{N}.mlp.up.output_min",
    "vision.layer{N}.mlp.up.output_max",
    "vision.layer{N}.mlp.down.weight",
    "vision.layer{N}.mlp.down.input_min",
    "vision.layer{N}.mlp.down.input_max",
    "vision.layer{N}.mlp.down.output_min",
    "vision.layer{N}.mlp.down.output_max",
    "vision.projector.weight",
    "vision.projector.std_bias",
    "vision.projector.std_scale",

    // Gemma 4 «unified» (12B): embebedor sin torre, parches crudos -> decoder.
    "vision.patch_ln1.weight",
    "vision.patch_ln1.bias",
    "vision.patch_dense.weight",
    "vision.patch_dense.bias",
    "vision.patch_ln2.weight",
    "vision.patch_ln2.bias",
    "vision.pos_embedding",
    "vision.pos_norm.weight",
    "vision.pos_norm.bias",
    "vision.mm_projection.weight",

    // §7 PROJECTOR
    "projector.vision.linear1.weight",
    "projector.vision.linear1.bias",
    "projector.vision.linear2.weight",
    "projector.vision.linear2.bias",
];

pub const AUDIO_PATTERNS: &[&str] = &[
    "audio.conv1.weight",
    "audio.conv1.bias",
    "audio.conv2.weight",
    "audio.conv2.bias",
    "audio.pos_embed.weight",
    "audio.layer{N}.attn.q_proj.weight",
    "audio.layer{N}.attn.k_proj.weight",
    "audio.layer{N}.attn.v_proj.weight",
    "audio.layer{N}.attn.o_proj.weight",
    "audio.layer{N}.ln1.weight",
    "audio.layer{N}.ln2.weight",
    "audio.layer{N}.mlp.fc1.weight",
    "audio.layer{N}.mlp.fc2.weight",
    "audio.ln_post.weight",
    "projector.audio.linear1.weight",
    "projector.audio.linear1.bias",
    "projector.audio.linear2.weight",
    "projector.audio.linear2.bias",
    // Gemma 4 «unified» (12B): la onda cruda entra por una sola proyección.
    "audio.mm_projection.weight",
];

pub const VIDEO_PATTERNS: &[&str] = &[
    "video.patch_embed.weight",
    "video.pos_embed.weight",
    "video.temporal_embed.weight",
    "video.layer{N}.attn.q_proj.weight",
    "video.layer{N}.attn.k_proj.weight",
    "video.layer{N}.attn.v_proj.weight",
    "video.layer{N}.attn.o_proj.weight",
    "video.layer{N}.temporal_attn.q_proj.weight",
    "video.layer{N}.temporal_attn.k_proj.weight",
    "video.layer{N}.temporal_attn.v_proj.weight",
    "video.layer{N}.temporal_attn.o_proj.weight",
    "video.layer{N}.mlp.fc1.weight",
    "video.layer{N}.mlp.fc2.weight",
    "video.ln_post.weight",
    "projector.video.linear1.weight",
    "projector.video.linear1.bias",
    "projector.video.linear2.weight",
    "projector.video.linear2.bias",
];

pub const SPATIAL_3D_PATTERNS: &[&str] = &[
    "spatial.point_embed.weight",
    "spatial.pos_embed.weight",
    "spatial.layer{N}.attn.q_proj.weight",
    "spatial.layer{N}.attn.k_proj.weight",
    "spatial.layer{N}.attn.v_proj.weight",
    "spatial.layer{N}.attn.o_proj.weight",
    "spatial.layer{N}.mlp.fc1.weight",
    "spatial.layer{N}.mlp.fc2.weight",
    "spatial.ln_post.weight",
];

pub const EXPERT_ROUTER_PATTERNS: &[&str] = &[
    "expert_router.global_gate.weight",
    "expert_router.expert_embeddings",
    "expert_router.layer{N}.aux_loss",
];

// ============================================================================
// VALIDADOR
// ============================================================================

fn pattern_to_regex(pattern: &str) -> String {
    let mut regex = pattern.replace('.', r"\.");
    regex = regex.replace("{N}", r"\d+");
    regex = regex.replace("{E}", r"\d+");
    format!("^{}$", regex)
}

static ALL_PATTERNS_REGEX: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let mut patterns = Vec::new();
    
    // Text model
    for p in TEXT_MODEL_PATTERNS {
        patterns.push(Regex::new(&pattern_to_regex(p)).unwrap());
    }
    
    // Vision
    for p in VISION_PATTERNS {
        patterns.push(Regex::new(&pattern_to_regex(p)).unwrap());
    }
    
    // Audio
    for p in AUDIO_PATTERNS {
        patterns.push(Regex::new(&pattern_to_regex(p)).unwrap());
    }
    
    // Video
    for p in VIDEO_PATTERNS {
        patterns.push(Regex::new(&pattern_to_regex(p)).unwrap());
    }
    
    // Spatial 3D
    for p in SPATIAL_3D_PATTERNS {
        patterns.push(Regex::new(&pattern_to_regex(p)).unwrap());
    }
    
    // Expert router
    for p in EXPERT_ROUTER_PATTERNS {
        patterns.push(Regex::new(&pattern_to_regex(p)).unwrap());
    }
    
    // Code exec (text model con prefijo "code.")
    for p in TEXT_MODEL_PATTERNS {
        let prefixed = format!("code.{}", p);
        patterns.push(Regex::new(&pattern_to_regex(&prefixed)).unwrap());
    }
    
    // Cortex (text model con prefijo "cortex.")
    for p in TEXT_MODEL_PATTERNS {
        let prefixed = format!("cortex.{}", p);
        patterns.push(Regex::new(&pattern_to_regex(&prefixed)).unwrap());
    }
    
    patterns
});

/// Valida que un nombre canónico exista en el diccionario HELIOS.
pub fn validate_tensor_name(name: &str) -> bool {
    for regex in ALL_PATTERNS_REGEX.iter() {
        if regex.is_match(name) {
            return true;
        }
    }
    false
}

/// Validador con caché y reporte de errores
#[derive(Default)]
pub struct DictionaryValidator {
    strict: bool,
    valid: HashSet<String>,
    invalid: HashSet<String>,
}

impl DictionaryValidator {
    pub fn new(strict: bool) -> Self {
        Self {
            strict,
            valid: HashSet::new(),
            invalid: HashSet::new(),
        }
    }
    
    pub fn validate(&mut self, name: &str) -> bool {
        if self.valid.contains(name) {
            return true;
        }
        if self.invalid.contains(name) {
            return false;
        }
        
        if validate_tensor_name(name) {
            self.valid.insert(name.to_string());
            true
        } else {
            self.invalid.insert(name.to_string());
            if self.strict {
                eprintln!("[DICT ERROR] Tensor no válido: {}", name);
            }
            false
        }
    }
    
    pub fn valid_count(&self) -> usize {
        self.valid.len()
    }
    
    pub fn invalid_count(&self) -> usize {
        self.invalid.len()
    }
    
    pub fn invalid_tensors(&self) -> Vec<&str> {
        self.invalid.iter().map(|s| s.as_str()).collect()
    }
    
    pub fn report(&self) {
        if !self.invalid.is_empty() {
            eprintln!("\n[DICTIONARY REPORT]");
            eprintln!("  Valid:   {}", self.valid.len());
            eprintln!("  Invalid: {}", self.invalid.len());
            if self.invalid.len() <= 10 {
                for name in &self.invalid {
                    eprintln!("    - {}", name);
                }
            } else {
                for name in self.invalid.iter().take(10) {
                    eprintln!("    - {}", name);
                }
                eprintln!("    ... and {} more", self.invalid.len() - 10);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_model_patterns() {
        assert!(validate_tensor_name("token_embedding.weight"));
        assert!(validate_tensor_name("lm_head.weight"));
        assert!(validate_tensor_name("final_norm.weight"));
        assert!(validate_tensor_name("layer0.attn.q_proj.weight"));
        assert!(validate_tensor_name("layer15.attn.o_proj.weight"));
        assert!(validate_tensor_name("layer0.mlp.gate.weight"));
        assert!(validate_tensor_name("layer0.mlp.down.weight"));
        assert!(validate_tensor_name("layer0.ln_attn_in.weight"));
    }
    
    #[test]
    fn test_vision_patterns() {
        assert!(validate_tensor_name("vision.patch_embed.weight"));
        assert!(validate_tensor_name("vision.pos_embed.weight"));
        assert!(validate_tensor_name("vision.layer0.attn.q_proj.weight"));
        assert!(validate_tensor_name("vision.layer23.attn.o_proj.weight"));
        assert!(validate_tensor_name("vision.layer0.mlp.fc1.weight"));
        assert!(validate_tensor_name("vision.post_layernorm.weight"));
        assert!(validate_tensor_name("vision.patch_embed.input_proj.weight"));
        assert!(validate_tensor_name("vision.layer12.attn.q_proj.input_max"));
        assert!(validate_tensor_name("vision.layer15.ln_mlp_post.weight"));
        assert!(validate_tensor_name("vision.layer3.mlp.gate.output_min"));
        assert!(validate_tensor_name("vision.projector.weight"));
    }
    
    #[test]
    fn test_invalid_patterns() {
        assert!(!validate_tensor_name("invalid.tensor.name"));
        assert!(!validate_tensor_name("layer0.attn.qproj.weight")); // falta _
        assert!(!validate_tensor_name("vision.layer0.attn.out_proj.weight")); // debe ser o_proj
    }
    
    #[test]
    fn test_code_exec_patterns() {
        assert!(validate_tensor_name("code.token_embedding.weight"));
        assert!(validate_tensor_name("code.layer0.attn.q_proj.weight"));
    }
    
    #[test]
    fn test_cortex_patterns() {
        assert!(validate_tensor_name("cortex.token_embedding.weight"));
        assert!(validate_tensor_name("cortex.layer0.mlp.down.weight"));
    }
}
