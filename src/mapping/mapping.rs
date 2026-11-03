// src/mapping/mod.rs
// ============================================================================
// MAPPING - Diccionario de nombres canónicos HELIOS
// ============================================================================

use std::path::Path;
use anyhow::Result;

/// Formato de cuantización específico
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorQuantFormat {
    FP16,
    HQ4K,
    HQ5K,
}

/// Resultado del mapeo de un tensor
#[derive(Debug, Clone)]
pub struct TensorMapping {
    pub canonical_name: String,
    pub block_id: usize,
    pub should_quantize: bool,
    pub quant_format: TensorQuantFormat,
}

/// Trait para mappers de diferentes arquitecturas
pub trait TensorMapper {
    fn name(&self) -> &str;
    fn map_tensor(&self, original_name: &str) -> Option<TensorMapping>;
}

/// Mapper genérico basado en patrones
pub struct GenericMapper {
    arch: String,
    patterns: Vec<(regex::Regex, String, usize, TensorQuantFormat)>,
}

impl GenericMapper {
    pub fn new(arch: &str) -> Self {
        let patterns = Self::build_patterns(arch);
        Self {
            arch: arch.to_string(),
            patterns,
        }
    }
    
    fn build_patterns(_arch: &str) -> Vec<(regex::Regex, String, usize, TensorQuantFormat)> {
        let mut patterns = Vec::new();
        
        // ============================================================
        // POLÍTICA DE CUANTIZACIÓN:
        // - Atención (q, k, v, o): HQ5K (más precisión)
        // - MLP (gate, up, down): HQ4K (más compresión)
        // - Embeddings, norms, biases: FP16 (sin cuantizar)
        // ============================================================
        
        let common_patterns: Vec<(&str, &str, usize, TensorQuantFormat)> = vec![
            // ========== FP16 (no cuantizar) ==========
            // Embeddings
            (r"^model\.embed_tokens\.weight$", "token_embedding.weight", 0, TensorQuantFormat::FP16),
            (r"^language_model\.model\.embed_tokens\.weight$", "token_embedding.weight", 0, TensorQuantFormat::FP16),
            
            // LM Head
            (r"^lm_head\.weight$", "lm_head.weight", 0, TensorQuantFormat::FP16),
            (r"^language_model\.lm_head\.weight$", "lm_head.weight", 0, TensorQuantFormat::FP16),
            
            // Final norm
            (r"^model\.norm\.weight$", "final_norm.weight", 0, TensorQuantFormat::FP16),
            (r"^language_model\.model\.norm\.weight$", "final_norm.weight", 0, TensorQuantFormat::FP16),
            
            // Layer norms
            (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", "layer{}.ln_attn_in.weight", 0, TensorQuantFormat::FP16),
            (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", "layer{}.ln_attn_out.weight", 0, TensorQuantFormat::FP16),
            (r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$", "layer{}.ln_pre_ffn.weight", 0, TensorQuantFormat::FP16),
            (r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$", "layer{}.ln_post_ffn.weight", 0, TensorQuantFormat::FP16),
            
            // Attention biases
            (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$", "layer{}.attn.q_proj.bias", 0, TensorQuantFormat::FP16),
            (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$", "layer{}.attn.k_proj.bias", 0, TensorQuantFormat::FP16),
            (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$", "layer{}.attn.v_proj.bias", 0, TensorQuantFormat::FP16),
            (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.bias$", "layer{}.attn.o_proj.bias", 0, TensorQuantFormat::FP16),
            
            // ========== HQ5K (atención - más precisión) ==========
            (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "layer{}.attn.q_proj.weight", 0, TensorQuantFormat::HQ5K),
            (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "layer{}.attn.k_proj.weight", 0, TensorQuantFormat::HQ5K),
            (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "layer{}.attn.v_proj.weight", 0, TensorQuantFormat::HQ5K),
            (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "layer{}.attn.o_proj.weight", 0, TensorQuantFormat::HQ5K),
            
            // ========== HQ4K (MLP - más compresión) ==========
            (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", "layer{}.mlp.gate.weight", 0, TensorQuantFormat::HQ4K),
            (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", "layer{}.mlp.up.weight", 0, TensorQuantFormat::HQ4K),
            (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", "layer{}.mlp.down.weight", 0, TensorQuantFormat::HQ4K),
        ];
        
        for (pattern, template, block_id, quant) in common_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                patterns.push((re, template.to_string(), block_id, quant));
            }
        }
        
        // Patrones específicos de visión (bloque 1)
        let vision_patterns: Vec<(&str, &str, usize, TensorQuantFormat)> = vec![
            (r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.weight$", "vision.patch_embed.weight", 1, TensorQuantFormat::HQ5K),
            (r"^vision_tower\.vision_model\.embeddings\.position_embedding\.weight$", "vision.pos_embed.weight", 1, TensorQuantFormat::FP16),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "vision.layer{}.attn.q_proj.weight", 1, TensorQuantFormat::HQ5K),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "vision.layer{}.attn.k_proj.weight", 1, TensorQuantFormat::HQ5K),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "vision.layer{}.attn.v_proj.weight", 1, TensorQuantFormat::HQ5K),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$", "vision.layer{}.attn.o_proj.weight", 1, TensorQuantFormat::HQ5K),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight$", "vision.layer{}.mlp.fc1.weight", 1, TensorQuantFormat::HQ4K),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.weight$", "vision.layer{}.mlp.fc2.weight", 1, TensorQuantFormat::HQ4K),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight$", "vision.layer{}.ln1.weight", 1, TensorQuantFormat::FP16),
            (r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.weight$", "vision.layer{}.ln2.weight", 1, TensorQuantFormat::FP16),
            (r"^vision_tower\.vision_model\.post_layernorm\.weight$", "vision.post_layernorm.weight", 1, TensorQuantFormat::FP16),
            
            // Projector
            (r"^multi_modal_projector\.linear_1\.weight$", "projector.vision.linear1.weight", 1, TensorQuantFormat::HQ5K),
            (r"^multi_modal_projector\.linear_1\.bias$", "projector.vision.linear1.bias", 1, TensorQuantFormat::FP16),
            (r"^multi_modal_projector\.linear_2\.weight$", "projector.vision.linear2.weight", 1, TensorQuantFormat::HQ5K),
            (r"^multi_modal_projector\.linear_2\.bias$", "projector.vision.linear2.bias", 1, TensorQuantFormat::FP16),
        ];
        
        for (pattern, template, block_id, quant) in vision_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                patterns.push((re, template.to_string(), block_id, quant));
            }
        }
        
        patterns
    }
}

impl TensorMapper for GenericMapper {
    fn name(&self) -> &str {
        &self.arch
    }
    
    fn map_tensor(&self, original_name: &str) -> Option<TensorMapping> {
        // Ignorar tensores que no deben mapearse
        if original_name.contains("rotary_emb") 
            || original_name.contains("inv_freq")
            || original_name.contains("_float_tensor")
            || original_name.contains("position_ids")
        {
            return None;
        }
        
        for (pattern, template, block_id, quant_format) in &self.patterns {
            if let Some(captures) = pattern.captures(original_name) {
                let mut canonical = template.clone();
                
                // Reemplazar {} con el número de capa capturado
                if let Some(layer_num) = captures.get(1) {
                    canonical = canonical.replace("{}", layer_num.as_str());
                }
                
                return Some(TensorMapping {
                    canonical_name: canonical,
                    block_id: *block_id,
                    should_quantize: *quant_format != TensorQuantFormat::FP16,
                    quant_format: *quant_format,
                });
            }
        }
        
        None
    }
}

/// Detecta la arquitectura de un modelo
pub fn detect_architecture(model_dir: impl AsRef<Path>) -> Result<String> {
    let config_path = model_dir.as_ref().join("config.json");
    
    if !config_path.exists() {
        anyhow::bail!("No config.json in {}", model_dir.as_ref().display());
    }
    
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)?
    )?;
    
    // Detectar por model_type o architectures
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        return Ok(model_type.to_lowercase());
    }
    
    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            if arch_lower.contains("llama") {
                return Ok("llama".to_string());
            } else if arch_lower.contains("qwen") {
                return Ok("qwen2".to_string());
            } else if arch_lower.contains("gemma") {
                return Ok("gemma".to_string());
            } else if arch_lower.contains("mistral") {
                return Ok("mistral".to_string());
            } else if arch_lower.contains("phi") {
                return Ok("phi".to_string());
            }
        }
    }
    
    Ok("generic".to_string())
}

/// Crea el mapper apropiado para un modelo
pub fn get_mapper(model_dir: impl AsRef<Path>) -> Result<Box<dyn TensorMapper>> {
    let arch = detect_architecture(&model_dir)?;
    Ok(Box::new(GenericMapper::new(&arch)))
}
