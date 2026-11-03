// src/mapping/clip.rs
// ============================================================================
// CLIP MAPPER - Mapea tensores CLIP/SigLIP/ViT a nombres canónicos
// ============================================================================
//
// Soporta: CLIP, OpenCLIP, SigLIP, ViT
// Vision encoders con arquitectura transformer.
//
// v9.0.5: Mejora encoder_variant dinámico
//
// Nombres canónicos según HELIOS_DICTIONARY v9.0.2:
//   vision.patch_embed.weight
//   vision.pos_embed.weight
//   vision.cls_token
//   vision.layer{N}.attn.{q,k,v,o}_proj.{weight,bias}
//   vision.layer{N}.mlp.fc1.{weight,bias}
//   vision.layer{N}.mlp.fc2.{weight,bias}
//   vision.layer{N}.ln1.{weight,bias}
//   vision.layer{N}.ln2.{weight,bias}
//   vision.pre_layernorm.{weight,bias}
//   vision.post_layernorm.{weight,bias}
//
// ============================================================================

use regex::Regex;
use serde_json::{json, Value};

use super::traits::ModelMapper;
use super::types::{TensorMapping, QuantHint, TensorCategory};

#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    pub projection_dim: Option<usize>,  // v9.0.5: Para detectar variante
}

impl ClipConfig {
    pub fn from_json(config: &Value) -> Self {
        // CLIP puede tener la config en "vision_config" o en la raíz
        let vision_config = config.get("vision_config").unwrap_or(config);
        
        Self {
            num_hidden_layers: vision_config["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
            hidden_size: vision_config["hidden_size"].as_u64().unwrap_or(1024) as usize,
            intermediate_size: vision_config["intermediate_size"].as_u64().unwrap_or(4096) as usize,
            num_attention_heads: vision_config["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            image_size: vision_config["image_size"].as_u64().unwrap_or(224) as usize,
            patch_size: vision_config["patch_size"].as_u64().unwrap_or(14) as usize,
            num_channels: vision_config["num_channels"].as_u64().unwrap_or(3) as usize,
            layer_norm_eps: vision_config["layer_norm_eps"].as_f64().unwrap_or(1e-5),
            projection_dim: vision_config["projection_dim"].as_u64().map(|x| x as usize),
        }
    }
    
    /// Detecta la variante del modelo basándose en las dimensiones
    pub fn detect_variant(&self) -> &'static str {
        match (self.hidden_size, self.num_hidden_layers, self.patch_size) {
            // ViT-B variants
            (768, 12, 32) => "vit-b-32",
            (768, 12, 16) => "vit-b-16",
            (768, 12, 14) => "vit-b-14",
            // ViT-L variants
            (1024, 24, 14) => "vit-l-14",
            (1024, 24, 16) => "vit-l-16",
            // ViT-H variants
            (1280, 32, 14) => "vit-h-14",
            // ViT-g variants (OpenCLIP)
            (1408, 40, 14) => "vit-g-14",
            // ViT-G variants (OpenCLIP bigG)
            (1664, 48, 14) => "vit-bigg-14",
            // Default
            _ => "vit-unknown",
        }
    }
}

pub struct ClipMapper {
    config: ClipConfig,
    // Embeddings
    re_patch_embed: Regex,
    re_pos_embed: Regex,
    re_class_embed: Regex,
    // Encoder layers - attention
    re_attn_qkv: Regex,
    re_attn_out: Regex,
    // Encoder layers - MLP
    re_mlp_fc1: Regex,
    re_mlp_fc2: Regex,
    // Encoder layers - norms
    re_ln1: Regex,
    re_ln2: Regex,
    // Pre/post norms
    re_pre_norm: Regex,
    re_post_norm: Regex,
    // Projection head
    re_projection: Regex,
}

impl ClipMapper {
    pub fn new(config: ClipConfig) -> Self {
        Self {
            config,
            // Embeddings
            re_patch_embed: Regex::new(r"^vision_model\.embeddings\.patch_embedding\.weight$").unwrap(),
            re_pos_embed: Regex::new(r"^vision_model\.embeddings\.position_embedding\.weight$").unwrap(),
            re_class_embed: Regex::new(r"^vision_model\.embeddings\.class_embedding$").unwrap(),
            // Attention - QKV separados
            re_attn_qkv: Regex::new(r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$").unwrap(),
            // Attention - output projection
            re_attn_out: Regex::new(r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$").unwrap(),
            // MLP
            re_mlp_fc1: Regex::new(r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)$").unwrap(),
            re_mlp_fc2: Regex::new(r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)$").unwrap(),
            // Layer norms
            re_ln1: Regex::new(r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.(weight|bias)$").unwrap(),
            re_ln2: Regex::new(r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.(weight|bias)$").unwrap(),
            // Pre/post norms (nota: algunos modelos tienen typo "layrnorm")
            re_pre_norm: Regex::new(r"^vision_model\.pre_layr?norm\.(weight|bias)$").unwrap(),
            re_post_norm: Regex::new(r"^vision_model\.post_layernorm\.(weight|bias)$").unwrap(),
            // Projection
            re_projection: Regex::new(r"^visual_projection\.weight$").unwrap(),
        }
    }
    
    pub fn from_json(config: &Value) -> Self {
        Self::new(ClipConfig::from_json(config))
    }
}

impl ModelMapper for ClipMapper {
    fn name(&self) -> &str {
        "clip"
    }
    
    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        if self.should_ignore(name) {
            return None;
        }
        
        // Ignorar text_model si existe (solo queremos vision)
        if name.starts_with("text_model.") || name.starts_with("text_projection") {
            return None;
        }
        
        // ══════════════════════════════════════════════════════════════
        // EMBEDDINGS (FP16)
        // ══════════════════════════════════════════════════════════════
        
        if self.re_patch_embed.is_match(name) {
            return Some(TensorMapping::new(
                "vision.patch_embed.weight",
                QuantHint::FP16,
                TensorCategory::VisionPatch,
            ));
        }
        
        if self.re_pos_embed.is_match(name) {
            return Some(TensorMapping::new(
                "vision.pos_embed.weight",
                QuantHint::FP16,
                TensorCategory::Embedding,
            ));
        }
        
        if self.re_class_embed.is_match(name) {
            return Some(TensorMapping::new(
                "vision.cls_token",  // Diccionario usa cls_token
                QuantHint::FP16,
                TensorCategory::Embedding,
            ));
        }
        
        // ══════════════════════════════════════════════════════════════
        // ATTENTION (HQ5K para weights, FP16 para biases)
        // ══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_attn_qkv.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let proj = &caps[2];  // q, k, v
            let kind = &caps[3];  // weight or bias
            
            let hint = if kind == "bias" { QuantHint::FP16 } else { QuantHint::HQ5K };
            
            return Some(TensorMapping::new(
                format!("vision.layer{}.attn.{}_proj.{}", layer, proj, kind),
                hint,
                TensorCategory::Attention,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_attn_out.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let kind = &caps[2];
            
            let hint = if kind == "bias" { QuantHint::FP16 } else { QuantHint::HQ5K };
            
            // Diccionario usa o_proj, no out_proj
            return Some(TensorMapping::new(
                format!("vision.layer{}.attn.o_proj.{}", layer, kind),
                hint,
                TensorCategory::Attention,
            ).with_layer(layer));
        }
        
        // ══════════════════════════════════════════════════════════════
        // MLP (HQ4K para weights, FP16 para biases)
        // ══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_mlp_fc1.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let kind = &caps[2];
            
            let hint = if kind == "bias" { QuantHint::FP16 } else { QuantHint::HQ4K };
            
            return Some(TensorMapping::new(
                format!("vision.layer{}.mlp.fc1.{}", layer, kind),
                hint,
                TensorCategory::MLP,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_mlp_fc2.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let kind = &caps[2];
            
            let hint = if kind == "bias" { QuantHint::FP16 } else { QuantHint::HQ4K };
            
            return Some(TensorMapping::new(
                format!("vision.layer{}.mlp.fc2.{}", layer, kind),
                hint,
                TensorCategory::MLP,
            ).with_layer(layer));
        }
        
        // ══════════════════════════════════════════════════════════════
        // LAYER NORMS (FP16)
        // ══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_ln1.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let kind = &caps[2];
            return Some(TensorMapping::new(
                format!("vision.layer{}.ln1.{}", layer, kind),
                QuantHint::FP16,
                TensorCategory::Norm,
            ).with_layer(layer));
        }
        
        if let Some(caps) = self.re_ln2.captures(name) {
            let layer: usize = caps[1].parse().ok()?;
            let kind = &caps[2];
            return Some(TensorMapping::new(
                format!("vision.layer{}.ln2.{}", layer, kind),
                QuantHint::FP16,
                TensorCategory::Norm,
            ).with_layer(layer));
        }
        
        // ══════════════════════════════════════════════════════════════
        // PRE/POST LAYERNORMS (FP16)
        // ══════════════════════════════════════════════════════════════
        
        if let Some(caps) = self.re_pre_norm.captures(name) {
            let kind = &caps[1];
            return Some(TensorMapping::new(
                format!("vision.pre_layernorm.{}", kind),
                QuantHint::FP16,
                TensorCategory::Norm,
            ));
        }
        
        if let Some(caps) = self.re_post_norm.captures(name) {
            let kind = &caps[1];
            return Some(TensorMapping::new(
                format!("vision.post_layernorm.{}", kind),
                QuantHint::FP16,
                TensorCategory::Norm,
            ));
        }
        
        // ══════════════════════════════════════════════════════════════
        // PROJECTION HEAD (HQ5K) - mapea a vision.head
        // ══════════════════════════════════════════════════════════════
        
        if self.re_projection.is_match(name) {
            return Some(TensorMapping::new(
                "vision.head.weight",
                QuantHint::HQ5K,
                TensorCategory::VisionProjector,
            ));
        }
        
        None
    }
    
    fn execution_hints(&self) -> Value {
        let c = &self.config;
        let head_dim = c.hidden_size / c.num_attention_heads;
        let num_patches = (c.image_size / c.patch_size).pow(2);
        
        // v9.0.5: Detectar variante automáticamente
        let variant = c.detect_variant();
        
        // Según spec v1.2, vision debe devolver vision_config
        json!({
            "encoder_arch": "clip",
            "encoder_variant": variant,
            "image_size": c.image_size,
            "patch_size": c.patch_size,
            "num_channels": c.num_channels,
            "hidden_size": c.hidden_size,
            "num_hidden_layers": c.num_hidden_layers,
            "num_attention_heads": c.num_attention_heads,
            "head_dim": head_dim,
            "intermediate_size": c.intermediate_size,
            "attention_type": "mha",
            "mlp_type": "standard",
            "mlp_activation": "quick_gelu",
            "norm_type": "layernorm",
            "layer_norm_eps": c.layer_norm_eps,
            "num_image_tokens": num_patches + 1,  // +1 for CLS token
            "projector": {
                "type": "mlp",
                "input_dim": c.hidden_size,
                "output_dim": c.projection_dim.unwrap_or(c.hidden_size),
                "depth": 2
            }
        })
    }
    
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
    
    fn vocab_size(&self) -> usize {
        0 // Vision encoder, no vocab
    }
    
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}
