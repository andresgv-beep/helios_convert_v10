// ============================================================================
// GEMMA 4 VISION MAPPER
// ============================================================================
//
// This mapper is deliberately separate from the Gemma 4 text mapper. The
// checkpoint contains text, vision and audio under one config and selecting a
// mapper by architecture alone is therefore unsafe.
// ============================================================================

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::{json, Value};

use super::traits::ModelMapper;
use super::types::{QuantHint, TensorCategory, TensorMapping};

pub struct Gemma4VisionMapper {
    vision: Value,
    image_processor: Value,
    projection_dim: usize,
    image_token_id: i64,
    boi_token_id: i64,
    eoi_token_id: i64,
    re_layer: Regex,
    re_attn_linear: Regex,
    re_mlp_linear: Regex,
}

impl Gemma4VisionMapper {
    pub fn from_model_path(model_path: &Path, root: &Value) -> Result<Self> {
        let vision = root
            .get("vision_config")
            .cloned()
            .context("Gemma 4 config is missing vision_config")?;
        if vision.get("model_type").and_then(Value::as_str) != Some("gemma4_vision") {
            anyhow::bail!("vision_config is not a Gemma 4 vision encoder");
        }
        let num_layers = vision
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if num_layers == 0 {
            anyhow::bail!("Gemma 4 vision_config has no valid num_hidden_layers");
        }
        // E2B/E4B: clamps aprendidos en cada lineal. 26B: sin clamps, con los
        // buffers de standardize (std_bias, std_scale) tras el pool.
        if vision.get("use_clipped_linears").and_then(Value::as_bool).is_none()
            || vision.get("standardize").and_then(Value::as_bool).is_none() {
            anyhow::bail!("Gemma 4 vision_config needs use_clipped_linears and standardize");
        }

        let processor_path = model_path.join("processor_config.json");
        let processor_root: Value = serde_json::from_str(
            &fs::read_to_string(&processor_path)
                .with_context(|| format!("failed to read {}", processor_path.display()))?,
        )
        .with_context(|| format!("invalid JSON in {}", processor_path.display()))?;
        let image_processor = processor_root
            .get("image_processor")
            .cloned()
            .context("processor_config.json is missing image_processor")?;

        let projection_dim = root
            .get("text_config")
            .and_then(|v| v.get("hidden_size"))
            .and_then(Value::as_u64)
            .context("Gemma 4 text_config is missing hidden_size for the vision projector")?
            as usize;

        Ok(Self {
            vision,
            image_processor,
            projection_dim,
            image_token_id: root.get("image_token_id").and_then(Value::as_i64).unwrap_or(-1),
            boi_token_id: root.get("boi_token_id").and_then(Value::as_i64).unwrap_or(-1),
            eoi_token_id: root.get("eoi_token_id").and_then(Value::as_i64).unwrap_or(-1),
            re_layer: Regex::new(r"^model\.vision_tower\.encoder\.layers\.(\d+)\.(.+)$")
                .unwrap(),
            re_attn_linear: Regex::new(
                r"^self_attn\.(q|k|v|o)_proj\.(linear\.weight|input_min|input_max|output_min|output_max)$",
            )
            .unwrap(),
            re_mlp_linear: Regex::new(
                r"^mlp\.(gate|up|down)_proj\.(linear\.weight|input_min|input_max|output_min|output_max)$",
            )
            .unwrap(),
        })
    }

    fn flag(&self, key: &str) -> bool {
        self.vision.get(key).and_then(Value::as_bool).unwrap_or(false)
    }

    fn layer_mapping(&self, layer: usize, suffix: &str) -> Option<TensorMapping> {
        if let Some(caps) = self.re_attn_linear.captures(suffix) {
            let projection = caps.get(1)?.as_str();
            let field = match caps.get(2)?.as_str() {
                "linear.weight" => "weight",
                other => other,
            };
            return Some(
                TensorMapping::new(
                    format!("vision.layer{layer}.attn.{projection}_proj.{field}"),
                    QuantHint::FP16,
                    TensorCategory::Attention,
                )
                .with_layer(layer),
            );
        }

        if let Some(caps) = self.re_mlp_linear.captures(suffix) {
            let projection = caps.get(1)?.as_str();
            let field = match caps.get(2)?.as_str() {
                "linear.weight" => "weight",
                other => other,
            };
            return Some(
                TensorMapping::new(
                    format!("vision.layer{layer}.mlp.{projection}.{field}"),
                    QuantHint::FP16,
                    TensorCategory::MLP,
                )
                .with_layer(layer),
            );
        }

        let canonical = match suffix {
            "input_layernorm.weight" => "ln_attn_in.weight",
            "post_attention_layernorm.weight" => "ln_attn_post.weight",
            "pre_feedforward_layernorm.weight" => "ln_mlp_in.weight",
            "post_feedforward_layernorm.weight" => "ln_mlp_post.weight",
            "self_attn.q_norm.weight" => "attn.q_norm.weight",
            "self_attn.k_norm.weight" => "attn.k_norm.weight",
            _ => return None,
        };
        Some(
            TensorMapping::new(
                format!("vision.layer{layer}.{canonical}"),
                QuantHint::FP16,
                TensorCategory::Norm,
            )
            .with_layer(layer),
        )
    }
}

impl ModelMapper for Gemma4VisionMapper {
    fn name(&self) -> &str {
        "gemma4_vision"
    }

    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        let global = match name {
            "model.vision_tower.patch_embedder.input_proj.weight" => Some(TensorMapping::new(
                "vision.patch_embed.input_proj.weight",
                QuantHint::FP16,
                TensorCategory::VisionPatch,
            )),
            "model.vision_tower.patch_embedder.position_embedding_table" => {
                Some(TensorMapping::new(
                    "vision.patch_embed.position_embedding.weight",
                    QuantHint::FP16,
                    TensorCategory::Embedding,
                ))
            }
            "model.embed_vision.embedding_projection.weight" => Some(TensorMapping::new(
                "vision.projector.weight",
                QuantHint::FP16,
                TensorCategory::VisionProjector,
            )),
            // Son de la torre, pero se aplican justo antes del proyector: con ese
            // prefijo entran en su fase de pesos en el motor.
            "model.vision_tower.std_bias" if self.flag("standardize") => Some(TensorMapping::new(
                "vision.projector.std_bias",
                QuantHint::FP16,
                TensorCategory::VisionProjector,
            )),
            "model.vision_tower.std_scale" if self.flag("standardize") => Some(TensorMapping::new(
                "vision.projector.std_scale",
                QuantHint::FP16,
                TensorCategory::VisionProjector,
            )),
            _ => None,
        };
        if global.is_some() {
            return global;
        }

        let caps = self.re_layer.captures(name)?;
        let layer = caps.get(1)?.as_str().parse().ok()?;
        self.layer_mapping(layer, caps.get(2)?.as_str())
    }

    fn execution_hints(&self) -> Value {
        let v = &self.vision;
        let p = &self.image_processor;
        let patch_size = v.get("patch_size").and_then(Value::as_u64).unwrap_or(0);
        let pooling = v
            .get("pooling_kernel_size")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let max_soft_tokens = p
            .get("max_soft_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| {
                v.get("default_output_length")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            });

        json!({
            "encoder_arch": "gemma4_vision",
            "encoder_type": "gemma4",
            "image_size": 0,
            "patch_size": patch_size,
            "hidden_size": v["hidden_size"],
            "num_hidden_layers": v["num_hidden_layers"],
            "num_attention_heads": v["num_attention_heads"],
            "num_key_value_heads": v["num_key_value_heads"],
            "head_dim": v["head_dim"],
            "intermediate_size": v["intermediate_size"],
            "num_channels": 3,
            "rms_norm_eps": v["rms_norm_eps"],
            "layer_norm_eps": v["rms_norm_eps"],
            "position_embedding_size": v["position_embedding_size"],
            "position_type": "learned_2d_plus_rope_2d",
            "rope_type": "default_2d",
            "rope_theta": v["rope_parameters"]["rope_theta"],
            "attention_type": "mha_bidirectional",
            "attention_scale": 1.0,
            "mlp_type": "geglu",
            "hidden_activation": v["hidden_activation"],
            "pooling_kernel_size": pooling,
            "max_soft_tokens": max_soft_tokens,
            "max_patches": max_soft_tokens * pooling * pooling,
            "num_image_tokens": max_soft_tokens,
            "dynamic_image_tokens": true,
            "use_clipped_linears": v["use_clipped_linears"],
            "standardize": v["standardize"],
            "qk_norm_weighted": true,
            "v_norm_weighted": false,
            "pool_norm_weighted": false,
            "projection_dim": self.projection_dim,
            "projector_type": "linear",
            "image_token_id": self.image_token_id,
            "boi_token_id": self.boi_token_id,
            "eoi_token_id": self.eoi_token_id,
            "pad_token_id": 0,
            "preprocess": {
                "do_convert_rgb": p["do_convert_rgb"],
                "do_normalize": p["do_normalize"],
                "do_rescale": p["do_rescale"],
                "do_resize": p["do_resize"],
                "rescale_factor": p["rescale_factor"],
                "resample": p["resample"],
                "antialias": true,
                "resize_multiple": patch_size * pooling,
                "patch_order": "hwc",
                "position_order": "xy"
            }
        })
    }

    fn num_layers(&self) -> usize {
        self.vision["num_hidden_layers"].as_u64().unwrap_or(0) as usize
    }

    fn vocab_size(&self) -> usize {
        0
    }

    fn hidden_size(&self) -> usize {
        self.vision["hidden_size"].as_u64().unwrap_or(0) as usize
    }

    fn expected_tensor_count(&self) -> Option<usize> {
        let per_layer = if self.flag("use_clipped_linears") { 41 } else { 13 };
        Some(3 + self.num_layers() * per_layer + if self.flag("standardize") { 2 } else { 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::dictionary::validate_tensor_name;
    use crate::mapping::{create_mapper_for_block, BlockType};
    use crate::safetensor::SafetensorReader;

    #[test]
    fn maps_clipped_linears_and_rejects_text() {
        let Ok(path) = std::env::var("GEMMA4_MODEL_DIR") else {
            return;
        };
        let mapper = create_mapper_for_block(Path::new(&path), BlockType::Vision).unwrap();
        assert_eq!(mapper.name(), "gemma4_vision");
        assert_eq!(
            mapper
                .map_tensor("model.vision_tower.encoder.layers.12.self_attn.q_proj.input_max")
                .unwrap()
                .canonical_name,
            "vision.layer12.attn.q_proj.input_max",
        );
        assert_eq!(
            mapper
                .map_tensor("model.vision_tower.encoder.layers.3.mlp.gate_proj.linear.weight")
                .unwrap()
                .canonical_name,
            "vision.layer3.mlp.gate.weight",
        );
        assert!(mapper
            .map_tensor("model.language_model.layers.0.self_attn.q_proj.weight")
            .is_none());
    }

    #[test]
    fn audits_local_gemma4_vision_inventory_when_available() {
        let Ok(path) = std::env::var("GEMMA4_MODEL_DIR") else {
            return;
        };
        let path = Path::new(&path);
        let mapper = create_mapper_for_block(path, BlockType::Vision).unwrap();
        let reader = SafetensorReader::from_folder(path).unwrap();
        let mapped: Vec<_> = reader
            .iter_tensors()
            .filter_map(|(name, _)| mapper.map_tensor(name))
            .collect();
        assert_eq!(mapped.len(), 659);
        assert_eq!(mapper.expected_tensor_count(), Some(659));
        assert!(mapped.iter().all(|m| m.quant_hint == QuantHint::FP16));
        assert!(mapped
            .iter()
            .all(|m| validate_tensor_name(&m.canonical_name)));
        assert_eq!(
            mapped
                .iter()
                .map(|m| &m.canonical_name)
                .collect::<HashSet<_>>()
                .len(),
            mapped.len(),
        );
        let hints = mapper.execution_hints();
        assert_eq!(hints["encoder_type"], "gemma4");
        assert_eq!(hints["max_patches"], 2520);
        assert_eq!(hints["projection_dim"], 1536);
        assert_eq!(hints["preprocess"]["resize_multiple"], 48);
    }
}
