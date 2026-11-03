// ============================================================================
// GEMMA 4 MAPPER - Text decoder only
// ============================================================================
//
// Gemma 4 is not a uniform Llama-like decoder.  Attention geometry, RoPE and
// MLP width may change per layer, and the small models carry Per-Layer
// Embeddings (PLE).  This mapper therefore derives every layer descriptor from
// the actual safetensors shapes instead of guessing from config.json.
//
// Vision/audio towers are intentionally not mapped here.  A positional/--text
// conversion produces a faithful text-model block; the other towers will get
// dedicated mappers and HNF blocks later.
// ============================================================================

use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::{json, Value};

use crate::safetensor::SafetensorReader;

use super::traits::ModelMapper;
use super::types::{QuantHint, TensorCategory, TensorMapping};

#[derive(Debug, Clone)]
struct Gemma4LayerConfig {
    layer_type: String,
    head_dim: usize,
    intermediate_size: usize,
    rope_theta: f64,
    rope_type: String,
    partial_rotary_factor: f64,
    /// KV heads de ESTA capa. Gemma 4 «unified» mezcla GQA en las capas
    /// deslizantes con una sola KV head en las globales, asi que el valor
    /// global de text_config no describe el modelo.
    num_kv_heads: usize,
    /// `attention_k_eq_v`: la capa no tiene v_proj y reutiliza K como V.
    k_eq_v: bool,
}

pub struct Gemma4Mapper {
    text: Value,
    layers: Vec<Gemma4LayerConfig>,
    re_layer_tensor: Regex,
}

impl Gemma4Mapper {
    pub fn from_model_path(model_path: &Path, root_config: &Value) -> Result<Self> {
        let text = root_config
            .get("text_config")
            .cloned()
            .context("Gemma 4 config is missing text_config")?;
        // Validate the MoE topology before opening any weight shards.
        let moe = super::gemma4_moe::MoeSpec::from_text_config(&text)?;
        let num_layers = text["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        if num_layers == 0 {
            anyhow::bail!("Gemma 4 text_config has no valid num_hidden_layers");
        }

        let reader = SafetensorReader::from_folder(model_path)?;
        let layer_types = text["layer_types"].as_array();
        let rope_parameters = text.get("rope_parameters");
        let mut layers = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let base = format!("model.language_model.layers.{layer}");
            let q_norm = reader
                .shape(&format!("{base}.self_attn.q_norm.weight"))
                .with_context(|| format!("missing q_norm for Gemma 4 layer {layer}"))?;
            let mlp_gate = reader
                .shape(&format!("{base}.mlp.gate_proj.weight"))
                .with_context(|| format!("missing MLP gate for Gemma 4 layer {layer}"))?;
            let k_proj = reader
                .shape(&format!("{base}.self_attn.k_proj.weight"))
                .with_context(|| format!("missing k_proj for Gemma 4 layer {layer}"))?;
            // Ausencia de v_proj == attention_k_eq_v en esta capa.  Se mira el
            // checkpoint y no el config: es el checkpoint quien manda.
            let k_eq_v = reader
                .shape(&format!("{base}.self_attn.v_proj.weight"))
                .is_none();

            let layer_type = layer_types
                .and_then(|v| v.get(layer))
                .and_then(Value::as_str)
                .unwrap_or("sliding_attention")
                .to_string();
            let rope = rope_parameters
                .and_then(|v| v.get(&layer_type))
                .unwrap_or(&Value::Null);

            layers.push(Gemma4LayerConfig {
                layer_type,
                head_dim: q_norm.first().copied().unwrap_or(0),
                intermediate_size: mlp_gate.first().copied().unwrap_or(0),
                rope_theta: rope["rope_theta"].as_f64().unwrap_or(10_000.0),
                rope_type: rope["rope_type"].as_str().unwrap_or("default").to_string(),
                partial_rotary_factor: rope["partial_rotary_factor"].as_f64().unwrap_or(1.0),
                num_kv_heads: {
                    let head_dim = q_norm.first().copied().unwrap_or(0);
                    let k_rows = k_proj.first().copied().unwrap_or(0);
                    if head_dim > 0 { k_rows / head_dim } else { 0 }
                },
                k_eq_v,
            });
        }

        if let Some(spec) = &moe {
            for (name, shape) in spec.required_tensor_shapes() {
                let actual = reader.shape(&name)
                    .with_context(|| format!("missing Gemma 4 MoE tensor {name}"))?;
                anyhow::ensure!(actual == shape.as_slice(),
                    "{name}: expected shape {shape:?}, got {actual:?}");
            }
        }

        let mapper = Self {
            text,
            layers,
            re_layer_tensor: Regex::new(
                r"^model\.language_model\.layers\.(\d+)\.(.+)$",
            )
            .unwrap(),
        };
        if moe.is_some() {
            for (name, _) in reader.iter_tensors() {
                anyhow::ensure!(!name.starts_with("model.language_model.")
                    || mapper.map_tensor(name).is_some(),
                    "unmapped Gemma 4 MoE language tensor: {name}");
            }
        }
        Ok(mapper)
    }

    /// Name-only preflight from an index; never supplies execution geometry.
    /// Useful while weight shards are still downloading.
    pub fn inventory_mappings(root: &Value, names: &[String]) -> Result<Vec<(String, TensorMapping)>> {
        let text = root.get("text_config").context("missing text_config")?.clone();
        super::gemma4_moe::MoeSpec::from_text_config(&text)?;
        let layers = text["num_hidden_layers"].as_u64().context("missing num_hidden_layers")?;
        let mapper = Self { text, layers: vec![],
            re_layer_tensor: Regex::new(r"^model\.language_model\.layers\.(\d+)\.(.+)$")? };
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for name in names {
            if !name.starts_with("model.language_model.") { continue; }
            let m = mapper.map_tensor(name).with_context(|| format!("unmapped language tensor: {name}"))?;
            anyhow::ensure!(m.layer_idx.map_or(true, |i| (i as u64) < layers), "layer out of range: {name}");
            anyhow::ensure!(seen.insert(m.canonical_name.clone()), "duplicate canonical name: {}", m.canonical_name);
            out.push((name.clone(), m));
        }
        Ok(out)
    }

    fn layer_mapping(&self, layer: usize, suffix: &str) -> Option<TensorMapping> {
        if self.is_moe() {
            if let Some(mapping) = super::gemma4_moe::map_extra_tensor(layer, suffix) {
                return Some(mapping);
            }
        }
        let (canonical, quant, category) = match suffix {
            "self_attn.q_proj.weight" => ("attn.q_proj.weight", QuantHint::HQ5K, TensorCategory::Attention),
            "self_attn.k_proj.weight" => ("attn.k_proj.weight", QuantHint::HQ5K, TensorCategory::Attention),
            "self_attn.v_proj.weight" => ("attn.v_proj.weight", QuantHint::HQ5K, TensorCategory::Attention),
            "self_attn.o_proj.weight" => ("attn.o_proj.weight", QuantHint::HQ5K, TensorCategory::Attention),
            "self_attn.q_norm.weight" => ("attn.q_norm.weight", QuantHint::FP16, TensorCategory::Norm),
            "self_attn.k_norm.weight" => ("attn.k_norm.weight", QuantHint::FP16, TensorCategory::Norm),
            "mlp.gate_proj.weight" => ("mlp.gate.weight", QuantHint::HQ4K, TensorCategory::MLP),
            "mlp.up_proj.weight" => ("mlp.up.weight", QuantHint::HQ4K, TensorCategory::MLP),
            "mlp.down_proj.weight" => ("mlp.down.weight", QuantHint::HQ4K, TensorCategory::MLP),
            "input_layernorm.weight" => ("ln_attn_in.weight", QuantHint::FP16, TensorCategory::Norm),
            "post_attention_layernorm.weight" => ("ln_attn_post.weight", QuantHint::FP16, TensorCategory::Norm),
            "pre_feedforward_layernorm.weight" => ("ln_mlp_in.weight", QuantHint::FP16, TensorCategory::Norm),
            "post_feedforward_layernorm.weight" => ("ln_mlp_post.weight", QuantHint::FP16, TensorCategory::Norm),
            "post_per_layer_input_norm.weight" => ("ln_ple_post.weight", QuantHint::FP16, TensorCategory::Norm),
            "per_layer_input_gate.weight" => ("ple.input_gate.weight", QuantHint::HQ5K, TensorCategory::Other),
            "per_layer_projection.weight" => ("ple.projection.weight", QuantHint::HQ5K, TensorCategory::Other),
            "layer_scalar" => ("layer_scalar", QuantHint::FP16, TensorCategory::Other),
            _ => return None,
        };

        Some(
            TensorMapping::new(format!("layer{layer}.{canonical}"), quant, category)
                .with_layer(layer),
        )
    }
}

impl ModelMapper for Gemma4Mapper {
    fn name(&self) -> &str {
        "gemma4"
    }

    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        let global = match name {
            "model.language_model.embed_tokens.weight" => Some(TensorMapping::new(
                "token_embedding.weight",
                QuantHint::FP16,
                TensorCategory::Embedding,
            )),
            "model.language_model.embed_tokens_per_layer.weight" => Some(TensorMapping::new(
                "ple.token_embedding.weight",
                // PLE dominates Gemma 4 E2B (~4.4 GiB in BF16). It is an
                // embedding lookup table, but HNF keeps it resident in its
                // quantized form and the runtime only dequantizes requested
                // rows. --quant FP16 still overrides this hint for the golden
                // reference conversion.
                QuantHint::HQ5K,
                TensorCategory::Embedding,
            )),
            "model.language_model.norm.weight" => Some(TensorMapping::new(
                "final_norm.weight",
                QuantHint::FP16,
                TensorCategory::Norm,
            )),
            "model.language_model.per_layer_model_projection.weight" => Some(TensorMapping::new(
                "ple.model_projection.weight",
                QuantHint::HQ5K,
                TensorCategory::Other,
            )),
            "model.language_model.per_layer_projection_norm.weight" => Some(TensorMapping::new(
                "ple.projection_norm.weight",
                QuantHint::FP16,
                TensorCategory::Norm,
            )),
            _ => None,
        };
        if global.is_some() {
            return global;
        }

        let caps = self.re_layer_tensor.captures(name)?;
        let layer = caps.get(1)?.as_str().parse().ok()?;
        self.layer_mapping(layer, caps.get(2)?.as_str())
    }

    fn execution_hints(&self) -> Value {
        let t = &self.text;
        let layer_hints: Vec<Value> = self
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                json!({
                    "index": index,
                    "attention_type": layer.layer_type,
                    "sliding_window": if layer.layer_type == "sliding_attention" {
                        t["sliding_window"].as_u64().unwrap_or(0)
                    } else { 0 },
                    "head_dim": layer.head_dim,
                    "intermediate_size": layer.intermediate_size,
                    "rope_type": layer.rope_type,
                    "rope_theta": layer.rope_theta,
                    "partial_rotary_factor": layer.partial_rotary_factor,
                    "num_key_value_heads": layer.num_kv_heads,
                    "k_eq_v": layer.k_eq_v,
                })
            })
            .collect();

        let local_head_dim = self.layers.first().map(|l| l.head_dim).unwrap_or(0);
        let max_head_dim = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(local_head_dim);
        let max_intermediate = self.layers.iter().map(|l| l.intermediate_size).max().unwrap_or(0);

        let mut hints = json!({
            "arch": "gemma4",
            "dtype": "bf16",
            "num_hidden_layers": self.layers.len(),
            "hidden_size": t["hidden_size"],
            "intermediate_size": t["intermediate_size"],
            "max_intermediate_size": max_intermediate,
            "vocab_size": t["vocab_size"],
            "max_position_embeddings": t["max_position_embeddings"],
            "num_attention_heads": t["num_attention_heads"],
            "num_key_value_heads": t["num_key_value_heads"],
            "head_dim": local_head_dim,
            "global_head_dim": max_head_dim,
            "attention_type": "mqa",
            "attention_pattern": "hybrid_per_layer",
            "qkv_layout": "separate",
            "attention_bias": t["attention_bias"],
            "use_qk_norm": true,
            "kv_layout": "BHSD",
            "num_kv_shared_layers": t["num_kv_shared_layers"],
            "mlp_type": "geglu",
            "mlp_activation": "gelu_new",
            "hidden_activation": t["hidden_activation"],
            "use_double_wide_mlp": t["use_double_wide_mlp"],
            "norm_type": "rmsnorm",
            "rms_norm_eps": t["rms_norm_eps"],
            "norm_placement": "gemma4_four_norm",
            "tie_word_embeddings": t["tie_word_embeddings"],
            "final_logit_softcapping": t["final_logit_softcapping"],
            "embedding_scale": (t["hidden_size"].as_u64().unwrap_or(1) as f64).sqrt(),
            "per_layer_embeddings": {
                // «unified» (12B) trae hidden_size_per_layer_input = 0: no hay
                // PLE.  Las E-series (E2B/E4B) si lo traen.
                "enabled": t["hidden_size_per_layer_input"].as_u64().unwrap_or(0) > 0,
                "hidden_size": t["hidden_size_per_layer_input"],
                "vocab_size": t["vocab_size_per_layer_input"],
                "projection_width": self.layers.len() * t["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as usize,
            },
            "layer_scalar": true,
            "layer_configs": layer_hints,
        });
        if self.is_moe() {
            hints["moe_enabled"] = json!(true);
            hints["num_experts"] = self.text["num_experts"].clone();
            hints["num_experts_per_tok"] = self.text["top_k_experts"].clone();
            hints["moe_intermediate_size"] = self.text["moe_intermediate_size"].clone();
            hints["moe_layout"] = json!("gemma4_packed_experts_v1");
        }
        hints
    }

    fn is_moe(&self) -> bool {
        self.text["enable_moe_block"].as_bool().unwrap_or(false)
    }

    fn num_experts(&self) -> Option<usize> {
        if self.is_moe() { self.text["num_experts"].as_u64().map(|v| v as usize) }
        else { None }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn vocab_size(&self) -> usize {
        self.text["vocab_size"].as_u64().unwrap_or(0) as usize
    }

    fn hidden_size(&self) -> usize {
        self.text["hidden_size"].as_u64().unwrap_or(0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use crate::mapping::create_mapper;

    #[test]
    fn maps_text_tensor_names() {
        let mapper = Gemma4Mapper {
            text: json!({}),
            layers: vec![],
            re_layer_tensor: Regex::new(r"^model\.language_model\.layers\.(\d+)\.(.+)$").unwrap(),
        };
        assert_eq!(
            mapper
                .map_tensor("model.language_model.layers.4.self_attn.q_proj.weight")
                .unwrap()
                .canonical_name,
            "layer4.attn.q_proj.weight"
        );
        assert!(mapper.map_tensor("model.vision_tower.encoder.layers.0.input_layernorm.weight").is_none());
    }

    #[test]
    fn inventory_rejects_unknown_duplicate_and_out_of_range_language_tensors() {
        let root = json!({"text_config":{"num_hidden_layers":30,"enable_moe_block":true,
            "hidden_size":2816,"num_experts":128,"top_k_experts":8,"moe_intermediate_size":704}});
        let name = "model.language_model.layers.0.router.scale".to_string();
        assert_eq!(Gemma4Mapper::inventory_mappings(&root, &[name.clone()]).unwrap().len(), 1);
        for names in [vec![name.clone(),name],
            vec!["model.language_model.layers.30.router.scale".into()],
            vec!["model.language_model.layers.0.router.bias".into()]] {
            assert!(Gemma4Mapper::inventory_mappings(&root, &names).is_err());
        }
        let dense = json!({"text_config":{"num_hidden_layers":42}});
        assert!(Gemma4Mapper::inventory_mappings(&dense,
            &["model.language_model.layers.0.router.scale".into()]).is_err());
    }

    #[test]
    fn audits_local_gemma4_inventory_when_available() {
        let Ok(path) = std::env::var("GEMMA4_MODEL_DIR") else { return };
        let path = Path::new(&path);
        let mapper = create_mapper(path).unwrap();
        let reader = SafetensorReader::from_folder(path).unwrap();
        let mut mapped = Vec::new();
        let mut language = 0usize;
        for (name, _) in reader.iter_tensors() {
            if name.starts_with("model.language_model.") { language += 1; }
            if let Some(m) = mapper.map_tensor(name) { mapped.push(m.canonical_name); }
        }
        assert_eq!(language, 600);
        assert_eq!(mapped.len(), language, "all and only language tensors must be mapped");
        assert_eq!(mapped.iter().collect::<HashSet<_>>().len(), mapped.len());
        let hints = mapper.execution_hints();
        assert_eq!(hints["arch"], "gemma4");
        assert_eq!(hints["layer_configs"].as_array().unwrap().len(), 35);
        assert_eq!(hints["layer_configs"][4]["head_dim"], 512);
        assert_eq!(hints["layer_configs"][15]["intermediate_size"], 12288);
    }
}
