// ============================================================================
// GEMMA 4 «UNIFIED» VISION MAPPER — embebedor sin torre
// ============================================================================
//
// El 12B/31B no tienen encoder visual: no hay ni una capa de atencion.  Los
// parches crudos se proyectan directamente al espacio del decoder y entran
// como `num_soft_tokens` tokens mas.  Pipeline (modeling_gemma4_unified.py,
// Gemma4UnifiedVisionEmbedder):
//
//   parches[model_patch_size^2 * 3] -> LN1 -> Dense -> LN2
//     -> + posicion factorizada (x,y) -> LN3 -> RMSNorm sin escala -> Linear
//
// Diferencias con `Gemma4VisionMapper` (E2B/E4B), que mapea 659 tensores de
// una torre de 16 capas:
//   - aqui son 10 tensores y ninguno es de atencion;
//   - las normas son LayerNorm CON bias, no RMSNorm;
//   - la posicion es factorizada: pos_embedding[idx, eje] y se suman los dos
//     ejes, de ahi el shape (mm_posemb_size, 2, mm_embed_dim);
//   - la RMSNorm previa a la proyeccion final no tiene escala, por eso el
//     checkpoint no trae ningun peso para ella.
// ============================================================================

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::safetensor::SafetensorReader;

use super::traits::ModelMapper;
use super::types::{QuantHint, TensorCategory, TensorMapping};

pub struct Gemma4UnifiedVisionMapper {
    vision: Value,
    tokens: Value,
    text_hidden: usize,
    patch_dim: usize,
}

impl Gemma4UnifiedVisionMapper {
    pub fn from_model_path(model_path: &Path, root: &Value) -> Result<Self> {
        let vision = root
            .get("vision_config")
            .cloned()
            .context("Gemma 4 unified config is missing vision_config")?;
        if vision.get("model_type").and_then(Value::as_str) != Some("gemma4_unified_vision") {
            anyhow::bail!("vision_config is not a Gemma 4 unified vision embedder");
        }
        let text_hidden = root
            .get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let model_patch = vision
            .get("model_patch_size")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if text_hidden == 0 || model_patch == 0 {
            anyhow::bail!("Gemma 4 unified vision needs text hidden_size and model_patch_size");
        }
        let patch_dim = model_patch * model_patch * 3;
        let posemb = vision
            .get("mm_posemb_size")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if posemb == 0 {
            anyhow::bail!("Gemma 4 unified vision needs mm_posemb_size");
        }

        // El inventario es fijo: si falta uno, el bloque saldria mudo y es
        // mejor rechazar la conversion que escribir visión a medias.
        let reader = SafetensorReader::from_folder(model_path)?;
        for (name, expected) in Self::inventory(patch_dim, text_hidden, posemb) {
            let shape = reader
                .shape(&name)
                .with_context(|| format!("Gemma 4 unified vision is missing {name}"))?;
            if shape != expected.as_slice() {
                anyhow::bail!(
                    "Gemma 4 unified vision tensor {name} has shape {shape:?}, expected {expected:?}"
                );
            }
        }

        // El marcador y sus delimitadores son del checkpoint: el motor expande
        // `<|image|>` en BOI + IMAGE * N + EOI sin adivinar ids.
        let tokens = json!({
            "image_token_id": root["image_token_id"],
            "boi_token_id": root["boi_token_id"],
            "eoi_token_id": root["eoi_token_id"],
            "pad_token_id": root["text_config"]["pad_token_id"],
        });
        if tokens.as_object().is_some_and(|t| t.values().any(|v| !v.is_u64())) {
            anyhow::bail!("Gemma 4 unified config is missing image token ids");
        }

        Ok(Self { vision, tokens, text_hidden, patch_dim })
    }

    /// Los 10 tensores del embebedor, con la forma que debe traer cada uno.
    fn inventory(patch_dim: usize, hidden: usize, posemb: usize) -> Vec<(String, Vec<usize>)> {
        vec![
            ("model.vision_embedder.patch_ln1.weight".into(), vec![patch_dim]),
            ("model.vision_embedder.patch_ln1.bias".into(), vec![patch_dim]),
            ("model.vision_embedder.patch_dense.weight".into(), vec![hidden, patch_dim]),
            ("model.vision_embedder.patch_dense.bias".into(), vec![hidden]),
            ("model.vision_embedder.patch_ln2.weight".into(), vec![hidden]),
            ("model.vision_embedder.patch_ln2.bias".into(), vec![hidden]),
            ("model.vision_embedder.pos_embedding".into(), vec![posemb, 2, hidden]),
            ("model.vision_embedder.pos_norm.weight".into(), vec![hidden]),
            ("model.vision_embedder.pos_norm.bias".into(), vec![hidden]),
            ("model.embed_vision.embedding_projection.weight".into(), vec![hidden, hidden]),
        ]
    }
}

impl ModelMapper for Gemma4UnifiedVisionMapper {
    fn name(&self) -> &str {
        "gemma4_unified_vision"
    }

    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        // FP16 en todo el bloque: son ~105 MB y es la referencia con la que se
        // compara la salida.  Ningun flag de cuantizacion textual lo toca.
        let (canonical, category) = match name {
            "model.vision_embedder.patch_ln1.weight" => ("vision.patch_ln1.weight", TensorCategory::Norm),
            "model.vision_embedder.patch_ln1.bias" => ("vision.patch_ln1.bias", TensorCategory::Norm),
            "model.vision_embedder.patch_dense.weight" => ("vision.patch_dense.weight", TensorCategory::VisionPatch),
            "model.vision_embedder.patch_dense.bias" => ("vision.patch_dense.bias", TensorCategory::VisionPatch),
            "model.vision_embedder.patch_ln2.weight" => ("vision.patch_ln2.weight", TensorCategory::Norm),
            "model.vision_embedder.patch_ln2.bias" => ("vision.patch_ln2.bias", TensorCategory::Norm),
            "model.vision_embedder.pos_embedding" => ("vision.pos_embedding", TensorCategory::VisionPatch),
            "model.vision_embedder.pos_norm.weight" => ("vision.pos_norm.weight", TensorCategory::Norm),
            "model.vision_embedder.pos_norm.bias" => ("vision.pos_norm.bias", TensorCategory::Norm),
            "model.embed_vision.embedding_projection.weight" => {
                ("vision.mm_projection.weight", TensorCategory::VisionProjector)
            }
            _ => return None,
        };
        Some(TensorMapping::new(canonical, QuantHint::FP16, category))
    }

    fn execution_hints(&self) -> Value {
        let v = &self.vision;
        json!({
            "arch": "gemma4_unified_vision",
            "dtype": "fp16",
            "encoder_free": true,
            "patch_dim": self.patch_dim,
            "model_patch_size": v["model_patch_size"],
            "patch_size": v["patch_size"],
            "mm_embed_dim": v["mm_embed_dim"],
            "mm_posemb_size": v["mm_posemb_size"],
            "num_soft_tokens": v["num_soft_tokens"],
            "pooling_kernel_size": v["pooling_kernel_size"],
            "output_proj_dims": v["output_proj_dims"],
            "rms_norm_eps": v["rms_norm_eps"],
            "text_hidden_size": self.text_hidden,
            "norm_type": "layernorm_with_bias",
            "position_embedding": "factorized_xy",
            "projection_norm": "rmsnorm_no_scale",
            "image_token_id": self.tokens["image_token_id"],
            "boi_token_id": self.tokens["boi_token_id"],
            "eoi_token_id": self.tokens["eoi_token_id"],
            "pad_token_id": self.tokens["pad_token_id"],
        })
    }

    fn num_layers(&self) -> usize { 0 }
    fn vocab_size(&self) -> usize { 0 }
    fn hidden_size(&self) -> usize { self.text_hidden }
}
