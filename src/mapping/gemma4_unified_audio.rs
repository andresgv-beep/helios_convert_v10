// ============================================================================
// GEMMA 4 «UNIFIED» AUDIO MAPPER — embebedor sin torre
// ============================================================================
//
// El 12B no tiene encoder de audio: ni mel ni conformer.  La onda cruda a
// 16 kHz se trocea en marcos de `audio_samples_per_token` muestras (640 =
// 40 ms) y cada marco entra como un token más (modeling_gemma4_unified.py,
// Gemma4UnifiedModel.get_audio_features):
//
//   marco[640] -> RMSNorm sin escala -> Linear(640 -> hidden)
//
// Es el mismo `Gemma4UnifiedMultimodalEmbedder` que cierra la visión, así que
// el checkpoint trae un único tensor: la proyección.  La RMSNorm no tiene peso.
// ============================================================================

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::safetensor::SafetensorReader;

use super::traits::ModelMapper;
use super::types::{QuantHint, TensorCategory, TensorMapping};

pub struct Gemma4UnifiedAudioMapper {
    audio: Value,
    tokens: Value,
    text_hidden: usize,
    frame: usize,
    sampling_rate: u64,
    ms_per_token: u64,
    max_tokens: u64,
}

const PROJECTION: &str = "model.embed_audio.embedding_projection.weight";

impl Gemma4UnifiedAudioMapper {
    pub fn from_model_path(model_path: &Path, root: &Value) -> Result<Self> {
        let audio = root
            .get("audio_config")
            .cloned()
            .context("Gemma 4 unified config is missing audio_config")?;
        if audio.get("model_type").and_then(Value::as_str) != Some("gemma4_unified_audio") {
            anyhow::bail!("audio_config is not a Gemma 4 unified audio embedder");
        }
        let text_hidden = root
            .get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let frame = audio
            .get("audio_samples_per_token")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let projected = audio.get("output_proj_dims").and_then(Value::as_u64).unwrap_or(0) as usize;
        if text_hidden == 0 || frame == 0 || projected != frame {
            anyhow::bail!(
                "Gemma 4 unified audio needs text hidden_size and a frame equal to output_proj_dims \
                 (frame {frame}, output_proj_dims {projected})"
            );
        }

        // El muestreo y la duración máxima viven en el procesador, no en el
        // modelo: sin ellos el motor tendría que adivinar cómo trocear la onda.
        let processor: Value = std::fs::read_to_string(model_path.join("processor_config.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .context("Gemma 4 unified audio needs processor_config.json")?;
        let extractor = processor.get("feature_extractor").cloned().unwrap_or(Value::Null);
        let sampling_rate = extractor.get("sampling_rate").and_then(Value::as_u64).unwrap_or(0);
        let ms_per_token = processor.get("audio_ms_per_token").and_then(Value::as_u64).unwrap_or(0);
        let max_tokens = processor.get("audio_seq_length").and_then(Value::as_u64).unwrap_or(0);
        if sampling_rate == 0 || ms_per_token == 0 || max_tokens == 0
            || sampling_rate * ms_per_token / 1000 != frame as u64
        {
            anyhow::bail!(
                "Gemma 4 unified audio processor disagrees with the model: {sampling_rate} Hz x \
                 {ms_per_token} ms should give {frame} samples per token"
            );
        }

        let reader = SafetensorReader::from_folder(model_path)?;
        let shape = reader
            .shape(PROJECTION)
            .with_context(|| format!("Gemma 4 unified audio is missing {PROJECTION}"))?;
        if shape != [text_hidden, frame] {
            anyhow::bail!(
                "Gemma 4 unified audio tensor {PROJECTION} has shape {shape:?}, expected {:?}",
                [text_hidden, frame]
            );
        }

        let tokens = json!({
            "audio_token_id": root["audio_token_id"],
            "boa_token_id": root["boa_token_id"],
            "eoa_token_id": root["eoa_token_index"],
            "pad_token_id": root["text_config"]["pad_token_id"],
        });
        if tokens.as_object().is_some_and(|t| t.values().any(|v| !v.is_u64())) {
            anyhow::bail!("Gemma 4 unified config is missing audio token ids");
        }

        Ok(Self { audio, tokens, text_hidden, frame, sampling_rate, ms_per_token, max_tokens })
    }
}

impl ModelMapper for Gemma4UnifiedAudioMapper {
    fn name(&self) -> &str {
        "gemma4_unified_audio"
    }

    fn map_tensor(&self, name: &str) -> Option<TensorMapping> {
        // FP16 como la visión: 5 MB, y es la referencia contra la que se mide.
        (name == PROJECTION).then(|| {
            TensorMapping::new("audio.mm_projection.weight", QuantHint::FP16, TensorCategory::AudioMel)
        })
    }

    fn execution_hints(&self) -> Value {
        json!({
            "arch": "gemma4_unified_audio",
            "dtype": "fp16",
            "encoder_free": true,
            "samples_per_token": self.frame,
            "sampling_rate": self.sampling_rate,
            "ms_per_token": self.ms_per_token,
            "max_tokens": self.max_tokens,
            "rms_norm_eps": self.audio["rms_norm_eps"],
            "text_hidden_size": self.text_hidden,
            "projection_norm": "rmsnorm_no_scale",
            "audio_token_id": self.tokens["audio_token_id"],
            "boa_token_id": self.tokens["boa_token_id"],
            "eoa_token_id": self.tokens["eoa_token_id"],
            "pad_token_id": self.tokens["pad_token_id"],
        })
    }

    fn num_layers(&self) -> usize { 0 }
    fn vocab_size(&self) -> usize { 0 }
    fn hidden_size(&self) -> usize { self.text_hidden }
}
