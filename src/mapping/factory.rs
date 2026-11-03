// src/mapping/factory.rs
// ============================================================================
// MAPPER FACTORY - Crea el mapper correcto para cada arquitectura
// ============================================================================

use std::path::Path;
use anyhow::{Result, Context};
use serde_json::Value;

use super::traits::ModelMapper;
use super::qwen2::Qwen2Mapper;
use super::llama::LlamaMapper;
use super::clip::ClipMapper;
use super::phi::PhiMapper;  // AÑADIDO
use super::gemma4::Gemma4Mapper;
use super::gemma4_vision::Gemma4VisionMapper;
use super::gemma4_unified_vision::Gemma4UnifiedVisionMapper;
use super::gemma4_unified_audio::Gemma4UnifiedAudioMapper;
use super::types::BlockType;

/// Detecta la arquitectura de un modelo desde config.json
pub fn detect_architecture(config: &Value) -> String {
    // Por model_type
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        let mt = model_type.to_lowercase();
        
        // Vision encoders
        if mt.contains("clip") || mt.contains("siglip") {
            return "clip".to_string();
        }
        if mt.contains("vit") {
            return "vit".to_string();
        }
        
        // LLMs - Phi ANTES de Llama (phi3 contiene "phi")
        if mt.contains("phi") {
            return "phi".to_string();
        }
        if mt.contains("qwen") {
            return "qwen2".to_string();
        }
        if mt.contains("llama") || mt.contains("deepseek") || mt.contains("codellama") {
            return "llama".to_string();
        }
        if mt.contains("mistral") {
            return "mistral".to_string();
        }
        if mt.contains("gemma") {
            return "gemma".to_string();
        }
        
        return mt;
    }
    
    // Por architectures
    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            
            // Vision
            if arch_lower.contains("clip") || arch_lower.contains("siglip") {
                return "clip".to_string();
            }
            
            // Phi - detectar antes de Llama
            if arch_lower.contains("phi") {
                return "phi".to_string();
            }
            
            // Qwen
            if arch_lower.contains("qwen") {
                return "qwen2".to_string();
            }
            
            // Llama family (includes DeepSeek, CodeLlama, etc.)
            if arch_lower.contains("llama") 
                || arch_lower.contains("deepseek")
                || arch_lower.contains("mistral") {
                return "llama".to_string();
            }
            
            // Gemma
            if arch_lower.contains("gemma") {
                return "gemma".to_string();
            }
        }
    }
    
    "generic".to_string()
}

/// Lee config.json de un modelo
pub fn load_config(model_path: &Path) -> Result<Value> {
    let config_path = model_path.join("config.json");
    
    if !config_path.exists() {
        anyhow::bail!("No config.json found in {}", model_path.display());
    }
    
    let data = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    
    let config: Value = serde_json::from_str(&data)
        .with_context(|| "Invalid JSON in config.json")?;
    
    Ok(config)
}

/// Crea el mapper correcto para un modelo
/// Gemma 4 tiene dos familias con el MISMO formato de tensores en el decoder:
/// `gemma4` (E2B/E4B, con PLE y torre de vision propia) y `gemma4_unified`
/// (12B/31B, sin PLE y con la vision integrada en el decoder).  El mapper de
/// texto sirve para ambas porque deriva la geometria de las shapes.
fn is_gemma4_family(config: &Value) -> bool {
    matches!(
        config.get("model_type").and_then(Value::as_str),
        Some("gemma4") | Some("gemma4_unified"),
    )
}

pub fn create_mapper(model_path: &Path) -> Result<Box<dyn ModelMapper>> {
    let config = load_config(model_path)?;
    let arch = detect_architecture(&config);
    
    println!("[INFO] Detected architecture: {}", arch);
    
    match arch.as_str() {
        "qwen2" | "qwen" => {
            Ok(Box::new(Qwen2Mapper::from_json(&config)))
        }
        
        "llama" | "mistral" | "deepseek" | "codellama" => {
            Ok(Box::new(LlamaMapper::from_json(&config)))
        }
        
        "clip" | "siglip" | "vit" => {
            Ok(Box::new(ClipMapper::from_json(&config)))
        }
        
        // AÑADIDO: Phi family
        "phi" | "phi3" | "phi4" => {
            Ok(Box::new(PhiMapper::from_json(&config)))
        }
        
        "gemma" if is_gemma4_family(&config) => {
            Ok(Box::new(Gemma4Mapper::from_model_path(model_path, &config)?))
        }

        // TODO: Añadir más arquitecturas
        // "gemma" | "gemma2" => Ok(Box::new(GemmaMapper::from_json(&config))),
        // "whisper" => Ok(Box::new(WhisperMapper::from_json(&config))),
        
        _ => {
            eprintln!("[WARN] Unknown architecture '{}', trying llama mapper", arch);
            Ok(Box::new(LlamaMapper::from_json(&config)))
        }
    }
}

/// Crea un mapper para una arquitectura y un bloque HNF concretos.
///
/// Gemma 4 contiene texto, visión y audio bajo el mismo `config.json`; elegir
/// solo por `model_type` hacía que `--vision` convirtiera por error los 600
/// tensores del decoder. Las rutas multimodales deben pasar siempre por esta
/// función para que una modalidad sin mapper falle de forma explícita.
pub fn create_mapper_for_block(
    model_path: &Path,
    target_block: BlockType,
) -> Result<Box<dyn ModelMapper>> {
    let config = load_config(model_path)?;
    let arch = detect_architecture(&config);
    let is_gemma4 = is_gemma4_family(&config);

    println!(
        "[INFO] Detected architecture: {} (target: {})",
        arch,
        target_block.name(),
    );

    match target_block {
        // «unified» no tiene torre: su embebedor va por otro mapper.
        BlockType::Vision
            if config.get("model_type").and_then(Value::as_str) == Some("gemma4_unified") =>
        {
            Ok(Box::new(Gemma4UnifiedVisionMapper::from_model_path(
                model_path, &config,
            )?))
        }
        BlockType::Vision if is_gemma4 => Ok(Box::new(
            Gemma4VisionMapper::from_model_path(model_path, &config)?,
        )),
        BlockType::Vision if matches!(arch.as_str(), "clip" | "siglip" | "vit") => {
            Ok(Box::new(ClipMapper::from_json(&config)))
        }
        BlockType::Vision => anyhow::bail!(
            "architecture '{}' has no dedicated vision mapper; refusing to map another modality",
            arch,
        ),
        // Igual que la visión: sin torre, un embebedor de un solo tensor.
        BlockType::Audio
            if config.get("model_type").and_then(Value::as_str) == Some("gemma4_unified") =>
        {
            Ok(Box::new(Gemma4UnifiedAudioMapper::from_model_path(
                model_path, &config,
            )?))
        }
        BlockType::Audio => anyhow::bail!(
            "architecture '{}' has no dedicated audio mapper; refusing to map another modality",
            arch,
        ),
        BlockType::TextModel | BlockType::CodeExec | BlockType::Cortex => {
            if is_gemma4 {
                Ok(Box::new(Gemma4Mapper::from_model_path(model_path, &config)?))
            } else {
                create_mapper(model_path)
            }
        }
        _ => anyhow::bail!(
            "block '{}' does not have a model mapper",
            target_block.name(),
        ),
    }
}
