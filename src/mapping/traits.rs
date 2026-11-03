// src/mapping/traits.rs
// ============================================================================
// MAPPER TRAIT - Interfaz para mappers de arquitecturas
// ============================================================================

use super::types::TensorMapping;
use serde_json::Value;

/// Trait para mappers de diferentes arquitecturas.
/// 
/// El mapper es PURO:
/// - Solo conoce la arquitectura (nombres de tensores)
/// - NO decide el bloque destino (eso lo hace el CLI)
/// - Sugiere cuantización pero no la impone
pub trait ModelMapper: Send + Sync {
    /// Nombre de la arquitectura (ej: "qwen2", "llama", "phi3")
    fn name(&self) -> &str;
    
    /// Mapea un tensor original a nombre canónico HELIOS.
    /// Retorna None si el tensor debe ignorarse (rotary_emb, inv_freq, etc.)
    fn map_tensor(&self, original_name: &str) -> Option<TensorMapping>;
    
    /// Genera execution_hints para el runtime.
    /// Contiene info de arquitectura: attention_type, mlp_type, rope, etc.
    fn execution_hints(&self) -> Value;
    
    /// Lista de tensores que deben ignorarse (opcional, tiene default)
    fn should_ignore(&self, name: &str) -> bool {
        name.contains("rotary_emb")
            || name.contains("inv_freq")
            || name.contains("_float_tensor")
            || name.contains("position_ids")
    }
    
    /// Número de capas (para validación)
    fn num_layers(&self) -> usize;
    
    /// Tamaño del vocabulario
    fn vocab_size(&self) -> usize;
    
    /// Hidden size
    fn hidden_size(&self) -> usize;

    /// Número exacto de tensores que debe aceptar este mapper, cuando el
    /// contrato de la modalidad lo permite. El builder lo comprueba antes de
    /// escribir el primer byte del bloque.
    fn expected_tensor_count(&self) -> Option<usize> {
        None
    }
    
    /// Es MoE?
    fn is_moe(&self) -> bool {
        false
    }
    
    /// Número de expertos (si es MoE)
    fn num_experts(&self) -> Option<usize> {
        None
    }
}
