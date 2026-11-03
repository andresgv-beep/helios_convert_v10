//! HNFv10: contenedor de HELIOS y su conversor. Ver docs/HNFV10.md.
#![recursion_limit = "256"]

// Contenedor HNFv10.
pub mod format;
pub mod metadata;
pub mod file;
pub mod edit;
pub mod migrate;
pub mod convert;
pub mod fixtures;

// Traídos de helios_convert_v9.1 en 79835e4 (rama gemma4-26b-moe): codificadores HQS, mapeo
// de modelos, safetensors, tokenizer HTF3, configuración binaria y plan de Gemma 4 MoE. Desde
// aquí evolucionan en el v10; el v9.1 queda como pieza histórica y no se vuelve a tocar.
pub mod base_id;
pub mod dictionary;
pub mod hints;
pub mod hqs;
pub mod htf;
pub mod mapping;
pub mod moe_experimental;
pub mod safetensor;

pub use safetensor::SafetensorReader;
