// src/htf/mod.rs
// ============================================================================
// HTF - HELIOS Token Format v1.2.2 / v1.3.0 (MULTI-DOMAIN)
// ============================================================================
//
// SOPORTA DOS FORMATOS:
//   - HTF v1.2.x (magic "HTF2"): Config como JSON embebido (legacy)
//   - HTF v1.3.0 (magic "HTF3"): Config como estructuras binarias (nuevo)
//
// v1.3.0 CHANGES:
//   - config_json reemplazado por TextDomainConfigBin (32 bytes)
//   - added_tokens_decoder ahora es array binario de AddedTokenEntry
//   - Magic cambia de "HTF2" a "HTF3"
//   - Parsing O(1) en lugar de O(n)
//
// v1.2.2 CHANGES:
//   - FALLBACK: vocab.json cuando tokenizer.json["model"]["vocab"] vacío
//   - FALLBACK: merges.txt cuando tokenizer.json["model"]["merges"] vacío
//   - Soporte para Phi-4, GPT-2, y otros modelos con formato legacy
//
// FORMATO CONTRACTUAL (SPEC: HTF_v1_3_SPEC.txt, HTF_v1_2_1_SPEC.txt)
//
// HEADER (32 bytes):
//   [0:4]   magic           "HTF3" (v1.3) o "HTF2" (v1.2)
//   [4:6]   version         0x0130 (v1.3) o 0x0103 (v1.2.1)
//   [6:8]   flags           u16
//   [8:9]   num_domains     u8
//   [9:16]  reserved        7 bytes (0x00)
//   [16:24] total_size      u64
//   [24:32] checksum        u64 (XXH3-64)
//
// DOMAIN TABLE (32 bytes per domain):
//   [0:1]   domain_type     u8
//   [1:2]   domain_flags    u8
//   [2:4]   reserved        u16
//   [4:8]   vocab_size      u32
//   [8:16]  data_offset     u64
//   [16:24] data_size       u64
//   [24:32] name_hash       u64 (XXH3-64 of domain name)
//
// ============================================================================

pub mod binary;
pub mod validate;

use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use serde_json::Value;

use binary::{
    TextDomainConfigBin, VisionDomainConfigBin, AudioDomainConfigBin, CodeDomainConfigBin,
    AddedTokenEntry, extract_added_tokens,
    HTF3_MAGIC, HTF3_VERSION,
};

/// A BPE merge kept as two independent token strings.  Tokenizers >= 0.21
/// serialize merges as JSON pairs; keeping the pair avoids corrupting tokens
/// that themselves contain whitespace/newlines.
type MergePair = (String, String);

// ============================================================================
// CONSTANTS
// ============================================================================

// HTF v1.2 (legacy)
pub const HTF_MAGIC: &[u8; 4] = b"HTF2";
pub const HTF_VERSION: u16 = 0x0103;  // v1.2.1

// HTF v1.3 (nuevo) - re-exportados de binary.rs
pub use binary::{HTF3_MAGIC as HTF_MAGIC_V13, HTF3_VERSION as HTF_VERSION_V13};
pub const HTF_HEADER_SIZE: usize = 32;
pub const HTF_DOMAIN_ENTRY_SIZE: usize = 32;

// Domain types (§5)
pub const HTF_DOMAIN_TEXT: u8 = 0x00;
pub const HTF_DOMAIN_VISION: u8 = 0x01;
pub const HTF_DOMAIN_AUDIO: u8 = 0x02;
pub const HTF_DOMAIN_CODE: u8 = 0x03;
pub const HTF_DOMAIN_CORTEX: u8 = 0x04;

// Header flags (§4)
pub const HTF_HEADER_HAS_CODEBOOK: u16 = 0x0001;
pub const HTF_HEADER_HAS_MERGES: u16 = 0x0002;

// Domain flags (§6) - CORREGIDO según spec
pub const HTF_FLAG_HAS_VOCAB: u8 = 0x01;      // bit 0
pub const HTF_FLAG_HAS_CODEBOOK: u8 = 0x02;   // bit 1
pub const HTF_FLAG_HAS_MERGES: u8 = 0x04;     // bit 2
pub const HTF_FLAG_IS_PRIMARY: u8 = 0x08;     // bit 3 ← ERA 0x80, CORREGIDO
pub const HTF_FLAG_SHARED_SPECIAL: u8 = 0x10; // bit 4 ← ERA 0x40, CORREGIDO

// Token flags (§7)
pub const TOKEN_FLAG_SPECIAL: u8 = 0x01;  // bit 0: IS_SPECIAL
pub const TOKEN_FLAG_UNKNOWN: u8 = 0x02;  // bit 1: IS_UNKNOWN
pub const TOKEN_FLAG_CONTROL: u8 = 0x04;  // bit 2: IS_CONTROL
pub const TOKEN_FLAG_BYTE: u8 = 0x08;     // bit 3: IS_BYTE
pub const TOKEN_FLAG_ADDED: u8 = 0x10;    // bit 4: IS_ADDED (de Python)

// ============================================================================
// XXH3-64 hash (contractual)
// ============================================================================

fn xxh3_64(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

fn compute_htf_checksum(blob: &[u8]) -> u64 {
    // Regla 6 (HTF spec): XXH3-64 seed=0 sobre:
    //   header[0:24] + (8 bytes 0x00) + (domain_table + domain_data)
    use xxhash_rust::xxh3::Xxh3;
    
    if blob.len() < HTF_HEADER_SIZE {
        return 0;
    }
    
    let mut hasher = Xxh3::new();
    hasher.update(&blob[..24]);           // header sin checksum
    hasher.update(&[0u8; 8]);             // checksum como zeros
    hasher.update(&blob[HTF_HEADER_SIZE..]); // domain table + data
    hasher.digest()
}

// ============================================================================
// HELPERS
// ============================================================================

fn pad_to(buf: &mut Vec<u8>, alignment: usize) {
    let pad = (alignment - (buf.len() % alignment)) % alignment;
    buf.extend(std::iter::repeat(0u8).take(pad));
}

fn is_byte_token(token: &str) -> bool {
    // Formato <0xNN>
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        let hex = &token[3..5];
        hex.chars().all(|c| c.is_ascii_hexdigit())
    } else {
        false
    }
}

fn extract_control_ids(config: &Value) -> std::collections::HashSet<u32> {
    let mut ids = std::collections::HashSet::new();
    for key in &["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"] {
        if let Some(id) = config.get(*key).and_then(|v| v.as_u64()) {
            ids.insert(id as u32);
        }
    }
    // También añadir eos_token_ids si es lista (Qwen3, etc.)
    if let Some(eos_list) = config.get("eos_token_ids").and_then(|v| v.as_array()) {
        for eos in eos_list {
            if let Some(id) = eos.as_u64() {
                ids.insert(id as u32);
            }
        }
    }
    ids
}

fn extract_special_ids(added_tokens: Option<&serde_json::Map<String, Value>>) -> std::collections::HashSet<u32> {
    let mut ids = std::collections::HashSet::new();
    if let Some(tokens) = added_tokens {
        for (id_str, info) in tokens {
            if let Ok(id) = id_str.parse::<u32>() {
                if info.get("special").and_then(|v| v.as_bool()).unwrap_or(false) {
                    ids.insert(id);
                }
            }
        }
    }
    ids
}

fn extract_added_ids(added_tokens: Option<&serde_json::Map<String, Value>>) -> std::collections::HashSet<u32> {
    let mut ids = std::collections::HashSet::new();
    if let Some(tokens) = added_tokens {
        for id_str in tokens.keys() {
            if let Ok(id) = id_str.parse::<u32>() {
                ids.insert(id);
            }
        }
    }
    ids
}

// ============================================================================
// HTF WRITER
// ============================================================================

struct DomainEntry {
    domain_type: u8,
    domain_flags: u8,
    vocab_size: u32,
    data: Vec<u8>,
}

/// HTF Writer - soporta v1.2 (JSON) y v1.3 (binario)
pub struct HTFWriter {
    domains: Vec<DomainEntry>,
    use_v13: bool,  // true = HTF v1.3 binario, false = HTF v1.2 JSON
}

impl HTFWriter {
    /// Crea writer para HTF v1.2 (JSON, legacy)
    pub fn new() -> Self {
        Self { domains: Vec::new(), use_v13: false }
    }
    
    /// Crea writer para HTF v1.3 (binario, nuevo)
    pub fn new_v13() -> Self {
        Self { domains: Vec::new(), use_v13: true }
    }
    
    /// Configura la versión a usar
    pub fn set_version(&mut self, use_v13: bool) {
        self.use_v13 = use_v13;
    }
    
    /// Añade dominio TEXT con vocab y merges
    pub fn add_text_domain(
        &mut self,
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
        is_primary: bool,
    ) {
        self.add_domain_internal(HTF_DOMAIN_TEXT, vocab, merges, config, is_primary);
    }
    
    /// Añade dominio CODE con vocab y merges
    pub fn add_code_domain(
        &mut self,
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
        is_primary: bool,
    ) {
        self.add_domain_internal(HTF_DOMAIN_CODE, vocab, merges, config, is_primary);
    }

    /// Añade dominio CORTEX con el mismo layout binario que TEXT.
    /// El tipo distinto permite que el runtime seleccione su tokenizer sin
    /// confundirlo con el modelo conversacional primario.
    pub fn add_cortex_domain(
        &mut self,
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
        is_primary: bool,
    ) {
        self.add_domain_internal(HTF_DOMAIN_CORTEX, vocab, merges, config, is_primary);
    }
    
    /// Añade dominio AUDIO con vocab y merges
    pub fn add_audio_domain(
        &mut self,
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
        is_primary: bool,
    ) {
        self.add_domain_internal(HTF_DOMAIN_AUDIO, vocab, merges, config, is_primary);
    }
    
    /// Añade dominio genérico (interno)
    fn add_domain_internal(
        &mut self,
        domain_type: u8,
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
        is_primary: bool,
    ) {
        let mut flags: u8 = 0;
        if !vocab.is_empty() {
            flags |= HTF_FLAG_HAS_VOCAB;
        }
        if !merges.is_empty() {
            flags |= HTF_FLAG_HAS_MERGES;
        }
        if is_primary {
            flags |= HTF_FLAG_IS_PRIMARY;  // 0x08, bit 3
        }
        
        // Elegir formato según versión
        let data = if self.use_v13 {
            Self::build_domain_data_v13(domain_type, vocab, merges, config)
        } else {
            Self::build_domain_data_v12(vocab, merges, config)
        };
        
        self.domains.push(DomainEntry {
            domain_type,
            domain_flags: flags,
            vocab_size: vocab.len() as u32,
            data,
        });
    }
    
    /// Build domain data para HTF v1.2 (JSON config)
    fn build_domain_data_v12(
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        
        // Config JSON (compacto, sin espacios)
        let config_json = serde_json::to_string(config).unwrap_or_default();
        let config_bytes = config_json.as_bytes();
        buf.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(config_bytes);
        pad_to(&mut buf, 4);
        
        // Extraer info de tokens especiales del config
        let added_tokens = config.get("added_tokens_decoder")
            .and_then(|v| v.as_object());
        let unk_token_id = config.get("unk_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let control_ids = extract_control_ids(config);
        let special_ids = extract_special_ids(added_tokens);
        let added_ids = extract_added_ids(added_tokens);
        
        // Vocab
        if !vocab.is_empty() {
            buf.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
            
            // Ordenar por token_id (contractual)
            let mut sorted: Vec<_> = vocab.iter().collect();
            sorted.sort_by_key(|(_, &id)| id);
            
            for (token, &token_id) in sorted {
                let token_bytes = token.as_bytes();
                let token_len = token_bytes.len() as u16;
                
                // Token flags (§7 spec)
                let mut token_flags: u8 = 0;
                if special_ids.contains(&token_id) {
                    token_flags |= TOKEN_FLAG_SPECIAL;  // 0x01
                }
                if Some(token_id) == unk_token_id {
                    token_flags |= TOKEN_FLAG_UNKNOWN;  // 0x02
                }
                if control_ids.contains(&token_id) {
                    token_flags |= TOKEN_FLAG_CONTROL;  // 0x04
                }
                if is_byte_token(token) {
                    token_flags |= TOKEN_FLAG_BYTE;     // 0x08
                }
                if added_ids.contains(&token_id) {
                    token_flags |= TOKEN_FLAG_ADDED;    // 0x10
                }
                
                let score_type: u8 = 0; // NONE
                
                // TokenEntry: u32 token_id, u16 token_len, u8 flags, u8 score_type
                buf.extend_from_slice(&token_id.to_le_bytes());
                buf.extend_from_slice(&token_len.to_le_bytes());
                buf.push(token_flags);
                buf.push(score_type);
                buf.extend_from_slice(token_bytes);
                pad_to(&mut buf, 4);
            }
        }
        
        // Merges (contractual: pares de IDs)
        if !merges.is_empty() && !vocab.is_empty() {
            let mut merge_pairs: Vec<(u32, u32)> = Vec::new();
            
            for (left, right) in merges {
                if let (Some(&a), Some(&b)) = (vocab.get(left), vocab.get(right)) {
                    merge_pairs.push((a, b));
                }
            }
            
            buf.extend_from_slice(&(merge_pairs.len() as u32).to_le_bytes());
            for (a, b) in merge_pairs {
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
            }
        }
        
        buf
    }
    
    /// Build domain data para HTF v1.3 (config binario)
    fn build_domain_data_v13(
        domain_type: u8,
        vocab: &HashMap<String, u32>,
        merges: &[MergePair],
        config: &Value,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        
        // Extraer info de tokens especiales del config (para vocab flags)
        let added_tokens_map = config.get("added_tokens_decoder")
            .and_then(|v| v.as_object());
        let unk_token_id = config.get("unk_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let control_ids = extract_control_ids(config);
        let special_ids = extract_special_ids(added_tokens_map);
        let added_ids = extract_added_ids(added_tokens_map);
        
        // 1. Config binario según tipo de dominio
        match domain_type {
            HTF_DOMAIN_TEXT | HTF_DOMAIN_CORTEX => {
                // Extraer added tokens para el config
                let added_tokens = extract_added_tokens(config);
                let num_added = added_tokens.len() as u16;
                
                // TextDomainConfigBin (32 bytes)
                let text_config = TextDomainConfigBin::from_config(config, vocab.len() as u32, num_added);
                buf.extend_from_slice(&text_config.to_bytes());
                
                // Added tokens count + entries
                buf.extend_from_slice(&(added_tokens.len() as u32).to_le_bytes());
                for token in &added_tokens {
                    buf.extend_from_slice(&token.to_bytes());
                }
                
                // Pad to 8 bytes before vocab
                pad_to(&mut buf, 8);
            }
            HTF_DOMAIN_CODE => {
                // Extraer added tokens
                let added_tokens = extract_added_tokens(config);
                let num_added = added_tokens.len() as u16;
                
                // Para CODE, usamos TextDomainConfigBin también (tiene vocab)
                // Pero añadimos CodeDomainConfigBin adicional para FIM tokens
                let text_config = TextDomainConfigBin::from_config(config, vocab.len() as u32, num_added);
                buf.extend_from_slice(&text_config.to_bytes());
                
                // CodeDomainConfigBin (32 bytes) - información adicional FIM
                let code_config = CodeDomainConfigBin::from_config(config);
                buf.extend_from_slice(&code_config.to_bytes());
                
                // Added tokens
                buf.extend_from_slice(&(added_tokens.len() as u32).to_le_bytes());
                for token in &added_tokens {
                    buf.extend_from_slice(&token.to_bytes());
                }
                
                pad_to(&mut buf, 8);
            }
            HTF_DOMAIN_VISION => {
                // VisionDomainConfigBin (64 bytes)
                let vision_config = VisionDomainConfigBin::from_config(config);
                buf.extend_from_slice(&vision_config.to_bytes());
                // Vision no tiene vocab, retornar solo config
                return buf;
            }
            HTF_DOMAIN_AUDIO => {
                // AudioDomainConfigBin (64 bytes)
                let audio_config = AudioDomainConfigBin::from_config(config);
                buf.extend_from_slice(&audio_config.to_bytes());
                // Audio puede tener codebook, pero vocab normal no
                // Por ahora retornamos solo config
                return buf;
            }
            _ => {
                // Dominio desconocido, usar formato JSON como fallback
                let config_json = serde_json::to_string(config).unwrap_or_default();
                let config_bytes = config_json.as_bytes();
                buf.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(config_bytes);
                pad_to(&mut buf, 8);
            }
        }
        
        // 2. Vocab (mismo formato que v1.2)
        if !vocab.is_empty() {
            buf.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
            
            // Ordenar por token_id (contractual)
            let mut sorted: Vec<_> = vocab.iter().collect();
            sorted.sort_by_key(|(_, &id)| id);
            
            for (token, &token_id) in sorted {
                let token_bytes = token.as_bytes();
                let token_len = token_bytes.len() as u16;
                
                // Token flags (§7 spec)
                let mut token_flags: u8 = 0;
                if special_ids.contains(&token_id) {
                    token_flags |= TOKEN_FLAG_SPECIAL;
                }
                if Some(token_id) == unk_token_id {
                    token_flags |= TOKEN_FLAG_UNKNOWN;
                }
                if control_ids.contains(&token_id) {
                    token_flags |= TOKEN_FLAG_CONTROL;
                }
                if is_byte_token(token) {
                    token_flags |= TOKEN_FLAG_BYTE;
                }
                if added_ids.contains(&token_id) {
                    token_flags |= TOKEN_FLAG_ADDED;
                }
                
                let score_type: u8 = 0;
                
                buf.extend_from_slice(&token_id.to_le_bytes());
                buf.extend_from_slice(&token_len.to_le_bytes());
                buf.push(token_flags);
                buf.push(score_type);
                buf.extend_from_slice(token_bytes);
                pad_to(&mut buf, 4);
            }
        }
        
        // 3. Merges (mismo formato que v1.2)
        if !merges.is_empty() && !vocab.is_empty() {
            let mut merge_pairs: Vec<(u32, u32)> = Vec::new();
            
            for (left, right) in merges {
                if let (Some(&a), Some(&b)) = (vocab.get(left), vocab.get(right)) {
                    merge_pairs.push((a, b));
                }
            }
            
            buf.extend_from_slice(&(merge_pairs.len() as u32).to_le_bytes());
            for (a, b) in merge_pairs {
                buf.extend_from_slice(&a.to_le_bytes());
                buf.extend_from_slice(&b.to_le_bytes());
            }
        }
        
        buf
    }
    
    /// Construye el archivo HTF completo (contractual)
    pub fn build(&self) -> Vec<u8> {
        if self.domains.is_empty() {
            return self.build_empty();
        }
        
        let num_domains = self.domains.len() as u8;
        
        // HTF header flags (§4)
        let mut htf_flags: u16 = 0;
        if self.domains.iter().any(|d| d.domain_flags & HTF_FLAG_HAS_MERGES != 0) {
            htf_flags |= HTF_HEADER_HAS_MERGES;  // 0x0002
        }
        if self.domains.iter().any(|d| d.domain_flags & HTF_FLAG_HAS_CODEBOOK != 0) {
            htf_flags |= HTF_HEADER_HAS_CODEBOOK;  // 0x0001
        }
        
        let mut result = Vec::new();
        
        // HEADER placeholder (32 bytes)
        result.extend_from_slice(&[0u8; HTF_HEADER_SIZE]);
        
        // DOMAIN TABLE placeholder (N * 32)
        let domain_table_offset = result.len();
        result.extend(std::iter::repeat(0u8).take(num_domains as usize * HTF_DOMAIN_ENTRY_SIZE));
        
        // Alinear a 16 antes del primer dominio (Regla 4: data_offset alineado a 16)
        pad_to(&mut result, 16);
        
        let mut domain_offsets: Vec<u64> = Vec::new();
        let mut domain_sizes: Vec<u64> = Vec::new();
        
        // DOMAIN DATA, cada dominio empieza alineado a 16
        for domain in &self.domains {
            pad_to(&mut result, 16);
            domain_offsets.push(result.len() as u64);
            domain_sizes.push(domain.data.len() as u64);
            result.extend_from_slice(&domain.data);
        }
        
        // Añadir padding final para que el HTF sea múltiplo de 32 bytes
        // Esto evita que el HNF writer añada padding externo
        pad_to(&mut result, 32);
        
        let total_size = result.len() as u64;
        
        // Escribir domain table
        for (idx, domain) in self.domains.iter().enumerate() {
            let name = match domain.domain_type {
                HTF_DOMAIN_TEXT => "text",
                HTF_DOMAIN_VISION => "vision",
                HTF_DOMAIN_AUDIO => "audio",
                HTF_DOMAIN_CODE => "code",
                HTF_DOMAIN_CORTEX => "cortex",
                _ => "unknown",
            };
            let name_hash = xxh3_64(name.as_bytes());
            
            let start = domain_table_offset + idx * HTF_DOMAIN_ENTRY_SIZE;
            
            // Domain entry: type(1) + flags(1) + reserved(2) + vocab_size(4) + offset(8) + size(8) + hash(8)
            result[start] = domain.domain_type;
            result[start + 1] = domain.domain_flags;
            result[start + 2..start + 4].copy_from_slice(&0u16.to_le_bytes());
            result[start + 4..start + 8].copy_from_slice(&domain.vocab_size.to_le_bytes());
            result[start + 8..start + 16].copy_from_slice(&domain_offsets[idx].to_le_bytes());
            result[start + 16..start + 24].copy_from_slice(&domain_sizes[idx].to_le_bytes());
            result[start + 24..start + 32].copy_from_slice(&name_hash.to_le_bytes());
        }
        
        // Escribir HEADER (sin checksum primero)
        // Usar magic y version según versión configurada
        if self.use_v13 {
            result[0..4].copy_from_slice(HTF3_MAGIC);
            result[4..6].copy_from_slice(&HTF3_VERSION.to_le_bytes());  // 0x0130
        } else {
            result[0..4].copy_from_slice(HTF_MAGIC);
            result[4..6].copy_from_slice(&HTF_VERSION.to_le_bytes());  // 0x0103
        }
        result[6..8].copy_from_slice(&htf_flags.to_le_bytes());
        result[8] = num_domains;
        // [9:16] ya son zeros (reserved)
        result[16..24].copy_from_slice(&total_size.to_le_bytes());
        // [24:32] checksum placeholder (zeros)
        
        // Calcular checksum y parchear
        let checksum = compute_htf_checksum(&result);
        result[24..32].copy_from_slice(&checksum.to_le_bytes());
        
        result
    }
    
    fn build_empty(&self) -> Vec<u8> {
        // HTF mínimo con 1 dominio TEXT vacío
        // Tamaño: 32 (header) + 32 (domain) = 64 bytes (ya múltiplo de 32)
        let mut result = vec![0u8; HTF_HEADER_SIZE + HTF_DOMAIN_ENTRY_SIZE];
        
        // Header - usar magic y version según versión configurada
        if self.use_v13 {
            result[0..4].copy_from_slice(HTF3_MAGIC);
            result[4..6].copy_from_slice(&HTF3_VERSION.to_le_bytes());
        } else {
            result[0..4].copy_from_slice(HTF_MAGIC);
            result[4..6].copy_from_slice(&HTF_VERSION.to_le_bytes());
        }
        result[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags
        result[8] = 1; // num_domains
        
        let total_size = result.len() as u64;
        result[16..24].copy_from_slice(&total_size.to_le_bytes());
        
        // Domain entry para TEXT vacío con IS_PRIMARY
        let start = HTF_HEADER_SIZE;
        result[start] = HTF_DOMAIN_TEXT;
        result[start + 1] = HTF_FLAG_IS_PRIMARY;  // 0x08
        let name_hash = xxh3_64(b"text");
        result[start + 24..start + 32].copy_from_slice(&name_hash.to_le_bytes());
        
        // Checksum
        let checksum = compute_htf_checksum(&result);
        result[24..32].copy_from_slice(&checksum.to_le_bytes());
        
        result
    }
}

impl Default for HTFWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// INTERNAL: Load tokenizer from model directory
// ============================================================================

fn load_tokenizer_from_dir(dir: &Path) -> Result<(HashMap<String, u32>, Vec<MergePair>, serde_json::Map<String, Value>)> {
    // Leer tokenizer.json (puede no existir en modelos legacy)
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer: Value = if tokenizer_path.exists() {
        let data = std::fs::read_to_string(&tokenizer_path)?;
        serde_json::from_str(&data)?
    } else {
        Value::Null
    };
    
    // ════════════════════════════════════════════════════════════════════════
    // VOCAB: Intentar tokenizer.json["model"]["vocab"] primero
    // ════════════════════════════════════════════════════════════════════════
    let mut vocab: HashMap<String, u32> = tokenizer
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                .collect()
        })
        .unwrap_or_default();
    
    // ════════════════════════════════════════════════════════════════════════
    // v1.2.2 FALLBACK: Si vocab vacío, leer vocab.json (Phi-4, GPT-2 format)
    // ════════════════════════════════════════════════════════════════════════
    if vocab.is_empty() {
        let vocab_path = dir.join("vocab.json");
        if vocab_path.exists() {
            let vocab_data = std::fs::read_to_string(&vocab_path)?;
            let vocab_json: Value = serde_json::from_str(&vocab_data)?;
            if let Some(obj) = vocab_json.as_object() {
                vocab = obj.iter()
                    .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                    .collect();
                println!("  [HTF] Loaded vocab from vocab.json: {} tokens", vocab.len());
            }
        }
    }
    
    // Si aún vacío y no hay tokenizer.json, devolver vacío
    if vocab.is_empty() && tokenizer.is_null() {
        return Ok((HashMap::new(), Vec::new(), serde_json::Map::new()));
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // MERGES: Intentar tokenizer.json["model"]["merges"] primero
    // ════════════════════════════════════════════════════════════════════════
    let mut merges: Vec<MergePair> = tokenizer
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    if let Some(pair) = v.as_array() {
                        if pair.len() == 2 {
                            return Some((pair[0].as_str()?.to_string(), pair[1].as_str()?.to_string()));
                        }
                    }
                    // Legacy tokenizers encode a merge as "left right".
                    let legacy = v.as_str()?;
                    let (left, right) = legacy.split_once(' ')?;
                    Some((left.to_string(), right.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    
    // ════════════════════════════════════════════════════════════════════════
    // v1.2.2 FALLBACK: Si merges vacío, leer merges.txt (Phi-4, GPT-2 format)
    // ════════════════════════════════════════════════════════════════════════
    if merges.is_empty() {
        let merges_path = dir.join("merges.txt");
        if merges_path.exists() {
            let merges_data = std::fs::read_to_string(&merges_path)?;
            merges = merges_data
                .lines()
                .filter(|line| {
                    // Ignorar líneas de versión, comentarios y vacías
                    !line.is_empty() 
                    && !line.starts_with("#version")
                    && !line.starts_with("# ")
                    && !line.starts_with("#")
                })
                .filter_map(|s| s.split_once(' '))
                .map(|(left, right)| (left.to_string(), right.to_string()))
                .collect();
            println!("  [HTF] Loaded merges from merges.txt: {} merges", merges.len());
        }
    }
    
    // Construir config
    let mut config = serde_json::Map::new();
    
    // Leer tokenizer_config.json (prioridad 1 según §17)
    let tok_config_path = dir.join("tokenizer_config.json");
    if tok_config_path.exists() {
        let data = std::fs::read_to_string(&tok_config_path)?;
        let tok_config: Value = serde_json::from_str(&data)?;
        
        for key in &["bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id",
                     "bos_token", "eos_token", "unk_token", "pad_token", "add_bos_token",
                     "tokenizer_class", "added_tokens_decoder", "chat_template"] {
            if let Some(v) = tok_config.get(*key) {
                config.insert(key.to_string(), v.clone());
            }
        }
    }

    // Some modern tokenizer configs publish token strings but leave IDs in a
    // nested model config (Gemma 4).  Resolve IDs directly from the canonical
    // vocabulary so HTF never silently emits -1 for known special tokens.
    for (token_key, id_key) in [
        ("bos_token", "bos_token_id"),
        ("eos_token", "eos_token_id"),
        ("unk_token", "unk_token_id"),
        ("pad_token", "pad_token_id"),
    ] {
        if !config.contains_key(id_key) {
            if let Some(token) = config.get(token_key).and_then(Value::as_str) {
                if let Some(id) = vocab.get(token) {
                    config.insert(id_key.to_string(), Value::Number((*id).into()));
                }
            }
        }
    }
    
    // Leer config.json para tokens especiales
    let model_config_path = dir.join("config.json");
    if model_config_path.exists() {
        let data = std::fs::read_to_string(&model_config_path)?;
        let model_config: Value = serde_json::from_str(&data)?;
        
        for key in &["bos_token_id", "eos_token_id", "vocab_size"] {
            if !config.contains_key(*key) {
                if let Some(v) = model_config.get(*key) {
                    config.insert(key.to_string(), v.clone());
                }
            }
        }
    }
    
    // Leer generation_config.json
    let gen_config_path = dir.join("generation_config.json");
    if gen_config_path.exists() {
        let data = std::fs::read_to_string(&gen_config_path)?;
        let gen_config: Value = serde_json::from_str(&data)?;
        
        for key in &["bos_token_id", "eos_token_id", "pad_token_id"] {
            if let Some(v) = gen_config.get(*key) {
                // Manejar eos_token_id como array (Qwen3, etc.)
                if *key == "eos_token_id" && v.is_array() {
                    // Guardar el primero como eos_token_id
                    if !config.contains_key(*key) {
                        if let Some(first) = v.as_array().and_then(|a| a.first()) {
                            config.insert(key.to_string(), first.clone());
                        }
                    }
                    // TAMBIÉN guardar la lista completa como eos_token_ids
                    if !config.contains_key("eos_token_ids") {
                        config.insert("eos_token_ids".to_string(), v.clone());
                    }
                } else if !config.contains_key(*key) {
                    config.insert(key.to_string(), v.clone());
                }
            }
        }
    }
    
    // Leer added_tokens.json (prioridad 2 según §17)
    let added_tokens_path = dir.join("added_tokens.json");
    if added_tokens_path.exists() {
        let data = std::fs::read_to_string(&added_tokens_path)?;
        let added: Value = serde_json::from_str(&data)?;
        
        if let Some(obj) = added.as_object() {
            let mut decoder = config
                .get("added_tokens_decoder")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            
            for (content, id) in obj {
                if let Some(id_num) = id.as_u64() {
                    decoder.insert(
                        id_num.to_string(),
                        serde_json::json!({
                            "content": content,
                            "special": true
                        }),
                    );
                }
            }
            config.insert("added_tokens_decoder".to_string(), Value::Object(decoder));
        }
    }
    
    // Añadir added_tokens de tokenizer.json (prioridad 4 según §17, pero menor)
    if let Some(added_tokens) = tokenizer.get("added_tokens").and_then(|v| v.as_array()) {
        let mut decoder = config
            .get("added_tokens_decoder")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        
        for token_info in added_tokens {
            if let (Some(id), Some(content)) = (
                token_info.get("id").and_then(|v| v.as_u64()),
                token_info.get("content").and_then(|v| v.as_str()),
            ) {
                let special = token_info.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                let lstrip = token_info.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false);
                let rstrip = token_info.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false);
                let single_word = token_info.get("single_word").and_then(|v| v.as_bool()).unwrap_or(false);
                let normalized = token_info.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false);
                // Solo añadir si no existe (prioridad menor)
                if !decoder.contains_key(&id.to_string()) {
                    decoder.insert(
                        id.to_string(),
                        serde_json::json!({
                            "content": content,
                            "special": special,
                            "lstrip": lstrip,
                            "rstrip": rstrip,
                            "single_word": single_word,
                            "normalized": normalized
                        }),
                    );
                }
            }
        }
        
        config.insert("added_tokens_decoder".to_string(), Value::Object(decoder));
    }
    
    // Detectar encoding_type (§17)
    let tokenizer_class = config.get("tokenizer_class")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    let tok_model_exists = dir.join("tokenizer.model").exists();
    
    let tokenizer_model_type = tokenizer
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("");

    let encoding_type = if tokenizer_model_type.eq_ignore_ascii_case("bpe") {
        "bpe"
    } else if tok_model_exists {
        "sentencepiece"
    } else if tokenizer_class.to_lowercase().contains("sentencepiece") 
        || tokenizer_class == "LlamaTokenizer" 
        || tokenizer_class == "GemmaTokenizer" 
    {
        "sentencepiece"
    } else {
        "bpe"
    };
    
    // ByteLevel is a tokenizer operation, not a property that can be inferred
    // from arbitrary vocabulary contents.  Gemma 4 contains tokens such as Ġ
    // in its multilingual vocabulary but uses metaspace + byte fallback.
    fn contains_component_type(value: &Value, expected: &str) -> bool {
        match value {
            Value::Object(map) => {
                map.get("type").and_then(Value::as_str) == Some(expected)
                    || map.values().any(|v| contains_component_type(v, expected))
            }
            Value::Array(values) => values.iter().any(|v| contains_component_type(v, expected)),
            _ => false,
        }
    }
    let byte_level = tokenizer
        .get("pre_tokenizer")
        .map(|v| contains_component_type(v, "ByteLevel"))
        .unwrap_or(false);
    let byte_fallback = tokenizer
        .get("model")
        .and_then(|m| m.get("byte_fallback"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    
    config.insert("encoding_type".to_string(), Value::String(encoding_type.to_string()));
    config.insert("byte_level".to_string(), Value::Bool(byte_level));
    if let Some(pre) = tokenizer.get("pre_tokenizer") {
        config.insert("pre_tokenizer".to_string(), pre.clone());
    }
    if let Some(decoder) = tokenizer.get("decoder") {
        config.insert("decoder".to_string(), decoder.clone());
    }
    if let Some(normalizer) = tokenizer.get("normalizer") {
        config.insert("normalizer".to_string(), normalizer.clone());
    }
    config.insert("byte_fallback".to_string(), Value::Bool(byte_fallback));
    config.insert("vocab_size".to_string(), Value::Number(vocab.len().into()));
    
    Ok((vocab, merges, config))
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Tipo de dominio para build_htf_multi
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainType {
    Text,
    Code,
    Cortex,
    Audio,
    Vision,
}

impl DomainType {
    pub fn to_u8(&self) -> u8 {
        match self {
            DomainType::Text => HTF_DOMAIN_TEXT,
            DomainType::Code => HTF_DOMAIN_CODE,
            DomainType::Cortex => HTF_DOMAIN_CORTEX,
            DomainType::Audio => HTF_DOMAIN_AUDIO,
            DomainType::Vision => HTF_DOMAIN_VISION,
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            DomainType::Text => "text",
            DomainType::Code => "code",
            DomainType::Cortex => "cortex",
            DomainType::Audio => "audio",
            DomainType::Vision => "vision",
        }
    }
}

/// Construye HTF con MÚLTIPLES dominios/tokenizers
/// 
/// # Arguments
/// * `sources` - Lista de (path, domain_type, is_primary)
/// * `use_v13` - true para HTF v1.3 (binario), false para v1.2 (JSON)
/// 
/// # Example
/// ```ignore
/// let sources = vec![
///     (Path::new("./Qwen2.5-4B"), DomainType::Text, true),
///     (Path::new("./DeepSeek-Coder"), DomainType::Code, false),
/// ];
/// let htf_bytes = build_htf_multi_versioned(&sources, true)?;  // v1.3
/// ```
pub fn build_htf_multi_versioned(sources: &[(&Path, DomainType, bool)], use_v13: bool) -> Result<Vec<u8>> {
    let mut writer = if use_v13 {
        HTFWriter::new_v13()
    } else {
        HTFWriter::new()
    };
    
    for (dir, domain_type, is_primary) in sources {
        let (vocab, merges, mut config) = load_tokenizer_from_dir(dir)?;
        
        if vocab.is_empty() {
            eprintln!("[HTF] Warning: No tokenizer found in {}, skipping", dir.display());
            continue;
        }
        
        config.insert("is_primary".to_string(), Value::Bool(*is_primary));
        
        let config_value = Value::Object(config);
        
        match domain_type {
            DomainType::Text => {
                writer.add_text_domain(&vocab, &merges, &config_value, *is_primary);
                let version = if use_v13 { "v1.3" } else { "v1.2" };
                println!("  [HTF {}] Added TEXT domain: {} tokens", version, vocab.len());
            }
            DomainType::Code => {
                writer.add_code_domain(&vocab, &merges, &config_value, *is_primary);
                let version = if use_v13 { "v1.3" } else { "v1.2" };
                println!("  [HTF {}] Added CODE domain: {} tokens", version, vocab.len());
            }
            DomainType::Cortex => {
                writer.add_cortex_domain(&vocab, &merges, &config_value, *is_primary);
                let version = if use_v13 { "v1.3" } else { "v1.2" };
                println!("  [HTF {}] Added CORTEX domain: {} tokens", version, vocab.len());
            }
            DomainType::Audio => {
                writer.add_audio_domain(&vocab, &merges, &config_value, *is_primary);
                let version = if use_v13 { "v1.3" } else { "v1.2" };
                println!("  [HTF {}] Added AUDIO domain: {} tokens", version, vocab.len());
            }
            DomainType::Vision => {
                // Vision normalmente no tiene tokenizer de texto
                eprintln!("[HTF] Warning: Vision domain tokenizer not yet supported");
            }
        }
    }
    
    Ok(writer.build())
}

/// Construye HTF con MÚLTIPLES dominios/tokenizers (usa v1.3 por defecto)
/// 
/// # Arguments
/// * `sources` - Lista de (path, domain_type, is_primary)
/// 
/// # Example
/// ```ignore
/// let sources = vec![
///     (Path::new("./Qwen2.5-4B"), DomainType::Text, true),
///     (Path::new("./DeepSeek-Coder"), DomainType::Code, false),
/// ];
/// let htf_bytes = build_htf_multi(&sources)?;
/// ```
pub fn build_htf_multi(sources: &[(&Path, DomainType, bool)]) -> Result<Vec<u8>> {
    // Usar v1.3 por defecto
    build_htf_multi_versioned(sources, true)
}

/// Construye HTF binario desde un directorio de modelo HuggingFace (single domain)
/// Usa v1.3 por defecto
pub fn build_htf(model_dir: impl AsRef<Path>) -> Result<Vec<u8>> {
    build_htf_versioned(model_dir, true)
}

/// Construye HTF binario desde un directorio de modelo HuggingFace (single domain)
/// 
/// # Arguments
/// * `model_dir` - Directorio del modelo
/// * `use_v13` - true para HTF v1.3 (binario), false para v1.2 (JSON)
pub fn build_htf_versioned(model_dir: impl AsRef<Path>, use_v13: bool) -> Result<Vec<u8>> {
    let dir = model_dir.as_ref();
    
    let (vocab, merges, mut config) = load_tokenizer_from_dir(dir)?;
    
    if vocab.is_empty() {
        // Sin tokenizer.json, crear HTF mínimo
        let writer = if use_v13 { HTFWriter::new_v13() } else { HTFWriter::new() };
        return Ok(writer.build());
    }
    
    config.insert("is_primary".to_string(), Value::Bool(true));
    
    // Construir HTF con versión especificada
    let mut writer = if use_v13 { HTFWriter::new_v13() } else { HTFWriter::new() };
    writer.add_text_domain(&vocab, &merges, &Value::Object(config), true);
    
    let version = if use_v13 { "v1.3" } else { "v1.2" };
    println!("  [HTF {}] Built single TEXT domain: {} tokens", version, vocab.len());
    
    Ok(writer.build())
}

#[cfg(test)]
mod gemma4_tests {
    use super::*;

    #[test]
    fn cortex_is_a_distinct_text_layout_domain() {
        let text_vocab = HashMap::from([
            ("<unk>".to_string(), 0),
            ("texto".to_string(), 1),
        ]);
        let cortex_vocab = HashMap::from([
            ("<unk>".to_string(), 0),
            ("agente".to_string(), 1),
            ("herramienta".to_string(), 2),
        ]);
        let config = serde_json::json!({
            "model": {"type": "BPE"},
            "unk_token_id": 0,
            "encoding_type": "bpe"
        });

        let mut writer = HTFWriter::new_v13();
        writer.add_text_domain(&text_vocab, &[], &config, true);
        writer.add_cortex_domain(&cortex_vocab, &[], &config, false);
        let blob = writer.build();
        let result = validate::validate_htf(&blob);

        assert!(result.valid, "{:?}", result.errors);
        assert_eq!(result.info.domains.len(), 2);
        assert_eq!(result.info.domains[0].domain_type, "TEXT");
        assert_eq!(result.info.domains[0].vocab_size, 2);
        assert_eq!(result.info.domains[1].domain_type, "CORTEX");
        assert_eq!(result.info.domains[1].vocab_size, 3);
        assert!(!result.info.domains[1].is_primary);
    }

    #[test]
    fn builds_complete_local_gemma4_htf_when_available() {
        let Ok(path) = std::env::var("GEMMA4_MODEL_DIR") else { return };
        let path = Path::new(&path);
        let (vocab, merges, config) = load_tokenizer_from_dir(path).unwrap();
        assert_eq!(vocab.len(), 262_144);
        assert_eq!(merges.len(), 514_906);
        assert_eq!(config["encoding_type"], "bpe");
        assert_eq!(config["unk_token_id"], 3);
        let blob = build_htf(path).unwrap();
        let result = validate::validate_htf(&blob);
        assert!(result.valid, "{:?}", result.errors);
        assert!(result.info.domains[0].has_merges);
        let domain_offset = result.info.domains[0].data_offset as usize;
        assert_eq!(blob[domain_offset + 24], 1); // split-space pre-tokenizer
        assert_eq!(blob[domain_offset + 25], 1); // U+2581 decoder
        assert_eq!(blob[domain_offset + 26] & 1, 1); // add BOS
        assert_eq!(blob[domain_offset + 23] & 1, 0); // not GPT-2 ByteLevel
        assert_eq!(blob[domain_offset + 27], 1); // space -> U+2581 normalizer
        assert_eq!(blob[domain_offset + 28] & 1, 1); // byte fallback
    }
}
