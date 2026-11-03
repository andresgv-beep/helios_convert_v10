// src/htf/validate.rs
// ============================================================================
// HTF Validator - Valida archivos HTF v1.2 y v1.3
// ============================================================================

use super::*;
use super::binary::*;

/// Resultado de validación
#[derive(Debug)]
pub struct HTFValidationResult {
    pub valid: bool,
    pub version: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub info: HTFInfo,
}

/// Información extraída del HTF
#[derive(Debug, Default)]
pub struct HTFInfo {
    pub magic: String,
    pub version: u16,
    pub num_domains: u8,
    pub total_size: u64,
    pub checksum: u64,
    pub domains: Vec<DomainInfo>,
}

#[derive(Debug, Default)]
pub struct DomainInfo {
    pub domain_type: String,
    pub vocab_size: u32,
    pub data_offset: u64,
    pub data_size: u64,
    pub is_primary: bool,
    pub has_vocab: bool,
    pub has_merges: bool,
}

/// Valida un blob HTF y extrae información
pub fn validate_htf(data: &[u8]) -> HTFValidationResult {
    let mut result = HTFValidationResult {
        valid: true,
        version: String::new(),
        errors: Vec::new(),
        warnings: Vec::new(),
        info: HTFInfo::default(),
    };
    
    // 1. Verificar tamaño mínimo
    if data.len() < HTF_HEADER_SIZE {
        result.valid = false;
        result.errors.push(format!(
            "File too small: {} bytes (minimum {})",
            data.len(),
            HTF_HEADER_SIZE
        ));
        return result;
    }
    
    // 2. Verificar magic
    let magic = &data[0..4];
    result.info.magic = String::from_utf8_lossy(magic).to_string();
    
    let is_v13 = magic == b"HTF3";
    let is_v12 = magic == b"HTF2";
    let is_v11 = magic == b"HTF1";
    
    if !is_v13 && !is_v12 && !is_v11 {
        result.valid = false;
        result.errors.push(format!(
            "Invalid magic: {:?} (expected HTF3, HTF2, or HTF1)",
            magic
        ));
        return result;
    }
    
    // 3. Leer version
    result.info.version = u16::from_le_bytes([data[4], data[5]]);
    
    if is_v13 {
        result.version = format!("v1.3.{}", result.info.version & 0xFF);
        if result.info.version != HTF3_VERSION {
            result.warnings.push(format!(
                "Unexpected version 0x{:04X} for HTF3 (expected 0x{:04X})",
                result.info.version, HTF3_VERSION
            ));
        }
    } else if is_v12 {
        result.version = format!("v1.2.{}", result.info.version & 0xFF);
    } else {
        result.version = "v1.1".to_string();
    }
    
    // 4. Leer num_domains
    result.info.num_domains = data[8];
    
    if result.info.num_domains == 0 {
        result.valid = false;
        result.errors.push("num_domains is 0 (must be 1-8)".to_string());
        return result;
    }
    
    if result.info.num_domains > 8 {
        result.valid = false;
        result.errors.push(format!(
            "num_domains is {} (maximum 8)",
            result.info.num_domains
        ));
        return result;
    }
    
    // 5. Verificar reserved bytes son 0
    for i in 9..16 {
        if data[i] != 0 {
            result.warnings.push(format!("Reserved byte at offset {} is non-zero", i));
        }
    }
    
    // 6. Leer total_size y checksum
    result.info.total_size = u64::from_le_bytes(data[16..24].try_into().unwrap());
    result.info.checksum = u64::from_le_bytes(data[24..32].try_into().unwrap());
    
    if result.info.total_size != data.len() as u64 {
        result.errors.push(format!(
            "total_size mismatch: header says {} but file is {} bytes",
            result.info.total_size,
            data.len()
        ));
        result.valid = false;
    }
    
    // 7. Verificar checksum
    let computed_checksum = compute_checksum_for_validation(data);
    if computed_checksum != result.info.checksum {
        result.errors.push(format!(
            "Checksum mismatch: computed 0x{:016X} but header says 0x{:016X}",
            computed_checksum, result.info.checksum
        ));
        result.valid = false;
    }
    
    // 8. Leer domain table
    let domain_table_size = result.info.num_domains as usize * HTF_DOMAIN_ENTRY_SIZE;
    let expected_min_size = HTF_HEADER_SIZE + domain_table_size;
    
    if data.len() < expected_min_size {
        result.errors.push(format!(
            "File too small for domain table: {} bytes (need at least {})",
            data.len(),
            expected_min_size
        ));
        result.valid = false;
        return result;
    }
    
    let mut has_primary = false;
    
    for i in 0..result.info.num_domains as usize {
        let start = HTF_HEADER_SIZE + i * HTF_DOMAIN_ENTRY_SIZE;
        let domain_type = data[start];
        let domain_flags = data[start + 1];
        let vocab_size = u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap());
        let data_offset = u64::from_le_bytes(data[start + 8..start + 16].try_into().unwrap());
        let data_size = u64::from_le_bytes(data[start + 16..start + 24].try_into().unwrap());
        
        let domain_type_str = match domain_type {
            0 => "TEXT",
            1 => "VISION",
            2 => "AUDIO",
            3 => "CODE",
            4 => "CORTEX",
            _ => "UNKNOWN",
        };
        
        let is_primary = (domain_flags & HTF_FLAG_IS_PRIMARY) != 0;
        let has_vocab = (domain_flags & HTF_FLAG_HAS_VOCAB) != 0;
        let has_merges = (domain_flags & HTF_FLAG_HAS_MERGES) != 0;
        
        if is_primary {
            if has_primary {
                result.errors.push("Multiple domains marked as PRIMARY".to_string());
                result.valid = false;
            }
            has_primary = true;
            
            if domain_type != HTF_DOMAIN_TEXT {
                result.errors.push(format!(
                    "Primary domain must be TEXT, got {}",
                    domain_type_str
                ));
                result.valid = false;
            }
        }
        
        // Verificar que data_offset y data_size son válidos
        if data_offset as usize + data_size as usize > data.len() {
            result.errors.push(format!(
                "Domain {} data exceeds file bounds: offset {} + size {} > {}",
                i, data_offset, data_size, data.len()
            ));
            result.valid = false;
        }
        
        // Verificar alineación
        if data_offset % 16 != 0 && data_size > 0 {
            result.warnings.push(format!(
                "Domain {} data_offset {} not aligned to 16 bytes",
                i, data_offset
            ));
        }
        
        result.info.domains.push(DomainInfo {
            domain_type: domain_type_str.to_string(),
            vocab_size,
            data_offset,
            data_size,
            is_primary,
            has_vocab,
            has_merges,
        });
    }
    
    if !has_primary {
        result.errors.push("No domain marked as PRIMARY".to_string());
        result.valid = false;
    }
    
    // 9. Validar contenido de dominios según versión
    if is_v13 {
        validate_v13_domains(data, &mut result);
    }
    
    result
}

fn validate_v13_domains(data: &[u8], result: &mut HTFValidationResult) {
    for (i, domain) in result.info.domains.iter().enumerate() {
        if domain.data_size == 0 {
            continue;
        }
        
        let offset = domain.data_offset as usize;
        
        match domain.domain_type.as_str() {
            "TEXT" | "CORTEX" => {
                // Verificar TextDomainConfigBin
                if domain.data_size < TextDomainConfigBin::SIZE as u64 {
                    result.errors.push(format!(
                        "{} domain {} too small for config: {} bytes (need {})",
                        domain.domain_type, i, domain.data_size, TextDomainConfigBin::SIZE
                    ));
                    result.valid = false;
                    continue;
                }
                
                // Leer y validar config
                let vocab_size = u32::from_le_bytes(
                    data[offset + 16..offset + 20].try_into().unwrap()
                );
                let num_added = u16::from_le_bytes(
                    data[offset + 20..offset + 22].try_into().unwrap()
                );
                let encoding_type = data[offset + 22];
                
                if encoding_type > 3 {
                    result.errors.push(format!(
                        "{} domain {}: invalid encoding_type {}",
                        domain.domain_type, i, encoding_type
                    ));
                    result.valid = false;
                }
                
                if vocab_size != domain.vocab_size && domain.has_vocab {
                    result.warnings.push(format!(
                        "{} domain {}: config vocab_size {} != table vocab_size {}",
                        domain.domain_type, i, vocab_size, domain.vocab_size
                    ));
                }
                
                // Bytes 24..26 son metadatos de pre-tokenizer/decoder desde
                // HTF 1.3; solo 27..31 permanecen reservados.
                for j in 3..8 {
                    if data[offset + 24 + j] != 0 {
                        result.warnings.push(format!(
                            "{} domain {}: reserved byte {} is non-zero",
                            domain.domain_type, i, j
                        ));
                    }
                }
            }
            "VISION" => {
                if domain.data_size < VisionDomainConfigBin::SIZE as u64 {
                    result.errors.push(format!(
                        "VISION domain {} too small for config: {} bytes (need {})",
                        i, domain.data_size, VisionDomainConfigBin::SIZE
                    ));
                    result.valid = false;
                }
            }
            "AUDIO" => {
                if domain.data_size < AudioDomainConfigBin::SIZE as u64 {
                    result.errors.push(format!(
                        "AUDIO domain {} too small for config: {} bytes (need {})",
                        i, domain.data_size, AudioDomainConfigBin::SIZE
                    ));
                    result.valid = false;
                }
            }
            "CODE" => {
                // CODE tiene TextDomainConfigBin + CodeDomainConfigBin
                let min_size = TextDomainConfigBin::SIZE + CodeDomainConfigBin::SIZE;
                if domain.data_size < min_size as u64 {
                    result.errors.push(format!(
                        "CODE domain {} too small for config: {} bytes (need {})",
                        i, domain.data_size, min_size
                    ));
                    result.valid = false;
                }
            }
            _ => {}
        }
    }
}

fn compute_checksum_for_validation(data: &[u8]) -> u64 {
    use xxhash_rust::xxh3::Xxh3;
    
    if data.len() < HTF_HEADER_SIZE {
        return 0;
    }
    
    let mut hasher = Xxh3::new();
    hasher.update(&data[..24]);
    hasher.update(&[0u8; 8]);
    hasher.update(&data[HTF_HEADER_SIZE..]);
    hasher.digest()
}

/// Imprime un resumen de validación
pub fn print_validation_result(result: &HTFValidationResult) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    HTF VALIDATION RESULT                         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    
    if result.valid {
        println!("║  Status: ✓ VALID                                                 ║");
    } else {
        println!("║  Status: ✗ INVALID                                               ║");
    }
    
    println!("║  Magic: {}                                                       ║", result.info.magic);
    println!("║  Version: {}                                                     ║", result.version);
    println!("║  Domains: {}                                                        ║", result.info.num_domains);
    println!("║  Total Size: {} bytes                                       ║", result.info.total_size);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    
    println!("║  DOMAINS:                                                         ║");
    for (i, domain) in result.info.domains.iter().enumerate() {
        let primary = if domain.is_primary { " [PRIMARY]" } else { "" };
        println!("║    [{}] {} - vocab: {}, size: {}{}",
            i, domain.domain_type, domain.vocab_size, domain.data_size, primary);
    }
    
    if !result.errors.is_empty() {
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  ERRORS:                                                         ║");
        for err in &result.errors {
            println!("║    ✗ {}", err);
        }
    }
    
    if !result.warnings.is_empty() {
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  WARNINGS:                                                       ║");
        for warn in &result.warnings {
            println!("║    ⚠ {}", warn);
        }
    }
    
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_empty() {
        let result = validate_htf(&[]);
        assert!(!result.valid);
    }
    
    #[test]
    fn test_validate_bad_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX");
        let result = validate_htf(&data);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("Invalid magic")));
    }
}
