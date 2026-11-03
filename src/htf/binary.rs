// src/htf/binary.rs
// ============================================================================
// HTF v1.3 - Estructuras Binarias
// ============================================================================
//
// SPEC: HTF_v1_3_SPEC.txt
//
// Todas las estructuras están alineadas a 8 bytes para compatibilidad CUDA
// y mmap directo en el Engine.
//
// ============================================================================

use serde_json::Value;
use std::collections::HashMap;

// ============================================================================
// CONSTANTS
// ============================================================================

pub const HTF3_MAGIC: &[u8; 4] = b"HTF3";
pub const HTF3_VERSION: u16 = 0x0130;  // v1.3.0

// EncodingType enum (§4.2)
pub const ENCODING_BPE: u8 = 0;
pub const ENCODING_SENTENCEPIECE: u8 = 1;
pub const ENCODING_WORDPIECE: u8 = 2;
pub const ENCODING_UNIGRAM: u8 = 3;

// TextConfigFlags (§4.3)
pub const FLAG_BYTE_LEVEL: u8 = 0x01;
pub const FLAG_ADD_PREFIX_SPACE: u8 = 0x02;
pub const FLAG_TRIM_OFFSETS: u8 = 0x04;
pub const FLAG_LEGACY_BEHAVIOUR: u8 = 0x08;

// NormalizerType (TextDomainConfigBin byte 27)
pub const NORMALIZER_DEFAULT: u8 = 0;
pub const NORMALIZER_SPACE_TO_METASPACE: u8 = 1;

// ModelBehaviourFlags (TextDomainConfigBin byte 28)
pub const MODEL_FLAG_BYTE_FALLBACK: u8 = 0x01;

// AddedTokenFlags (§4.4)
pub const ADDED_FLAG_SPECIAL: u8 = 0x01;
pub const ADDED_FLAG_LSTRIP: u8 = 0x02;
pub const ADDED_FLAG_RSTRIP: u8 = 0x04;
pub const ADDED_FLAG_SINGLE_WORD: u8 = 0x08;
pub const ADDED_FLAG_NORMALIZED: u8 = 0x10;

// VisionEncoderType (§5)
pub const VISION_CLIP: u32 = 0;
pub const VISION_SIGLIP: u32 = 1;
pub const VISION_VIT: u32 = 2;
pub const VISION_EVA: u32 = 3;
pub const VISION_DINOV2: u32 = 4;

// ProjectorType (§5)
pub const PROJECTOR_LINEAR: u32 = 0;
pub const PROJECTOR_MLP: u32 = 1;
pub const PROJECTOR_RESAMPLER: u32 = 2;

// AudioEncoderType (§6)
pub const AUDIO_WHISPER: u32 = 0;
pub const AUDIO_ENCODEC: u32 = 1;
pub const AUDIO_SEAMLESS: u32 = 2;
pub const AUDIO_WAV2VEC2: u32 = 3;

// ============================================================================
// TEXT DOMAIN CONFIG (32 bytes)
// ============================================================================

/// TextDomainConfigBin - 32 bytes, alineado a 8
/// 
/// Layout:
///   [0:4]   bos_token_id    i32 (-1 si no definido)
///   [4:8]   eos_token_id    i32
///   [8:12]  pad_token_id    i32
///   [12:16] unk_token_id    i32
///   [16:20] vocab_size      u32
///   [20:22] num_added_tokens u16
///   [22]    encoding_type   u8
///   [23]    flags           u8
///   [24]    pre_tokenizer_type u8
///   [25]    decoder_type    u8
///   [26]    behaviour_flags u8
///   [27]    normalizer_type u8
///   [28]    model_flags     u8
///   [29:32] reserved        3 bytes (0x00)
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TextDomainConfigBin {
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub pad_token_id: i32,
    pub unk_token_id: i32,
    pub vocab_size: u32,
    pub num_added_tokens: u16,
    pub encoding_type: u8,
    pub flags: u8,
    pub pre_tokenizer_type: u8, // 0=default, 1=split-space-merged-with-previous
    pub decoder_type: u8,       // 0=default, 1=replace U+2581 with space
    pub behaviour_flags: u8,    // bit0=add_bos_token
    pub normalizer_type: u8,    // 1=replace ASCII space with U+2581
    pub model_flags: u8,        // bit0=byte_fallback
    pub reserved: [u8; 3],
}

impl TextDomainConfigBin {
    pub const SIZE: usize = 32;
    
    pub fn from_config(config: &Value, vocab_size: u32, num_added_tokens: u16) -> Self {
        let bos = config.get("bos_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(-1);
        
        let eos = config.get("eos_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(-1);
        
        let pad = config.get("pad_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(-1);
        
        let unk = config.get("unk_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(-1);
        
        let encoding_type = match config.get("encoding_type").and_then(|v| v.as_str()) {
            Some("bpe") => ENCODING_BPE,
            Some("sentencepiece") => ENCODING_SENTENCEPIECE,
            Some("wordpiece") => ENCODING_WORDPIECE,
            Some("unigram") => ENCODING_UNIGRAM,
            _ => ENCODING_BPE,
        };
        
        let mut flags: u8 = 0;
        if config.get("byte_level").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_BYTE_LEVEL;
        }
        if config.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_ADD_PREFIX_SPACE;
        }

        let pre_tokenizer_type = if config
            .get("pre_tokenizer")
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str) == Some("Split")
            && config
                .get("pre_tokenizer")
                .and_then(|v| v.get("behavior"))
                .and_then(Value::as_str) == Some("MergedWithPrevious")
        { 1 } else { 0 };
        let decoder_type = if config
            .get("decoder")
            .map(|v| v.to_string().contains('▁'))
            .unwrap_or(false)
        { 1 } else { 0 };
        let behaviour_flags = if config.get("add_bos_token").and_then(Value::as_bool).unwrap_or(false) {
            0x01
        } else { 0 };
        let normalizer_type = if config
            .get("normalizer")
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str) == Some("Replace")
            && config
                .get("normalizer")
                .and_then(|v| v.get("pattern"))
                .and_then(|v| v.get("String"))
                .and_then(Value::as_str) == Some(" ")
            && config
                .get("normalizer")
                .and_then(|v| v.get("content"))
                .and_then(Value::as_str) == Some("▁")
        { NORMALIZER_SPACE_TO_METASPACE } else { NORMALIZER_DEFAULT };
        let model_flags = if config.get("byte_fallback").and_then(Value::as_bool).unwrap_or(false) {
            MODEL_FLAG_BYTE_FALLBACK
        } else { 0 };
        
        Self {
            bos_token_id: bos,
            eos_token_id: eos,
            pad_token_id: pad,
            unk_token_id: unk,
            vocab_size,
            num_added_tokens,
            encoding_type,
            flags,
            pre_tokenizer_type,
            decoder_type,
            behaviour_flags,
            normalizer_type,
            model_flags,
            reserved: [0; 3],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.bos_token_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.eos_token_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.pad_token_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.unk_token_id.to_le_bytes());
        buf[16..20].copy_from_slice(&self.vocab_size.to_le_bytes());
        buf[20..22].copy_from_slice(&self.num_added_tokens.to_le_bytes());
        buf[22] = self.encoding_type;
        buf[23] = self.flags;
        buf[24] = self.pre_tokenizer_type;
        buf[25] = self.decoder_type;
        buf[26] = self.behaviour_flags;
        buf[27] = self.normalizer_type;
        buf[28] = self.model_flags;
        // [29:32] reserved
        buf
    }
}

// ============================================================================
// ADDED TOKEN ENTRY (8 + content_len bytes, aligned to 4)
// ============================================================================

/// AddedTokenEntry - variable size
/// 
/// Layout:
///   [0:4]   token_id      u32
///   [4:6]   content_len   u16
///   [6]     flags         u8
///   [7]     reserved      u8
///   [8:8+N] content       N bytes (UTF-8, no null terminator)
///   [...]   padding       0-3 bytes to align to 4
#[derive(Debug, Clone)]
pub struct AddedTokenEntry {
    pub token_id: u32,
    pub content: String,
    pub flags: u8,
}

impl AddedTokenEntry {
    pub fn new(
        token_id: u32,
        content: String,
        special: bool,
        lstrip: bool,
        rstrip: bool,
        single_word: bool,
        normalized: bool,
    ) -> Self {
        let mut flags: u8 = 0;
        if special { flags |= ADDED_FLAG_SPECIAL; }
        if lstrip { flags |= ADDED_FLAG_LSTRIP; }
        if rstrip { flags |= ADDED_FLAG_RSTRIP; }
        if single_word { flags |= ADDED_FLAG_SINGLE_WORD; }
        if normalized { flags |= ADDED_FLAG_NORMALIZED; }
        
        Self { token_id, content, flags }
    }
    
    /// Serializa a bytes con padding a 4 bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let content_bytes = self.content.as_bytes();
        let content_len = content_bytes.len() as u16;
        
        let mut buf = Vec::with_capacity(8 + content_bytes.len() + 4);
        buf.extend_from_slice(&self.token_id.to_le_bytes());
        buf.extend_from_slice(&content_len.to_le_bytes());
        buf.push(self.flags);
        buf.push(0); // reserved
        buf.extend_from_slice(content_bytes);
        
        // Pad to 4 bytes
        let pad = (4 - (buf.len() % 4)) % 4;
        buf.extend(std::iter::repeat(0u8).take(pad));
        
        buf
    }
}

/// Extrae added tokens del config JSON
pub fn extract_added_tokens(config: &Value) -> Vec<AddedTokenEntry> {
    let mut entries = Vec::new();
    
    if let Some(decoder) = config.get("added_tokens_decoder").and_then(|v| v.as_object()) {
        for (id_str, info) in decoder {
            if let Ok(token_id) = id_str.parse::<u32>() {
                let content = info.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                
                if content.is_empty() {
                    continue;
                }
                
                let special = info.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                let lstrip = info.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false);
                let rstrip = info.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false);
                let single_word = info.get("single_word").and_then(|v| v.as_bool()).unwrap_or(false);
                let normalized = info.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false);
                
                entries.push(AddedTokenEntry::new(
                    token_id,
                    content,
                    special,
                    lstrip,
                    rstrip,
                    single_word,
                    normalized,
                ));
            }
        }
    }
    
    // Ordenar por token_id para consistencia
    entries.sort_by_key(|e| e.token_id);
    
    entries
}

// ============================================================================
// VISION DOMAIN CONFIG (64 bytes)
// ============================================================================

/// VisionDomainConfigBin - 64 bytes, alineado a 8
/// 
/// Layout:
///   [0:4]   encoder_type        u32
///   [4:8]   image_size          u32
///   [8:12]  patch_size          u32
///   [12:16] num_channels        u32
///   [16:20] hidden_size         u32
///   [20:24] num_hidden_layers   u32
///   [24:28] num_attention_heads u32
///   [28:32] intermediate_size   u32
///   [32:34] image_mean_r        i16 (×1000)
///   [34:36] image_mean_g        i16
///   [36:38] image_mean_b        i16
///   [38:40] image_std_r         i16
///   [40:42] image_std_g         i16
///   [42:44] image_std_b         i16
///   [44:48] num_image_tokens    u32
///   [48:52] image_token_id      i32
///   [52:56] projection_dim      u32
///   [56:60] projector_type      u32
///   [60:62] flags               u16
///   [62:64] reserved            2 bytes
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct VisionDomainConfigBin {
    pub encoder_type: u32,
    pub image_size: u32,
    pub patch_size: u32,
    pub num_channels: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub intermediate_size: u32,
    pub image_mean_r: i16,
    pub image_mean_g: i16,
    pub image_mean_b: i16,
    pub image_std_r: i16,
    pub image_std_g: i16,
    pub image_std_b: i16,
    pub num_image_tokens: u32,
    pub image_token_id: i32,
    pub projection_dim: u32,
    pub projector_type: u32,
    pub flags: u16,
    pub reserved: [u8; 2],
}

impl VisionDomainConfigBin {
    pub const SIZE: usize = 64;
    
    pub fn from_config(config: &Value) -> Self {
        let encoder_type = match config.get("encoder_type").and_then(|v| v.as_str()) {
            Some("clip") => VISION_CLIP,
            Some("siglip") => VISION_SIGLIP,
            Some("vit") => VISION_VIT,
            Some("eva") => VISION_EVA,
            Some("dinov2") => VISION_DINOV2,
            _ => VISION_CLIP,
        };
        
        let image_size = config.get("image_size").and_then(|v| v.as_u64()).unwrap_or(224) as u32;
        let patch_size = config.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as u32;
        let num_channels = config.get("num_channels").and_then(|v| v.as_u64()).unwrap_or(3) as u32;
        let hidden_size = config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as u32;
        let num_hidden_layers = config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(12) as u32;
        let num_attention_heads = config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(12) as u32;
        let intermediate_size = config.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(3072) as u32;
        
        // Image mean/std como fixed point (×1000)
        let (mean_r, mean_g, mean_b) = extract_image_mean(config);
        let (std_r, std_g, std_b) = extract_image_std(config);
        
        let num_image_tokens = config.get("num_image_tokens").and_then(|v| v.as_u64()).unwrap_or(196) as u32;
        let image_token_id = config.get("image_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32;
        let projection_dim = config.get("projection_dim").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        
        let projector_type = match config.get("projector_type").and_then(|v| v.as_str()) {
            Some("linear") => PROJECTOR_LINEAR,
            Some("mlp") => PROJECTOR_MLP,
            Some("resampler") => PROJECTOR_RESAMPLER,
            _ => PROJECTOR_LINEAR,
        };
        
        Self {
            encoder_type,
            image_size,
            patch_size,
            num_channels,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            image_mean_r: mean_r,
            image_mean_g: mean_g,
            image_mean_b: mean_b,
            image_std_r: std_r,
            image_std_g: std_g,
            image_std_b: std_b,
            num_image_tokens,
            image_token_id,
            projection_dim,
            projector_type,
            flags: 0,
            reserved: [0; 2],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.encoder_type.to_le_bytes());
        buf[4..8].copy_from_slice(&self.image_size.to_le_bytes());
        buf[8..12].copy_from_slice(&self.patch_size.to_le_bytes());
        buf[12..16].copy_from_slice(&self.num_channels.to_le_bytes());
        buf[16..20].copy_from_slice(&self.hidden_size.to_le_bytes());
        buf[20..24].copy_from_slice(&self.num_hidden_layers.to_le_bytes());
        buf[24..28].copy_from_slice(&self.num_attention_heads.to_le_bytes());
        buf[28..32].copy_from_slice(&self.intermediate_size.to_le_bytes());
        buf[32..34].copy_from_slice(&self.image_mean_r.to_le_bytes());
        buf[34..36].copy_from_slice(&self.image_mean_g.to_le_bytes());
        buf[36..38].copy_from_slice(&self.image_mean_b.to_le_bytes());
        buf[38..40].copy_from_slice(&self.image_std_r.to_le_bytes());
        buf[40..42].copy_from_slice(&self.image_std_g.to_le_bytes());
        buf[42..44].copy_from_slice(&self.image_std_b.to_le_bytes());
        buf[44..48].copy_from_slice(&self.num_image_tokens.to_le_bytes());
        buf[48..52].copy_from_slice(&self.image_token_id.to_le_bytes());
        buf[52..56].copy_from_slice(&self.projection_dim.to_le_bytes());
        buf[56..60].copy_from_slice(&self.projector_type.to_le_bytes());
        buf[60..62].copy_from_slice(&self.flags.to_le_bytes());
        // [62:64] already zeros
        buf
    }
}

fn extract_image_mean(config: &Value) -> (i16, i16, i16) {
    if let Some(arr) = config.get("image_mean").and_then(|v| v.as_array()) {
        let r = arr.get(0).and_then(|v| v.as_f64()).unwrap_or(0.5);
        let g = arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.5);
        let b = arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.5);
        ((r * 1000.0) as i16, (g * 1000.0) as i16, (b * 1000.0) as i16)
    } else {
        (500, 500, 500) // Default 0.5
    }
}

fn extract_image_std(config: &Value) -> (i16, i16, i16) {
    if let Some(arr) = config.get("image_std").and_then(|v| v.as_array()) {
        let r = arr.get(0).and_then(|v| v.as_f64()).unwrap_or(0.5);
        let g = arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.5);
        let b = arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.5);
        ((r * 1000.0) as i16, (g * 1000.0) as i16, (b * 1000.0) as i16)
    } else {
        (500, 500, 500) // Default 0.5
    }
}

// ============================================================================
// AUDIO DOMAIN CONFIG (64 bytes)
// ============================================================================

/// AudioDomainConfigBin - 64 bytes, alineado a 8
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioDomainConfigBin {
    pub encoder_type: u32,
    pub sample_rate: u32,
    pub n_mels: u32,
    pub n_fft: u32,
    pub hop_length: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub chunk_length: u32,
    pub codebook_size: u32,
    pub codebook_dim: u32,
    pub num_codebooks: u16,
    pub reserved1: u16,
    pub audio_token_id: i32,
    pub sot_token_id: i32,
    pub eot_token_id: i32,
    pub flags: u16,
    pub reserved2: [u8; 2],
}

impl AudioDomainConfigBin {
    pub const SIZE: usize = 64;
    
    pub fn from_config(config: &Value) -> Self {
        let encoder_type = match config.get("encoder_type").and_then(|v| v.as_str()) {
            Some("whisper") => AUDIO_WHISPER,
            Some("encodec") => AUDIO_ENCODEC,
            Some("seamless") => AUDIO_SEAMLESS,
            Some("wav2vec2") => AUDIO_WAV2VEC2,
            _ => AUDIO_WHISPER,
        };
        
        Self {
            encoder_type,
            sample_rate: config.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(16000) as u32,
            n_mels: config.get("n_mels").and_then(|v| v.as_u64()).unwrap_or(128) as u32,
            n_fft: config.get("n_fft").and_then(|v| v.as_u64()).unwrap_or(400) as u32,
            hop_length: config.get("hop_length").and_then(|v| v.as_u64()).unwrap_or(160) as u32,
            hidden_size: config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1280) as u32,
            num_hidden_layers: config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(32) as u32,
            num_attention_heads: config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(20) as u32,
            chunk_length: config.get("chunk_length").and_then(|v| v.as_u64()).unwrap_or(30) as u32,
            codebook_size: config.get("codebook_size").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            codebook_dim: config.get("codebook_dim").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            num_codebooks: config.get("num_codebooks").and_then(|v| v.as_u64()).unwrap_or(0) as u16,
            reserved1: 0,
            audio_token_id: config.get("audio_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            sot_token_id: config.get("sot_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            eot_token_id: config.get("eot_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            flags: 0,
            reserved2: [0; 2],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.encoder_type.to_le_bytes());
        buf[4..8].copy_from_slice(&self.sample_rate.to_le_bytes());
        buf[8..12].copy_from_slice(&self.n_mels.to_le_bytes());
        buf[12..16].copy_from_slice(&self.n_fft.to_le_bytes());
        buf[16..20].copy_from_slice(&self.hop_length.to_le_bytes());
        buf[20..24].copy_from_slice(&self.hidden_size.to_le_bytes());
        buf[24..28].copy_from_slice(&self.num_hidden_layers.to_le_bytes());
        buf[28..32].copy_from_slice(&self.num_attention_heads.to_le_bytes());
        buf[32..36].copy_from_slice(&self.chunk_length.to_le_bytes());
        buf[36..40].copy_from_slice(&self.codebook_size.to_le_bytes());
        buf[40..44].copy_from_slice(&self.codebook_dim.to_le_bytes());
        buf[44..46].copy_from_slice(&self.num_codebooks.to_le_bytes());
        buf[46..48].copy_from_slice(&self.reserved1.to_le_bytes());
        buf[48..52].copy_from_slice(&self.audio_token_id.to_le_bytes());
        buf[52..56].copy_from_slice(&self.sot_token_id.to_le_bytes());
        buf[56..60].copy_from_slice(&self.eot_token_id.to_le_bytes());
        buf[60..62].copy_from_slice(&self.flags.to_le_bytes());
        // [62:64] already zeros
        buf
    }
}

// ============================================================================
// CODE DOMAIN CONFIG (32 bytes)
// ============================================================================

/// CodeDomainConfigBin - 32 bytes, alineado a 8
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CodeDomainConfigBin {
    pub base_domain_index: u32,
    pub fim_prefix_token_id: i32,
    pub fim_middle_token_id: i32,
    pub fim_suffix_token_id: i32,
    pub fim_pad_token_id: i32,
    pub indent_2spaces_id: i16,
    pub indent_4spaces_id: i16,
    pub indent_tab_id: i16,
    pub flags: u16,
    pub reserved: [u8; 4],
}

impl CodeDomainConfigBin {
    pub const SIZE: usize = 32;
    
    pub fn from_config(config: &Value) -> Self {
        Self {
            base_domain_index: config.get("base_domain_index").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            fim_prefix_token_id: config.get("fim_prefix_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            fim_middle_token_id: config.get("fim_middle_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            fim_suffix_token_id: config.get("fim_suffix_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            fim_pad_token_id: config.get("fim_pad_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            indent_2spaces_id: config.get("indent_2spaces_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i16,
            indent_4spaces_id: config.get("indent_4spaces_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i16,
            indent_tab_id: config.get("indent_tab_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i16,
            flags: if config.get("fim_enabled").and_then(|v| v.as_bool()).unwrap_or(false) { 0x0001 } else { 0 },
            reserved: [0; 4],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.base_domain_index.to_le_bytes());
        buf[4..8].copy_from_slice(&self.fim_prefix_token_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.fim_middle_token_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.fim_suffix_token_id.to_le_bytes());
        buf[16..20].copy_from_slice(&self.fim_pad_token_id.to_le_bytes());
        buf[20..22].copy_from_slice(&self.indent_2spaces_id.to_le_bytes());
        buf[22..24].copy_from_slice(&self.indent_4spaces_id.to_le_bytes());
        buf[24..26].copy_from_slice(&self.indent_tab_id.to_le_bytes());
        buf[26..28].copy_from_slice(&self.flags.to_le_bytes());
        // [28:32] already zeros
        buf
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_config_size() {
        assert_eq!(std::mem::size_of::<TextDomainConfigBin>(), 32);
        assert_eq!(TextDomainConfigBin::SIZE, 32);
    }
    
    #[test]
    fn test_vision_config_size() {
        assert_eq!(std::mem::size_of::<VisionDomainConfigBin>(), 64);
        assert_eq!(VisionDomainConfigBin::SIZE, 64);
    }
    
    #[test]
    fn test_audio_config_size() {
        assert_eq!(std::mem::size_of::<AudioDomainConfigBin>(), 64);
        assert_eq!(AudioDomainConfigBin::SIZE, 64);
    }
    
    #[test]
    fn test_code_config_size() {
        assert_eq!(std::mem::size_of::<CodeDomainConfigBin>(), 32);
        assert_eq!(CodeDomainConfigBin::SIZE, 32);
    }
    
    #[test]
    fn test_added_token_serialization() {
        let token = AddedTokenEntry::new(
            151643,
            "<|endoftext|>".to_string(),
            true,
            false,
            false,
            false,
            false,
        );
        let bytes = token.to_bytes();
        
        // 4 (id) + 2 (len) + 1 (flags) + 1 (reserved) + 13 (content) = 21
        // Padded to 24 (múltiplo de 4)
        assert_eq!(bytes.len(), 24);
        
        // Verify token_id
        let id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(id, 151643);
        
        // Verify content_len
        let len = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(len, 13);
        
        // Verify flags
        assert_eq!(bytes[6], ADDED_FLAG_SPECIAL);
    }
}
