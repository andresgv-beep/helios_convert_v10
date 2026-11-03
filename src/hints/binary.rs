// src/hints/binary.rs
// ============================================================================
// EXECUTION HINTS BINARY - Bloque [0xB] para HNFv9.1
// ============================================================================
//
// SPEC: PROPOSAL_EXECUTION_HINTS_BINARY.docx
//
// Formato binario paralelo al JSON [0xA] para parsing O(1).
// Engine prefiere [0xB] si existe, fallback a [0xA].
//
// ============================================================================

use anyhow::{Context, Result};
use serde_json::Value;

// ============================================================================
// CONSTANTS
// ============================================================================

pub const HINTS_MAGIC: u32 = 0x48494E54;  // "HINT"
pub const HINTS_VERSION_MAJOR: u16 = 1;
pub const HINTS_VERSION_MINOR: u16 = 1;

// Arch enum
pub const ARCH_UNKNOWN: u32 = 0;
pub const ARCH_LLAMA: u32 = 1;
pub const ARCH_LLAMA2: u32 = 2;
pub const ARCH_LLAMA3: u32 = 3;
pub const ARCH_QWEN: u32 = 4;
pub const ARCH_QWEN2: u32 = 5;
pub const ARCH_PHI3: u32 = 6;
pub const ARCH_PHI4: u32 = 7;
pub const ARCH_GEMMA: u32 = 8;
pub const ARCH_GEMMA2: u32 = 9;
pub const ARCH_MISTRAL: u32 = 10;
pub const ARCH_MIXTRAL: u32 = 11;
pub const ARCH_DEEPSEEK: u32 = 12;
pub const ARCH_CLIP: u32 = 13;
pub const ARCH_SIGLIP: u32 = 14;
pub const ARCH_GEMMA4: u32 = 15;

// DType enum
pub const DTYPE_FP16: u32 = 0;
pub const DTYPE_BF16: u32 = 1;
pub const DTYPE_FP32: u32 = 2;

// AttentionType enum
pub const ATTN_MHA: u32 = 0;
pub const ATTN_GQA: u32 = 1;
pub const ATTN_MQA: u32 = 2;

// QKVLayout enum
pub const QKV_SEPARATE: u32 = 0;
pub const QKV_FUSED: u32 = 1;

// MLPType enum
pub const MLP_SWIGLU: u32 = 0;
pub const MLP_SWIGLU_FUSED: u32 = 1;
pub const MLP_GEGLU: u32 = 2;
pub const MLP_GATED: u32 = 3;
pub const MLP_STANDARD: u32 = 4;

// Activation enum
pub const ACT_SILU: u32 = 0;
pub const ACT_GELU: u32 = 1;
pub const ACT_GELU_NEW: u32 = 2;
pub const ACT_RELU: u32 = 3;

// NormType enum
pub const NORM_RMSNORM: u32 = 0;
pub const NORM_LAYERNORM: u32 = 1;

// RoPEType enum
pub const ROPE_DEFAULT: u32 = 0;
pub const ROPE_LLAMA3: u32 = 1;
pub const ROPE_LINEAR: u32 = 2;
pub const ROPE_DYNAMIC: u32 = 3;
pub const ROPE_YARN: u32 = 4;
pub const ROPE_LONGROPE: u32 = 5;
pub const ROPE_SU: u32 = 6;
pub const ROPE_NONE: u32 = 7;
pub const ROPE_PROPORTIONAL: u32 = 8;

// Flags for TextModelConfigBin
pub const FLAG_ATTENTION_BIAS: u32 = 0x0001;
pub const FLAG_MLP_BIAS: u32 = 0x0002;
pub const FLAG_NORM_BIAS: u32 = 0x0004;
pub const FLAG_USE_QK_NORM: u32 = 0x0008;
pub const FLAG_PARALLEL_ATTENTION: u32 = 0x0010;
pub const FLAG_TIE_WORD_EMBEDDINGS: u32 = 0x0020;
pub const FLAG_ROPE_PARTIAL: u32 = 0x0040;

// ============================================================================
// HEADER (64 bytes)
// ============================================================================

/// ExecutionHintsBin header - 64 bytes, alineado a 8
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutionHintsBin {
    // Magic + Version (8 bytes)
    pub magic: u32,                     // 0x48494E54 = "HINT"
    pub version_major: u16,
    pub version_minor: u16,
    
    // Offsets to model configs (24 bytes)
    pub text_offset: u32,               // Offset a TextModelConfigBin
    pub vision_offset: u32,             // Offset a VisionModelConfigBin (0 si no hay)
    pub audio_offset: u32,              // Offset a AudioModelConfigBin (0 si no hay)
    pub code_offset: u32,               // Offset a CodeModelConfigBin (0 si no hay)
    pub cortex_offset: u32,             // Offset a CortexModelConfigBin (0 si no hay)
    pub spatial_offset: u32,            // Offset a SpatialModelConfigBin (0 si no hay)
    
    // Counts (8 bytes)
    pub num_text_models: u16,
    pub num_vision_models: u16,
    pub num_audio_models: u16,
    pub num_code_models: u16,
    
    // Global flags (4 bytes)
    pub flags: u32,
    // bit 0: text_enabled
    // bit 1: vision_enabled
    // bit 2: audio_enabled
    // bit 3: code_enabled
    // bit 4: cortex_enabled
    
    // Reserved (20 bytes)
    pub reserved: [u8; 20],
}

impl ExecutionHintsBin {
    pub const SIZE: usize = 64;
    
    pub fn new() -> Self {
        Self {
            magic: HINTS_MAGIC,
            version_major: HINTS_VERSION_MAJOR,
            version_minor: HINTS_VERSION_MINOR,
            ..Default::default()
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        buf[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        buf[8..12].copy_from_slice(&self.text_offset.to_le_bytes());
        buf[12..16].copy_from_slice(&self.vision_offset.to_le_bytes());
        buf[16..20].copy_from_slice(&self.audio_offset.to_le_bytes());
        buf[20..24].copy_from_slice(&self.code_offset.to_le_bytes());
        buf[24..28].copy_from_slice(&self.cortex_offset.to_le_bytes());
        buf[28..32].copy_from_slice(&self.spatial_offset.to_le_bytes());
        buf[32..34].copy_from_slice(&self.num_text_models.to_le_bytes());
        buf[34..36].copy_from_slice(&self.num_vision_models.to_le_bytes());
        buf[36..38].copy_from_slice(&self.num_audio_models.to_le_bytes());
        buf[38..40].copy_from_slice(&self.num_code_models.to_le_bytes());
        buf[40..44].copy_from_slice(&self.flags.to_le_bytes());
        buf[44..64].copy_from_slice(&self.reserved);
        buf
    }
}

// ============================================================================
// TEXT MODEL CONFIG (128 bytes)
// ============================================================================

/// TextModelConfigBin - 128 bytes, alineado a 8
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TextModelConfigBin {
    // Floats primero (24 bytes)
    pub rope_theta: f32,
    pub rope_scaling_factor: f32,
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub reserved_float: f32,
    
    // Dimensions (24 bytes)
    pub num_hidden_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rope_dim: u32,
    
    // Attention (20 bytes)
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub attention_type: u32,            // Enum: MHA=0, GQA=1, MQA=2
    pub qkv_layout: u32,                // Enum: SEPARATE=0, FUSED=1
    
    // Identity + Types (16 bytes)
    pub arch: u32,                      // Enum: LLAMA=1, QWEN2=5, etc.
    pub dtype: u32,                     // Enum: FP16=0, BF16=1, FP32=2
    pub mlp_type: u32,                  // Enum: SWIGLU=0, GEGLU=2, etc.
    pub mlp_activation: u32,            // Enum: SILU=0, GELU=1, etc.
    
    // More types (8 bytes)
    pub norm_type: u32,                 // Enum: RMSNORM=0, LAYERNORM=1
    pub rope_type: u32,                 // Enum: DEFAULT=0, LLAMA3=1, etc.
    
    // Flags (4 bytes)
    pub flags: u32,
    // bit 0: attention_bias
    // bit 1: mlp_bias
    // bit 2: norm_bias
    // bit 3: use_qk_norm
    // bit 4: parallel_attention
    // bit 5: tie_word_embeddings
    // bit 6: rope_partial
    
    // Reserved (28 bytes)
    pub reserved: [u8; 28],
}

impl TextModelConfigBin {
    pub const SIZE: usize = 128;
    
    pub fn from_json(config: &Value) -> Self {
        let arch_str = config.get("arch").and_then(|v| v.as_str()).unwrap_or("unknown");
        let arch = match arch_str {
            "llama" => ARCH_LLAMA,
            "llama2" => ARCH_LLAMA2,
            "llama3" | "llama3.1" | "llama3.2" => ARCH_LLAMA3,
            "qwen" => ARCH_QWEN,
            "qwen2" | "qwen2.5" => ARCH_QWEN2,
            "phi3" => ARCH_PHI3,
            "phi4" | "phi" => ARCH_PHI4,
            "gemma" => ARCH_GEMMA,
            "gemma2" => ARCH_GEMMA2,
            "gemma4" => ARCH_GEMMA4,
            "mistral" => ARCH_MISTRAL,
            "mixtral" => ARCH_MIXTRAL,
            "deepseek" | "deepseek2" => ARCH_DEEPSEEK,
            _ => ARCH_UNKNOWN,
        };
        
        let dtype = match config.get("dtype").and_then(|v| v.as_str()).unwrap_or("bf16") {
            "fp16" => DTYPE_FP16,
            "bf16" => DTYPE_BF16,
            "fp32" => DTYPE_FP32,
            _ => DTYPE_BF16,
        };
        
        let attention_type = match config.get("attention_type").and_then(|v| v.as_str()).unwrap_or("gqa") {
            "mha" => ATTN_MHA,
            "gqa" => ATTN_GQA,
            "mqa" => ATTN_MQA,
            _ => ATTN_GQA,
        };
        
        let qkv_layout = match config.get("qkv_layout").and_then(|v| v.as_str()).unwrap_or("separate") {
            "fused" => QKV_FUSED,
            _ => QKV_SEPARATE,
        };
        
        let mlp_type = match config.get("mlp_type").and_then(|v| v.as_str()).unwrap_or("swiglu") {
            "swiglu" => MLP_SWIGLU,
            "swiglu_fused" => MLP_SWIGLU_FUSED,
            "geglu" => MLP_GEGLU,
            "gated" => MLP_GATED,
            "standard" => MLP_STANDARD,
            _ => MLP_SWIGLU,
        };
        
        let mlp_activation = match config.get("mlp_activation").and_then(|v| v.as_str()).unwrap_or("silu") {
            "silu" => ACT_SILU,
            "gelu" => ACT_GELU,
            "gelu_new" => ACT_GELU_NEW,
            "relu" => ACT_RELU,
            _ => ACT_SILU,
        };
        
        let norm_type = match config.get("norm_type").and_then(|v| v.as_str()).unwrap_or("rmsnorm") {
            "layernorm" => NORM_LAYERNORM,
            _ => NORM_RMSNORM,
        };
        
        let rope_type = match config.get("rope_type").and_then(|v| v.as_str()).unwrap_or("default") {
            "llama3" => ROPE_LLAMA3,
            "linear" => ROPE_LINEAR,
            "dynamic" => ROPE_DYNAMIC,
            "yarn" => ROPE_YARN,
            "longrope" => ROPE_LONGROPE,
            "su" => ROPE_SU,
            "none" => ROPE_NONE,
            "proportional" => ROPE_PROPORTIONAL,
            _ => ROPE_DEFAULT,
        };
        
        // Build flags
        let mut flags: u32 = 0;
        if config.get("attention_bias").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_ATTENTION_BIAS;
        }
        if config.get("mlp_bias").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_MLP_BIAS;
        }
        if config.get("norm_bias").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_NORM_BIAS;
        }
        if config.get("use_qk_norm").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_USE_QK_NORM;
        }
        if config.get("parallel_attention").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_PARALLEL_ATTENTION;
        }
        if config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false) {
            flags |= FLAG_TIE_WORD_EMBEDDINGS;
        }
        
        Self {
            rope_theta: config.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32,
            rope_scaling_factor: config.get("rope_scaling_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            partial_rotary_factor: config.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            rms_norm_eps: config.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
            layer_norm_eps: config.get("layer_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
            reserved_float: 0.0,
            
            num_hidden_layers: config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(32) as u32,
            hidden_size: config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(4096) as u32,
            intermediate_size: config.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(11008) as u32,
            vocab_size: config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(32000) as u32,
            max_position_embeddings: config.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(4096) as u32,
            rope_dim: config.get("rope_dim").and_then(|v| v.as_u64())
                .or_else(|| config.get("head_dim").and_then(|v| v.as_u64()))
                .unwrap_or(128) as u32,
            
            num_attention_heads: config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(32) as u32,
            num_key_value_heads: config.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(8) as u32,
            head_dim: config.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as u32,
            attention_type,
            qkv_layout,
            
            arch,
            dtype,
            mlp_type,
            mlp_activation,
            
            norm_type,
            rope_type,
            
            flags,
            reserved: [0; 28],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        
        // Floats (24 bytes)
        buf[0..4].copy_from_slice(&self.rope_theta.to_le_bytes());
        buf[4..8].copy_from_slice(&self.rope_scaling_factor.to_le_bytes());
        buf[8..12].copy_from_slice(&self.partial_rotary_factor.to_le_bytes());
        buf[12..16].copy_from_slice(&self.rms_norm_eps.to_le_bytes());
        buf[16..20].copy_from_slice(&self.layer_norm_eps.to_le_bytes());
        buf[20..24].copy_from_slice(&self.reserved_float.to_le_bytes());
        
        // Dimensions (24 bytes)
        buf[24..28].copy_from_slice(&self.num_hidden_layers.to_le_bytes());
        buf[28..32].copy_from_slice(&self.hidden_size.to_le_bytes());
        buf[32..36].copy_from_slice(&self.intermediate_size.to_le_bytes());
        buf[36..40].copy_from_slice(&self.vocab_size.to_le_bytes());
        buf[40..44].copy_from_slice(&self.max_position_embeddings.to_le_bytes());
        buf[44..48].copy_from_slice(&self.rope_dim.to_le_bytes());
        
        // Attention (20 bytes)
        buf[48..52].copy_from_slice(&self.num_attention_heads.to_le_bytes());
        buf[52..56].copy_from_slice(&self.num_key_value_heads.to_le_bytes());
        buf[56..60].copy_from_slice(&self.head_dim.to_le_bytes());
        buf[60..64].copy_from_slice(&self.attention_type.to_le_bytes());
        buf[64..68].copy_from_slice(&self.qkv_layout.to_le_bytes());
        
        // Identity + Types (16 bytes)
        buf[68..72].copy_from_slice(&self.arch.to_le_bytes());
        buf[72..76].copy_from_slice(&self.dtype.to_le_bytes());
        buf[76..80].copy_from_slice(&self.mlp_type.to_le_bytes());
        buf[80..84].copy_from_slice(&self.mlp_activation.to_le_bytes());
        
        // More types (8 bytes)
        buf[84..88].copy_from_slice(&self.norm_type.to_le_bytes());
        buf[88..92].copy_from_slice(&self.rope_type.to_le_bytes());
        
        // Flags (4 bytes)
        buf[92..96].copy_from_slice(&self.flags.to_le_bytes());
        
        // Reserved [96..128] already zeros
        buf
    }
}

// ============================================================================
// VISION MODEL CONFIG (64 bytes)
// ============================================================================

/// VisionModelConfigBin - 64 bytes
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct VisionModelConfigBin {
    pub encoder_type: u32,              // CLIP=0, SigLIP=1, ViT=2, EVA=3
    pub image_size: u32,
    pub patch_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub intermediate_size: u32,
    pub num_channels: u32,
    pub layer_norm_eps: f32,
    pub projection_dim: u32,
    pub projector_type: u32,            // LINEAR=0, MLP=1, RESAMPLER=2
    pub num_image_tokens: u32,
    pub image_token_id: i32,
    pub flags: u32,
    pub reserved: [u8; 8],
}

impl VisionModelConfigBin {
    pub const SIZE: usize = 64;
    
    pub fn from_json(config: &Value) -> Self {
        let encoder_type = match config.get("encoder_type").and_then(|v| v.as_str()).unwrap_or("clip") {
            "siglip" => 1,
            "vit" => 2,
            "eva" => 3,
            "gemma4" | "gemma4_vision" => 4,
            _ => 0,
        };
        
        Self {
            encoder_type,
            image_size: config.get("image_size").and_then(|v| v.as_u64()).unwrap_or(224) as u32,
            patch_size: config.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as u32,
            hidden_size: config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as u32,
            num_hidden_layers: config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(12) as u32,
            num_attention_heads: config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(12) as u32,
            intermediate_size: config.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(3072) as u32,
            num_channels: config.get("num_channels").and_then(|v| v.as_u64()).unwrap_or(3) as u32,
            layer_norm_eps: config.get("layer_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
            projection_dim: config.get("projection_dim").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            projector_type: match config.get("projector_type").and_then(|v| v.as_str()).unwrap_or("linear") {
                "mlp" => 1,
                "resampler" => 2,
                _ => 0,
            },
            num_image_tokens: config.get("num_image_tokens").and_then(|v| v.as_u64()).unwrap_or(196) as u32,
            image_token_id: config.get("image_token_id").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
            flags: 0,
            reserved: [0; 8],
        }
    }
    
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.encoder_type.to_le_bytes());
        buf[4..8].copy_from_slice(&self.image_size.to_le_bytes());
        buf[8..12].copy_from_slice(&self.patch_size.to_le_bytes());
        buf[12..16].copy_from_slice(&self.hidden_size.to_le_bytes());
        buf[16..20].copy_from_slice(&self.num_hidden_layers.to_le_bytes());
        buf[20..24].copy_from_slice(&self.num_attention_heads.to_le_bytes());
        buf[24..28].copy_from_slice(&self.intermediate_size.to_le_bytes());
        buf[28..32].copy_from_slice(&self.num_channels.to_le_bytes());
        buf[32..36].copy_from_slice(&self.layer_norm_eps.to_le_bytes());
        buf[36..40].copy_from_slice(&self.projection_dim.to_le_bytes());
        buf[40..44].copy_from_slice(&self.projector_type.to_le_bytes());
        buf[44..48].copy_from_slice(&self.num_image_tokens.to_le_bytes());
        buf[48..52].copy_from_slice(&self.image_token_id.to_le_bytes());
        buf[52..56].copy_from_slice(&self.flags.to_le_bytes());
        // [56..64] reserved
        buf
    }
}

// ============================================================================
// BUILD BINARY HINTS
// ============================================================================

/// Construye bloque binario [0xB] desde JSON de execution_hints
pub fn build_execution_hints_binary(hints_json: &Value) -> Result<Vec<u8>> {
    // Legacy callers pass a bare text config. Combined multimodal hints always
    // wrap each modality under its own key.
    let text_json = hints_json.get("text").or_else(|| {
        if hints_json.get("vision").is_none() { Some(hints_json) } else { None }
    });
    let vision_json = hints_json.get("vision");
    if text_json.is_none() && vision_json.is_none() {
        anyhow::bail!("execution hints contain neither text nor vision config");
    }

    anyhow::ensure!(!text_json.map(|t| t["arch"] == "gemma4" && t["moe_enabled"] == true).unwrap_or(false),
        "Gemma 4 MoE binary execution contract is not implemented; refusing dense GM4X serialization");
    let gemma4_extension = text_json.map(build_gemma4_extension).unwrap_or_default();
    let gemma4_vision_extension = match vision_json {
        Some(vision) => build_gemma4_vision_extension(vision)?,
        None => Vec::new(),
    };

    // Reserve the header and append each present config. Offsets derive from
    // the real buffer so vision-only HNF cannot acquire a phantom text config.
    let mut buf = vec![0u8; ExecutionHintsBin::SIZE];
    let mut header = ExecutionHintsBin::new();

    if let Some(text) = text_json {
        header.num_text_models = 1;
        header.flags |= 0x0001;
        header.text_offset = buf.len() as u32;
        buf.extend_from_slice(&TextModelConfigBin::from_json(text).to_bytes());
        if !gemma4_extension.is_empty() {
            let extension_offset = buf.len() as u32;
            header.reserved[0..4].copy_from_slice(&extension_offset.to_le_bytes());
            header.reserved[4..8].copy_from_slice(&(gemma4_extension.len() as u32).to_le_bytes());
            header.reserved[8..12].copy_from_slice(b"GM4X");
            buf.extend_from_slice(&gemma4_extension);
        }
    }

    if let Some(vision) = vision_json {
        header.num_vision_models = 1;
        header.flags |= 0x0002;
        header.vision_offset = buf.len() as u32;
        buf.extend_from_slice(&VisionModelConfigBin::from_json(vision).to_bytes());
        if !gemma4_vision_extension.is_empty() {
            let extension_offset = buf.len() as u32;
            header.reserved[12..16].copy_from_slice(&extension_offset.to_le_bytes());
            header.reserved[16..20].copy_from_slice(&(gemma4_vision_extension.len() as u32).to_le_bytes());
            buf.extend_from_slice(&gemma4_vision_extension);
        }
    }

    buf[0..ExecutionHintsBin::SIZE].copy_from_slice(&header.to_bytes());
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.extend(std::iter::repeat(0u8).take(pad));
    Ok(buf)
}

/// Gemma 4 vision extension, fixed 96-byte record.
///
/// The parent header uses the eight bytes still free after the GM4X pointer:
/// reserved[12..16] = offset and reserved[16..20] = length. The extension
/// carries its own GM4V magic, so legacy readers can ignore it safely.
fn build_gemma4_vision_extension(vision: &Value) -> Result<Vec<u8>> {
    if vision.get("encoder_type").and_then(Value::as_str) != Some("gemma4") {
        return Ok(Vec::new());
    }

    fn u32_field(value: &Value, key: &str) -> Result<u32> {
        Ok(value.get(key).and_then(Value::as_u64)
            .with_context(|| format!("Gemma 4 vision hints are missing integer '{key}'"))? as u32)
    }
    fn i32_field(value: &Value, key: &str) -> Result<i32> {
        Ok(value.get(key).and_then(Value::as_i64)
            .with_context(|| format!("Gemma 4 vision hints are missing integer '{key}'"))? as i32)
    }
    fn f32_field(value: &Value, key: &str) -> Result<f32> {
        Ok(value.get(key).and_then(Value::as_f64)
            .with_context(|| format!("Gemma 4 vision hints are missing number '{key}'"))? as f32)
    }
    fn bool_field(value: &Value, key: &str) -> Result<bool> {
        value.get(key).and_then(Value::as_bool)
            .with_context(|| format!("Gemma 4 vision hints are missing bool '{key}'"))
    }

    let preprocess = vision.get("preprocess")
        .context("Gemma 4 vision hints are missing preprocess")?;
    let activation = match vision.get("hidden_activation").and_then(Value::as_str)
        .context("Gemma 4 vision hints are missing hidden_activation")? {
        "gelu_pytorch_tanh" => 1u32,
        other => anyhow::bail!("unsupported Gemma 4 vision activation '{other}'"),
    };
    if preprocess.get("patch_order").and_then(Value::as_str) != Some("hwc") {
        anyhow::bail!("Gemma 4 vision requires HWC patch order");
    }
    if preprocess.get("position_order").and_then(Value::as_str) != Some("xy") {
        anyhow::bail!("Gemma 4 vision requires XY position order");
    }

    let mut flags = 0u32;
    flags |= (bool_field(vision, "use_clipped_linears")? as u32) << 0;
    flags |= (bool_field(vision, "standardize")? as u32) << 1;
    flags |= (bool_field(vision, "dynamic_image_tokens")? as u32) << 2;
    flags |= 1 << 3; // learned XY position table
    flags |= 1 << 4; // 2D RoPE
    flags |= 1 << 5; // bidirectional / non-causal attention
    flags |= (bool_field(preprocess, "do_convert_rgb")? as u32) << 6;
    flags |= (bool_field(preprocess, "do_rescale")? as u32) << 7;
    flags |= (bool_field(preprocess, "do_resize")? as u32) << 8;
    flags |= (bool_field(preprocess, "do_normalize")? as u32) << 9;
    flags |= (bool_field(preprocess, "antialias")? as u32) << 10;
    flags |= (bool_field(vision, "qk_norm_weighted")? as u32) << 11;
    flags |= (!bool_field(vision, "v_norm_weighted")? as u32) << 12;
    flags |= (!bool_field(vision, "pool_norm_weighted")? as u32) << 13;
    flags |= 1 << 14; // pool scale sqrt(hidden_size), accumulated in FP32
    flags |= 1 << 15; // model input transform is 2*x-1

    let mut out = Vec::with_capacity(96);
    out.extend_from_slice(b"GM4V");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&96u16.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "num_key_value_heads")?.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "head_dim")?.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "position_embedding_size")?.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "pooling_kernel_size")?.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "max_soft_tokens")?.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "projection_dim")?.to_le_bytes());
    out.extend_from_slice(&u32_field(vision, "patch_size")?.to_le_bytes());
    out.extend_from_slice(&i32_field(vision, "image_token_id")?.to_le_bytes());
    out.extend_from_slice(&i32_field(vision, "boi_token_id")?.to_le_bytes());
    out.extend_from_slice(&i32_field(vision, "eoi_token_id")?.to_le_bytes());
    out.extend_from_slice(&i32_field(vision, "pad_token_id")?.to_le_bytes());
    out.extend_from_slice(&f32_field(vision, "rope_theta")?.to_le_bytes());
    out.extend_from_slice(&f32_field(vision, "attention_scale")?.to_le_bytes());
    out.extend_from_slice(&f32_field(vision, "rms_norm_eps")?.to_le_bytes());
    out.extend_from_slice(&f32_field(preprocess, "rescale_factor")?.to_le_bytes());
    out.extend_from_slice(&u32_field(preprocess, "resize_multiple")?.to_le_bytes());
    out.extend_from_slice(&u32_field(preprocess, "resample")?.to_le_bytes());
    out.extend_from_slice(&activation.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // HWC patch order
    out.extend_from_slice(&1u32.to_le_bytes()); // XY position order
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), 96);
    Ok(out)
}

/// Bit 0 de `Gemma4LayerConfigBin.flags`: la capa reutiliza K como V.
pub const GEMMA4_LAYER_FLAG_K_EQ_V: u32 = 1 << 0;

/// Gemma4 binary extension.  Header is 32 bytes followed by 40-byte records:
/// [kind, window, head_dim, intermediate, rope_type, flags,
///  rope_theta:f32, partial_rotary:f32, kv_share_group:i32, reserved].
pub(crate) fn build_gemma4_extension(text: &Value) -> Vec<u8> {
    if text.get("arch").and_then(Value::as_str) != Some("gemma4") {
        return Vec::new();
    }
    let Some(layers) = text.get("layer_configs").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(32 + layers.len() * 40);
    out.extend_from_slice(b"GM4X");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&40u16.to_le_bytes());
    out.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    let mut flags = 0u32;
    if text.get("per_layer_embeddings").and_then(|v| v.get("enabled")).and_then(Value::as_bool).unwrap_or(false) { flags |= 1 << 0; }
    if text.get("layer_scalar").and_then(Value::as_bool).unwrap_or(false) { flags |= 1 << 1; }
    if text.get("final_logit_softcapping").and_then(Value::as_f64).unwrap_or(0.0) != 0.0 { flags |= 1 << 2; }
    if text.get("num_kv_shared_layers").and_then(Value::as_u64).unwrap_or(0) != 0 { flags |= 1 << 3; }
    if text.get("use_double_wide_mlp").and_then(Value::as_bool).unwrap_or(false) { flags |= 1 << 4; }
    flags |= 1 << 5; // Gemma 4 four-norm decoder block
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(text.get("global_head_dim").and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
    out.extend_from_slice(&(text.get("per_layer_embeddings").and_then(|v| v.get("hidden_size")).and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
    out.extend_from_slice(&(text.get("num_kv_shared_layers").and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    for layer in layers {
        let kind = if layer.get("attention_type").and_then(Value::as_str) == Some("full_attention") { 1u32 } else { 0u32 };
        let rope_type = if layer.get("rope_type").and_then(Value::as_str) == Some("proportional") { ROPE_PROPORTIONAL } else { ROPE_DEFAULT };
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&(layer.get("sliding_window").and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
        out.extend_from_slice(&(layer.get("head_dim").and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
        out.extend_from_slice(&(layer.get("intermediate_size").and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
        out.extend_from_slice(&rope_type.to_le_bytes());
        // flags por capa: bit 0 = K_EQ_V (la capa no tiene v_proj).
        let mut layer_flags = 0u32;
        if layer.get("k_eq_v").and_then(Value::as_bool).unwrap_or(false) {
            layer_flags |= GEMMA4_LAYER_FLAG_K_EQ_V;
        }
        out.extend_from_slice(&layer_flags.to_le_bytes());
        out.extend_from_slice(&(layer.get("rope_theta").and_then(Value::as_f64).unwrap_or(10_000.0) as f32).to_le_bytes());
        out.extend_from_slice(&(layer.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0) as f32).to_le_bytes());
        out.extend_from_slice(&(-1i32).to_le_bytes());
        // El antiguo `reserved` pasa a llevar los KV heads de la capa: en
        // «unified» las globales tienen 1 y las deslizantes 8.  0 = sin dato,
        // el runtime cae al valor global (HNF anteriores a este campo).
        out.extend_from_slice(&(layer.get("num_key_value_heads").and_then(Value::as_u64).unwrap_or(0) as u32).to_le_bytes());
    }
    out
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn refuses_moe_hints_in_dense_binary_contract() {
        let t = serde_json::json!({"arch":"gemma4","moe_enabled":true});
        assert!(build_execution_hints_binary(&t).is_err());
        assert!(build_execution_hints_binary(&serde_json::json!({"text":t})).is_err());
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<ExecutionHintsBin>(), 64);
        assert_eq!(ExecutionHintsBin::SIZE, 64);
    }
    
    #[test]
    fn test_text_config_size() {
        assert_eq!(std::mem::size_of::<TextModelConfigBin>(), 128);
        assert_eq!(TextModelConfigBin::SIZE, 128);
    }
    
    #[test]
    fn test_vision_config_size() {
        assert_eq!(std::mem::size_of::<VisionModelConfigBin>(), 64);
        assert_eq!(VisionModelConfigBin::SIZE, 64);
    }
    
    #[test]
    fn test_build_binary() {
        let json = serde_json::json!({
            "arch": "qwen2",
            "num_hidden_layers": 24,
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_key_value_heads": 2
        });
        
        let binary = build_execution_hints_binary(&json).unwrap();
        
        // Verificar magic
        assert_eq!(&binary[0..4], &HINTS_MAGIC.to_le_bytes());
        
        // Verificar tamaño mínimo: header(64) + text(128) = 192, padded to 224
        assert!(binary.len() >= 192);
        assert_eq!(binary.len() % 32, 0);
    }

    #[test]
    fn test_gemma4_layer_extension() {
        let json = serde_json::json!({
            "text_enabled": true,
            "text": {
                "arch": "gemma4",
                "num_hidden_layers": 1,
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "vocab_size": 262144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "per_layer_embeddings": { "enabled": true, "hidden_size": 256 },
                "layer_scalar": true,
                "layer_configs": [{
                    "attention_type": "full_attention",
                    "sliding_window": 0,
                    "head_dim": 512,
                    "intermediate_size": 12288,
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25
                }]
            }
        });
        let binary = build_execution_hints_binary(&json).unwrap();
        let extension_offset = u32::from_le_bytes(binary[44..48].try_into().unwrap()) as usize;
        assert_eq!(&binary[52..56], b"GM4X");
        assert_eq!(&binary[extension_offset..extension_offset + 4], b"GM4X");
        assert_eq!(u32::from_le_bytes(binary[extension_offset + 8..extension_offset + 12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(binary[extension_offset + 32..extension_offset + 36].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(binary[extension_offset + 40..extension_offset + 44].try_into().unwrap()), 512);
    }

    fn gemma4_vision_json() -> Value {
        serde_json::json!({
            "encoder_type": "gemma4",
            "image_size": 0,
            "patch_size": 16,
            "hidden_size": 768,
            "num_hidden_layers": 16,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "intermediate_size": 3072,
            "num_channels": 3,
            "layer_norm_eps": 1e-6,
            "rms_norm_eps": 1e-6,
            "position_embedding_size": 10240,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "num_image_tokens": 280,
            "projection_dim": 1536,
            "projector_type": "linear",
            "image_token_id": 258880,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "pad_token_id": 0,
            "rope_theta": 100.0,
            "attention_scale": 1.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "use_clipped_linears": true,
            "standardize": false,
            "dynamic_image_tokens": true,
            "qk_norm_weighted": true,
            "v_norm_weighted": false,
            "pool_norm_weighted": false,
            "preprocess": {
                "do_convert_rgb": true,
                "do_normalize": false,
                "do_rescale": true,
                "do_resize": true,
                "rescale_factor": 1.0 / 255.0,
                "resample": 3,
                "antialias": true,
                "resize_multiple": 48,
                "patch_order": "hwc",
                "position_order": "xy"
            }
        })
    }

    #[test]
    fn test_gemma4_vision_only_binary_has_no_phantom_text() {
        let json = serde_json::json!({
            "vision_enabled": true,
            "vision": gemma4_vision_json(),
        });
        let binary = build_execution_hints_binary(&json).unwrap();
        assert_eq!(u32::from_le_bytes(binary[8..12].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(binary[32..34].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(binary[12..16].try_into().unwrap()), 64);
        assert_eq!(u16::from_le_bytes(binary[34..36].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(binary[64..68].try_into().unwrap()), 4);
        let extension_offset = u32::from_le_bytes(binary[56..60].try_into().unwrap()) as usize;
        assert_eq!(u32::from_le_bytes(binary[60..64].try_into().unwrap()), 96);
        assert_eq!(&binary[extension_offset..extension_offset + 4], b"GM4V");
        assert_eq!(u32::from_le_bytes(binary[extension_offset + 16..extension_offset + 20].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(binary[extension_offset + 28..extension_offset + 32].try_into().unwrap()), 280);
    }

    #[test]
    fn test_gm4x_and_gm4v_pointers_do_not_overlap() {
        let json = serde_json::json!({
            "text": {
                "arch": "gemma4",
                "num_hidden_layers": 0,
                "layer_configs": []
            },
            "vision": gemma4_vision_json(),
        });
        let binary = build_execution_hints_binary(&json).unwrap();
        let text_extension = u32::from_le_bytes(binary[44..48].try_into().unwrap()) as usize;
        let vision_extension = u32::from_le_bytes(binary[56..60].try_into().unwrap()) as usize;
        assert_eq!(&binary[52..56], b"GM4X");
        assert_eq!(&binary[text_extension..text_extension + 4], b"GM4X");
        assert_eq!(&binary[vision_extension..vision_extension + 4], b"GM4V");
        assert!(vision_extension > text_extension);
    }

    #[test]
    fn test_incomplete_gemma4_vision_contract_is_rejected() {
        let json = serde_json::json!({
            "vision": {
                "encoder_type": "gemma4",
                "hidden_size": 768
            }
        });
        assert!(build_execution_hints_binary(&json).is_err());
    }
}
