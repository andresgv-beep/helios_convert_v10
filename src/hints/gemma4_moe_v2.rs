//! GM4X v2 contract adopted by Hector-engine at 7cabb92.
//! Full-model production HNF emission remains separate and disabled for MoE.
use anyhow::{ensure, Context, Result};
use serde_json::Value;

pub const HEADER_BYTES: usize = 64;
pub const LAYER_BYTES: usize = 208;
pub const DESCRIPTOR_BYTES: usize = 48;
/// Header offsets; values describe expert-major order and gate then up per expert.
pub const EXPERT_ORDER_OFFSET: usize = 48;
pub const EXPERT_ORDER_EXPERT_MAJOR: u32 = 1;
pub const GATE_UP_ORDER_OFFSET: usize = 52;
pub const GATE_UP_ORDER_GATE_THEN_UP: u32 = 1;
/// Layer-relative offset. Bit (role - 1): gate_up, expert down, shared down.
pub const DESCRIPTOR_MASK_OFFSET: usize = 52;
pub const DESCRIPTOR_MASK_ALL: u32 = (1 << (1 - 1)) | (1 << (2 - 1)) | (1 << (3 - 1));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage { Fp16, Bf16, Fp32, Hq44, Hq8 }
impl Storage {
    pub fn code(self) -> u32 { match self {Self::Fp16=>0,Self::Bf16=>1,Self::Fp32=>2,Self::Hq44=>3,Self::Hq8=>4} }
    /// Descriptor layout field: 0 floating point, 1 HQ44, 2 HQ8K.
    pub fn layout(self) -> u32 { match self {Self::Hq44=>1,Self::Hq8=>2,_=>0} }
    fn row_bytes(self, cols: u32) -> u64 {
        match self {Self::Fp16|Self::Bf16=>cols as u64*2,Self::Fp32=>cols as u64*4,Self::Hq44=>cols as u64/256*152,
            Self::Hq8=>cols as u64/256*crate::hqs::HQ8K_BLOCK_SIZE as u64}
    }
}
fn u32_at(b: &mut [u8], o: usize, v: u32) {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
fn u64_at(b: &mut [u8], o: usize, v: u64) {b[o..o+8].copy_from_slice(&v.to_le_bytes());}
fn field(v: &Value, key: &str) -> Result<u32> {
    let n=v[key].as_u64().with_context(|| format!("missing {key}"))?;
    ensure!(n>0 && n<=u32::MAX as u64,"invalid {key}");Ok(n as u32)
}
/// Validated physical layout shared by GM4X v2 headers and payload writers.
/// Fields are private so byte counts cannot drift after construction.
#[derive(Clone, Debug)]
pub struct MatrixLayout {
    role: u32,
    storage: Storage,
    rows: u32,
    logical_cols: u32,
    physical_cols: u32,
    experts: u32,
    row_bytes: u64,
    expert_stride: u64,
    tensor_bytes: u64,
}

impl MatrixLayout {
    pub fn new(role: u32, storage: Storage, rows: u32, logical_cols: u32, experts: u32) -> Result<Self> {
        ensure!((1..=3).contains(&role), "unknown matrix role");
        ensure!(rows > 0 && logical_cols > 0 && experts > 0, "empty matrix layout");
        ensure!(role != 1 || rows % 2 == 0, "gate_up requires equal gate/up halves");
        ensure!(role != 3 || experts == 1, "shared down must have one matrix");
        // Los kernels MoE del motor solo leen expertos FP16 o HQ44.
        ensure!(storage != Storage::Hq8 || role == 3, "HQ8K is only for the shared down");
        let physical_cols = logical_cols.checked_add(255).context("padding overflow")? / 256 * 256;
        let row_bytes = storage.row_bytes(physical_cols);
        let expert_stride = row_bytes.checked_mul(rows as u64).context("expert stride overflow")?;
        let tensor_bytes = expert_stride.checked_mul(experts as u64).context("tensor size overflow")?;
        Ok(Self { role, storage, rows, logical_cols, physical_cols, experts,
            row_bytes, expert_stride, tensor_bytes })
    }

    pub fn storage(&self) -> Storage { self.storage }
    pub fn rows(&self) -> u32 { self.rows }
    pub fn logical_cols(&self) -> u32 { self.logical_cols }
    pub fn physical_cols(&self) -> u32 { self.physical_cols }
    pub fn experts(&self) -> u32 { self.experts }
    pub fn row_bytes(&self) -> u64 { self.row_bytes }
    pub fn expert_stride(&self) -> u64 { self.expert_stride }
    pub fn tensor_bytes(&self) -> u64 { self.tensor_bytes }

    /// Absolute range, checked against the containing file/buffer and u64 overflow.
    pub fn expert_range(&self, tensor_offset: u64, expert: u32, container_bytes: u64)
        -> Result<std::ops::Range<u64>> {
        ensure!(expert < self.experts, "expert index out of range");
        let tensor_end = tensor_offset.checked_add(self.tensor_bytes).context("tensor offset overflow")?;
        ensure!(tensor_end <= container_bytes, "tensor exceeds container");
        let start = tensor_offset.checked_add(self.expert_stride.checked_mul(expert as u64)
            .context("expert offset overflow")?).context("expert offset overflow")?;
        let end = start.checked_add(self.expert_stride).context("expert end overflow")?;
        Ok(start..end)
    }

    pub fn to_bytes(&self) -> [u8; DESCRIPTOR_BYTES] {
        let mut b = [0u8; DESCRIPTOR_BYTES];
        for (offset, value) in [(0, self.role), (4, self.storage.code()), (8, self.rows),
            (12, self.logical_cols), (16, self.rows), (20, self.physical_cols),
            (24, self.experts), (28, self.storage.layout())] {
            u32_at(&mut b, offset, value);
        }
        u64_at(&mut b, 32, self.row_bytes);
        u64_at(&mut b, 40, self.expert_stride);
        b
    }
}

fn descriptor(role: u32, dtype: Storage, rows: u32, logical: u32, count: u32) -> Result<[u8; 48]> {
    Ok(MatrixLayout::new(role, dtype, rows, logical, count)?.to_bytes())
}

/// The three storages per layer are gate_up, expert down, shared down.
/// Source logical shapes remain authoritative; physical columns include zero padding.
pub fn build_extension_v2(text:&Value, storage:&[[Storage;3]]) -> Result<Vec<u8>> {
    ensure!(text["arch"]=="gemma4" && text["moe_enabled"]==true,"requires Gemma 4 MoE hints");
    let layers=text["layer_configs"].as_array().context("missing layer_configs")?;
    let n=field(text,"num_hidden_layers")? as usize;
    ensure!(n==layers.len() && n==storage.len(),"layer/storage count mismatch");
    let e=field(text,"num_experts")?;let k=field(text,"num_experts_per_tok")?;
    ensure!(k<=e,"top-k exceeds experts");
    let i=field(text,"moe_intermediate_size")?;let h=field(text,"hidden_size")?;
    let gate_rows=i.checked_mul(2).context("gate rows overflow")?;
    let old=super::binary::build_gemma4_extension(text);
    ensure!(old.len()==32+n*40,"invalid base GM4X");
    let size=n.checked_mul(LAYER_BYTES).and_then(|v| v.checked_add(HEADER_BYTES)).context("extension overflow")?;
    let mut out=vec![0u8;size];out[..32].copy_from_slice(&old[..32]);
    out[4..6].copy_from_slice(&2u16.to_le_bytes());
    out[6..8].copy_from_slice(&(LAYER_BYTES as u16).to_le_bytes());
    let flags=u32::from_le_bytes(out[12..16].try_into().unwrap()) | (1<<6);
    u32_at(&mut out,12,flags);u32_at(&mut out,28,HEADER_BYTES as u32);
    u32_at(&mut out,32,DESCRIPTOR_BYTES as u32);u32_at(&mut out,36,3);
    u64_at(&mut out,40,size as u64);
    u32_at(&mut out,EXPERT_ORDER_OFFSET,EXPERT_ORDER_EXPERT_MAJOR);
    u32_at(&mut out,GATE_UP_ORDER_OFFSET,GATE_UP_ORDER_GATE_THEN_UP);
    for (idx,layer) in layers.iter().enumerate() {
        let start=HEADER_BYTES+idx*LAYER_BYTES;
        out[start..start+40].copy_from_slice(&old[32+idx*40..32+(idx+1)*40]);
        for (o,v) in [(40,e),(44,k),(48,i),(DESCRIPTOR_MASK_OFFSET,DESCRIPTOR_MASK_ALL),(56,3)] {u32_at(&mut out,start+o,v);}
        let shared=field(layer,"intermediate_size")?;
        let descriptors=[descriptor(1,storage[idx][0],gate_rows,h,e)?,
            descriptor(2,storage[idx][1],h,i,e)?,descriptor(3,storage[idx][2],h,shared,1)?];
        for (slot,d) in descriptors.iter().enumerate() {
            let at=start+64+slot*DESCRIPTOR_BYTES;out[at..at+48].copy_from_slice(d);
        }
    }
    Ok(out)
}

/// Explicit experimental text-only hints; dense/production callers still reject MoE.
pub fn build_experimental_hints(text: &Value, storage: &[[Storage; 3]]) -> Result<Vec<u8>> {
    use super::binary::{ExecutionHintsBin, TextModelConfigBin};
    let extension = build_extension_v2(text, storage)?;
    let mut header = ExecutionHintsBin::new();
    header.num_text_models = 1;
    header.flags = 1;
    header.text_offset = ExecutionHintsBin::SIZE as u32;
    let extension_offset = ExecutionHintsBin::SIZE + TextModelConfigBin::SIZE;
    header.reserved[..4].copy_from_slice(&(extension_offset as u32).to_le_bytes());
    header.reserved[4..8].copy_from_slice(&u32::try_from(extension.len())?.to_le_bytes());
    header.reserved[8..12].copy_from_slice(b"GM4X");
    let mut bytes = header.to_bytes().to_vec();
    bytes.extend_from_slice(&TextModelConfigBin::from_json(text).to_bytes());
    bytes.extend_from_slice(&extension);
    bytes.resize((bytes.len()+31)/32*32, 0);
    Ok(bytes)
}

/// Compatibility alias for callers and fixture tooling created before adoption.
pub use build_extension_v2 as build_candidate;

#[cfg(test)]
mod tests {
    use super::*;
    fn hints()->Value {serde_json::json!({"arch":"gemma4","moe_enabled":true,"num_hidden_layers":1,
        "num_experts":128,"num_experts_per_tok":8,"moe_intermediate_size":704,"hidden_size":2816,
        "global_head_dim":512,"layer_configs":[{"intermediate_size":2112,"head_dim":256,
            "attention_type":"sliding_attention","sliding_window":1024,"rope_type":"default",
            "rope_theta":10000.0,"partial_rotary_factor":1.0,"num_key_value_heads":8}]})}
    fn u32v(b:&[u8],o:usize)->u32 {u32::from_le_bytes(b[o..o+4].try_into().unwrap())}
    fn u64v(b:&[u8],o:usize)->u64 {u64::from_le_bytes(b[o..o+8].try_into().unwrap())}
    #[test] fn fixed_offsets_and_padding() {
        let b=build_candidate(&hints(),&[[Storage::Bf16;3]]).unwrap();
        assert_eq!(b.len(),272);assert_eq!(&b[..6],b"GM4X\x02\x00");
        assert_eq!(u32v(&b,EXPERT_ORDER_OFFSET),EXPERT_ORDER_EXPERT_MAJOR);
        assert_eq!(u32v(&b,GATE_UP_ORDER_OFFSET),GATE_UP_ORDER_GATE_THEN_UP);
        assert_eq!(u32v(&b,64+DESCRIPTOR_MASK_OFFSET),7);
        assert_eq!(u32v(&b,28),64);assert_eq!(u64v(&b,40),272);
        assert_eq!(u32v(&b,64+40),128);assert_eq!(u32v(&b,64+44),8);
        assert_eq!(u32v(&b,64+112+12),704);assert_eq!(u32v(&b,64+112+20),768);
        assert_eq!(u64v(&b,64+112+32),1536);assert_eq!(u64v(&b,64+112+40),4325376);
        assert_eq!(u32v(&b,64+160+12),2112);assert_eq!(u32v(&b,64+160+20),2304);
    }
    #[test] fn independent_dtype_per_matrix() {
        let b=build_candidate(&hints(),&[[Storage::Hq44,Storage::Fp32,Storage::Fp16]]).unwrap();
        assert_eq!(u64v(&b,64+64+32),11*152);assert_eq!(u64v(&b,64+112+32),768*4);
        assert_eq!(u32v(&b,64+64+28),1);assert_eq!(u32v(&b,64+112+28),0);
    }
    #[test] fn rejects_bad_topology_and_overflow() {
        assert!(build_candidate(&hints(),&[]).is_err());
        let mut h=hints();h["num_experts_per_tok"]=129.into();assert!(build_candidate(&h,&[[Storage::Bf16;3]]).is_err());
        let mut h=hints();h["hidden_size"]=u32::MAX.into();assert!(build_candidate(&h,&[[Storage::Bf16;3]]).is_err());
    }
}
