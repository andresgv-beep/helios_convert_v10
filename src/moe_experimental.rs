//! Explicit full-text MoE laboratory export; ordinary production conversion stays guarded.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path, io::Write};
use half::{bf16, f16};
use crate::{mapping::{ModelMapper, gemma4::Gemma4Mapper, gemma4_moe::MoeSpec, moe_inventory::inventory},
    SafetensorReader, hints::{gemma4_moe_v2::{MatrixLayout, Storage, build_experimental_hints},
    gemma4_moe_payload::encode_rows_with_importance}};

#[derive(Clone, Debug, serde::Serialize)]
pub struct PlannedTensor {
    pub source: String,
    pub canonical: String,
    pub source_dtype: String,
    pub logical_shape: Vec<usize>,
    pub physical_shape: Vec<usize>,
    pub dtype: String,
    pub bytes: u64,
}

/// Formato de la atención y la MLP compartida (y del embedding con Hq8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseFormat { Hq44, Fp16, Hq8 }

pub struct Plan { pub tensors: Vec<PlannedTensor>, pub text: Value, pub inventory: Value, pub dense: DenseFormat }
impl Plan {
    pub fn report(&self) -> Value {
        json!({"kind":"experimental_gemma4_moe_text_plan","production":false,
            "calibration":"none_unit_importance","layers":self.text["num_hidden_layers"],
            "text_hints":self.text,"inventory":{
                "language_tensors_mapped":self.inventory["language_tensors_mapped"],
                "other_tensors_excluded":self.inventory["other_tensors"],
                "pending_shards":self.inventory["pending_shards"]},
            "total_tensor_bytes":self.tensors.iter().map(|t| t.bytes).sum::<u64>(),"tensors":self.tensors})
    }
    pub fn binary_hints(&self) -> Result<Vec<u8>> {
        let count = self.text["num_hidden_layers"].as_u64().context("missing layer count")? as usize;
        let shared = match self.dense {DenseFormat::Fp16=>Storage::Fp16, DenseFormat::Hq8=>Storage::Hq8, DenseFormat::Hq44=>Storage::Hq44};
        build_experimental_hints(&self.text, &vec![[Storage::Hq44,Storage::Hq44,shared]; count])
    }
}

pub fn make_plan(model: &Path) -> Result<Plan> {make_plan_with(model, DenseFormat::Hq44)}

/// Atención y MLP compartida densas en FP16, sin cuantizar: medido sobre los pesos
/// originales, son las piezas que más alejan el modelo del checkpoint.
fn dense(name: &str) -> bool {
    (name.contains(".attn.") && name.ends_with("_proj.weight")) || name.contains(".mlp.")
}

pub fn make_plan_with(model: &Path, dense_format: DenseFormat) -> Result<Plan> {
    let audit = inventory(model)?;
    ensure!(audit["pending_shards"].as_array().context("invalid inventory")?.is_empty(), "source download is incomplete");
    let config: Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
    let spec = MoeSpec::from_text_config(&config["text_config"])?.context("requires Gemma4 MoE")?;
    ensure!(config["text_config"]["tie_word_embeddings"] == true, "experimental profile requires tied embedding");
    ensure!(config["text_config"]["hidden_size_per_layer_input"].as_u64().unwrap_or(0) == 0,
        "PLE is outside this experimental profile");
    ensure!(model.join("tokenizer.json").is_file(), "full experimental HNF requires tokenizer.json");
    ensure!(config["model_type"] == "gemma4", "requires Gemma4 config");
    let mapper = Gemma4Mapper::from_model_path(model, &config)?;
    ensure!(mapper.is_moe(), "expected a MoE mapper");
    let reader = SafetensorReader::from_folder(model)?;
    let mut text = mapper.execution_hints();
    let mut tensors = Vec::new();
    let mut shapes = BTreeMap::new();
    for (source, info) in reader.iter_tensors() {
        let Some(mapping) = mapper.map_tensor(source) else {continue};
        let name = mapping.canonical_name;
        ensure!(crate::dictionary::validate_tensor_name(&name), "unregistered tensor: {name}");
        ensure!(shapes.insert(name.clone(),info.shape.clone()).is_none(), "duplicate canonical tensor");
        ensure!(matches!(info.dtype.as_str(), "BF16"|"F16"|"F32"), "unsupported source dtype");
        ensure!(!info.shape.is_empty() && info.shape.iter().all(|n| *n>0), "invalid source shape");
        let mut physical = info.shape.clone();
        let dtype = if name == "token_embedding.weight" {
            ensure!(physical.len()==2 && physical[1]%256==0, "shared embedding requires aligned rows");
            if dense_format == DenseFormat::Hq8 {"hq8k"} else {"hq52k"}
        } else if name.contains(".moe.router.") || physical.len()==1 {"fp16"}
        else if dense_format != DenseFormat::Hq44 && dense(&name) {
            ensure!(physical.len()==2, "dense format expects a matrix");
            // GM4X v2 redondea a 256 columnas en todos los storages, también FP16.
            let last = physical.last_mut().unwrap();
            *last = last.checked_add(255).context("padding overflow")?/256*256;
            if dense_format == DenseFormat::Hq8 {"hq8k"} else {"fp16"}
        }
        else {
            ensure!(physical.len()==2 || (physical.len()==3 && name.contains(".moe.experts.")), "unexpected matrix rank");
            let last = physical.last_mut().unwrap();
            *last = last.checked_add(255).context("padding overflow")?/256*256;
            "hq44k_g16"
        };
        let count = physical.iter().try_fold(1u64,|a,b| a.checked_mul(*b as u64)).context("numel overflow")?;
        let bytes = match dtype {
            "fp16"=>count.checked_mul(2),
            "hq52k"=> (count/256).checked_mul(crate::hqs::HQ52K_BLOCK_SIZE as u64),
            "hq8k"=> (count/256).checked_mul(crate::hqs::HQ8K_BLOCK_SIZE as u64),
            _=>(count/256).checked_mul(crate::hqs::v4::BLOCK_BYTES as u64),
        }.context("tensor byte overflow")?;
        tensors.push(PlannedTensor {source:source.into(),canonical:format!("text.{name}"),source_dtype:info.dtype.clone(),
            logical_shape:info.shape.clone(),physical_shape:physical,dtype:dtype.into(),bytes});
    }
    let need = |name: &str, expected: &[usize]| -> Result<()> {
        ensure!(shapes.get(name).map(Vec::as_slice)==Some(expected), "missing or wrong shape: {name}, expected {expected:?}");Ok(())
    };
    let embedding = shapes.get("token_embedding.weight").context("missing embedding")?;
    ensure!(embedding.len()==2 && embedding[1]==spec.hidden, "embedding shape mismatch");
    ensure!(!shapes.contains_key("lm_head.weight"), "tied embedding must not duplicate the output head");
    need("final_norm.weight",&[spec.hidden])?;
    let heads = text["num_attention_heads"].as_u64().context("missing attention heads")? as usize;
    for layer in 0..spec.layers {
        let prefix = format!("layer{layer}.");
        let cfg = &text["layer_configs"][layer];
        let head = cfg["head_dim"].as_u64().context("missing head_dim")? as usize;
        let kv = cfg["num_key_value_heads"].as_u64().context("missing kv heads")? as usize;
        let shared = cfg["intermediate_size"].as_u64().context("missing shared width")? as usize;
        for norm in ["ln_attn_in.weight","ln_attn_post.weight","ln_mlp_in.weight","ln_mlp_post.weight",
            "ln_moe_in.weight","ln_mlp_branch_post.weight","ln_moe_branch_post.weight"] {
            need(&format!("{prefix}{norm}"),&[spec.hidden])?;
        }
        need(&format!("{prefix}layer_scalar"),&[1])?;
        for norm in ["q_norm","k_norm"] {need(&format!("{prefix}attn.{norm}.weight"),&[head])?;}
        need(&format!("{prefix}attn.q_proj.weight"),&[heads*head,spec.hidden])?;
        need(&format!("{prefix}attn.k_proj.weight"),&[kv*head,spec.hidden])?;
        if cfg["k_eq_v"] != true {need(&format!("{prefix}attn.v_proj.weight"),&[kv*head,spec.hidden])?;}
        need(&format!("{prefix}attn.o_proj.weight"),&[spec.hidden,heads*head])?;
        for part in ["gate","up"] {need(&format!("{prefix}mlp.{part}.weight"),&[shared,spec.hidden])?;}
        need(&format!("{prefix}mlp.down.weight"),&[spec.hidden,shared])?;
    }
    ensure!(tensors.len() == audit["language_tensors_mapped"].as_u64().unwrap() as usize, "inventory/mapping mismatch");
    text["vocab_size"] = json!(embedding[0]);
    text["dtype"] = json!("fp16"); // compute precision, not packed weight storage
    let plan = Plan {tensors,text,inventory:audit,dense:dense_format};
    plan.binary_hints()?;
    Ok(plan)
}

fn source_value(raw: &[u8], dtype: &str, index: usize) -> f32 {
    if dtype=="F32" {let at=index*4;f32::from_le_bytes(raw[at..at+4].try_into().unwrap())}
    else {let at=index*2;let bytes=[raw[at],raw[at+1]];
        if dtype=="BF16" {bf16::from_le_bytes(bytes).to_f32()} else {f16::from_le_bytes(bytes).to_f32()}}
}

/// Bounded batches; never flatten rank-three logical rows across padding.
pub fn emit_tensor(writer: &mut dyn Write, tensor: &PlannedTensor, raw: &[u8], batch_rows: usize) -> Result<()> {
    emit_tensor_with_importance(writer, tensor, raw, batch_rows, None)
}

/// `importance`: HQ44 experts only, [experts, physical columns] as one slice; a batch
/// never crosses an expert, so every row is weighted by its own expert.
pub fn emit_tensor_with_importance(writer: &mut dyn Write, tensor: &PlannedTensor, raw: &[u8], batch_rows: usize,
                                   importance: Option<&[f32]>) -> Result<()> {
    ensure!(batch_rows > 0 && batch_rows <= 4096, "invalid row batch");
    ensure!(importance.is_none() || (tensor.dtype=="hq44k_g16" && tensor.physical_shape.len()==3),
        "importance only applies to rank-three HQ44 experts");
    let count = tensor.logical_shape.iter().product::<usize>();
    let source_width = if tensor.source_dtype=="F32" {4} else {2};
    ensure!(raw.len()==count*source_width,"source byte count mismatch");
    let padded = tensor.logical_shape.last()!=tensor.physical_shape.last();
    if tensor.dtype=="fp16" && padded {
        let logical = *tensor.logical_shape.last().unwrap();
        let physical = *tensor.physical_shape.last().unwrap();
        for row in 0..count/logical {
            let mut bytes=vec![0u8;physical*2];
            for col in 0..logical {
                let x=source_value(raw,&tensor.source_dtype,row*logical+col);let y=f16::from_f32(x);
                ensure!(x.is_finite() && y.is_finite(),"nonfinite or overflowing FP16 source");
                bytes[col*2..col*2+2].copy_from_slice(&y.to_le_bytes());
            }
            writer.write_all(&bytes)?;
        }
        return Ok(());
    }
    if tensor.dtype=="fp16" {
        for start in (0..count).step_by(batch_rows*256) {
            let end = count.min(start+batch_rows*256);
            let mut bytes=Vec::with_capacity((end-start)*2);
            for index in start..end {
                let x=source_value(raw,&tensor.source_dtype,index);let y=f16::from_f32(x);
                ensure!(x.is_finite() && y.is_finite(),"nonfinite or overflowing FP16 source");
                bytes.extend_from_slice(&y.to_le_bytes());
            }
            writer.write_all(&bytes)?;
        }
        return Ok(());
    }
    let logical = *tensor.logical_shape.last().unwrap();
    let physical = *tensor.physical_shape.last().unwrap();
    let rows = count/logical;
    let layout=MatrixLayout::new(2,Storage::Hq44,1,u32::try_from(logical)?,1)?;
    let expert_rows = if tensor.logical_shape.len()==3 {tensor.logical_shape[1]} else {rows};
    if let Some(imp)=importance {
        ensure!(imp.len()==tensor.physical_shape[0]*physical, "importance must be [experts, physical columns]");
    }
    let mut first=0;
    while first<rows {
        let n=batch_rows.min(rows-first).min(expert_rows-first%expert_rows);
        let mut values=vec![0f32;n*physical];
        for row in 0..n {for col in 0..logical {
            values[row*physical+col]=source_value(raw,&tensor.source_dtype,(first+row)*logical+col);
        }}
        ensure!(values.iter().all(|v|v.is_finite()),"nonfinite source matrix");
        let encoded=if tensor.dtype=="hq8k" {
            let bytes=crate::hqs::quantize_hq8k(&values);
            ensure!(crate::hqs::dequantize_hq8k(&bytes,values.len()).iter().all(|v|v.is_finite()),"HQ8K nonfinite result");
            bytes
        } else if tensor.dtype=="hq52k" {
            let bytes=crate::hqs::quantize_hq52k(&values);
            ensure!(crate::hqs::dequantize_hq52k(&bytes,values.len()).iter().all(|v|v.is_finite()),"HQ52 nonfinite result");
            bytes
        } else {
            let expert=first/expert_rows;
            encode_rows_with_importance(&layout,&values,importance.map_or(&[][..],|imp|&imp[expert*physical..(expert+1)*physical]))?
        };
        writer.write_all(&encoded)?;
        first+=n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn importance_is_per_expert_and_batches_never_cross_experts() {
        let (experts, rows, cols) = (2usize, 5usize, 256usize);
        let values: Vec<f32> = (0..experts*rows*cols).map(|i| ((i as f32*0.37).sin()*0.02) + if i%41==0 {0.1} else {0.0}).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect();
        let exact: Vec<f32> = values.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
        let tensor = PlannedTensor {source:"s".into(),canonical:"text.layer0.moe.experts.down.weight".into(),source_dtype:"BF16".into(),
            logical_shape:vec![experts,rows,cols],physical_shape:vec![experts,rows,cols],dtype:"hq44k_g16".into(),bytes:0};
        let mut importance = vec![1.0f32; experts*cols];
        for c in 0..cols { importance[cols+c] = 1.0 + (c % 7) as f32 * 3.0; }
        let mut by_three = Vec::new();
        emit_tensor_with_importance(&mut by_three, &tensor, &raw, 3, Some(&importance)).unwrap();
        let mut by_one = Vec::new();
        emit_tensor_with_importance(&mut by_one, &tensor, &raw, 1, Some(&importance)).unwrap();
        assert_eq!(by_three, by_one);
        let half = rows*cols;
        let unit = crate::hqs::v4::quantize(&exact[..half], cols, &[]).unwrap();
        let weighted = crate::hqs::v4::quantize(&exact[half..], cols, &importance[cols..]).unwrap();
        assert_eq!(&by_three[..unit.len()], &unit[..]);
        assert_eq!(&by_three[unit.len()..], &weighted[..]);
        assert_ne!(weighted, crate::hqs::v4::quantize(&exact[half..], cols, &[]).unwrap());
        let mut plain = Vec::new();
        emit_tensor(&mut plain, &tensor, &raw, 3).unwrap();
        assert_eq!(&plain[unit.len()..], &crate::hqs::v4::quantize(&exact[half..], cols, &[]).unwrap()[..]);
    }
}
