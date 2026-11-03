//! Source inventory shared by the read-only CLI and experimental full text export.
use anyhow::{bail, ensure, Context, Result};
use crate::mapping::{gemma4::Gemma4Mapper, gemma4_moe::MoeSpec};
use serde_json::{json, Value};
use std::{collections::BTreeSet, io::Read, path::Path};

pub fn inventory(path: &Path) -> Result<Value> {
    let config: Value=serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let spec=MoeSpec::from_text_config(&config["text_config"])?.context("not Gemma 4 MoE")?;
    let index: Value=serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json"))?)?;
    let weights=index["weight_map"].as_object().context("missing weight_map")?;
    let required=spec.required_tensor_shapes();
    for (name, _) in &required { ensure!(weights.contains_key(name), "missing MoE tensor in index: {name}"); }
    let names=weights.keys().cloned().collect::<Vec<_>>();
    let mappings=Gemma4Mapper::inventory_mappings(&config,&names)?;
    let shards=weights.values().map(|v| v.as_str().context("invalid shard name"))
        .collect::<Result<BTreeSet<_>>>()?;
    let mut pending=Vec::new(); let mut verified=0usize;
    for shard in shards {
        ensure!(std::path::Path::new(shard).components().count()==1 && !shard.starts_with('.'), "invalid shard path");
        let file_path=path.join(shard);
        if !file_path.exists() { pending.push(shard); continue; }
        let mut file=std::fs::File::open(&file_path)?;
        let size=file.metadata()?.len();
        let mut n=[0u8;8];file.read_exact(&mut n)?;
        let n=u64::from_le_bytes(n);
        ensure!(n<=64*1024*1024 && n+8<=size,"invalid header in {shard}");
        let mut bytes=vec![0u8;n as usize];file.read_exact(&mut bytes)?;
        let header:Value=serde_json::from_slice(&bytes)?;
        let entries=header.as_object().context("invalid tensor header")?;
        let mut ranges=Vec::new();
        for (name, info) in entries {
            if name=="__metadata__" {continue;}
            ensure!(weights.get(name).and_then(Value::as_str)==Some(shard),"tensor not assigned to shard: {name}");
            let offsets=info["data_offsets"].as_array().context("missing data offsets")?;
            ensure!(offsets.len()==2,"invalid offsets for {name}");
            let start=offsets[0].as_u64().context("invalid start")?;
            let end=offsets[1].as_u64().context("invalid end")?;
            ensure!(start<=end && end<=size-8-n,"truncated/invalid tensor: {name}");
            let dims=info["shape"].as_array().context("missing shape")?;
            let mut count=1u64;
            for d in dims {count=count.checked_mul(d.as_u64().context("invalid dimension")?).context("shape overflow")?;}
            let width=match info["dtype"].as_str() {Some("BF16"|"F16")=>2,Some("F32")=>4,_=>bail!("unsupported source dtype: {name}")};
            ensure!(count.checked_mul(width)==Some(end-start),"byte count mismatch: {name}");
            ranges.push((start,end));
        }
        ranges.sort_unstable();let mut next=0;
        for (start,end) in ranges {ensure!(start==next,"gap or overlap in {shard}");next=end;}
        ensure!(next==size-8-n,"unexpected trailing bytes in {shard}");
        for (name, assigned) in weights {
            if assigned.as_str()==Some(shard) {ensure!(entries.contains_key(name),"indexed tensor absent: {name}");}
        }
        for (name,shape) in &required {
            if weights[name].as_str()!=Some(shard) {continue;}
            ensure!(header[name]["shape"]==json!(shape),"MoE shape mismatch: {name}");verified+=1;
        }
    }
    Ok(json!({
        "stage":"inventory_only_not_runtime_support", "source":path,
        "source_bytes":index["metadata"]["total_size"],"layers":spec.layers,
        "experts":spec.experts,"top_k":spec.top_k,"expert_width":spec.intermediate,
        "language_tensors_mapped":mappings.len(),"other_tensors":weights.len()-mappings.len(),
        "moe_tensors_expected":required.len(),"moe_shapes_verified":verified,"pending_shards":pending,
        "hqs_v4_down_row_aligned":spec.intermediate%256==0,
        "mappings":mappings.iter().map(|(src,m)|json!({"source":src,"canonical":format!("text.{}",m.canonical_name),
            "suggested_precision":format!("{:?}",m.quant_hint),"category":format!("{:?}",m.category)})).collect::<Vec<_>>()
    }))
}
